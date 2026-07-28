import json
import os
import re
import io
import uuid
import contextlib
import traceback
from datetime import datetime, timezone

import requests
import pandas as pd
from bs4 import BeautifulSoup
from openai import OpenAI
from duckduckgo_search import DDGS

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

MAX_STEPS = 10

SYSTEM_PROMPT = """You are a careful data-analysis agent.

You will be given one or more chat messages from a user. Only the LAST message
is the actual question you must answer; earlier messages are context for a
multi-turn conversation.

The question will specify an exact JSON reply shape, something like:
  Reply with ONLY this JSON object: {"answer": <...>, "log_url": "<...>"}

Your job:
1. Figure out what the "answer" field must look like (its keys/shape) from
   the question text.
2. Actually work out the correct value using the tools available to you
   (fetch_url, web_search, run_python). Use MOSPI (mospi.gov.in) or other
   public sources when the question references public datasets. If data is
   given inline in the question, use that directly instead of searching.
3. When you are confident in the final answer, respond with ONLY a JSON
   object of the form:
     {"answer": <value shaped exactly as the question requires>}
   Do NOT include "log_url" yourself - the system adds it afterward.
   Do NOT wrap it in markdown code fences. Do NOT add any explanation text
   outside the JSON object in your final message.

Be precise about data types (numbers vs strings), rounding, and key names -
match whatever the question's example shape uses exactly.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return a list of {title, url, snippet}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch a URL. If it's an HTML page, returns cleaned visible "
                "text plus any HTML tables found (as CSV text). If it's a "
                "CSV/XLS/XLSX file, returns a preview and a local file path "
                "you can pass to run_python to load with pandas."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code for data analysis. pandas (pd), numpy "
                "(np), json, re, math are pre-imported. Use print() to "
                "produce output - only stdout is returned to you. Files "
                "downloaded via fetch_url are available at the paths it "
                "returned. There is no internet access inside run_python; "
                "use fetch_url/web_search for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
]

_download_dir = "downloads"
os.makedirs(_download_dir, exist_ok=True)


def _tool_web_search(query: str, max_results: int = 5):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
            for r in results
        ]
    except Exception as e:
        return {"error": f"web_search failed: {e}"}


def _tool_fetch_url(url: str):
    try:
        resp = requests.get(
            url,
            timeout=25,
            headers={"User-Agent": "Mozilla/5.0 (data-analysis-bot)"},
        )
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")

        # Binary / tabular files -> save locally, let run_python load them
        if any(ext in url.lower() for ext in [".xlsx", ".xls", ".csv"]) or \
           "spreadsheet" in content_type or "excel" in content_type or "csv" in content_type:
            ext = ".xlsx" if ".xlsx" in url.lower() else (
                ".xls" if ".xls" in url.lower() else ".csv"
            )
            fname = os.path.join(_download_dir, f"{uuid.uuid4().hex}{ext}")
            with open(fname, "wb") as f:
                f.write(resp.content)
            preview = ""
            try:
                if ext == ".csv":
                    df = pd.read_csv(fname, nrows=10)
                else:
                    df = pd.read_excel(fname, nrows=10)
                preview = df.to_string()
            except Exception as e:
                preview = f"(could not preview: {e})"
            return {
                "type": "file",
                "local_path": fname,
                "preview": preview,
            }

        # HTML -> text + tables
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()

        tables_csv = []
        try:
            dfs = pd.read_html(io.StringIO(resp.text))
            for i, df in enumerate(dfs[:5]):
                tables_csv.append({"table_index": i, "csv": df.to_csv(index=False)})
        except ValueError:
            pass

        return {
            "type": "html",
            "text": text[:8000],
            "tables": tables_csv,
        }
    except Exception as e:
        return {"error": f"fetch_url failed: {e}"}


def _tool_run_python(code: str):
    g = {"pd": pd, "json": json, "re": re, "math": __import__("math")}
    try:
        import numpy as np
        g["np"] = np
    except ImportError:
        pass
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, g)
        out = buf.getvalue()
        if len(out) > 6000:
            out = out[:6000] + "\n...[truncated]"
        return {"stdout": out}
    except Exception:
        return {"stdout": buf.getvalue(), "error": traceback.format_exc()[-2000:]}


TOOL_IMPL = {
    "web_search": _tool_web_search,
    "fetch_url": _tool_fetch_url,
    "run_python": _tool_run_python,
}


def _extract_json_object(text: str):
    """Pull the first {...} JSON object out of a string, tolerating code fences."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unbalanced JSON in model output")


def solve_question(question: str, history: list[str] | None = None, chat_id=None):
    """
    Runs a tool-using agent loop to answer `question`.
    Returns (result_dict, log_path) where result_dict = {"answer": ...}
    (log_url is added by the caller, who knows the public base URL).
    """
    run_id = str(uuid.uuid4())
    log_path = os.path.join(LOG_DIR, f"{run_id}.jsonl")

    def log(entry: dict):
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        entry["run_id"] = run_id
        if chat_id is not None:
            entry["chat_id"] = chat_id
        with open(log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    log({"event": "start", "question": question, "history": history or []})

    convo_context = ""
    if history and len(history) > 1:
        convo_context = (
            "Earlier messages in this conversation (for context only):\n"
            + "\n".join(f"- {h}" for h in history[:-1])
            + "\n\n"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{convo_context}Question to answer:\n{question}",
        },
    ]

    final_answer = None
    try:
        for step in range(MAX_STEPS):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    fname = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    log({"event": "tool_call", "step": step, "tool": fname, "args": args})
                    impl = TOOL_IMPL.get(fname)
                    result = impl(**args) if impl else {"error": f"unknown tool {fname}"}
                    log({"event": "tool_result", "step": step, "tool": fname, "result": result})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, default=str)[:6000],
                        }
                    )
                continue

            # No tool calls -> model believes it has the final answer
            content = msg.content or ""
            log({"event": "model_final_message", "step": step, "content": content})
            try:
                parsed = _extract_json_object(content)
                if "answer" in parsed:
                    final_answer = {"answer": parsed["answer"]}
                else:
                    final_answer = {"answer": parsed}
                break
            except Exception as e:
                log({"event": "parse_error", "step": step, "error": str(e)})
                # Nudge the model to comply and try once more
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last message was not a single valid JSON object "
                            'of the form {"answer": ...}. Reply again with ONLY '
                            "that JSON object, no markdown, no explanation."
                        ),
                    }
                )
                continue

        if final_answer is None:
            final_answer = {
                "answer": {"error": "Agent could not produce a final answer in time"}
            }
            log({"event": "give_up"})

    except Exception as e:
        log({"event": "fatal_error", "error": str(e), "trace": traceback.format_exc()[-2000:]})
        final_answer = {"answer": {"error": str(e)}}

    log({"event": "done", "final_answer": final_answer})
    return final_answer, log_path
