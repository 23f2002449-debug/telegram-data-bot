import os
import json
import uuid
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from telegram import Bot
from dotenv import load_dotenv

from agent import solve_question, LOG_DIR

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # e.g. https://your-app.onrender.com/webhook
BASE_URL = os.getenv("BASE_URL") or (
    WEBHOOK_URL.replace("/webhook", "") if WEBHOOK_URL else None
)

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not found in environment variables.")
if not BASE_URL:
    raise ValueError("BASE_URL (or WEBHOOK_URL) not set - needed to build log_url.")

bot = Bot(BOT_TOKEN)
app = FastAPI()

os.makedirs(LOG_DIR, exist_ok=True)
# Serve log files publicly at https://<BASE_URL>/logs/<file>.jsonl
app.mount("/logs", StaticFiles(directory=LOG_DIR), name="logs")

# chat_id -> list of recent message texts
conversation_history: dict[int, list[str]] = defaultdict(list)
MAX_HISTORY = 5


@app.on_event("startup")
async def set_webhook_on_startup():
    if WEBHOOK_URL:
        try:
            await bot.set_webhook(url=WEBHOOK_URL)
        except Exception as e:
            print(f"Warning: could not set webhook: {e}")


@app.get("/")
async def root():
    return {"status": "running"}


@app.post("/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()

    try:
        if "message" not in update:
            return JSONResponse({"ok": True})

        message = update["message"]
        chat_id = message["chat"]["id"]
        text = message.get("text", "")
        if not text:
            return JSONResponse({"ok": True})

        conversation_history[chat_id].append(text)
        conversation_history[chat_id] = conversation_history[chat_id][-MAX_HISTORY:]
        history = list(conversation_history[chat_id])

        result, log_path = solve_question(text, history=history, chat_id=chat_id)

        log_filename = os.path.basename(log_path)
        log_url = f"{BASE_URL.rstrip('/')}/logs/{log_filename}"

        final_payload = {"answer": result.get("answer"), "log_url": log_url}
        reply_text = json.dumps(final_payload, ensure_ascii=False)

        await bot.send_message(chat_id=chat_id, text=reply_text)

    except Exception as e:
        error = {"answer": {"error": str(e)}, "log_url": ""}
        try:
            await bot.send_message(chat_id=chat_id, text=json.dumps(error))
        except Exception:
            pass

    return JSONResponse({"ok": True})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
