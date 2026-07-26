import os
import json
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from telegram import Bot
from dotenv import load_dotenv

from agent import solve_question

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not found in environment variables.")

bot = Bot(BOT_TOKEN)

app = FastAPI()

# Stores recent conversation history for each user
conversation_history = defaultdict(list)


@app.get("/")
async def root():
    return {"status": "running"}


@app.post("/webhook")
async def telegram_webhook(request: Request):
    """
    Telegram sends every incoming message here.
    """

    update = await request.json()

    try:
        if "message" not in update:
            return JSONResponse({"ok": True})

        message = update["message"]

        chat_id = message["chat"]["id"]

        text = message.get("text", "")

        if not text:
            return JSONResponse({"ok": True})

        # Save history
        conversation_history[chat_id].append(text)

        # Keep only last 5 messages
        conversation_history[chat_id] = conversation_history[chat_id][-5:]

        conversation = "\n".join(conversation_history[chat_id])

        result = solve_question(conversation)

        # Telegram expects plain text.
        # The result MUST already be a JSON string.
        await bot.send_message(
            chat_id=chat_id,
            text=result
        )

    except Exception as e:

        error = {
            "answer": {
                "error": str(e)
            },
            "log_url": ""
        }

        await bot.send_message(
            chat_id=chat_id,
            text=json.dumps(error)
        )

    return JSONResponse({"ok": True})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
