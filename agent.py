import json
import os
import uuid
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def solve_question(question: str):
    """
    Dummy agent.
    Returns a JSON string expected by app.py.
    """

    run_id = str(uuid.uuid4())
    log_path = os.path.join(LOG_DIR, f"{run_id}.jsonl")

    with open(log_path, "w") as f:
        f.write(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "question": question
        }) + "\n")

        f.write(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed"
        }) + "\n")

    result = {
        "answer": {
            "message": "Bot received the question successfully.",
            "question": question
        },
        "log_url": f"https://example.com/logs/{run_id}.jsonl"
    }

    return json.dumps(result)
