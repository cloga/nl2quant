import json
import os
from datetime import datetime
import uuid

HISTORY_DIR = "history"

def ensure_history_dir():
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)

def get_session_file(session_id):
    return os.path.join(HISTORY_DIR, f"{session_id}.json")

def save_session(session_id, messages, title=None):
    ensure_history_dir()
    file_path = get_session_file(session_id)
    
    # If title is not provided and it's a new file, try to generate one from the first message
    if not title:
        # Try to find existing title
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    current_data = json.load(f)
                    title = current_data.get("title")
            except:
                pass
        
        # If still no title, generate from first user message
        if not title and messages:
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    title = content[:20] + "..." if len(content) > 20 else content
                    break
    
    data = {
        "id": session_id,
        "title": title or "New Session",
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_session(session_id):
    file_path = get_session_file(session_id)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def list_sessions():
    ensure_history_dir()
    sessions = []
    for filename in os.listdir(HISTORY_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(HISTORY_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "id": data.get("id", filename.replace(".json", "")),
                        "title": data.get("title", "Untitled"),
                        "timestamp": data.get("timestamp", "")
                    })
            except:
                continue
    # Sort by timestamp desc
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return sessions

def create_new_session():
    return str(uuid.uuid4())
