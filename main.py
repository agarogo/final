from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypedDict

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


# -----------------------------
# Config
# -----------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
UPLOADS_DIR = BASE_DIR / "uploads"
DIAGRAMS_DIR = BASE_DIR / "diagrams"
LOGS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
DIAGRAMS_DIR.mkdir(exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/tasks_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QA_MODEL = os.getenv("QA_MODEL", "llama3.2:3b")
ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", "llama3.2:3b")

qa_llm = ChatOllama(model=QA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
assistant_llm = ChatOllama(model=ASSISTANT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)


# -----------------------------
# State
# -----------------------------
class AppState(TypedDict, total=False):
    user_message: str
    intent: Literal["qa", "create_task", "create_reminder", "list_tasks", "update_task", "read_file", "clarify"]
    route: Literal["tools", "clarify", "end"]
    title: str
    description: str
    priority: str
    deadline: str
    status: str
    reminder_text: str
    remind_at: str
    target_id: int
    file_path: str
    tool_result: dict[str, Any]
    final_answer: str


# -----------------------------
# Debug helpers
# -----------------------------
def dump_json(name: str, data: Any) -> None:
    path = LOGS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def debug(label: str, payload: Any) -> None:
    print(f"\n[DEBUG] {label}")
    try:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    except Exception:
        print(payload)


# -----------------------------
# DB
# -----------------------------
def get_conn() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def init_db() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    priority TEXT DEFAULT 'medium',
                    status TEXT DEFAULT 'new',
                    deadline TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id SERIAL PRIMARY KEY,
                    reminder_text TEXT NOT NULL,
                    remind_at TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_events (
                    id SERIAL PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    payload JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
                """
            )
        conn.commit()


def save_event(event_type: str, payload: dict[str, Any]) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agent_events (event_type, payload) VALUES (%s, %s)",
                (event_type, json.dumps(payload, ensure_ascii=False)),
            )
        conn.commit()


# -----------------------------
# Intent / extraction
# -----------------------------
def classify_message(text: str) -> str:
    t = text.lower().strip()

    if not t:
        return "clarify"

    if t.startswith("/file") or "прочитай файл" in t or "read file" in t:
        return "read_file"
    if any(x in t for x in ["напомни", "напомин", "remind"]):
        return "create_reminder"
    if any(x in t for x in ["покажи задачи", "список задач", "show tasks", "list tasks"]):
        return "list_tasks"
    if any(x in t for x in ["отметь задачу", "измени статус", "выполненн", "completed"]):
        return "update_task"
    if any(x in t for x in ["создай задачу", "задача", "заявка", "task"]):
        return "create_task"
    return "qa"


def extract_priority(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ["высок", "high", "срочно", "urgent"]):
        return "high"
    if any(x in t for x in ["низк", "low"]):
        return "low"
    return "medium"


def extract_deadline(text: str) -> str:
    t = text.lower()
    now = datetime.now()
    if "завтра" in t:
        return (now + timedelta(days=1)).strftime("%Y-%m-%d 09:00")
    if "сегодня" in t:
        return now.strftime("%Y-%m-%d 18:00")

    m = re.search(r"(\d{1,2})[:.](\d{2})", t)
    if m:
        hh, mm = m.groups()
        return now.strftime(f"%Y-%m-%d {int(hh):02d}:{int(mm):02d}")

    weekdays = {
        "понедельник": 0,
        "вторник": 1,
        "среда": 2,
        "четверг": 3,
        "пятница": 4,
        "суббота": 5,
        "воскресенье": 6,
    }
    for word, wd in weekdays.items():
        if word in t:
            days_ahead = (wd - now.weekday()) % 7
            days_ahead = 7 if days_ahead == 0 else days_ahead
            dt = now + timedelta(days=days_ahead)
            return dt.strftime("%Y-%m-%d 18:00")
    return ""


def extract_task_fields(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^(создай задачу:?|задача:?|заявка:?)\s*", "", text, flags=re.I).strip()
    return {
        "title": cleaned or text,
        "description": text,
        "priority": extract_priority(text),
        "deadline": extract_deadline(text),
        "status": "new",
    }


def extract_reminder_fields(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^(напомни:?|создай напоминание:?|remind me:? )\s*", "", text, flags=re.I).strip()
    return {
        "reminder_text": cleaned or text,
        "remind_at": extract_deadline(text) or (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
    }


def extract_update_fields(text: str) -> dict[str, Any]:
    m = re.search(r"(\d+)", text)
    task_id = int(m.group(1)) if m else 0
    status = "completed" if any(x in text.lower() for x in ["выполн", "completed", "done"]) else "in_progress"
    return {"target_id": task_id, "status": status}


def extract_file_path(text: str) -> str:
    if text.lower().startswith("/file"):
        return text[5:].strip().strip('"')
    m = re.search(r"файл\s+(.+)$", text, flags=re.I)
    return m.group(1).strip().strip('"') if m else ""


# -----------------------------
# Tools
# -----------------------------
def tool_create_task(state: AppState) -> dict[str, Any]:
    data = extract_task_fields(state["user_message"])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tasks (title, description, priority, status, deadline)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, title, priority, status, deadline
                """,
                (data["title"], data["description"], data["priority"], data["status"], data["deadline"] or None),
            )
            row = cur.fetchone()
        conn.commit()
    return {"action": "create_task", "task": row}


def tool_create_reminder(state: AppState) -> dict[str, Any]:
    data = extract_reminder_fields(state["user_message"])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reminders (reminder_text, remind_at)
                VALUES (%s, %s)
                RETURNING id, reminder_text, remind_at, status
                """,
                (data["reminder_text"], data["remind_at"]),
            )
            row = cur.fetchone()
        conn.commit()
    return {"action": "create_reminder", "reminder": row}


def tool_list_tasks(state: AppState) -> dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, priority, status, COALESCE(deadline, '') AS deadline
                FROM tasks
                ORDER BY id DESC
                LIMIT 10
                """
            )
            tasks = cur.fetchall()
            cur.execute(
                """
                SELECT id, reminder_text, COALESCE(remind_at, '') AS remind_at, status
                FROM reminders
                ORDER BY id DESC
                LIMIT 10
                """
            )
            reminders = cur.fetchall()
    return {"action": "list_tasks", "tasks": tasks, "reminders": reminders}


def tool_update_task(state: AppState) -> dict[str, Any]:
    data = extract_update_fields(state["user_message"])
    if not data["target_id"]:
        return {"action": "update_task", "error": "Не удалось определить id задачи"}

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tasks
                SET status = %s, updated_at = NOW()
                WHERE id = %s
                RETURNING id, title, priority, status, COALESCE(deadline, '') AS deadline
                """,
                (data["status"], data["target_id"]),
            )
            row = cur.fetchone()
        conn.commit()
    if not row:
        return {"action": "update_task", "error": f"Задача {data['target_id']} не найдена"}
    return {"action": "update_task", "task": row}


def tool_read_file(state: AppState) -> dict[str, Any]:
    file_path = extract_file_path(state["user_message"])
    if not file_path:
        return {"action": "read_file", "error": "Путь к файлу не найден"}

    p = Path(file_path)
    if not p.is_absolute():
        p = BASE_DIR / p
    if not p.exists():
        return {"action": "read_file", "error": f"Файл не найден: {str(p)}"}

    text = p.read_text(encoding="utf-8")
    return {"action": "read_file", "file_path": str(p), "content": text[:4000]}


# -----------------------------
# Graph nodes
# -----------------------------
def assistant_node(state: AppState) -> AppState:
    if state.get("tool_result"):
        tool_result = state["tool_result"]
        debug("assistant_after_tools_input", tool_result)
        answer = summarize_tool_result(tool_result)
        payload = {
            "model": ASSISTANT_MODEL,
            "phase": "after_tools",
            "tool_result": tool_result,
            "final_answer": answer,
        }
        dump_json("assistant_final_response", payload)
        return {"final_answer": answer, "route": "end"}

    user_message = state["user_message"]
    intent = classify_message(user_message)

    payload = {
        "model": ASSISTANT_MODEL,
        "user_message": user_message,
        "intent": intent,
    }
    debug("assistant_decision", payload)
    dump_json("assistant_decision", payload)

    if intent == "qa":
        response = qa_llm.invoke(
            f"Ответь коротко и по делу на русском языке. Вопрос пользователя: {user_message}"
        )
        raw = {
            "model": QA_MODEL,
            "user_message": user_message,
            "response": response.content,
        }
        dump_json("qa_response", raw)
        return {"intent": "qa", "final_answer": response.content, "route": "end"}

    if intent == "clarify":
        return {
            "intent": "clarify",
            "final_answer": "Не до конца понял запрос. Уточни: это задача, напоминание, чтение файла или обычный вопрос?",
            "route": "clarify",
        }

    return {"intent": intent, "route": "tools"}


def tools_node(state: AppState) -> AppState:
    intent = state["intent"]
    if intent == "create_task":
        result = tool_create_task(state)
    elif intent == "create_reminder":
        result = tool_create_reminder(state)
    elif intent == "list_tasks":
        result = tool_list_tasks(state)
    elif intent == "update_task":
        result = tool_update_task(state)
    elif intent == "read_file":
        result = tool_read_file(state)
    else:
        result = {"action": "unknown", "error": f"Неизвестный intent: {intent}"}

    debug("tool_result", result)
    dump_json("tool_result", result)
    return {"tool_result": result}


def sync_state_node(state: AppState) -> AppState:
    tool_result = state.get("tool_result", {})
    save_event("sync_state", tool_result)
    dump_json("sync_state", {"saved": True, "payload": tool_result})
    return {}


def clarify_node(state: AppState) -> AppState:
    dump_json("clarify", state)
    return state


def route_after_assistant(state: AppState) -> str:
    return state.get("route", "end")


# -----------------------------
# Result formatting
# -----------------------------
def summarize_tool_result(tool_result: dict[str, Any]) -> str:
    action = tool_result.get("action")
    if tool_result.get("error"):
        return f"Ошибка: {tool_result['error']}"

    if action == "create_task":
        task = tool_result["task"]
        return (
            f"Задача создана: #{task['id']} — {task['title']}. "
            f"Приоритет: {task['priority']}. Статус: {task['status']}. "
            f"Дедлайн: {task.get('deadline') or 'не указан'}."
        )

    if action == "create_reminder":
        r = tool_result["reminder"]
        return f"Напоминание создано: #{r['id']} — {r['reminder_text']}. Время: {r.get('remind_at') or 'не указано'}."

    if action == "list_tasks":
        tasks = tool_result.get("tasks", [])
        reminders = tool_result.get("reminders", [])
        parts: list[str] = []
        if tasks:
            parts.append("Задачи:")
            for t in tasks:
                parts.append(f"- #{t['id']} {t['title']} [{t['status']}] priority={t['priority']} deadline={t['deadline'] or '-'}")
        else:
            parts.append("Задач пока нет.")
        if reminders:
            parts.append("Напоминания:")
            for r in reminders:
                parts.append(f"- #{r['id']} {r['reminder_text']} [{r['status']}] at={r['remind_at'] or '-'}")
        return "\n".join(parts)

    if action == "update_task":
        task = tool_result["task"]
        return f"Задача обновлена: #{task['id']} — {task['title']}. Новый статус: {task['status']}."

    if action == "read_file":
        content = tool_result.get("content", "")
        prompt = (
            "Кратко перескажи содержимое файла на русском. "
            f"Путь: {tool_result.get('file_path', '')}\n\n"
            f"Содержимое:\n{content}"
        )
        response = qa_llm.invoke(prompt)
        dump_json("file_summary", {"model": QA_MODEL, "response": response.content})
        return response.content

    return f"Инструмент выполнился: {json.dumps(tool_result, ensure_ascii=False)}"


# -----------------------------
# Build graph
# -----------------------------
def build_graph():
    graph = StateGraph(AppState)
    graph.add_node("assistant", assistant_node)
    graph.add_node("tools", tools_node)
    graph.add_node("sync_state", sync_state_node)
    graph.add_node("clarify", clarify_node)

    graph.add_edge(START, "assistant")
    graph.add_conditional_edges(
        "assistant",
        route_after_assistant,
        {
            "tools": "tools",
            "clarify": "clarify",
            "end": END,
        },
    )
    graph.add_edge("tools", "sync_state")
    graph.add_edge("sync_state", "assistant")
    graph.add_edge("clarify", END)

    return graph.compile()


# -----------------------------
# Render graph
# -----------------------------
def render_graph() -> None:
    app = build_graph()
    graph_obj = app.get_graph()

    mermaid = graph_obj.draw_mermaid()
    (DIAGRAMS_DIR / "graph.mmd").write_text(mermaid, encoding="utf-8")
    print(f"Mermaid сохранён: {DIAGRAMS_DIR / 'graph.mmd'}")

    try:
        png = graph_obj.draw_mermaid_png()
        with open(DIAGRAMS_DIR / "graph.png", "wb") as f:
            f.write(png)
        print(f"PNG сохранён: {DIAGRAMS_DIR / 'graph.png'}")
    except Exception as e:
        print(f"PNG автоматически не отрисовался: {e}")


# -----------------------------
# CLI
# -----------------------------
def run_cli() -> None:
    init_db()
    app = build_graph()

    print("AI Task Dispatcher (one-file version)")
    print(f"Assistant model: {ASSISTANT_MODEL}")
    print(f"QA model: {QA_MODEL}")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print("Команды:")
    print("- Создай задачу: подготовить отчет до пятницы, приоритет высокий")
    print("- Напомни завтра в 8 утра позвонить маме")
    print("- Покажи задачи")
    print("- Отметь задачу 1 как выполненную")
    print("- /file uploads/sample_note.txt")
    print("- /render  (сохранить Mermaid/PNG)")
    print("- exit")
    print("-" * 60)

    while True:
        user_message = input("Вы: ").strip()
        if not user_message:
            continue
        if user_message.lower() == "exit":
            print("Выход.")
            break
        if user_message.lower() == "/render":
            render_graph()
            continue

        debug("new_request", {"user_message": user_message, "model": ASSISTANT_MODEL})
        result = app.invoke({"user_message": user_message})
        dump_json("last_result", result)

        final_answer = result.get("final_answer", "Нет ответа")
        print(f"\nАссистент: {final_answer}")
        save_event("final_answer", {"user_message": user_message, "final_answer": final_answer})


if __name__ == "__main__":
    run_cli()
