# One-file AI Task Dispatcher

Максимально упрощённый проект, где **вся логика приложения лежит в `main.py`**.

Что умеет:
- обычный вопрос
- создать задачу
- создать напоминание
- показать задачи
- обновить статус задачи
- прочитать файл
- сохранить Mermaid/PNG диаграмму графа

## Структура

```text
onefile_task_dispatcher/
├── main.py
├── requirements.txt
├── docker-compose.yml
├── .env.example
├── .gitignore
├── uploads/
├── logs/
└── diagrams/
```

## Запуск с нуля

### 1. Создать venv

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Поднять PostgreSQL и Ollama

```powershell
docker compose up -d
```

### 3. Скачать модель

```powershell
docker exec -it onefile_ollama ollama pull llama3.2:3b
```

### 4. Создать .env

```powershell
Copy-Item .env.example .env
```

### 5. Запустить проект

```powershell
python main.py
```

## Команды в приложении

- `Создай задачу: подготовить отчет до пятницы, приоритет высокий`
- `Напомни завтра в 8 утра позвонить маме`
- `Покажи задачи`
- `Отметь задачу 1 как выполненную`
- `/file uploads/sample_note.txt`
- `/render`
- `exit`

## Что делает граф

- `assistant` — решает: ответить самому, уйти в tools или попросить уточнение
- `tools` — создаёт/читает задачи и напоминания, читает файл
- `sync_state` — пишет событие в БД и возвращает управление в assistant
- `clarify` — ветка уточнения

Схема:

`START -> assistant -> (tools | clarify | END)`

`tools -> sync_state -> assistant`

## Логи

JSON-логи сохраняются в `logs/`.

## Mermaid / PNG

Команда `/render` создаёт:
- `diagrams/graph.mmd`
- `diagrams/graph.png` (если авто-рендер доступен)
