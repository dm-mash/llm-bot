# MCP notes demo — live transcript

```json
{
  "turns": [
    {
      "user": "Запиши в CRM заметку «купить корм коту» с тегом «покупки».",
      "reply": "Заметка успешно сохранена в CRM: «купить корм коту», тег «покупки», идентификатор записи 1.",
      "mcp": [
        {
          "tool": "add_note",
          "server": "notes",
          "ok": true,
          "result": "Заметка сохранена: id=1, теги: покупки",
          "error": ""
        }
      ],
      "usage": {
        "context_tokens": 647,
        "reply_tokens": 82
      }
    },
    {
      "user": "Выведи список всех заметок, которые сейчас лежат в CRM.",
      "reply": "Список текущих заметок в CRM:\n\n1. **Текст:** купить корм коту  \n   **Теги:** покупки  \n   **ID:** 1",
      "mcp": [],
      "usage": {
        "context_tokens": 528,
        "reply_tokens": 99
      }
    }
  ],
  "notes_db": [
    {
      "id": 1,
      "text": "купить корм коту",
      "tags": [
        "покупки"
      ]
    }
  ]
}
```
