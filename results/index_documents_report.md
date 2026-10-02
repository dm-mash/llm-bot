# Индексация документов: локальный индекс с эмбеддингами

- Дата: 2026-10-01T23:19:01
- Команда: `scripts/index_documents.py --input_dir . --extensions .md --index_dir data/emb --out results/index_documents_report.md --strategy structure`
- Модель эмбеддингов: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Корпус: **12 документов**, 157964 символов ≈ 52.7 страниц (3000 символов/страница)
- Расширения: .md
- Параметры chunking: `structure` = {"max_tokens": 126, "min_chars": 200}

## Метаданные чанка

Каждый чанк хранит источник, заголовок документа, путь секции и диапазон строк — любой ответ индекса можно проверить по `file:line`.

```json
{
  "id": "structure-000-000",
  "metadata": {
    "chunk_id": "structure-000-000",
    "source": "README.md",
    "title": "llm-bot",
    "section": "llm-bot",
    "section_title": "llm-bot",
    "document_id": 0,
    "chunk_position": 0,
    "start_line": 1,
    "end_line": 9,
    "char_start": 0,
    "char_end": 290,
    "char_count": 290,
    "token_count": 76,
    "content_hash": "b3783153e93a75e1",
    "chunking_strategy": "structure"
  },
  "text": "# llm-bot\n\nA provider-agnostic Python client for OpenAI-compatible LLM APIs, extended with\nan **agent layer**. Beyond single prompts, it supports multiple LLM providers,\nnamed agents (roles), and persistent multi-turn ch…"
}
```

## Сравнение стратегий chunking

| Метрика | structure |
|---|---:|
| чанков | 579 |
| символов на чанк (среднее) | 270.6 |
| символов на чанк (медиана) | 271 |
| мин. размер, символов | 22 |
| макс. размер, символов | 485 |
| разброс (σ), символов | 94.1 |
| токенов на чанк (среднее) | 84.9 |
| макс. чанк, токенов | 141 |
| доля чанков длиннее бюджета стратегии | 0.4% |
| …из них одиночная строка (не режется) | 0.4% |
| …из них многострочных (должно быть 0) | 0.0% |
| доля чанков длиннее окна модели | 0.4% |
| доля чанков < min_chars | 23.1% |
| доля дубликатов | 0.0% |
| покрытие корпуса | 99.2% |
| время эмбеддингов, с | 10.71 |
| размер индекса, КБ | 2335.4 |

### Retrieval-бенчмарк

Запросов: **120** (эталон — точный диапазон символов раздела; hit — чанк пересекается с ним; 0 вручную, 120 из заголовков).

| Метрика | structure |
|---|---:|
| recall@1 | 63.3% |
| recall@3 | 75.8% |
| recall@5 | 79.2% |
| MRR@10 | 69.8% |

## Как читать индекс

```python
import json, math

index = json.load(open('data/emb/index_structure.json', encoding='utf-8'))
query = index['chunks'][0]   # в проде здесь вектор настоящего вопроса
ranked = sorted(
    ((math.fsum(a * b for a, b in zip(row['embedding'], query['embedding'])),
      row['metadata']) for row in index['chunks']),
    key=lambda pair: pair[0],   # ключ обязателен: при равных score
    reverse=True,               # сравниваются dict-ы, и сортировка падает
)
for score, meta in ranked[:3]:
    print(f"{score:.3f} {meta['source']}:{meta['start_line']}-"
          f"{meta['end_line']}  {meta['section']}")
```

Векторы нормализованы, поэтому скалярное произведение — это косинусная
близость.

