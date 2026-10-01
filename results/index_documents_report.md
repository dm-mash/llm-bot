# Индексация документов: локальный индекс с эмбеддингами

- Дата: 2026-10-01T11:02:32
- Команда: `scripts/index_documents.py --input_dir . --extensions .md`
- Модель эмбеддингов: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Корпус: **11 документов**, 153750 символов ≈ 51.2 страниц (3000 символов/страница)
- Расширения: .md
- Параметры chunking: `fixed_size` = {"unit": "tokens", "chunk_size": 120, "chunk_overlap": 20}; `structure` = {"max_tokens": 126, "min_chars": 200}

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

| Метрика | fixed_size | structure |
|---|---:|---:|
| чанков | 482 | 582 |
| символов на чанк (среднее) | 370.6 | 261.9 |
| символов на чанк (медиана) | 368.5 | 265.5 |
| мин. размер, символов | 94 | 8 |
| макс. размер, символов | 534 | 485 |
| разброс (σ), символов | 58 | 100.3 |
| токенов на чанк (среднее) | 116.1 | 82.3 |
| макс. чанк, токенов | 120 | 141 |
| доля чанков длиннее бюджета стратегии | 0.0% | 0.3% |
| …из них одиночная строка (не режется) | 0.0% | 0.3% |
| …из них многострочных (должно быть 0) | 0.0% | 0.0% |
| доля чанков длиннее окна модели | 0.0% | 0.3% |
| доля чанков < min_chars | 0.0% | 25.9% |
| доля дубликатов | 0.0% | 0.0% |
| покрытие корпуса | 100.0% | 99.1% |
| время эмбеддингов, с | 19.29 | 20.12 |
| размер индекса, КБ | 1933.3 | 2343.5 |

### Retrieval-бенчмарк

Запросов: **119** (эталон — точный диапазон символов раздела; hit — чанк пересекается с ним; 0 вручную, 119 из заголовков).

| Метрика | fixed_size | structure |
|---|---:|---:|
| recall@1 | 36.1% | 60.5% |
| recall@3 | 46.2% | 76.5% |
| recall@5 | 56.3% | 79.0% |
| MRR@10 | 43.9% | 68.6% |

### Вывод

- recall@1: лучше **structure** (60.5% против 36.1%).
- recall@5: лучше **structure** (79.0% против 56.3%).
- Медианный чанк: 368.5 символов у fixed_size против 265.5 у structure — структурный чанк ближе к смысловому блоку, поэтому запрос про заголовок попадает в него целиком.
- Плата за структуру: 25.9% чанков короче min_chars против 0.0% — это короткие разделы целиком: склеивать их можно только с соседним разделом, а это уже ломает метаданные `section`. Поднимать `--structure_min_chars` вверх бесполезно, опускать — вниз.
- Цена: structure даёт 582 чанков (2343.5 КБ) против 482 (1933.3 КБ) у fixed_size.

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

