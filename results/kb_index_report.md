# Индексация документов: локальный индекс с эмбеддингами

- Дата: 2026-10-01T23:19:36
- Команда: `scripts/index_documents.py --input_dir knowledge_base --extensions .md --index_dir data/kb_emb --out results/kb_index_report.md --strategy structure`
- Модель эмбеддингов: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Корпус: **2 документов**, 3470 символов ≈ 1.2 страниц (3000 символов/страница)
- Расширения: .md
- Параметры chunking: `structure` = {"max_tokens": 126, "min_chars": 200}

## Метаданные чанка

Каждый чанк хранит источник, заголовок документа, путь секции и диапазон строк — любой ответ индекса можно проверить по `file:line`.

```json
{
  "id": "structure-000-000",
  "metadata": {
    "chunk_id": "structure-000-000",
    "source": "delivery.md",
    "title": "Доставка и самовывоз — Зёрна",
    "section": "Доставка и самовывоз — Зёрна > Зоны доставки",
    "section_title": "Зоны доставки",
    "document_id": 0,
    "chunk_position": 0,
    "start_line": 1,
    "end_line": 9,
    "char_start": 0,
    "char_end": 306,
    "char_count": 306,
    "token_count": 78,
    "content_hash": "dbeb9497b5acaf71",
    "chunking_strategy": "structure"
  },
  "text": "# Доставка и самовывоз — Зёрна\n\n## Зоны доставки\n\n- Собственная доставка курьером кофейни: Малый проспект и прилегающие улицы.\n- Минимальная сумма заказа для собственной доставки — 700 ₽.\n- Стоимость доставки — 150 ₽, от…"
}
```

## Сравнение стратегий chunking

| Метрика | structure |
|---|---:|
| чанков | 17 |
| символов на чанк (среднее) | 202.2 |
| символов на чанк (медиана) | 198 |
| мин. размер, символов | 20 |
| макс. размер, символов | 433 |
| разброс (σ), символов | 88.8 |
| токенов на чанк (среднее) | 61.3 |
| макс. чанк, токенов | 121 |
| доля чанков длиннее бюджета стратегии | 0.0% |
| …из них одиночная строка (не режется) | 0.0% |
| …из них многострочных (должно быть 0) | 0.0% |
| доля чанков длиннее окна модели | 0.0% |
| доля чанков < min_chars | 52.9% |
| доля дубликатов | 0.0% |
| покрытие корпуса | 99.1% |
| время эмбеддингов, с | 0.36 |
| размер индекса, КБ | 68.9 |

### Retrieval-бенчмарк

Запросов: **3** (эталон — точный диапазон символов раздела; hit — чанк пересекается с ним; 0 вручную, 3 из заголовков).

| Метрика | structure |
|---|---:|
| recall@1 | 100.0% |
| recall@3 | 100.0% |
| recall@5 | 100.0% |
| MRR@10 | 100.0% |

## Как читать индекс

```python
import json, math

index = json.load(open('data/kb_emb/index_structure.json', encoding='utf-8'))
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

