# Minimal Vectorless RAG (Interview Demo)

This is a **solid, minimal, explainable RAG pipeline** inspired by PageIndex’s “vectorless, reasoning-based retrieval” idea:

- **Indexing unit**: page-level (PDF) or section-level (Markdown)
- **No vector DB**: retrieval uses **BM25** (lexical ranking)
- **LLM use**: (1) make short per-page summaries for a human-readable index, (2) answer using only retrieved context with citations

## Files (kept intentionally tiny)

- `rag.py`: ingest + ask CLI (single script)
- `requirements.txt`: dependencies
- `demo_doc.md`: a small doc for a guaranteed demo
- `.env.example`: environment template

## Setup (one-time)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set your **OpenRouter API key** either via `.env` or directly in PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-or-v1-..."
$env:OPENROUTER_MODEL="openai/gpt-oss-120b:free"
```

## 60-second interview demo script

### 1) Build an index (from the included markdown doc)

```bash
python rag.py ingest --src demo_doc.md --out index.json
```

Open `index.json` to show:
- `pages[].summary` (human-readable)
- `pages[].text` (ground truth)

### 2) Ask a question

```bash
python rag.py ask --index index.json --q "What are the key constraints and why avoid vectors?"
```

You should see:
- **Retrieved pages** (ids like `p001, p002, ...`)
- An answer with **citations** like `[p003]`

## Using a PDF instead

```bash
python rag.py ingest --src path\to\your.pdf --out index.json --max-pages 8
python rag.py ask --index index.json --q "What does the document say about X?"
```

## What to say to the interviewer (tight)

- **“Index = pages/sections, not chunks.”** That keeps it explainable and avoids heavy chunking logic.
- **“Retrieval = BM25, no vectors.”** Still strong baselines, especially for exact terms and short corpora.
- **“LLM does two jobs.”** Summaries make an inspectable index; answer step is grounded + cited.
- **“Failure mode is explicit.”** If context is insufficient, it says “I don’t know…”.

