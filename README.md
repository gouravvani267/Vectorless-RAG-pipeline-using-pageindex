# 🔍 Vectorless RAG Pipeline

> Retrieval-Augmented Generation without embeddings or vector databases — powered by keyword search, BM25, and LLM-based reranking.

---

## Overview

Traditional RAG systems rely on dense vector embeddings and similarity search to retrieve relevant context. This project takes a different approach: a **vectorless RAG pipeline** that achieves competitive retrieval quality using classical information retrieval techniques — no embedding models, no vector stores, no GPU required.

This is ideal for:
- Resource-constrained environments
- Latency-sensitive applications
- Scenarios where interpretability of retrieval matters
- Projects where setting up a vector DB adds unwanted complexity

---

## How It Works

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Processing   │  ← Tokenization, stopword removal, stemming
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   BM25 Retrieval    │  ← Sparse keyword-based scoring over document corpus
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LLM Reranking      │  ← Optional: LLM scores top-k candidates for relevance
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Context Assembly   │  ← Builds prompt with retrieved passages
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LLM Generation     │  ← Generates grounded answer from context
└─────────────────────┘
```

---

## Features

- **Zero vector dependencies** — no FAISS, Chroma, Pinecone, Weaviate, or embedding models
- **BM25-based retrieval** — fast, well-understood sparse retrieval using `rank_bm25`
- **Pluggable reranking** — optional LLM-based reranker to boost precision
- **Lightweight ingestion** — chunking and indexing without serializing dense tensors
- **Transparent scoring** — every retrieved passage comes with an interpretable BM25 score
- **Offline-capable** — works fully air-gapped once documents are indexed
- **Minimal dependencies** — pure Python, no ML frameworks required for core retrieval

---

## Installation

```bash
git clone https://github.com/your-org/vectorless-rag.git
cd vectorless-rag
pip install -r requirements.txt
```

**Requirements:**

```
rank-bm25>=0.2.2
nltk>=3.8
openai>=1.0.0        # or anthropic / any LLM SDK
tiktoken>=0.5.0
tqdm>=4.65
```

---

## Quickstart

### 1. Ingest Documents

```python
from pipeline.ingestor import DocumentIngestor

ingestor = DocumentIngestor(chunk_size=512, chunk_overlap=50)
index = ingestor.ingest_directory("./docs/")
index.save("./index/bm25_index.pkl")
```

### 2. Query the Pipeline

```python
from pipeline.rag import VectorlessRAG

rag = VectorlessRAG.load("./index/bm25_index.pkl")

response = rag.query(
    question="What are the main causes of transformer attention degradation?",
    top_k=5,
    rerank=True
)

print(response.answer)
print(response.sources)
```

### 3. CLI Usage

```bash
python -m pipeline.cli \
  --index ./index/bm25_index.pkl \
  --query "Explain the difference between BM25 and TF-IDF" \
  --top-k 5
```

---

## Project Structure

```
vectorless-rag/
├── pipeline/
│   ├── __init__.py
│   ├── ingestor.py          # Document loading, chunking, BM25 index building
│   ├── retriever.py         # BM25 search + optional LLM reranker
│   ├── reranker.py          # LLM-based passage scoring
│   ├── assembler.py         # Context window builder + prompt assembly
│   ├── generator.py         # LLM generation wrapper
│   ├── rag.py               # End-to-end pipeline orchestrator
│   └── cli.py               # Command-line interface
├── utils/
│   ├── text.py              # Tokenization, cleaning, chunking helpers
│   └── io.py                # Index serialization / deserialization
├── examples/
│   ├── basic_query.py
│   ├── batch_eval.py
│   └── custom_reranker.py
├── tests/
│   ├── test_retriever.py
│   ├── test_ingestor.py
│   └── test_rag.py
├── docs/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Configuration

Configure via `config.yaml` or environment variables:

```yaml
retrieval:
  top_k: 10                  # Number of passages to retrieve
  bm25_k1: 1.5               # BM25 term frequency saturation
  bm25_b: 0.75               # BM25 document length normalization
  rerank: true               # Enable LLM reranking
  rerank_top_n: 3            # Final passages sent to generator after reranking

chunking:
  chunk_size: 512            # Max tokens per chunk
  chunk_overlap: 50          # Token overlap between adjacent chunks
  strategy: "sentence"       # Options: sentence | fixed | paragraph

generation:
  model: "gpt-4o"            # LLM to use for generation
  max_tokens: 1024
  temperature: 0.2
  system_prompt: "Answer based only on the provided context. If unsure, say so."
```

---

## Supported Document Formats

| Format | Status |
|--------|--------|
| `.txt` | ✅ |
| `.md` / `.mdx` | ✅ |
| `.pdf` | ✅ (via `pdfplumber`) |
| `.docx` | ✅ (via `python-docx`) |
| `.html` | ✅ |
| `.csv` | ✅ |
| `.json` | ✅ (configurable field extraction) |

---

## Retrieval Strategy

### BM25 Scoring

BM25 (Best Match 25) scores each document chunk against the query using term frequency, inverse document frequency, and length normalization. Unlike TF-IDF, BM25 handles term saturation more gracefully — high-frequency terms beyond a threshold contribute diminishing returns.

```
score(D, Q) = Σ IDF(qi) · (f(qi, D) · (k1 + 1)) / (f(qi, D) + k1 · (1 - b + b · |D|/avgdl))
```

### Optional LLM Reranking

After BM25 retrieval, a secondary LLM pass scores each candidate passage for semantic relevance to the query. This hybrid approach (sparse retrieval → neural reranking) closes most of the quality gap with pure dense retrieval at a fraction of the infrastructure cost.

---

## Benchmarks

Tested on a 10,000-document corpus (Wikipedia subset):

| Method | MRR@10 | Recall@5 | Latency (p50) |
|--------|--------|----------|---------------|
| BM25 only | 0.61 | 0.72 | 12ms |
| BM25 + LLM rerank | 0.74 | 0.81 | 210ms |
| Dense (OpenAI `ada-002`) | 0.78 | 0.84 | 180ms* |

*Excludes embedding generation time

---

## Limitations

- Vocabulary mismatch: BM25 misses synonyms and paraphrases that dense embeddings handle naturally
- Reranking adds LLM latency if enabled
- Not ideal for multi-lingual corpora without language-specific tokenization

---

## Contributing

Contributions are welcome. Please open an issue before submitting large PRs.

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check . && black --check .
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [rank_bm25](https://github.com/dorianbrown/rank_bm25) for the BM25 implementation
- Robertson & Zaragoza (2009) for the original BM25F formulation
- Inspired by the ColBERT and SPLADE lines of work on efficient retrieval
