# Demo Doc (Interview RAG)

## What is this?
This is a tiny document you can use to demo a retrieval-augmented generation (RAG) pipeline.
The pipeline ingests this file, builds a **page/section index** (no vectors), retrieves the most relevant
pages with BM25, and asks an LLM to answer using only retrieved context.

## Product overview
Vectify is building an internal assistant for policy and engineering docs.

Key constraints:
- No vector DB in the first milestone.
- Index must be inspectable by humans.
- Retrieval should be fast enough for interactive Q&A.

## Security policy
We never store API keys in the repo.
We do not send customer PII to external services.
For demos, use synthetic data.

## System design notes
The index is stored as JSON and contains:
- metadata (source, created_at)
- pages: each page has text and a short summary

Retrieval uses a lexical scorer (BM25) over page text + summary.

## FAQ
**Q: Why not vector search?**
A: It’s great, but sometimes you want a minimal system that still works and can be explained quickly.

**Q: How do you avoid hallucinations?**
A: The prompt forces citations to retrieved page ids, and answers must be grounded in retrieved text.
