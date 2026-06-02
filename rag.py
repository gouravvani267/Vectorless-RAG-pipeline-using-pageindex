from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get_env(name: str) -> str:
    v = os.getenv(name)
    if v is not None and v.strip():
        return v.strip()

    # Fallback: some Windows editors save `.env` as UTF-16, which python-dotenv may not parse.
    candidates = [
        Path(".env"),
        Path(__file__).resolve().parent / ".env",
    ]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        return ""

    raw = p.read_bytes()
    for enc in ("utf-16", "utf-16-le", "utf-16-be", "utf-8-sig", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    else:
        return ""

    # If the wrong codec “succeeds”, it often leaves NUL separators.
    if "\x00" in text:
        text = text.replace("\x00", "")

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, val = line.split("=", 1)
        key_name = k.strip().lstrip("\ufeff")
        if key_name == name:
            return val.strip().strip('"').strip("'")
    return ""


def _require_real_gemini_key() -> str:
    # Backwards-compat shim (older versions of this repo used Gemini).
    return ""


def _require_openrouter_key() -> str:
    # OpenRouter uses OpenAI-compatible keys like: sk-or-v1-...
    key = _get_env("OPENROUTER_API_KEY") or _get_env("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "Missing OPENAI_API_KEY / OPENROUTER_API_KEY. Put your OpenRouter key in .env, or run with --offline."
        )
    if key.lower() == "replace_me":
        raise SystemExit("API key is still 'replace_me'. Edit .env and paste a real key, or run with --offline.")
    return key


def _tokenize(text: str) -> list[str]:
    # Small + explainable tokenization for BM25.
    return re.findall(r"[a-z0-9]+", text.lower())


def _simple_summary(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip())
    if not t:
        return "- (empty page)"
    snippet = t[:240]
    return f"- {snippet}{'…' if len(t) > 240 else ''}"


def _read_pdf_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for p in reader.pages:
        pages.append((p.extract_text() or "").strip())
    return pages


def _read_md_sections(md_path: Path) -> list[str]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    # Split on markdown headings to create “page-like” sections.
    parts = re.split(r"(?m)^(?=#{1,6}\s)", text)
    sections = [p.strip() for p in parts if p.strip()]
    return sections


@dataclass(frozen=True)
class Page:
    page_id: str
    text: str
    summary: str


def _normalize_model(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return "openai/gpt-oss-120b:free"
    return m


def _openrouter_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def _llm_generate(*, client: OpenAI, model: str, input_text: str) -> str:
    # OpenRouter is OpenAI-compatible; chat.completions is the most widely supported surface.
    resp = client.chat.completions.create(
        model=_normalize_model(model),
        messages=[{"role": "user", "content": input_text}],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def _llm_summarize(*, client: OpenAI, model: str, text: str) -> str:
    # Keep summaries short to make index inspectable + keep prompt small.
    prompt = (
        "Summarize the following page/section in 1-2 bullet points. "
        "Be factual and keep it short.\n\n"
        f"TEXT:\n{text[:8000]}"
    )
    return _llm_generate(client=client, model=model, input_text=prompt)


def build_index(
    *,
    src_path: Path,
    out_path: Path,
    model: str,
    max_pages: int | None,
    offline: bool,
) -> None:
    load_dotenv()
    client: OpenAI | None = None
    if not offline:
        key = _require_openrouter_key()
        client = _openrouter_client(key)

    if src_path.suffix.lower() == ".pdf":
        raw_pages = _read_pdf_pages(src_path)
        kind = "pdf"
    elif src_path.suffix.lower() in {".md", ".markdown"}:
        raw_pages = _read_md_sections(src_path)
        kind = "markdown"
    else:
        raise SystemExit("Unsupported input. Use a .pdf or .md file.")

    raw_pages = [p for p in raw_pages if p.strip()]
    if max_pages is not None:
        raw_pages = raw_pages[: max_pages]

    pages: list[Page] = []
    for i, text in enumerate(raw_pages, start=1):
        page_id = f"p{i:03d}"
        if offline:
            summary = _simple_summary(text)
        else:
            assert client is not None
            summary = _llm_summarize(client=client, model=model, text=text)
        pages.append(Page(page_id=page_id, text=text, summary=summary))

    index: dict[str, Any] = {
        "created_at": _utc_iso(),
        "source": {"path": str(src_path), "type": kind},
        "pages": [{"id": p.page_id, "summary": p.summary, "text": p.text} for p in pages],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: wrote index with {len(pages)} pages -> {out_path}")


def _load_index(index_path: Path) -> dict[str, Any]:
    return json.loads(index_path.read_text(encoding="utf-8"))


def _bm25_rank(pages: list[dict[str, Any]], query: str, k: int) -> list[dict[str, Any]]:
    corpus = [f"{p.get('summary','')}\n{p.get('text','')}" for p in pages]
    tokenized = [_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(pages, scores), key=lambda x: x[1], reverse=True)
    return [p for p, _ in ranked[:k]]


def _format_context(pages: Iterable[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for p in pages:
        pid = p["id"]
        summary = p.get("summary", "").strip()
        text = (p.get("text", "") or "").strip()
        blocks.append(
            f"[{pid}] SUMMARY:\n{summary}\n\n[{pid}] TEXT:\n{text[:6000]}".strip()
        )
    return "\n\n---\n\n".join(blocks)


def ask(
    *,
    index_path: Path,
    question: str,
    model: str,
    top_k: int,
    offline: bool,
) -> None:
    load_dotenv()
    client: OpenAI | None = None
    if not offline:
        key = _require_openrouter_key()
        client = _openrouter_client(key)
    index = _load_index(index_path)
    pages: list[dict[str, Any]] = index.get("pages", [])
    if not pages:
        raise SystemExit("Index has no pages.")

    hits = _bm25_rank(pages, question, k=top_k)
    context = _format_context(hits)

    prompt = (
        "You are a careful RAG assistant.\n"
        "Answer the question using ONLY the provided context.\n"
        "If the context is insufficient, say: \"I don't know based on the provided document.\".\n"
        "Add citations like [p001] at the end of sentences.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}"
    )

    if offline:
        answer = (
            "OFFLINE MODE (no LLM): showing retrieved page summaries.\n\n"
            + "\n".join([f"- {p['id']}: {p.get('summary','').strip()}" for p in hits])
        ).strip()
    else:
        assert client is not None
        answer = _llm_generate(client=client, model=model, input_text=prompt)
    print("\n=== Retrieved pages ===")
    print(", ".join([p["id"] for p in hits]))
    print("\n=== Answer ===")
    print(answer)


def main(argv: list[str]) -> int:
    # Windows PowerShell consoles often default to a legacy code page (cp1252),
    # which can crash on perfectly valid model output. Make printing robust.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    p = argparse.ArgumentParser(
        prog="rag.py",
        description="Minimal vectorless RAG (page/section index + BM25 + LLM).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Build an index JSON from a PDF/MD")
    ingest.add_argument("--src", required=True, help="Path to .pdf or .md")
    ingest.add_argument("--out", default="index.json", help="Output index path (json)")
    ingest.add_argument("--model", default=_get_env("OPENROUTER_MODEL") or "openai/gpt-oss-120b:free")
    ingest.add_argument("--max-pages", type=int, default=None, help="Limit pages/sections for demo")
    ingest.add_argument("--offline", action="store_true", help="Run without Gemini (heuristic summaries)")

    askp = sub.add_parser("ask", help="Ask a question using an index JSON")
    askp.add_argument("--index", default="index.json", help="Path to index JSON")
    askp.add_argument("--q", required=True, help="Question")
    askp.add_argument("--top-k", type=int, default=4, help="Retrieved pages to include")
    askp.add_argument("--model", default=_get_env("OPENROUTER_MODEL") or "openai/gpt-oss-120b:free")
    askp.add_argument("--offline", action="store_true", help="Run without Gemini (no generated answer)")

    args = p.parse_args(argv)

    if args.cmd == "ingest":
        build_index(
            src_path=Path(args.src),
            out_path=Path(args.out),
            model=args.model,
            max_pages=args.max_pages,
            offline=args.offline,
        )
        return 0

    if args.cmd == "ask":
        ask(
            index_path=Path(args.index),
            question=args.q,
            model=args.model,
            top_k=args.top_k,
            offline=args.offline,
        )
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

