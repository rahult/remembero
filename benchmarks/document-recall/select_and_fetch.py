#!/usr/bin/env python3
"""Pick the tier documents out of XL-DocBench, fetch their PDFs, and write the run spec.

XL-DocBench ships labels only: gold answers, evidence page numbers and verbatim quotes for 331
public PDFs that it references by URL. So this script does three things and records what it did.

  select  documents whose real page count lands in a tier's band, preferring the ones with the
          most labelled questions, on hosts that serve a PDF to a script (government, IGO, open
          access). Hosts that are unauthorised copies of copyrighted books are skipped by name.
  fetch   the PDF, with a browser user-agent and the document's own host as referer, because a
          third of the hosts refuse a bare request. Every file is checked with pdfinfo: a
          download whose page count disagrees with the label is dropped, not scored.
  emit    benchmarks/document-recall/spec.json: per document, its sha256 and its questions with
          evidence pages, ready for `npm run eval:doc-recall`.

Nothing here judges or answers anything; it only decides which pages the benchmark will read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path(".cache/xl-docbench")
PDF_DIR = CACHE / "pdfs"
SPEC = Path("benchmarks/document-recall/spec.json")

# tiers as page bands: a real document of about this length, not a synthetic concatenation
BANDS = {"100p": (85, 130), "500p": (430, 600), "1000p": (700, 1250)}

# hosts that serve an unauthorised copy of a copyrighted book: not ours to benchmark on
BLOCKED_HOSTS = ("libcats.org", "files.wordpress.com", "rexresearch1.com", "rexresearch.com")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def load_labels() -> tuple[dict, dict]:
    documents = {}
    for line in (CACHE / "documents.jsonl").read_text().splitlines():
        record = json.loads(line)
        documents[record["document_id"]] = record
    questions: dict[str, list] = {}
    for line in (CACHE / "qa_single_doc.jsonl").read_text().splitlines():
        record = json.loads(line)
        questions.setdefault(record["document"]["document_id"], []).append(record)
    return documents, questions


def document_url(record: dict) -> str:
    return record.get("url") or record.get("public_metadata", {}).get("url") or ""


def candidates(documents: dict, questions: dict, band: tuple[int, int]) -> list[dict]:
    low, high = band
    rows = []
    for document_id, record in documents.items():
        if not low <= record["page_count"] <= high:
            continue
        url = document_url(record)
        host = urllib.parse.urlparse(url).netloc
        if not url or any(blocked in host for blocked in BLOCKED_HOSTS):
            continue
        rows.append(
            {
                "document_id": document_id,
                "pages": record["page_count"],
                "url": url,
                "host": host,
                "title": record.get("public_metadata", {}).get("title") or document_id,
                "questions": questions.get(document_id, []),
            }
        )
    rows.sort(key=lambda row: (-len(row["questions"]), row["pages"]))
    return rows


def fetch(url: str, target: Path) -> tuple[bool, str]:
    if target.exists() and target.stat().st_size > 10_000:
        return True, "cached"
    parsed = urllib.parse.urlparse(url)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf,*/*",
            "Referer": f"{parsed.scheme}://{parsed.netloc}/",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
        return False, f"{type(error).__name__}: {error}"
    if not body.startswith(b"%PDF"):
        return False, f"not a pdf ({body[:16]!r})"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    return True, f"{len(body) / 1_048_576:.1f}MB"


def pdf_pages(path: Path) -> int | None:
    try:
        info = subprocess.run(
            ["pdfinfo", str(path)], capture_output=True, text=True, timeout=120, check=True
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    return int(match.group(1)) if match else None


def spec_question(record: dict) -> dict | None:
    """One XL-DocBench QA row as this benchmark's labelled question."""
    answer = record.get("answer", {})
    value = answer.get("value")
    unanswerable = answer.get("format") == "None" or record.get("metadata", {}).get("is_unanswerable")
    pages: list[int] = []
    for item in record["document"].get("evidence_items", []):
        pages.extend(int(page) for page in item.get("pages", []) if isinstance(page, (int, float)))
    if not unanswerable and (value is None or not pages):
        return None  # an answerable question with no page label cannot score retrieval
    return {
        "id": record["question_id"],
        "question": record["question"],
        "answer": None if unanswerable else str(value),
        "evidencePages": sorted(set(pages)),
        "datasetKind": record.get("metadata", {}).get("reasoning_type"),
        "answerFormat": answer.get("format"),
        "verificationRule": answer.get("verification_rule"),
        "difficulty": record.get("metadata", {}).get("difficulty"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-tier", type=int, default=6, help="documents to land per tier")
    parser.add_argument("--attempts", type=int, default=14, help="documents to try per tier")
    parser.add_argument("--tiers", default="100p,500p,1000p")
    args = parser.parse_args()

    documents, questions = load_labels()
    sources: list[dict] = []
    tiers: list[dict] = []
    report: list[str] = []

    for tier in args.tiers.split(","):
        band = BANDS[tier]
        landed: list[str] = []
        for row in candidates(documents, questions, band)[: args.attempts]:
            if len(landed) >= args.per_tier:
                break
            target = PDF_DIR / f"{row['document_id']}.pdf"
            ok, note = fetch(row["url"], target)
            if not ok:
                report.append(f"{tier} {row['document_id']} SKIP fetch {note} [{row['host']}]")
                continue
            pages = pdf_pages(target)
            if pages is None:
                report.append(f"{tier} {row['document_id']} SKIP unreadable pdf")
                target.unlink(missing_ok=True)
                continue
            if abs(pages - row["pages"]) > 2:
                report.append(
                    f"{tier} {row['document_id']} SKIP page count {pages} != label {row['pages']}"
                )
                target.unlink(missing_ok=True)
                continue
            labelled = [spec_question(record) for record in row["questions"]]
            labelled = [question for question in labelled if question is not None]
            if not labelled:
                report.append(f"{tier} {row['document_id']} SKIP no scorable questions")
                continue
            sources.append(
                {
                    "id": row["document_id"],
                    "title": row["title"],
                    "sourceUrl": row["url"],
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "pdf": str(target),
                    "labelPages": pages,
                    "questions": labelled,
                }
            )
            landed.append(row["document_id"])
            report.append(
                f"{tier} {row['document_id']} ok {pages}pp {len(labelled)}q {note} [{row['host']}]"
            )
        tiers.append({"name": tier, "documents": landed, "band": list(band)})

    SPEC.parent.mkdir(parents=True, exist_ok=True)
    SPEC.write_text(
        json.dumps(
            {
                "name": "xl-docbench-page-tiers",
                "labels": "XL-DocBench (arXiv:2608.00036), data/qa_single_doc.jsonl",
                "tiers": tiers,
                "sources": sources,
            },
            indent=2,
        )
        + "\n"
    )
    print("\n".join(report))
    for tier in tiers:
        got = [source for source in sources if source["id"] in tier["documents"]]
        print(
            f"\n{tier['name']}: {len(got)} documents, "
            f"{sum(source['labelPages'] for source in got)} pages, "
            f"{sum(len(source['questions']) for source in got)} questions"
        )
    print(f"\nwrote {SPEC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
