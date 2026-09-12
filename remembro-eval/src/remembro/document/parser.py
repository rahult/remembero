"""Layer 1: what is physically in the document.

V0 reads the Markdown fixture directly. Docling sits behind the same Protocol
when PDFs arrive; the domain model never sees either.
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class Span(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int
    paragraph: int  # 1-based within the document
    section: str | None
    kind: str  # paragraph | table_row | heading | bullet
    text: str
    start_offset: int
    end_offset: int
    authority: str  # body | appendix | table
    header: str | None = None  # for a table row: the column names, "Role | Operational | Capital"


class ParsedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    language: str = "en"
    spans: list[Span]
    text: str


class DocumentParser(Protocol):
    def parse(self, document_id: str, source: bytes) -> ParsedDocument: ...


PAGE_MARK = re.compile(r"<!--\s*page\s+(\d+)\s*-->")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


class MarkdownParser:
    """Paragraph-level spans with page and section, table rows as their own spans."""

    def parse(self, document_id: str, source: bytes) -> ParsedDocument:
        text = source.decode("utf-8")
        spans: list[Span] = []
        page = 1
        section: str | None = None
        paragraph = 0
        in_appendix = False
        offset = 0
        for block in re.split(r"\n\s*\n", text):
            start = text.find(block, offset)
            if start < 0:
                start = offset
            end = start + len(block)
            offset = end
            stripped = block.strip()
            if not stripped:
                continue
            mark = PAGE_MARK.search(stripped)
            if mark:
                page = int(mark.group(1))
                stripped = PAGE_MARK.sub("", stripped).strip()
                if not stripped:
                    continue
            heading = HEADING.match(stripped.splitlines()[0])
            if heading:
                section = heading.group(2).strip()
                in_appendix = section.lower().startswith("appendix")
                rest = "\n".join(stripped.splitlines()[1:]).strip()
                if not rest:
                    continue
                stripped = rest
            authority = "appendix" if in_appendix else "body"
            if stripped.startswith("|"):
                lines = stripped.splitlines()
                # the row before a |---| separator is the header; it names the columns of every row
                header: str | None = None
                for i, line in enumerate(lines):
                    if re.match(r"^\|\s*-", line) and i > 0:
                        header = " | ".join(c.strip() for c in lines[i - 1].strip().strip("|").split("|"))
                        break
                for line in lines:
                    if re.match(r"^\|\s*-", line) or not line.strip():
                        continue
                    paragraph += 1
                    cells = [c.strip() for c in line.strip().strip("|").split("|")]
                    row = " | ".join(cells)
                    is_header = header is not None and row == header
                    spans.append(
                        Span(
                            page=page,
                            paragraph=paragraph,
                            section=section,
                            kind="table_header" if is_header else "table_row",
                            text=row,
                            start_offset=start,
                            end_offset=end,
                            authority="appendix" if in_appendix else "table",
                            header=None if is_header else header,
                        )
                    )
                continue
            paragraph += 1
            kind = "bullet" if stripped.startswith("- ") else "paragraph"
            spans.append(
                Span(
                    page=page,
                    paragraph=paragraph,
                    section=section,
                    kind=kind,
                    text=re.sub(r"\s*\n\s*", " ", stripped),
                    start_offset=start,
                    end_offset=end,
                    authority=authority,
                )
            )
        return ParsedDocument(id=document_id, spans=spans, text=text)
