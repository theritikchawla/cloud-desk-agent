"""Loads the Markdown documents and cuts them into "chunks".

Why chunks? Gemini only needs the few paragraphs that answer the question, not
whole documents. Small chunks make search more accurate and save free-tier tokens.

How we cut: every "## Heading" section of a document becomes one chunk. If a
section is very long, we split it further at blank lines.
"""
from dataclasses import dataclass
from pathlib import Path

import config


@dataclass(frozen=True)
class Chunk:
    chunk_id: str    # e.g. "billing_policy.md#3"
    source: str      # file name, shown to the user as a reference
    doc_title: str   # the "# Title" line of the file
    heading: str     # the "## Heading" of the section
    text: str        # what we search and send to Gemini


def load_documents(kb_dir) -> list[dict]:
    """Read every .md file in the folder. Returns [{'source': name, 'text': content}]."""
    folder = Path(kb_dir)
    files = sorted(folder.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No Markdown files found in {folder}")
    return [{"source": f.name, "text": f.read_text(encoding="utf-8")} for f in files]


def split_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (document title, [(heading, body), ...]) for one Markdown file."""
    title = "Untitled"
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Overview"
    current_lines: list[str] = []

    for line in markdown.splitlines():
        if line.startswith("# ") and title == "Untitled":
            title = line[2:].strip()
        elif line.startswith("## "):
            if "".join(current_lines).strip():
                sections.append((current_heading, current_lines))
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if "".join(current_lines).strip():
        sections.append((current_heading, current_lines))

    return title, [(h, "\n".join(lines).strip()) for h, lines in sections]


def split_long_text(body: str, max_chars: int) -> list[str]:
    """Split text longer than max_chars at blank lines (and, if needed, at lines)."""
    if len(body) <= max_chars:
        return [body]
    pieces: list[str] = []
    current = ""
    for paragraph in body.split("\n\n"):
        for line in ([paragraph] if len(paragraph) <= max_chars else paragraph.split("\n")):
            if current and len(current) + len(line) + 2 > max_chars:
                pieces.append(current.strip())
                current = ""
            current += line + "\n\n"
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_document(source: str, markdown: str, max_chars: int = config.MAX_CHUNK_CHARS) -> list[Chunk]:
    title, sections = split_sections(markdown)
    chunks: list[Chunk] = []
    for heading, body in sections:
        for piece in split_long_text(body, max_chars):
            text = f"{title} - {heading}\n{piece}"
            chunks.append(Chunk(f"{source}#{len(chunks)}", source, title, heading, text))
    return chunks


def load_and_chunk(kb_dir=config.KB_DIR, max_chars: int = config.MAX_CHUNK_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in load_documents(kb_dir):
        chunks.extend(chunk_document(doc["source"], doc["text"], max_chars))
    return chunks
