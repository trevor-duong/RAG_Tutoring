"""Turn a retrieval hit into something a student can actually go and read.

Retrieval returns a ``source`` (the filename stem) and a ``page`` (pypdf's page
index). Neither is fit to show as-is, and the page is the dangerous one: it is
*not* the number printed on the page. Front matter makes those differ by 18 in
*Dive into Deep Learning*, so citing a bare "p. 305" sends a student to the wrong
page of the book with no hint that anything is off -- a wrong answer wearing the
costume of a precise one.

This module is deliberately separate from ``store.py``. The store's job is to
find passages; how a passage is described to a human is a presentation concern
that the API and the notebook should share rather than each reinvent.
"""

from __future__ import annotations

from dataclasses import dataclass

# Filenames cannot contain ":", so a colon in a title was replaced with "-" when
# the PDF was saved: "GloVe- Global Vectors..." was "GloVe: Global Vectors...".
# Verified against all 37 filenames in the corpus -- each has at most one
# hyphen-space, every one of them a stripped colon, and every genuine hyphen is
# intra-word ("Fine-Tuning", "DECODING-ENHANCED"), so it never has a space after
# it. Only the first occurrence is replaced, which is what the corpus supports;
# a title with two would be a case this rule has not been checked against.
_STRIPPED_COLON = "- "

SNIPPET_CHARS = 320


@dataclass(frozen=True)
class Citation:
    """A retrieval hit, described for a reader.

    ``page`` is kept alongside ``page_label`` because they serve different
    callers: the label is for a human, the raw index is what the eval set
    matches on and what any future deep link would use.
    """

    document: str
    page: int
    page_label: str
    source_type: str
    score: float
    snippet: str


def document_title(source: str) -> str:
    """Restore the colon a filename could not carry.

    Cosmetic, and kept that way on purpose: no case normalisation, no attempt to
    fix the corpus's own typos. Titles that are shouted in the original stay
    shouted, because "correcting" them would mangle acronyms.
    """
    return source.replace(_STRIPPED_COLON, ": ", 1)


def page_label(page: int) -> str:
    """Describe the page in terms that survive the front-matter offset.

    ``load_pdf`` numbers pages from 1 in file order, which is exactly what a PDF
    viewer's page counter shows -- so this label is directly actionable ("open
    the PDF, go to page 305") even though the number printed on that page may be
    different. Saying "PDF page" rather than "p." is the whole point: it is the
    difference between a citation a student can follow and one that quietly
    misdirects them.

    Printing the *printed* number would need a per-document front-matter offset,
    and there is no honest way to fill that in for 37 documents right now --
    offsets are not uniform (roman-numeral front matter, unnumbered plates), so a
    single constant per document would be wrong for part of each one.
    """
    return f"PDF page {page}"


def snippet(text: str, max_chars: int = SNIPPET_CHARS) -> str:
    """Collapse whitespace and truncate at a word boundary.

    PDF extraction leaves ragged internal whitespace, and cutting at a fixed
    character count lands mid-word often enough to look broken. Truncation is
    marked with an ellipsis so a clipped passage is never mistaken for a
    complete one.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    cut = collapsed[:max_chars]
    # rsplit on the last space, unless the window has no space at all (one very
    # long token), in which case a hard cut is the only option.
    head = cut.rsplit(" ", 1)[0] if " " in cut else cut
    return f"{head}..."


def cite(hit, max_chars: int = SNIPPET_CHARS) -> Citation:
    """Build a ``Citation`` from a ``store.Retrieved``.

    Typed loosely on purpose: this needs only ``text``/``source``/``page``/
    ``source_type``/``score``, so it works for anything with that shape and does
    not drag a Chroma-backed import into presentation code.
    """
    return Citation(
        document=document_title(hit.source),
        page=hit.page,
        page_label=page_label(hit.page),
        source_type=hit.source_type,
        score=round(float(hit.score), 4),
        snippet=snippet(hit.text, max_chars),
    )
