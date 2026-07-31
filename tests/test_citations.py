"""Tests for how a hit is described to a reader.

Presentation code looks too trivial to test, which is exactly why the page label
is worth pinning: getting it wrong produces a citation that is precise, readable,
and points at the wrong page.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_tutoring.citations import cite, document_title, page_label, snippet


@dataclass(frozen=True)
class FakeHit:
    """The shape ``cite`` needs -- deliberately not a ``store.Retrieved``, since
    that would import Chroma to test string formatting."""

    text: str = "some passage"
    source: str = "GloVe- Global Vectors for Word Representation"
    source_type: str = "paper"
    page: int = 2
    score: float = 0.654321


def test_stripped_colon_is_restored():
    assert document_title("GloVe- Global Vectors") == "GloVe: Global Vectors"


def test_intraword_hyphens_survive():
    """A hyphen with no space after it is part of a word, not a stripped colon.

    ``Composable Sparse Fine-Tuning`` and ``DECODING-ENHANCED`` are real corpus
    titles; a blind hyphen replace would corrupt both.
    """
    assert document_title("Composable Sparse Fine-Tuning for Cross-Lingual Transfer") == (
        "Composable Sparse Fine-Tuning for Cross-Lingual Transfer"
    )
    assert document_title("DEBERTA- DECODING-ENHANCED BERT") == "DEBERTA: DECODING-ENHANCED BERT"


def test_only_the_first_hyphen_space_is_replaced():
    """The corpus has at most one per title, so a second is an unverified case.

    Replacing every occurrence would be a rule this has never been checked
    against; leaving the rest alone keeps the damage to one known-safe edit.
    """
    assert document_title("A- B- C") == "A: B- C"


def test_title_case_is_left_alone():
    """Shouted titles stay shouted -- normalising case would mangle acronyms."""
    assert document_title("LORA- LOW-RANK ADAPTATION") == "LORA: LOW-RANK ADAPTATION"


def test_page_label_says_which_numbering_it_means():
    """The label must not read as the number printed on the page.

    ``Chunk.page`` is pypdf's index; front matter puts it 18 off the printed
    number in *Dive into Deep Learning*. A bare "p. 305" is followable and wrong,
    which is worse than being explicit -- so the label names the numbering it is
    using, and "PDF page 305" stays true because it matches a viewer's counter.
    """
    assert page_label(305) == "PDF page 305"
    assert "p. 305" != page_label(305)


def test_snippet_collapses_extraction_whitespace():
    assert snippet("ragged\n  text   here") == "ragged text here"


def test_short_text_is_not_marked_as_truncated():
    """An ellipsis on a complete passage would claim there is more to read."""
    assert snippet("short enough", max_chars=50) == "short enough"


def test_snippet_truncates_on_a_word_boundary_and_says_so():
    out = snippet("alpha beta gamma delta", max_chars=14)
    assert out == "alpha beta..."
    assert not out.removesuffix("...").endswith(" ")


def test_snippet_hard_cuts_a_token_with_no_boundary_to_use():
    """One unbroken 40-character token has no space to back off to.

    Backing off to "the last space" when there is none would return an empty
    string, silently dropping the passage instead of clipping it.
    """
    out = snippet("x" * 40, max_chars=10)
    assert out == "x" * 10 + "..."


def test_cite_maps_a_hit_onto_a_reader_facing_citation():
    c = cite(FakeHit())
    assert c.document == "GloVe: Global Vectors for Word Representation"
    assert c.page == 2, "the raw index is kept for the eval set to match on"
    assert c.page_label == "PDF page 2"
    assert c.score == 0.6543, "scores are rounded for display, not carried at full precision"
    assert c.snippet == "some passage"
