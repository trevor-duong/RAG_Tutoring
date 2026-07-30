"""Tests for chunking, focused on the token budget the embedding model imposes.

``pack_words`` takes token costs rather than text, so these run without loading
a tokenizer: a fake counter makes each word's cost explicit and the edge cases
(a word larger than the whole budget, an overlap that would fail to advance)
reproducible.
"""

from __future__ import annotations

from rag_tutoring.ingest import _EXTRACTION_JUNK, Chunk, chunk_text, pack_words


def spans_fit(costs: list[int], spans: list[tuple[int, int]], budget: int) -> bool:
    """Every span is within budget, except a lone word that cannot be split."""
    return all(sum(costs[s:e]) <= budget or e - s == 1 for s, e in spans)


def test_packs_up_to_the_budget_and_no_further():
    costs = [10] * 10
    spans = pack_words(costs, max_tokens=30, overlap_tokens=0)
    assert spans == [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert spans_fit(costs, spans, 30)


def test_every_word_appears_at_least_once():
    costs = [3, 7, 1, 9, 4, 2, 8, 5, 6, 1, 3, 7]
    spans = pack_words(costs, max_tokens=15, overlap_tokens=5)
    covered = {i for s, e in spans for i in range(s, e)}
    assert covered == set(range(len(costs)))


def test_overlap_carries_context_between_windows():
    costs = [5] * 12
    spans = pack_words(costs, max_tokens=20, overlap_tokens=10)
    # each window holds 4 words; 10 tokens of overlap re-includes the last 2
    assert spans[0] == (0, 4)
    assert spans[1][0] == 2, "second window should start inside the first"


def test_word_larger_than_budget_is_emitted_alone_and_does_not_stall():
    costs = [4, 500, 4]  # a mis-extracted table can produce one enormous "word"
    spans = pack_words(costs, max_tokens=50, overlap_tokens=10)
    assert (1, 2) in spans, "the oversized word must be isolated in its own span"
    assert {i for s, e in spans for i in range(s, e)} == {0, 1, 2}


def test_expensive_words_cannot_make_overlap_stall():
    # Each word nearly fills the budget, so the overlap step-back would land on
    # the current start and loop forever if it were not clamped.
    costs = [24] * 6
    spans = pack_words(costs, max_tokens=25, overlap_tokens=20)
    starts = [s for s, _ in spans]
    assert starts == sorted(set(starts)), "window starts must strictly increase"
    assert len(spans) == 6


def test_chunk_text_preserves_original_casing_and_punctuation():
    text = "Multi-Head Attention (Vaswani et al., 2017) improves BLEU by 2.0."
    chunks = chunk_text(text, count_tokens=lambda w: 1, max_tokens=1000, overlap_tokens=0)
    assert chunks == [text], "chunk text is quoted as a citation and must not be normalised"


def test_chunk_text_respects_budget_with_a_realistic_cost_function():
    words = [f"word{i}" for i in range(200)]
    costs = {w: (4 if i % 7 == 0 else 1) for i, w in enumerate(words)}
    chunks = chunk_text(
        " ".join(words), count_tokens=costs.__getitem__, max_tokens=20, overlap_tokens=5
    )
    assert all(sum(costs[w] for w in c.split()) <= 20 for c in chunks)
    assert " ".join(chunks).split()[0] == "word0"


def test_empty_text_produces_no_chunks():
    assert chunk_text("   ", count_tokens=lambda w: 1) == []


def clean(text: str) -> str:
    """What ``load_pdf`` does to raw extracted text."""
    return " ".join(_EXTRACTION_JUNK.sub(" ", text).split())


def test_extraction_junk_never_fuses_two_values_together():
    # A control char standing in for a table glyph, with no spaces around it.
    # Deleting it would invent the number 547.0; replacing it must not.
    assert clean("Top-5\x1747.0") == "Top-5 47.0"


def test_extraction_junk_is_removed():
    assert clean("F(x)\x01+\x01x") == "F(x) + x"
    assert clean("null\x00byte and \x94C1\x93 controls") == "null byte and C1 controls"
    assert clean("lone \ud835 surrogate") == "lone surrogate"


def test_ordinary_whitespace_survives_cleaning():
    assert clean("tabs\tand\nnewlines\rcollapse") == "tabs and newlines collapse"


def test_chunk_id_is_stable_and_sorts_by_page():
    a = Chunk(text="x", source="Paper", source_type="paper", page=2, chunk_index=0)
    b = Chunk(text="y", source="Paper", source_type="paper", page=10, chunk_index=3)
    assert a.id == "Paper::p0002::c00"
    assert a.id < b.id, "zero-padding keeps ids ordered by page"
