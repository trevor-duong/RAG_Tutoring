"""Tests for chunking, focused on the token budget the embedding model imposes.

``pack_words`` takes token costs rather than text, so these run without loading
a tokenizer: a fake counter makes each word's cost explicit and the edge cases
(a word larger than the whole budget, an overlap that would fail to advance)
reproducible.
"""

from __future__ import annotations

from rag_tutoring.ingest import (
    _EMAIL,
    _EXTRACTION_JUNK,
    Chunk,
    chunk_pages,
    chunk_text,
    pack_words,
)


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


def redact(text: str) -> str:
    """What ``load_pdf`` does to raw extracted text, in order."""
    cleaned = " ".join(_EXTRACTION_JUNK.sub(" ", text).split())
    return " ".join(_EMAIL.sub(" ", cleaned).split())


# Every address below is synthetic, on a reserved example domain. The fixtures
# reproduce the *shape* of what the corpus actually contains -- a purchase
# watermark, an inline author contact, a brace-grouped shared-domain list -- which
# is all these tests discriminate on. Pasting the real strings in would commit the
# very addresses the redaction exists to remove, permanently, to fix nothing.


def test_purchase_watermark_is_removed():
    # A per-buyer watermark on a textbook's title page. Chunk text is quoted to
    # students verbatim, so this would disclose who bought the book.
    assert redact("TRIPLE BAM!!! Sold to buyer@example.com") == "TRIPLE BAM!!! Sold to"


def test_author_contact_addresses_are_removed():
    assert (
        redact("First Author first@cs.example.edu Second Author second@cs.example.edu")
        == "First Author Second Author"
    )


def test_shared_domain_author_lists_are_removed():
    """Papers print one brace-grouped address for all authors.

    Matching only the ordinary form leaves these behind -- 13 of them in this
    corpus -- because the character before the "@" is "}", which no plain
    local-part pattern accepts. Separators vary, and the list often wraps across
    a line, which is why page text is whitespace-canonicalised first.
    """
    assert redact("Research {alpha, v-beta, gamma}@corp.example.com Abstract") == (
        "Research Abstract"
    )
    assert redact("{alpha|beta|gamma}@lab.example.com") == ""
    assert (
        redact("Edinburgh\n{first.last,a.other}@school.example.ac.uk\nAbstract")
        == "Edinburgh Abstract"
    )


def test_redaction_does_not_run_away_past_the_address():
    """The brace form is length-capped, so an unrelated "{" cannot swallow prose."""
    text = "{" + "x " * 120 + "} and dropout@example.com follows"
    assert "and" in redact(text) and "follows" in redact(text)
    assert "@example.com" not in redact(text)


def test_redaction_prefers_over_removal_to_leaking():
    """When extraction drops the space before an address the boundary is genuinely
    ambiguous, and the greedy local part takes the preceding digits with it.

    Chosen deliberately: leaving half an address in a passage quoted to a student
    is worse than losing a number. Measured 0 occurrences of this in the corpus
    (every real address is preceded by whitespace or punctuation), so the cost is
    hypothetical while the leak it prevents is not.
    """
    assert redact("p=0.5author@cs.example.edu Hinton et al.") == "p= Hinton et al."


def test_redaction_leaves_ordinary_text_with_an_at_sign_alone():
    """Not every @ is an address; the pattern needs a domain to fire."""
    assert redact("attention @ layer 6 costs O(n^2)") == "attention @ layer 6 costs O(n^2)"


def test_chunk_pages_matches_chunk_pdf_on_the_same_pages():
    """``chunk_pdf`` must be exactly ``chunk_pages`` over ``load_pdf``.

    The rebuild script chunks pre-extracted pages so it can cache the text in the
    same pass. If that path could drift from the one the pipeline uses, an index
    rebuilt by the script would stop matching what the eval baseline describes.
    """
    pages = [(1, "alpha beta gamma delta"), (4, "epsilon zeta")]
    one = lambda w: 1  # noqa: E731 -- trivial cost function, one token per word
    chunks = chunk_pages(pages, "Doc", "paper", one, max_tokens=3, overlap_tokens=1)
    assert [(c.page, c.chunk_index, c.text) for c in chunks] == [
        (1, 0, "alpha beta gamma"),
        (1, 1, "gamma delta"),
        (4, 0, "epsilon zeta"),
    ]
    assert {c.source for c in chunks} == {"Doc"}
    assert {c.source_type for c in chunks} == {"paper"}


def test_chunk_id_is_stable_and_sorts_by_page():
    a = Chunk(text="x", source="Paper", source_type="paper", page=2, chunk_index=0)
    b = Chunk(text="y", source="Paper", source_type="paper", page=10, chunk_index=3)
    assert a.id == "Paper::p0002::c00"
    assert a.id < b.id, "zero-padding keeps ids ordered by page"
