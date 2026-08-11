"""Tests for telling a document's apparatus from its exposition.

Every fixture here is **synthetic** -- invented authors, invented titles, prose
written for the test. The corpus is copyrighted and this file is committed, the same
rule that keeps retrieved text out of ``eval/baseline.json``. What each one
reproduces is the *shape* that the real chunk had, and the docstring says which real
chunk that was, so a rule can be argued with without the source text being here.

The assertions that earn their place are the negative ones. A false positive deletes
a real explanation from the index and nothing downstream reports it -- the retrieval
eval only checks 32 labelled pages, so content lost on any of the other 2,600 is
invisible to it. Every rule below was tightened because of a specific false positive,
and each of those is now a test.
"""

from __future__ import annotations

from rag_tutoring.structure import (
    MIN_WORDS,
    REFERENCE_CITE_DENSITY,
    _densities,
    classify,
    is_structural,
)


def pad(text: str, words: int = MIN_WORDS + 20) -> str:
    """Extend ``text`` past the minimum length with neutral filler.

    Filler is deliberately function-word-free and citation-free, so it dilutes both
    densities equally and cannot be what makes a case pass or fail.
    """
    filler = " ".join(f"term{i}" for i in range(max(words - len(text.split()), 0)))
    return f"{text} {filler}".strip()


REFERENCES = pad(
    "A. Okonkwo, B. Lindqvist, and C. Ferreira. Learning representations from weak "
    "supervision. In Proceedings of the 2019 Conference on Machine Learning, pages "
    "1102-1115, 2019. D. Nakamura and E. Sorenson. Scaling laws for tabular models. "
    "arXiv preprint arXiv:1907.04412, 2019. F. Adeyemi, G. Vasquez, and H. Park. "
    "Gradient routing in sparse networks. Journal of Applied Learning, 12(3):88-104, "
    "2020."
)

# Deliberately as citation-dense as a bibliography (0.115 markers per word, against a
# 0.06 threshold), so the *only* thing keeping it out of the filter is that it is
# written in sentences. An earlier version of this fixture cited too sparsely: it
# passed with the function-word rule deleted, which made it a test of nothing.
RELATED_WORK = (
    "These approaches have been generalized to coarser granularities, such as sentence "
    "embeddings (Okonkwo et al., 2019; Lindqvist and Ferreira, 2019) or paragraph "
    "embeddings (Nakamura and Sorenson, 2018). To train sentence representations, prior "
    "work has used objectives that rank candidate next sentences (Adeyemi et al., 2020; "
    "Vasquez et al., 2020), or that generate the words of the following sentence given a "
    "representation of the previous one (Park et al., 2017). We follow the second of "
    "these and show in Section 4 that it transfers better to small datasets."
)


def test_a_reference_list_is_flagged():
    """The failure this whole module exists for.

    A bibliography is the most confident-looking evidence a tutoring tool can hand a
    student for a topic it does not cover: every word of the question matches, and
    the passage explains nothing. Three of the corpus's five unanswerable questions
    appear in it only as reference entries.
    """
    assert classify(REFERENCES) == "reference-list"


def test_a_paragraph_that_cites_its_sources_is_not_a_reference_list():
    """The load-bearing negative, and the reason the rule is a conjunction.

    A related-work paragraph and a bibliography are equally dense in years and author
    names. Citation density alone flagged both -- and a related-work section is
    exactly the kind of passage a student asking "what came before transformers?"
    needs. What separates them is that one is written in sentences, so function-word
    density is the second half of the test.

    The two asserts are one test on purpose: the first is what the filter must do,
    the second is what makes the first able to fail. Without it this passes on a
    fixture that never came near the threshold in the first place.
    """
    cite_density, _ = _densities(RELATED_WORK)
    assert cite_density >= REFERENCE_CITE_DENSITY, "fixture must clear the citation bar"
    assert classify(RELATED_WORK) is None


def test_code_is_not_a_reference_list():
    """A textbook's code listings are content, and they look structural on paper:
    no function words at all, and an early version of the author-initials pattern
    matched the ``X.`` in ``X.reshape``. Reproduces Dive into Deep Learning p412,
    the multi-head attention implementation.
    """
    code = pad(
        "def transpose_qkv(X, num_heads): X = X.reshape(X.shape[0], X.shape[1], "
        "num_heads, -1) X = X.transpose(0, 2, 1, 3) output = X.reshape(-1, "
        "X.shape[2], X.shape[3]) return output",
        words=MIN_WORDS + 5,
    )
    assert classify(code) is None


def test_a_short_chunk_is_judged_on_a_stricter_bar_rather_than_exempted():
    """Density over a page-tail chunk is a noisy estimate, so the bar rises with the
    noise. Both halves of that are load-bearing, and each comes from a real chunk.

    The prose case reproduces Longformer p8's last chunk: one ordinary sentence
    citing three papers, 34 words, which clears the *long*-chunk citation bar purely
    because it is short.

    The reference case reproduces the 46-word tail of a reference page in the
    Compute-Optimal paper -- the top hit at 0.536 for a question about
    mixture-of-experts, which nothing in this corpus explains. An earlier version
    exempted short chunks outright and let exactly that one through, which is how a
    precision guard turns into a hole.
    """
    prose = (
        "Nevertheless our method outperforms the alternatives (Okonkwo et al., 2019; "
        "Park et al., 2020; Vasquez et al., 2020). 8"
    )
    tail = (
        "A. Okonkwo, B. Lindqvist, C. Ferreira, and D. Nakamura, editors, Advances in "
        "Applied Learning, volume 32. Northgate Press, 2019. URL "
        "https://example.org/proceedings/2019/0412.pdf. E. Sorenson, F. Adeyemi, and "
        "G. Vasquez, 2020."
    )
    assert len(prose.split()) < MIN_WORDS and len(tail.split()) < MIN_WORDS
    assert classify(prose) is None
    assert classify(tail) == "reference-list"


def test_an_acknowledgments_section_is_flagged():
    ack = pad(
        "ACKNOWLEDGMENTS We gratefully acknowledge the compute contributed by our "
        "institutional partners, and we thank R. Okonkwo, S. Lindqvist and T. Ferreira "
        "for generous feedback and helpful discussions throughout this project."
    )
    assert classify(ack) == "acknowledgments"


def test_a_heading_late_in_a_chunk_does_not_condemn_the_chunk_in_front_of_it():
    """Chunks do not respect section boundaries, so a heading lands mid-chunk with a
    paper's conclusion in front of it.

    Measured: of the 37 chunks in this corpus carrying an apparatus heading, 24 carry
    it past the 30% mark. Flagging all 37 would have deleted 24 conclusions to remove
    24 tails. The tail stays indexed instead, which is only the status quo.
    """
    conclusion = pad(
        "We have shown that the method converges under the assumptions of Section 3, "
        "and that the same argument applies when the batch size is small, which is "
        "the setting most practitioners actually work in. Future work should address "
        "the case where the gradients are stale.",
        words=140,
    )
    assert classify(f"{conclusion} ACKNOWLEDGMENTS We thank our reviewers.") is None


def test_the_verb_acknowledge_in_prose_is_not_a_heading():
    """A heading is capitalised; a verb in a sentence is not, and that case is the
    only thing separating them once whitespace is squeezed out. A case-insensitive
    version of this rule flagged the RAG paper's broader-impacts section, which is
    ordinary prose about misuse risks.
    """
    prose = pad(
        "In order to mitigate these risks we acknowledge that a deployed system needs "
        "monitoring, and that the safeguards described above are only partial."
    )
    assert classify(prose) is None


def test_a_contents_page_is_flagged():
    contents = pad(
        "4.7 Forward Propagation, Backward Propagation . . . . . . . . 167 4.7.1 "
        "Forward Propagation . . . . . . . . . . . . . . . 168 4.7.2 Computational "
        "Graph . . . . . . . . . . . . . . . 168 4.7.3 Backpropagation . . . . . . . "
        ". . . . . . . . . . 170"
    )
    assert classify(contents) == "table-of-contents"


def test_ellipses_in_a_figure_are_not_a_contents_page():
    """Reproduces BERT p13, whose architecture diagram extracts as a row of labels
    separated by ellipses. Dot leaders alone matched it; requiring the run of dots to
    arrive at a page number is what a contents line has and a figure does not.
    """
    figure = pad(
        "Trm Trm Trm Trm ... ... Trm Trm Trm Trm ... ... E1 E2 EN ... ... T1 T2 TN "
        "... ... Figure 3: Differences in pre-training model architectures."
    )
    assert classify(figure) is None


def test_classify_names_the_rule_that_fired():
    """A reason rather than a boolean, so a flagged chunk can be argued with. "This
    240-word passage was dropped" is not reviewable; "dropped as a reference list"
    points at the rule to go and check."""
    assert classify(REFERENCES) == "reference-list"
    assert is_structural(REFERENCES) is True
    assert is_structural(RELATED_WORK) is False
