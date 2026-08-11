"""Tell a document's apparatus apart from its exposition.

Every document here carries text that is not about deep learning at all: reference
lists, acknowledgments, copyright notices, tables of contents. It is chunked and
embedded exactly like prose, and it competes with prose for the five slots a student
sees. Measured against the unfiltered arm, which reproduces the frozen 2026-07-30
baseline to the digit -- so these are the numbers from before this module existed,
and ``run_eval.py --include-structural`` still reproduces them:

* "How does a sparsely gated mixture-of-experts layer route tokens to experts?" --
  a question the corpus cannot answer -- returned three reference lists as its top
  three hits, at 0.536, 0.486 and 0.481. The topic appears in this corpus *only* as
  a bibliography entry, so retrieval was matching the title of a paper nobody here
  explains.
* "What is the alignment model in attention-based translation?" spent 4 of its top
  10 slots on reference lists and a copyright notice.

That second failure is the dangerous one. A reference list is the most confident-
looking evidence a tutoring tool can hand a student for a topic it does not cover:
the words all match, and the passage says nothing.

**Precision over recall, deliberately.** A false positive deletes a real explanation
from the index and no test will ever notice; a false negative leaves one more
competing chunk, which is the status quo. So every rule here is a *conjunction*, and
the thresholds were set from the corpus's own distributions rather than guessed --
see ``scripts/audit_structure.py``, which prints the per-document flagged fraction
that would expose a rule misfiring on one document's formatting.

Heuristics rather than a model. 9,169 chunks is 9,169 classification calls, it would
have to be redone on every re-chunk, and the result would not be reproducible run to
run. These rules are free, deterministic, and each one can be explained to a student
whose passage went missing.
"""

from __future__ import annotations

import re

# Author initials, as they appear in a bibliography: "J. Goodman", "A. Beygelzimer",
# "J.L. Elman". The lookahead for a following capital is what keeps code out -- an
# early version matched "X." in ``X.reshape(X.shape[0], ...)`` and flagged a page of
# Dive into Deep Learning's multi-head attention implementation as a reference list.
_INITIALS = re.compile(r"\b[A-Z]\.(?:\s?[A-Z]\.)*(?=\s+[A-Z])")

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

# A numbered citation marker standing on its own -- "[63]", "[LYN+20]", "[Mac92]".
# The lookbehind excludes subscripts, which follow an identifier: "X.shape[2]" and
# "logits[0]" are code, not citations.
_MARKER = re.compile(r"(?<![\w)\]])\[(?:\d{1,3}|[A-Za-z][A-Za-z0-9+']{0,8}\d{2})\]")

_ET_AL = re.compile(r"\bet al\b", re.IGNORECASE)

# Where a reference points: a venue, a preprint id, a URL, a page range.
_VENUE = re.compile(
    r"\barXiv\b|\bdoi\b|https?://|\bIn Proceedings\b|\bpreprint\b|\bpp\.\s?\d|"
    r"\bvolume \d|\beditors\b|\bAdvances in Neural\b|\bConference on\b|\bJournal\b",
    re.IGNORECASE,
)

# The skeleton of a sentence. A bibliography entry is titles and surnames; it has
# almost no "is", "that", "we", "which". This is the feature that separates a
# reference list from a related-work paragraph -- both are dense in years and author
# names, but only one of them is written in sentences.
_FUNCTION_WORDS = frozenset(
    "the a an of to in is are was were be been being that this these those it its "
    "as at by for from with without on onto into over under we our us you your they "
    "them their he she his her i but or nor so if then than because while when where "
    "how why what which who whom can could may might must shall should will would do "
    "does did done have has had not no all any each both few more most other some "
    "such only own same too very just about after before again further once here "
    "there".split()
)
_ALPHA = re.compile(r"[A-Za-z']+")

_GRATITUDE = re.compile(
    r"\bthanks?\b|\bthank\b|\bgrateful\w*\b|\backnowledge\w*\b|\bfor (?:helpful|"
    r"useful|valuable|insightful) \w+|\bfor (?:discussions?|feedback|comments?)\b|"
    r"\b(?:supported|funded|sponsored) (?:by|in part)\b|\bgrant (?:no|number|#)\b",
    re.IGNORECASE,
)
# The heading itself, tolerant of the spaced small-caps PDF extraction produces
# ("A CKNOWLEDGMENTS", "A UTHOR CONTRIBUTIONS"). Spaces are squeezed out before
# matching, so this also catches "Acknowledge ments" from a bad line break.
#
# Case-sensitive, and only the noun. An earlier case-insensitive version that also
# accepted the bare verb flagged the RAG paper's broader-impacts section, which
# contains "we acknowledge" in ordinary prose -- a heading is capitalised and a verb
# in a sentence is not, so the case is the signal, not an accident of formatting.
_APPARATUS_HEADING = re.compile(
    r"ACKNOWLEDGE?MENTS?|Acknowledge?ments?|"
    r"AUTHORCONTRIBUTIONS?|AuthorContributions?|FUNDINGSTATEMENT"
)

# A table-of-contents line: dot leaders running to a page number. Both halves are
# required. Leaders alone matched BERT's Figure 3, whose architecture diagram
# extracts as "Trm Trm ... ... T1 T2 TN... ..." -- ellipses in a figure, read as a
# contents page. A run of dots that arrives at a number is a contents line and
# almost nothing else.
_CONTENTS_LINE = re.compile(r"(?:\.\s?){4,}\s*\d{1,4}\b")

# Below this many words a chunk is judged on a stricter pair of thresholds rather
# than exempted outright. A density over 40 words is a noisy estimate -- one ordinary
# paragraph citing three papers scores like a bibliography purely because it is
# short -- so the bar goes up with the noise rather than the rule switching off.
#
# Exempting short chunks entirely was the first attempt, and it let through the worst
# offender in the whole corpus: the 46-word tail of a reference page, which is the
# top hit at 0.536 for "how does a sparsely gated mixture-of-experts layer route
# tokens to experts?" -- a question nothing here answers. It scores 0.348 citation
# density against 0.055 function words, which is not a marginal call at any length.
MIN_WORDS = 60
SHORT_CITE_DENSITY = 0.20
SHORT_FUNCTION_DENSITY = 0.10

# Thresholds, from the corpus distribution (n=9,169). Citation density is at the 95th
# percentile around 0.085 and function-word density at the 10th around 0.107, so the
# conjunction selects a small, deliberately conservative corner of the joint
# distribution rather than either tail on its own.
REFERENCE_CITE_DENSITY = 0.06
REFERENCE_FUNCTION_DENSITY = 0.18
GRATITUDE_DENSITY = 0.025
CONTENTS_LINES = 2

# How far into a chunk an apparatus heading may sit before the chunk is majority
# content. Chunks do not respect section boundaries, so "ACKNOWLEDGMENTS" lands
# mid-chunk with a paper's conclusion in front of it. Measured: of the 37 chunks
# containing such a heading, 24 carry it past this mark -- flagging all 37 would have
# deleted 24 conclusions to remove 24 tails. The tail stays indexed instead, which is
# only the status quo.
HEADING_POSITION = 0.30


def _densities(text: str) -> tuple[float, float]:
    """``(citation markers per word, function words per alphabetic word)``."""
    words = text.split()
    alpha = _ALPHA.findall(text.lower())
    cites = (
        len(_INITIALS.findall(text))
        + len(_YEAR.findall(text))
        + len(_MARKER.findall(text))
        + len(_ET_AL.findall(text))
        + len(_VENUE.findall(text))
    )
    functions = sum(w in _FUNCTION_WORDS for w in alpha)
    return cites / max(len(words), 1), functions / max(len(alpha), 1)


def classify(text: str) -> str | None:
    """Name the kind of apparatus ``text`` is, or ``None`` if it is exposition.

    The reason is returned rather than a bare boolean so a flagged chunk can be
    argued with. "This 240-word passage was dropped" is not reviewable; "dropped as
    a reference list" points at the rule to check.
    """
    words = text.split()
    if not words:
        return None
    short = len(words) < MIN_WORDS

    # The heading and contents-line rules match a *pattern*, not a density, so they
    # are as reliable on a short chunk as on a long one and carry no length guard.
    squeezed = "".join(words)
    heading = _APPARATUS_HEADING.search(squeezed)
    if heading and heading.start() <= HEADING_POSITION * len(squeezed):
        return "acknowledgments"

    if len(_CONTENTS_LINE.findall(text)) >= CONTENTS_LINES:
        return "table-of-contents"

    cite_limit = SHORT_CITE_DENSITY if short else REFERENCE_CITE_DENSITY
    function_limit = SHORT_FUNCTION_DENSITY if short else REFERENCE_FUNCTION_DENSITY
    cite_density, function_density = _densities(text)
    if cite_density >= cite_limit and function_density <= function_limit:
        return "reference-list"

    # No short-chunk variant: gratitude markers are sparse even in a real
    # acknowledgments section, so the density is meaningless over 40 words and there
    # is no stricter bar that would be anything but arbitrary.
    if not short and len(_GRATITUDE.findall(text)) / len(words) >= GRATITUDE_DENSITY:
        return "acknowledgments"

    # There is deliberately no copyright/licence rule. One was written (>=2 hits on
    # "all rights reserved", "grants permission", "ISBN", "Creative Commons", ...)
    # and deleted after the audit: it fired 4 times in 9,169 chunks, and 2 of those
    # were LoRA's appendix describing which licence each *dataset* is released
    # under -- real content, and exactly the kind of loss nothing downstream would
    # have reported. The other 2 were reference-list entries the rule above already
    # catches. A rule with no unique true positives and two false ones is worse than
    # no rule.
    #
    # It leaves a known gap: the "Attention Is All You Need" title page, which ranks
    # 9th for a question about alignment models, opens with Google's permission
    # notice. That chunk also carries the title, the author list and the first
    # sentence of the abstract, so flagging it would delete abstract text to remove a
    # notice. Mixed chunks like it are the limit of a chunk-level classifier, and
    # tightening the rule until it caught that one is how the LoRA appendix
    # disappears.
    return None


def is_structural(text: str) -> bool:
    """Whether ``text`` is a document's apparatus rather than its exposition."""
    return classify(text) is not None
