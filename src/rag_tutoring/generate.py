"""Synthesise a short answer from retrieved passages, cited back to them.

This layer sits *over* retrieval and consumes what ``/ask`` already returns. It adds
an answer; it does not replace the passages, which keep rendering underneath. That
shape is the whole design, and it comes from the baseline: a correct page is in the
top 5 half the time but is the *top* hit only 19% of the time, so an answer built
from these passages is sometimes built from the wrong ones. Showing the sources it
was built from is what keeps that auditable by the student rather than invisible.

Two rules are enforced here rather than hoped for:

* **Grounding is instructed, and citations are checked.** The model is told to answer
  only from the passages, and every ``[source N]`` marker it emits is validated
  against the set it was actually handed. Prompting for citations and never checking
  them would be a check that cannot fail.
* **The model does not abstain, but it does not invent either.** Thin passages
  produce an answer that says what they do and do not cover, plus the closest thing
  they genuinely say. "Always answer" must not decay into "answer from pretraining",
  because a hedged sentence sourced from the model's own memory is a confabulation
  wearing a disclaimer -- and the student has no way to tell the difference.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag_tutoring.citations import Citation
from rag_tutoring.config import (
    GENERATION_MAX_TOKENS,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
)

# Bumped whenever the prompt changes in a way that could move answer quality, and
# kept in this file rather than config.py precisely so it cannot drift away from the
# text it versions. Stamped into every eval report.
PROMPT_VERSION = "1"

# Citations are marked "[source 3]", not "[3]". Measured against the extracted corpus
# before choosing: bare "[N]" occurs 1,342 times across 11.6% of the 2,639 pages,
# because papers are full of bracketed reference numbers. Validating a bare marker
# would therefore collide with the corpus's own citations -- a quoted "[15]" would
# either fail validation for no reason or, with a larger k, validate as a genuine
# index pointing at an unrelated source. "[source N]" and "[[N]]" both occur zero
# times; the labelled form wins for being self-explanatory to a student.
_MARKER = re.compile(r"\[source (\d{1,2})\]", re.IGNORECASE)

SYSTEM_PROMPT = """\
You help a student understand deep learning, using only the numbered passages \
below, which come from their tutor's own course materials.

Answer from the passages and nothing else. Do not add facts from your own \
knowledge, even when you are confident they are correct: the student is shown \
these exact passages beneath your answer and can check you against them, and an \
unsupported claim is worse than an incomplete one precisely because it looks the \
same as a supported one.

Cite as you go. Put [source N] immediately after each claim, using the passage \
number it comes from. Cite only numbers that appear in the passages given to you.

When the passages only partly cover the question, do not refuse and do not fill \
the gap from memory. Say plainly what they do and do not address, then give the \
closest thing they genuinely do say, marked as related rather than as the answer. \
The student can then decide whether to read further.

Keep it short -- a few sentences, or one short paragraph. The passages are shown \
directly below your answer, so you are orienting the student toward the right \
reading, not replacing it.

The passages are text extracted from PDFs, so expect missing spaces, broken \
mathematical notation and interleaved columns. Read through those artifacts; do \
not quote them back or treat them as meaningful.\
"""


class UngroundedCitation(Exception):
    """The model cited a passage number it was not given.

    Raised rather than repaired. Stripping the bad marker would leave prose that
    reads as sourced while pointing at nothing, and renumbering it would invent a
    source the model never claimed -- both convert a detectable failure into an
    undetectable one. The caller drops the answer and shows the passages alone,
    which is the behaviour the system had before generation existed.
    """


@dataclass(frozen=True)
class Answer:
    """A synthesised answer and the provenance needed to reproduce it."""

    text: str
    cited: tuple[int, ...]  # 1-based passage numbers, in first-appearance order
    model: str
    prompt_version: str


def format_passages(citations: Sequence[Citation]) -> str:
    """Number the passages for the model the same way a student sees them.

    The document title and page label go in alongside the text so the model can say
    "the Dropout paper" rather than "passage 3", which reads better and lets a
    student match a sentence to the citation block below it.
    """
    return "\n\n".join(
        f"[source {i}] {c.document}, {c.page_label}\n{c.snippet}"
        for i, c in enumerate(citations, start=1)
    )


def cited_sources(text: str, available: int) -> tuple[int, ...]:
    """Extract the passage numbers an answer cites, rejecting any that don't exist.

    Order is first appearance, deduplicated -- useful for rendering, and stable
    enough to store in an eval result.
    """
    seen: list[int] = []
    for match in _MARKER.finditer(text):
        n = int(match.group(1))
        if not 1 <= n <= available:
            raise UngroundedCitation(
                f"answer cites [source {n}] but only {available} passages were provided"
            )
        if n not in seen:
            seen.append(n)
    return tuple(seen)


class Generator:
    """Turns a question plus retrieved passages into a cited answer.

    Takes an already-constructed client rather than building one, so tests inject a
    stub and the suite needs no API key -- the same inversion that lets the rest of
    the suite run with no index and no embedding model.
    """

    def __init__(
        self,
        client,
        model: str = GENERATION_MODEL,
        temperature: float = GENERATION_TEMPERATURE,
        max_tokens: int = GENERATION_MAX_TOKENS,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer(self, question: str, citations: Sequence[Citation]) -> Answer:
        """Answer ``question`` from ``citations``.

        Raises ``ValueError`` on an empty passage set: there is nothing to ground an
        answer in, and asking the model anyway would be inviting exactly the
        unsourced answer this module exists to prevent.
        """
        if not citations:
            raise ValueError("cannot generate an answer with no retrieved passages")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\nPassages:\n\n{format_passages(citations)}"
                    ),
                }
            ],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
        return Answer(
            text=text,
            cited=cited_sources(text, len(citations)),
            model=self.model,
            prompt_version=PROMPT_VERSION,
        )


def generator_from_env() -> Generator | None:
    """Build a generator if a key is configured, otherwise ``None``.

    ``None`` means generation is off and the API serves passages alone -- which is a
    working product, not a degraded one, since that is exactly what it did before
    this module existed. Returning ``None`` rather than raising is what keeps the key
    optional: the tests, the notebook and a local run without a key all still work,
    and nothing has to be stubbed out to achieve that.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None

    import anthropic  # imported lazily so the package is only needed when used

    return Generator(anthropic.Anthropic(api_key=key))
