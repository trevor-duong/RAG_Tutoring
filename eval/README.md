# Retrieval eval set

`questions.jsonl` holds tutoring questions with known-correct sources, so
retrieval quality is a number rather than an impression. Run it with:

```bash
python scripts/run_eval.py --save eval/baseline.json
```

## What a label is

Each question lists the `(document, page)` pairs that genuinely answer it. A
question is a hit at `k` if any labelled page appears in the top `k` results.

`page` is the **indexer's** page number — pypdf's page index, which is what
`Chunk.page` stores — not the number printed on the page. In *Dive into Deep
Learning* those differ by 18, so every label here was derived from `load_pdf`
output. Reading page numbers off a PDF viewer would offset every textbook label
and look exactly like a retrieval failure.

Labels are pages rather than chunk ids on purpose. A chunk id encodes the chunk
setting (`source::pNNNN::cNN`), so re-chunking the corpus renames every chunk and
would invalidate the whole set. Page numbers survive re-chunking, which is what
lets this set score a chunking change — the main reason to have it.

## How the set was built

Questions are stratified over the topics actually tutored — 20 of them, from
backpropagation through decoding — with each topic drawn from the document that
covers it best, across both papers and textbooks.

Ground-truth pages were found by **lexical** search over the extracted text,
never by running the retriever. Labelling with the retriever would make the eval
circular: it could only ever confirm that retrieval finds what retrieval found.

Each question is tagged with how it is phrased, because that is the axis that
most changes the score:

| `style` | meaning |
| --- | --- |
| `verbatim-term` | uses the source's own term ("what is gradient clipping?") |
| `paraphrase` | asks for the concept in different words |
| `symptom` | describes what the student is seeing, never naming the concept |

Results are reported broken out by style. An aggregate over a set that is mostly
`verbatim-term` looks rigorous and measures very little, since matching a term
against a passage containing it is the easy case.

Five topics carry a `verbatim-term` and a `symptom`/`paraphrase` question with
**identical labels** (`sgd-minibatch` / `sgd-batch-tradeoff`, `bpe-subwords` /
`oov-garbage`, and three more). Those pairs isolate phrasing: same target pages,
different wording, so the rank difference between them is not confounded with
topic difficulty. The runner finds them automatically — any two questions sharing
a label set — and prints their ranks side by side.

## Questions the corpus cannot answer

Five entries have `"sources": []`. They are on-domain questions a student might
plausibly ask that nothing here answers — diffusion models, state space models,
graph neural networks, mixture-of-experts, knowledge distillation — each verified
absent by lexical search. Three of them appear in bibliographies but are never
explained, which makes them the hard cases: a confident citation to a reference
list is exactly the failure a tutoring tool must not make.

These are scored **separately** and never folded into recall or MRR. Mixing
questions with no correct answer into an aggregate drags it toward zero and makes
it uninterpretable. They answer a different question — can the system tell when
it has nothing? — and the runner reports the two score distributions side by side
so it is visible whether they separate at all.

## Reading the numbers honestly

**Recall here is a lower bound.** The 37 documents overlap heavily on these
topics, so a retrieved passage that genuinely answers a question but was never
labelled counts as a miss. That is the standard position with incomplete
relevance judgements. The set is fixed, so the numbers stay comparable across
chunking and model changes even where they understate absolute quality.

**A score means nothing without the index that produced it.** Every report
stamps the embedding model, chunk budget, overlap, collection name, document
count, and chunk count. Without that, `recall@5 = 0.72` cannot answer "did my
change help?"

**No abstention threshold is recommended from five negatives.** The runner
reports what a given cutoff would cost (answerable questions wrongly refused)
alongside what it would buy (unanswerable questions correctly refused). Picking a
single constant off a handful of examples would be false precision.

## Baseline (2026-07-30)

Against 9,169 chunks from all 37 documents, 240-word-piece chunks, MiniLM-L6-v2:

| | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| all 32 labelled | 0.19 | 0.41 | **0.50** | 0.69 | 0.319 |
| `verbatim-term` (n=18) | 0.22 | 0.50 | 0.61 | 0.83 | 0.390 |
| `paraphrase` (n=5) | 0.20 | 0.40 | 0.40 | 0.80 | 0.317 |
| `symptom` (n=9) | 0.11 | 0.22 | 0.33 | 0.33 | 0.176 |

Half the time a student's question puts a correct source in the top 5. Three
things the breakdown shows that the aggregate hides:

- **Phrasing dominates.** `symptom` recall does not improve at all between k=5
  and k=10 — when the wording never names the concept, the right page is not
  merely ranked low, it is absent from the candidate set. Four of the five
  controlled pairs split in the expected direction. The fifth reversed:
  `oov-garbage` ("nonsense for words not in the vocabulary") hit at rank 3 while
  `bpe-subwords` ("what is byte pair encoding") missed entirely, because that
  paper's abstract is itself written in symptom language. So the effect is not
  "students phrase things badly" — it is that retrieval matches surface framing,
  whichever direction that cuts.
- **Non-content pages compete with content.** The top hit for "training loss down,
  validation error up" is page 14 of the Lottery Ticket paper at 0.498 — above the
  lowest score of any question that did retrieve a correct source. Acknowledgments,
  reference lists, tables of contents and title pages are all chunked and indexed
  like prose.

  **Corrected 2026-08-10.** This entry originally called that hit "the
  acknowledgments section", because page 14 *is* the acknowledgments page and this
  report identifies hits by page. At chunk level it is not: page 14 holds five
  chunks, and the one that scored 0.498 is appendix prose about validation loss
  rising as a model overfits — a defensible hit from an unlabelled document. The
  finding survives (see the four verified cases in the filter section below); the
  example was wrong, and the reason it was wrong is exactly why the filter that
  came out of it classifies chunks and not pages.
- **A similarity threshold cannot detect out-of-corpus questions here.** The two
  score distributions overlap: the 16 questions that retrieved a correct source in
  the top 5 span 0.476–0.766, the 5 unanswerable ones 0.421–0.600. No cutoff
  separates them, and the non-content-page finding is part of why. Abstention will
  need something other than raw top-1 similarity. (The cost side counts those 16,
  not all 32 — refusing a question retrieval already got wrong discards nothing.)

`paraphrase` is n=5 — too few to support a conclusion in either direction.

Labels were **not** revised after seeing these results, and two known
consequences are left standing rather than papered over: `attention-alignment`
labels only Bahdanau p3 while the alignment model continues onto p4, which
retrieval ranks 2nd; and the corpus overlaps enough that several misses retrieved
a reasonable passage from an unlabelled document. Both push recall down. Adding a
label because retrieval happened to surface it is how a set stops measuring
anything. The five highest-scoring misses were audited against the source pages
before freezing: four were genuine failures (fastText subwords for a BPE
question, BERT *encoder code* for the MLM objective, batch norm for a layer norm
question, an acknowledgments page for overfitting), one borderline.

### These numbers are post-redaction, and redaction cost a little recall

Email addresses are stripped at extraction (see `ingest.load_pdf`) because one
textbook's title page carries a per-buyer purchase watermark and paper title
pages carry author contact addresses — both would be quoted verbatim to a
student. That changed 23 of 2,639 pages, including 7 labelled ones, and the index
went from 9,174 chunks to 9,169.

recall@5 held at 0.50, but recall@10 fell (0.75 → 0.69) and MRR fell
(0.359 → 0.319). The mechanism is worth recording: removing chunks can only
*promote* things into a top-10, so a demotion has to come from a changed chunk
scoring *higher*. The changed chunks are title pages, and stripping a block of
email garbage out of a page that is otherwise title-plus-abstract makes it a
better match for almost any on-topic query. `Attention Is All You Need` p1 now
appears at rank 9 for a question about alignment models. In other words the
privacy fix made non-content pages *more* competitive — the same finding arriving
from the other direction, and the reason a structural filter was the next
retrieval experiment rather than a nice-to-have.

## Structural filter (2026-08-10)

Reference lists, acknowledgments and contents pages are 8.5% of the corpus (782 of
9,169 chunks) and are now excluded from retrieval by default. Same index, same
embeddings, filtered at query time — so the pre-filter arm is a flag away rather
than a rebuild:

```bash
python scripts/run_eval.py                        # 8,387 chunks searchable
python scripts/run_eval.py --include-structural   # all 9,169
```

| | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| all 32, filtered | 0.19 | 0.41 | 0.50 | **0.72** | **0.323** |
| all 32, unfiltered | 0.19 | 0.41 | 0.50 | 0.69 | 0.319 |

**The unfiltered arm reproduces the 07-30 baseline to the digit** — every rate,
every per-question rank, every top-1 score. That is what makes the comparison mean
anything: the index was rebuilt, and the check proves the rebuild changed nothing
except the flag.

**Three questions moved, and naming them is the honest report.** n=32, so one
question is 0.031 of recall and a rate hides how little happened:

- `attention-alignment` — **miss → rank 10**. Four of its top ten were apparatus:
  two reference-list chunks, a third from another paper, and the *Attention Is All
  You Need* title page. Removing three of them let a labelled page in.
- `momentum-method` — rank 5 → 4, a reference-list chunk cleared out from above it.
- `neg-moe` — top-1 fell from 0.536 to 0.477. The 0.536 was a bibliography entry:
  the corpus mentions mixture-of-experts *only* in reference lists, so retrieval had
  been matching the title of a paper nobody here explains. This is the case the
  filter exists for and it moves no rate at all, because the question has no
  labelled page to hit.

**Recall was never the right instrument for this.** A boilerplate chunk crowding
out a fourth good passage changes no rate, and on the five negatives there is no
rate to change. So the report counts slots instead: before the filter, **11 of 185
top-5 slots on 7 of 37 questions** were apparatus, including 3 of 5 for both
`neg-moe` and `neg-distillation`. After, zero by construction.

**Abstention separated slightly better, and it is still not separable.** The
negatives' median top-1 fell 0.536 → 0.509 while the answered questions' band held
at 0.476–0.766, so a 0.50 cutoff now refuses 2 of 5 negatives instead of 1, at the
same cost of 2 of 16. The distributions still overlap. The 07-30 conclusion stands:
abstention needs something other than raw top-1 similarity.

### What the filter is allowed to break, and how that is checked

A false positive deletes a real explanation, and the retrieval eval **cannot see
it** — it checks 32 labelled pages, so content lost on any of the other 2,600 is
invisible. `scripts/audit_structure.py` is the check that can:

- **Flagged fraction per document.** A rule misfiring on one document's formatting
  shows up as that document sitting far above the rest. Papers land at 6–22%,
  *Dive into Deep Learning* at 5.7%, Nielsen at 0.7%, StatQuest at 0.3% — the shape
  you would predict, since a paper's bibliography is a real fraction of its pages
  and a textbook's is not. The one document above 30% is the RAG paper at 40.7%,
  and it is genuine: pages 10–16 of 16 are references and appendix.
- **Labelled pages losing every chunk.** Zero, and the script exits non-zero on any.
- **Reading the flags.** All 14 short-chunk flags were read individually; all 14 are
  reference tails, footnote-URL lists or a title page.

Three rules were cut or tightened *because* of that audit, and each is now a test:
a copyright/licence rule that fired four times and was wrong twice (LoRA's appendix
describing which licence each dataset is released under is content); a
case-insensitive acknowledgments heading that matched "we acknowledge" in the RAG
paper's broader-impacts prose; and dot-leader detection that matched the ellipses
in BERT's Figure 3.

Two known gaps are left standing rather than closed by tightening until they went
away. The *Attention Is All You Need* title page still ranks for alignment
questions: its chunk carries Google's permission notice *and* the title, authors
and first sentence of the abstract, so flagging it would delete abstract text.
And one 89-word reference chunk scores 0.189 function-word density against a 0.18
threshold — a near miss that stays indexed, because moving the threshold to catch
it would leave a 0.02 margin on real related-work prose.

## Generation baseline (2026-08-10, filtered index)

The second instrument, `claude-sonnet-5` at k=5 over the same 37 questions. Re-run
after the structural filter, because a change in which chunks are searchable changes
what generation is scored on without a line of the prompt moving:

| partition | n | answered | uncited | ungrounded | truncated | cited-label | hedged\* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hit | 16 | 16 | 0 | 0 | 0 | 15 | 8 |
| miss | 16 | 16 | 0 | 0 | 0 | 0 | 13 |
| negative | 5 | 5 | 0 | 0 | 0 | 0 | 5 |

The 08-05 run on the unfiltered index was identical except `hedged` on the hit
partition, which read 10. That is inside the run-to-run band described below, so it
is not a finding.

**Zero ungrounded citations and zero uncited answers across 37 questions.** Every
`[source N]` marker resolved to a passage actually supplied. That is the mechanical
property the validation exists to enforce, and on this run it never had to fire.

**All five negatives declined, and the filter did not make that harder.** This was
the specific risk in filtering bibliographies out: three of the five negatives used
to decline partly *because* their top passage was visibly a reference list, and
removing it hands the model something that reads more like prose. The
mixture-of-experts answer now retrieves a passage about "products of experts" —
a genuine near-miss — and names it as a different concept from sparse gating rather
than answering from it. n=5, so this is an existence proof that the mechanism can
work, not a rate.

**One cosmetic artifact.** The state-space answer emitted `[source none]`, which
looks like a citation and is not one. Validation ignores it (the marker pattern
requires a number), so it is not an ungrounded citation — but a student sees
something citation-shaped pointing at nothing. Not chased yet; recorded so it is not
rediscovered as a validation bug.

**The hedge proxy under-counts.** All four `miss` answers it scored as unhedged do
hedge, in wording it does not match — "the passages don't spell out", "this is a
partial picture", "the direct comparison is limited". Treat 13/16 as a floor.

**A "miss" is not a bad answer.** Several answers on labelled-miss questions are
substantively right, because the corpus overlaps and retrieval surfaced correct but
unlabelled pages — the incomplete-judgements caveat above, showing up from the
generation side. `cited_a_labelled_page` is 0 on that partition by construction; it
measures labelling, not quality.

**These numbers are a sample, not a constant.** `claude-sonnet-5` rejects `temperature`
and `top_p`, so decoding cannot be pinned. Between two runs an hour apart the hit
partition's `cited-label` moved 16 → 15 and `hedged` moved 12 → 13 with no code change
in between. Only a movement well outside that band is evidence of anything.

\*hedged is a keyword proxy, not a judgement. The 21 miss and negative answers are
short; reading them is the measurement.

## Files

- `questions.jsonl` — the set. Notes paraphrase why a page answers a question;
  no source text is quoted, since the corpus is copyrighted.
- `baseline.json` — a saved report to compare future runs against. Scores and
  citations only, no retrieved text (which would carry both copyrighted material
  and, before redaction, an email address). Its `provenance` records
  `structural_filter`, because two reports with the same chunk count can still
  describe different candidate sets.
- `generation-baseline.json` — a **separate** instrument, written by
  `run_generation_eval.py`. It scores what the model does with these passages, over
  the same question set: mainly the ~16 questions where retrieval put no correct page
  in the top 5, and the 5 negatives. Kept in its own file so a generation change
  cannot move the numbers above. The committed copy was produced before the
  `chunks_retrievable` provenance field was added, within this same change, so it
  does not carry it: the run used the filtered index (8,387 searchable), and the
  next run will record that itself rather than the field being backfilled by hand
  into a machine-written artifact.
- `generation-answers.jsonl` — answer prose, **gitignored**. Answers may quote their
  passages, so the same rule that keeps retrieved text out of `baseline.json` keeps it
  out of git here. This is the file to actually read: the mechanical rates say whether
  citations resolve, not whether an answer is any good.

Regenerate both the index and the numbers with the sequence in the top-level
README: `build_index.py` → `audit_corpus.py` → `audit_structure.py` →
`run_eval.py`, then `run_generation_eval.py` for the generation side.
