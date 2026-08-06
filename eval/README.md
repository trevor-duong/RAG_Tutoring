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
  validation error up" is the *acknowledgments section* of the Lottery Ticket
  paper at 0.498 — above the lowest score of any question that did retrieve a
  correct source. Acknowledgments, reference lists, tables of contents and title
  pages are all chunked and indexed like prose.
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
privacy fix made non-content pages *more* competitive — the same finding as the
acknowledgments hit, arriving from the other direction, and the reason a
structural filter at ingest is the next retrieval experiment rather than a
nice-to-have.

## Files

- `questions.jsonl` — the set. Notes paraphrase why a page answers a question;
  no source text is quoted, since the corpus is copyrighted.
- `baseline.json` — a saved report to compare future runs against. Scores and
  citations only, no retrieved text (which would carry both copyrighted material
  and, before redaction, an email address).
- `generation-baseline.json` — a **separate** instrument, written by
  `run_generation_eval.py`. It scores what the model does with these passages, over
  the same question set: mainly the ~16 questions where retrieval put no correct page
  in the top 5, and the 5 negatives. Kept in its own file so a generation change
  cannot move the numbers above; `baseline.json` stays frozen.
- `generation-answers.jsonl` — answer prose, **gitignored**. Answers may quote their
  passages, so the same rule that keeps retrieved text out of `baseline.json` keeps it
  out of git here. This is the file to actually read: the mechanical rates say whether
  citations resolve, not whether an answer is any good.

Regenerate both the index and the numbers with the sequence in the top-level
README: `build_index.py` → `audit_corpus.py` → `run_eval.py`, then
`run_generation_eval.py` for the generation side.
