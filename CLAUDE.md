# RAG Tutoring Project

## What this is
A RAG system grounded in my deep learning tutoring materials. Students ask 
questions about DL concepts (backprop, loss functions, CNNs/RNNs/Transformers, 
etc.) and get answers cited to real source passages — not generic LLM output.
Pilot with actual tutoring students this summer.

## Build order (do NOT skip ahead)
1. Ingest 10-15 docs, get basic retrieval working at notebook level, no API
2. Wrap in FastAPI, add citation formatting, build a 20-30 question eval set 
   with known correct sources
3. Simple frontend, deploy (Railway/Fly.io), test with 2-3 real students
4. THEN layer in infra, in this order: caching → async ingestion (job queue) 
   → observability/logging → rate limiting/vector DB scaling (understand, 
   lower priority to build)

## Stack
FastAPI, pgvector or Chroma (undecided — help me pick), sentence-transformers 
or an embedding API (undecided), Next.js or minimal HTML/JS for frontend.

## Constraints
- This doubles as my system design study — flag when a decision (caching 
  strategy, async pattern, consistency tradeoffs) maps to something I should 
  be able to explain out loud in an interview.
- This is the flagship piece of my SWE internship portfolio. Code should be 
  clean enough to walk through in an interview, not just "working."
- I have ~8-10 hrs/week for this. Bias toward finishing the current phase 
  over gold-plating.

## Current phase
Phase 3 — frontend + deploy. Ingestion, retrieval, the API, the eval set and the 
single-page frontend are done. Deployed to Fly.io behind a shared password, with 
the index baked into the image. Remaining in this phase: 2–3 real students. Then 
Phase 4 infra, starting with caching.