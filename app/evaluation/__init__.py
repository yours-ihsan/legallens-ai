"""Groundedness evaluation — isolated from the normal answering pipeline.

Nothing in this package is imported by the PDF -> extraction -> chunking ->
retrieval -> threshold -> LLM -> validate path. It is imported ONLY by
app/main.py behind the separate /api/evaluation* endpoints.

Flow (evaluation only):
    question -> retrieve -> baseline generate          (Baseline RAG)
    question -> retrieve -> verify -> generate/refuse  (LegalLens)
then deterministic claim extraction + lexical grounding check against the
retrieved evidence, then metrics. No number in this module is hardcoded —
every score is calculated from an actual run.
"""
