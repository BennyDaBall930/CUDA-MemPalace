# AGENTS.md

This fork is the active implementation target for the CUDA Fresh Start integration.

## Mission

Build MemPalace into a CUDA-accelerated persistent memory system for AI agents.

Current priorities:

- preserve Chroma as durable source of truth
- preserve the persisted exact sidecar as deterministic retrieval surface
- keep CUDA embeddings and CUDA exact search optional and local-first
- add compiled custom CUDA scoring and fused top-k kernels
- integrate Fresh Start memory innovations only with tests

## Kernel Rules

- torch CUDA exact search is the reference implementation
- custom score kernels must match torch scores within strict tolerance
- fused top-k kernels must match torch top-k IDs exactly
- deterministic tie behavior must remain stable
- performance claims require parity first

## Fresh Start Import Rules

Use Fresh Start for:

- exact top-k and stale/finality tests
- structured fact/relation semantics
- scope isolation
- truth supersession
- maintenance cleanup, replay, and consolidation
- evidence governance

Do not use Fresh Start for:

- hidden answer-deciding sidecars
- unsupported fixed-state claims
- historical wins that have not been integrated into MemPalace tests

## Test Expectations

Before broad claims, run focused backend/config tests.

Before integration claims, run the full MemPalace suite.

Before kernel claims, run exact score and top-k parity tests against torch CUDA.
