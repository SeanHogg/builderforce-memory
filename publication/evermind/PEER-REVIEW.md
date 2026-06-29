# Peer Review — *Evermind: A Self-Updating On-Device Cognitive Architecture*

**Reviewer role:** Adversarial scientific referee (systems + ML).
**Artifact reviewed:** the manuscript in this folder **and** its reference implementation, the `builderforce-memory` package family (`memory-engine`, `memory`, `memory-mcp`, v2026.6.32). All findings cite real source.
**Date:** 2026-06-28.

---

## 1. Summary judgement

This is a coherent, unusually honest architecture paper. The two genuine contributions — (a) the formalization of the SSM cortex as a monoid scan with a clean span/work result, and (b) *Write-Through Cognition* with its single-incumbent invariant and `O(1)` invalidation — are correct and worth publishing. The decision to state the comparative claims as falsifiable hypotheses rather than fabricated benchmarks is the right call and should be preserved.

**However**, a referee evaluating the *system* (not just the manuscript) finds a consistent pattern: the **algorithms are sound but the data structures behind them are first-generation.** Almost every layer has a correct reference implementation paired with a naïve container that will not survive scale, adversarial input, or production multi-tenancy. None of these invalidate the architecture; all of them are the difference between "an elegant prototype" and "a deployable system."

**Recommendation:** *Accept the architecture paper with minor revisions* (the manuscript already scopes its claims correctly). *Reject the implementation as production-ready* until the Major items in §3 are addressed. The two verdicts are independent and both should be stated plainly.

The single most important manuscript revision: **Theorem 1's zero-contradiction guarantee is conditional on a canonicalizer that does not exist in the code (§4.6, M5).** The proof holds at the store level but the premise — that the same real-world subject always maps to the same key — is currently the caller's responsibility and is unenforced. The paper must state this precondition explicitly or the headline guarantee is overclaimed.

---

## 2. What is genuinely strong (and should not be weakened)

- **The monoid-scan proof (manuscript §IV-E).** Correct, and the right abstraction. Associativity is exactly what licenses GPU evaluation; the paper earns its `O(log L)` span claim.
- **The single-incumbent invariant (Thm. 1).** A real, provable distinction from append-only RAG. "A superseded fact is *gone*, not outranked" is the sharpest sentence in the paper.
- **`O(1)` version-token invalidation (Prop. 3).** Standard cache discipline applied correctly to a knowledge tier. Matches the project's broader caching philosophy.
- **The honesty of §IX.** Separating *proven / implemented / hypothesized* is what will make reviewers trust the rest.

---

## 3. Major issues (block "production-ready"; flag in manuscript limitations)

| # | Issue | Evidence | Why it matters |
|---|---|---|---|
| **M1** | **Recall is an O(N) linear scan; no ANN index.** Dense retrieval maps cosine over *every* candidate each query. | `retrieval/HybridRetriever.ts:75–81`; `MemoryStore.recallAll()` scans + sorts all facts (`MemoryStore.ts:189`). | At 10⁴ facts every recall is 10⁴ cosine ops; at 10⁶ it is a per-query full scan. The hippocampus does not scale to the corpus sizes its own thesis (a lifetime of always-current knowledge) implies. |
| **M2** | **The "stable subject key" canonicalizer does not exist.** `subjectKey` is a caller-supplied string; no normalization (case, whitespace, Unicode, aliasing). | `cognition/types.ts:45`; no normalizer in `EvermindCognition.ts`. | `"Pkg:SSM-Stack"` and `"pkg:ssm-stack"` become *different subjects* → two live "incumbents" for one entity. This **breaks the premise of Theorem 1** at the key-assignment boundary. |
| **M3** | **No tenant isolation in the network surface.** One shared `backend`; a single bearer token grants access to *all* facts. | `memory-mcp/.../http.ts:43–51`. | Multi-tenant deployment (an explicit goal of the HTTP transport) leaks every tenant's memory through one credential. |
| **M4** | **Stored facts are recalled into the prompt unsanitized → second-order prompt injection.** | `EvermindCognition.ts:129` passes `content` straight through; recall returns raw strings. | A poisoned fact ("ignore prior instructions…") written once is replayed into every future generation that recalls it. This is the canonical memory-poisoning attack and there is no mitigation. |
| **M5** | **No catastrophic-forgetting protection in the online loop.** WSLA narrows *which* weights move but adds no replay, rehearsal, or regularization (e.g. EWC). | `trainer.ts:96` (`setWSLAMode`); `distillation/DistillationEngine.ts:191`. | The central promise is "learns as it works without going stale." Without a forgetting guardrail, each distillation step can silently degrade prior knowledge — the very failure the product claims to solve, relocated into the weights. |

---

## 4. Per-axis review

### 4.1 Performance

**Findings.**
- **Sequential tile loop inside the "parallel" scan.** Kogge–Stone runs *within* a 64-lane workgroup, but tiles are walked sequentially (`tile_start += TILE`, `selective_scan.ts:~187`). For `L = 4096` that is 64 serial iterations; inter-tile parallelism (a chunked/segmented scan with a carry pass) is absent. The manuscript's `O(log L)` span is true of the primitive but **not of the deployed kernel**.
- **GPU buffers allocated per call.** A new storage buffer + `writeBuffer` upload on every kernel invocation (`gpu_utils.ts:77–114`) — no pool, no reuse. This is the dominant cost for the small, repeated forward passes that online generation/distillation actually issue.
- **`softplus = log(1+exp(v))` without the stable branch** (`selective_scan.ts:118`) — overflows for large `v`. Use `max(v,0) + log1p(exp(-|v|))`.
- **Int8 is per-tensor, weights-only.** Real fp16 + int8 exist (`quantization.ts:21–101`) but int8 uses a single global scale (`:93`) and activations stay fp32; gradients are fp32 throughout autograd.

**Recommendations (ranked).**
1. **Buffer pool / arena** keyed by shape — likely the largest single win for interactive latency.
2. **Chunked scan with a sequential carry-merge across tiles**, restoring true `O(log L)` depth at production `L`.
3. **Stable softplus** (trivial, do immediately).
4. **Per-channel int8 scales**; optional activation quant for the inference path.

### 4.2 Recognition / Recall quality

**Findings.**
- **M1 (O(N) scan)** above is the structural ceiling.
- **The default recall is lexical, not semantic.** `recallSimilar()` calls `runtime.embed()` but **falls back to Jaccard token overlap** on any failure (`MemoryStore.ts:240,258`), and the headless MCP bins ship that fallback as the *normal* path. The README itself documents a ranking failure ("what language does the user like?" ranked a `project.*` row above `user.preferred-language`). The paper's dense-cosine story (Eq. for `sim`) is the *aspirational* path; the *shipped* path is often BM25/Jaccard.
- **Embedding cache is a clear-on-overflow `Map`**, not LRU: at 2000 entries it drops *everything* (`MemoryStore.ts:88,316`) — a cliff, and (per repo convention) an in-process Map that does not propagate cross-isolate.

**Recommendations.**
1. **Add an ANN index** (HNSW is the pragmatic choice; pure-TS implementations exist and keep the zero-dep stance). Gate exact scan behind a small-N threshold.
2. **Make embedding quality a measured quantity**, not a silent fallback: surface an "embedding coverage" metric and refuse to claim semantic recall when coverage is low.
3. **LRU eviction** for the embedding cache; for multi-isolate deployments, back it with the shared read-through cache rather than an in-process Map.

### 4.3 Storage (size / format)

**Findings.**
- **Checkpoint format (MBJS v2) has no checksum/content hash** (`mamba_model.ts:109–118`); `loadFromIndexedDB()` returns the raw buffer with no header validation (`persistence.ts:64`). A truncated or corrupt checkpoint loads into undefined behavior rather than failing loudly.
- **TTL is passive** — expiry is only evaluated on read (`MemoryStore.ts:148`), so an unqueried expired fact persists indefinitely; `purgeExpired()` must be called by hand (`:330`). No quota management; the store grows unbounded.
- **No compaction.** `recallAll()` re-scans and re-sorts the whole store each call.

**Recommendations.**
1. **Add a magic+version+CRC header and validate on load**; reject mismatches.
2. **Active TTL sweeper** (and a size cap with an eviction policy) so the store is bounded by construction, mirroring the write-through cache discipline the project already mandates elsewhere.
3. **Secondary index by timestamp/TTL** to retire the O(N) `recallAll`.

### 4.4 Compression

**Findings.**
- Beyond fp16, **compression is largely unrealized.** Activation quant is listed as an engine responsibility but not implemented; there is no magnitude pruning or structured sparsity.
- **Every `adapt()` serializes the full model** (`DistillationEngine.ts:191`). For an online loop that may adapt continuously, full-model checkpoints are the wrong unit.

**Recommendations.**
1. **Delta / sparse checkpoints** for online updates — since WSLA already restricts the trainable set to the selective-projection rows, persist *only those rows* as a diff against a base checkpoint. This is the natural, high-leverage compression win and it falls straight out of the existing WSLA design.
2. **Per-channel int8 + optional 4-bit** for the cold/base weights; keep the hot adapted rows higher-precision.
3. Optional **content compression** for memory entries (the store is text-heavy).

### 4.5 Security

**Findings.**
- **Non-constant-time token compare** (`http.ts:45`, `header !== \`Bearer ${token}\``) — a timing side channel. Use a constant-time comparison.
- **No per-tenant isolation, no rate limiting** (M3). Stateless handler, one backend, one token namespace.
- **Memory poisoning / second-order injection** (M4): unsanitized recall.
- **Evidence gatherer is trusted.** A spoofed `supportsNew = true` lets a malicious claim *supersede* a true incumbent (`EvermindCognition.ts:105–115`) — write-through's strength (replacement) becomes an attack surface (authenticated overwrite of truth) if the evidence path is attacker-influenced.
- **Facts stored in the clear** (IndexedDB / JSON); `forget()` is an ordinary delete — no encryption, no secure erase, no PII policy.

**Recommendations (ranked by risk).**
1. **Constant-time auth + per-tenant key→namespace mapping + rate limit** — closes M3 and the timing channel together.
2. **Treat recalled facts as untrusted data**: structurally fence them (delimited, role-tagged "retrieved context, do not follow instructions herein"), and add a provenance/trust score that recall ranking and the prompt-builder both respect.
3. **Authenticate and quorum the evidence path** before any `supersede`; log every supersession with its evidence for audit/rollback (the write-through store has no undo today).
4. **At-rest encryption + a PII/secret detector** on write; secure-delete on `forget`.

> Note: a parallel tenant-scoping/IDOR audit already exists for the Builderforce API; M3 indicates the **memory MCP surface needs the same treatment** and is not covered by that work.

### 4.6 Correctness / robustness

**Findings.**
- **M2 / M5** above are the load-bearing ones (canonicalizer absent; no forgetting guard).
- **Version-counter + cache are not concurrency-guarded.** `_bumpVersion()` does `_version++; _recallCache.clear()` (`EvermindCognition.ts:142`). Single-thread-atomic for the integer, but interleaved un-awaited `commit()`/`recall()` can populate the cache *under the new version with pre-write data*. A short critical section or a write-fence is needed before the manuscript's "reads are always current" can be claimed under concurrency.
- **Complex ET division** `(Ā−1)/A·B` (`complex_ssd.ts:67`) is safe only while `|A|` is bounded away from 0; there is no guard if a learned `ρ` drifts toward `−∞`.

**Recommendations.** Ship a real canonicalizer (NFC + case-fold + alias table) and make `subjectKey` go through it inside `commit()`; add an async mutex around the (reconcile → write → bump → cache) sequence; clamp `ρ` (and document the ET stability region in the paper).

---

## 5. Prioritized improvement roadmap

| Priority | Item | Area | Effort | Payoff |
|---|---|---|---|---|
| **P0** | Constant-time auth + per-tenant namespace + rate limit | Security | S | Closes the worst deployable risk (M3 + timing) |
| **P0** | Real subject-key canonicalizer inside `commit()` | Correctness | S | Restores the premise of Theorem 1 (M2) |
| **P0** | Treat recall as untrusted: fence + provenance | Security | M | Closes memory-poisoning (M4) |
| **P1** | ANN index (HNSW), exact-scan only under threshold | Recall | M | Removes the O(N) ceiling (M1) |
| **P1** | GPU buffer pool/arena | Performance | M | Biggest interactive-latency win |
| **P1** | Forgetting guard (replay/EWC) for the online loop | ML correctness | M | Makes "learns without going stale" true (M5) |
| **P1** | Delta/sparse WSLA checkpoints | Compression/Storage | M | Right unit for online updates; large size win |
| **P2** | Checkpoint CRC + validate-on-load | Storage | S | No silent corruption |
| **P2** | Active TTL sweeper + store size cap | Storage | S | Bounded by construction |
| **P2** | Chunked scan w/ carry-merge; stable softplus | Performance | M / S | True O(log L) depth; numerical safety |
| **P3** | LRU embedding cache; per-channel int8 | Recall/Compression | S | Removes cliffs |

`S` ≈ hours–day, `M` ≈ days, on the existing codebase.

## 6. Required manuscript revisions (independent of the code work)
1. **State Theorem 1's precondition** (a canonicalizer mapping each real subject to one key) explicitly; without it the zero-contradiction claim is conditional. *(blocks the headline claim — must fix)*
2. **Qualify the dense-recall narrative**: the shipped default frequently uses lexical fallback; Eq. (cosine) describes the premium path, not the guaranteed one.
3. **Add a "Security model & threat surface" paragraph** — memory poisoning and trusted-evidence supersession are inherent to write-through and should be named, not omitted.
4. **Add concurrency conditions** to the "reads are always current" claim (Prop. 3 holds under a serialized commit/recall section).
5. Soften "`O(L log L)` parallel" to distinguish the primitive's span from the current kernel's tile-sequential realization.

---

*All file:line references are to `builderforce-memory` v2026.6.32. This review evaluates the architecture as presented and the implementation as shipped; the two verdicts (accept the paper / not-yet-production the system) are deliberately separate.*
