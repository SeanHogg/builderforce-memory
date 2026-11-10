# Changelog — @seanhogg/builderforce-memory

## 2026.10.75

### Added — stats: `meanAndSd` (`./stats`, PRD 41 §D13)

- stats: `meanAndSd` (Welford, sample sd, non-finite skipped) — the one mean/sd for predicted measures (PRD 41 §D13). Returns `{ n, mean, sd, nonFinite }` (type `MeanAndSd`); n = 0 gives mean NaN, sd 0; n = 1 gives sd 0.

## 2026.10.74

### Fixed — Wilcoxon signed-rank, paired Hodges–Lehmann and the paired bootstrap (`./stats`, shipped wrong in 2026.10.72)

- **`wilcoxonSignedRank` p-values were wrong for every n.** The exact null CDF was built once for n = 60 and indexed with W+ for any n, so p ≈ 0 for every input with n ≤ 60 (a null 10-pair series gave 2.2e-15; n = 3 gave 8.7e-18). Above 60 the "normal" branch used exp(−z²/2), not a normal tail. Now:
  - **n ≤ 50** (`WILCOXON_EXACT_MAX_N`): an EXACT two-sided p from the permutation distribution of W+ built for this n by DP over the (doubled, so mid-ranks are integers) ranks — conditional on ties.
  - **n > 50**: normal approximation with continuity correction ½ and the tie variance correction Σ(t³ − t)/48, two-sided normal tail. The tail uses the upper incomplete gamma Q directly, so a far tail is a tiny positive number (n = 100 all positive ≈ 4e-18), not 0. At n = 50 the two methods agree within 0.001 for p < 0.1.
  - **Non-finite differences** (NaN, ±Infinity) are dropped from the test, the estimate and `pairedBootstrapCi`. `WilcoxonResult` gains a required **`nonFinite`** count; it is additive for readers.
  - **Zeros:** Wilcoxon's rule, as before: zero differences are dropped before ranking; `n` counts the rest.
  - Any caller that read a tiny p from null-ish data was reading the bug. Data where every pair moves the same way gets the same verdict as before (p = 2/2ⁿ, still tiny).
- **The paired Hodges–Lehmann estimate dropped zero differences**, biasing it away from 0. It is now the median of the Walsh averages of ALL paired differences, zeros included (Hollander & Wolfe; R's `wilcox.test` drops zeros for its estimate). It is computed by exact value-space selection in O(n log n), not by sorting the n(n+1)/2 averages. The result is the same double as the full sort. At n = 200 the Hodges–Lehmann bootstrap takes about 0.12 s, down from 6.3 s.
- **`pairedBootstrapCi` gains `statistic: 'mean' | 'hodges-lehmann'`** (default `'mean'`, unchanged). With `'hodges-lehmann'` it resamples pairs and reports the percentile CI of the paired Hodges–Lehmann estimate (PRD 41 C6.2), centred on that estimate. Still seeded with a fixed resample count. New type `PairedStatistic`.
- Regression tests: `src/contract/regression.wilcoxon.test.ts`. They cover null data (p = 1), all-positive n = 10 (p = 2/1024), the α = 0.05 critical values for n = 5..20, ties, exact vs normal at n = 50, the far tail without underflow, non-finite pairs, Hodges–Lehmann with zeros, the selection against a brute-force median (200 seeded inputs with ties, zero clusters and mixed scales), and the Hodges–Lehmann bootstrap with its cost.

## 2026.10.73

### Added — a review's versions are recorded when it runs and carried by the pack (`./knowledge`, PRD 41 §C9)

- **`MeasuredWith`** `{ extractorVersion: string; toolVersions: Record<string, string> }` and **`ScopeRecord.measured?: MeasuredWith`**: what the review at `lastDigest` was measured with (the extractor version and analyzer name → version; the packs stay in `ScopeRecord.packs`).
- **`ReviewArgs.measured?`**: a finished (not aborted) review stamps it on the scope together with `lastDigest`. `LocalSnapshotStore.write` drops a stored `measured` when a write records a `lastDigest` (moved or not) without one, so an unmeasured review never inherits the versions of the review before it (absent = unknown); a write without a digest (mid-run, aborted) keeps it.
- **BFKP: a new `MEAS` section (tag 14)** carries `scope.measured` (`extractorVersion`, then the tools in sorted order). SCOPE is closed, so this is a new tag: a reader older than this release skips it and decodes the scope unmeasured; a pack without it (every older pack) decodes with no `measured`. A MEAS without a SCOPE, a tool named twice (or `__proto__`), or trailing bytes is a `PackFormatError`; so is ENCODING a `measured` that is not all strings (a scope loaded from an unvalidated snapshot).
- **Why**: hosts stated their versions when they BUILT a pack, so a review queued offline across an extension upgrade was stored with the newer versions and compared as "the same". Regression: `src/contract/regression.scopeMeasured.test.ts`.
- Minor, non-breaking: optional fields and a new section tag only.

## 2026.10.72

### Added — FileFacts v3, FactRule and the incremental evaluator (`./knowledge`, PRD 41 WS4 phase 1, §B1/§B3/§B5/§B8.6)

- **`FileFacts` v3** (types for the Phase 2 tree-sitter / regex extractors): `symbols`, `imports`, `exports` (name, kind, signature hash), `entryPoints` (§C5.1, with the guard `chain` files), `calls` (callee, enclosing symbol, `awaited`, `inLoop`, count — no positions), `complexity`, `capabilityImports` (§C5.3), optional `testFacts` (§C8.2). Fact kinds `structure` / `body` / `content` (`FACT_KIND_FIELDS`), each with its own hash (`factKindHashes`); `canonicalFileFacts` sorts every array; `factsHash` is the content address; `readFileFacts` reads a v2 object as `factsVersion: 2` with empty v3 lists.
- **Composite extractor version**: `FACTS_EXTRACTOR_VERSION = 3`; `factsExtractorTag(stamp, language?)` composes family, version.revision, the language's grammar version and the EOL rule (`tree-sitter@3.0/typescript=0.23.2/eol-lf@1`), so a grammar bump owes only that language (`isFactsCurrent`); `extractorStampMeta` feeds `comparable()` (reason `tool-version-changed:grammar:<lang>`). Separate from `EXTRACTOR_VERSION` (6, the packs' `Extracted`), which is unchanged.
- **`FactRule`** (`defineRule`; `scope: file | neighbourhood | global`, `reads`, `hops` ≤ 2, `readsGuardChain`, `extraInputs`) and **`evaluateRules(rules, world, previous?)`**: verifying traces with early cutoff — a row reruns only when the read kind hashes in its scope (or a declared extra input's hash) moved; returns `{ rows, trace, findings }`. The trace is enforced: `derive` sees each file projected to its read kinds (`projectFileFacts`) and `extra` cut to its declared inputs. `reads: []` is allowed for a global rule reading extra inputs only (the knowledge rules). `knowledgeInputHash(k)` is the hash a host passes for the `knowledge` input (graph digest + registries + near-duplicate pairs). Canonical facts normalize their path fields, break sort ties on the whole item and are deep-frozen.
- **Fact graph**: `emptyFactGraph`, `updateFactGraph`, `buildFactGraph`, `factGraphDump` (sorted canonical JSON of manifest, objects and derived rows). The determinism oracle (500 seeded edit sequences, CRLF on the incremental side) asserts incremental dump === scratch dump.
- **The generic findings are one rule per file** under `knowledge/rules/` (cycle, clone-group, near-duplicate, partition-violation, duplicate-copy, unpartitioned, stale, god-module, possible-same), registered in `GENERIC_FINDING_RULES` / `KNOWLEDGE_FINDING_RULES`; `genericFindings` / `computeFindings` are the unchanged facade. `genericFindings` now emits near-duplicate findings before partition violations instead of interleaving them per edge (`computeFindings` sorts, so its output is unchanged).
- Minor, non-breaking: new names only.
- `Sha256.digest` zero-pads in one update instead of byte by byte (same digest, far cheaper per small hash).

### Added — the D13 consolidation primitives (`./stats`, PRD 41 WS15)

- **`sigmoid(z)`** (the stats module's numerically stable sigmoid; `fitting.ts` now uses it), **`logisticProbability(weights, bias, x)`** (σ(b + w·x), shape check left to the caller) and **`onlineLogisticStep(weights, bias, x, label, learningRate)`** (one SGD step on the log-loss, returns a new head). This binary head is float64 and engine-free on purpose; batch fitting stays in memory-engine's `LogisticModel`. Builderforce migrates onto these: prefetch calibration (scoring and its online update), the difficulty head's SCORING only (its training stays on the engine's injected trainer), and the pipeline router's sigmoid.
- **`wilsonInterval(successes, n, z = 1.96)`**: the Wilson score interval; n not > 0 (or NaN), or non-finite `successes`, is `[0, 1]`; a `z` that is not a finite positive number throws `RangeError`. Builderforce's replay impact / issue-cluster and canvas proportion copies migrate onto it.
- **`betaMoments(alpha, beta)`** (mean, variance). The learned-routing Thompson draw migrates onto it.
- **`BetaPosterior` gains a REQUIRED `variance` field.** Additive for every reader of `betaPosterior(...)`; code that constructs a `BetaPosterior` object literal itself must now supply `variance` (none does in this repo or Builderforce).
- **`ledoitWolfCovariance` moved to memory-engine** (`ml/classical/covariance.ts`, where the anomaly composite's Mahalanobis term fits on it) and is re-exported from `./stats` unchanged, with `CovarianceResult`. `./stats` imports it through the engine's NEW light subpath **`@seanhogg/builderforce-memory-engine/ml/classical`** (pure classical maths: no WebGPU, kernels, training loop, I/O or `node:` imports), never the engine root, so browser bundles that import `./stats` stay small. The engine is already a peer dependency of this package. A consumer of `./stats` must now have `@seanhogg/builderforce-memory-engine` installed, at a version that has the `./ml/classical` export. Regression: `src/contract/regression.statsEngineFree.test.ts` (the stats import graph reaches the engine only through that subpath, and the subpath never reaches the engine root).
- Additive: new names, one new required field on a result type.

### Fixed (`./stats`)

- **Thompson sampling is deterministic under its own seed.** `thompsonSample` / `thompsonRank` drew normals from a polar Box–Muller that cached its spare at module scope, so one caller's spare leaked into the next caller's schedule and the same seed could give a different pick. They now use the module's one stateless normal (`normalDraw`). Draws differ from before for the same seed. Regression: `src/contract/regression.thompsonDeterminism.test.ts`.

### Fixed (`./knowledge`, PRD 38 P4)

- **`encodeCoChange` never writes a line `decodeCoChange` would reject.** It drops an entry with an empty ref (or one holding a newline), a `together` that is not an integer ≥ 1 or exceeds `ofSelf`, or a `low` / `high` that is NaN, outside [0, 1] or with low > high. Before, one such entry voided the whole badge on read (and `coChangeHidden` with it). Encoder and decoder now share one line validator.

### Added — the caller's extractor key (`./knowledge`, PRD 43)

- **`extractorKey?: string`** on `ExtractFileArgs`, `PlanOptions` and `ReviewArgs`: an input the host's extractors read besides a file's bytes (agent-tools passes its tsconfig path-alias digest). A fresh extract is stamped with it (`Extracted.extractorKey?`), and the root's extractor entry records it (`{ id: 'extractorKey' }`; `ExtractorState.extractorKey?` / `extractorKeyNow?`). A cached extract made under another key — or under none when one is now given — is not reused, and a changed key owes every file like a version bump, so files whose bytes did not change still re-extract when the tsconfig did. `refreshPlan(detail, manifest, extractorKey?)` takes it for *Continue* (default: the plan's — a host must pass a freshly computed key; `''` is an explicit "no key").
- **A host pool echoes its own key.** `ExtractJob.extractorKey?` carries the requested key; the worker extracts with what it holds and returns the key it actually used in `ExtractJobResult.extractorKey?` (`extractJob(job, packs, workerKey?)` echoes `workerKey`, never the job's). A different echo is neither cached nor used (`EXTRACTOR_KEY_MISMATCH`, transient): the file stays owed until the pool has rebuilt its aliases.
- **A key-only change estimates no model calls** (the files re-extract; their judgements are re-anchored). A version upgrade keeps its estimate. An aborted run keeps the previously recorded key, so every file stays owed until a run under the new key finishes.
- **No key: unchanged.** Records and extracts without a key match a call without one, and nothing new is written. The extract cache and the owed state decide by ONE comparison (`isCurrentExtractor`: version and key; an empty key reads as none).
- Not carried by the BFKP pack: an extract that went through a pack reads as keyless, so a keyed host re-extracts it once.
- Minor, non-breaking: optional fields and an optional parameter only.

## 2026.10.71

### Added — predicted measures P4 co-change and P7 transitions (`./knowledge`, PRD 38)

- **`coChangePartners(commits, opts?)`** (P4): per file, the files changed with it — `{ ref, together, ofSelf, confidence = P(B|A), lift, low, high }`, with the confidence's credible interval — kept at support ≥ `COCHANGE_MIN_SUPPORT` (3) and lift ≥ `COCHANGE_MIN_LIFT` (2), at most `COCHANGE_MAX_PARTNERS` (8) ranked confidence, lift, ref. A commit over `COCHANGE_MAX_COMMIT_FILES` (50) files — its whole-repo `fileCount` when given, else its distinct refs — is excluded as bulk; the newest `COCHANGE_MAX_COMMITS` (5000) are counted (by parsed `at` when every commit is dated, else input order, oldest first). Deterministic in any input order.
- **`encodeCoChange` / `decodeCoChange`**: the ONLY writer and reader of the `coChange` badge string — a `v1` line, then `together\tofSelf\tlow\thigh\tref` per partner; decode gives `[]` for a missing or unknown version and anything malformed. **`coChangeRecallAtK`**: the accept measure, `{ recall, cases, skipped }` (recall@k on held-out commits; `skipped` = held-out files with no partners) or `{ status: 'insufficient-data' }`.
- **`applyCoChangeHidden`**: `buildGraph` now stamps `coChangeHidden` on files carrying a caller-stamped `coChange` badge: partners (that are graph nodes) sharing no live `references`/`links` edge either way, both ends rolled up to their file (symbol refs and out-of-root placeholders explain coupling). Absent when 0; cleared from a file without `coChange`.
- **`OUTLINE_BADGE_KEYS` gains `coChange`, `coChangeHidden`**, so the outline carries both.
- **Station transitions** (P7): `transitionsFromWeights`, `transitionsFromSequences`, `stationTransitions({ sequences?, weights })`, `transitionOf`, `mostLikelyWalk` (greedy per-hop argmax, not Viterbi). A table row is `{ source: 'traces' | 'references', transitions: StationTransition[] }`; `stationTransitions` picks per station (its trace row when it has trace hops, else its reference row). A hop is `{ to, p, k, n, low?, high? }`: trace hops carry the interval, reference hops (not observations) do not.
- **`./stats`: `rateInterval(k, n)`, `RATE_PRIOR` (0.5), `RATE_CREDIBLE_MASS` (0.9)**: THE interval of a shown rate — the equal-tailed 90% Jeffreys interval with the boundary rule (low 0 at k = 0, high 1 at k = n). Used by co-change confidence and trace hops.
- **`./stats`: `inverseRegularizedIncompleteBeta` (so `betaPosterior(...).quantile`) is Halley steps inside a bisection bracket**, instead of 100 fixed bisections: far fewer incomplete-beta evaluations per quantile. It stops when the Newton step is below 1e-10 relative to min(x, 1 − x), so quantiles near 0 or 1 keep their accuracy; a step landing on a bracket end counts as converged; and if the step cap is reached, bisection finishes the bracket, so an unconverged x is never returned. **Fixed:** the first Halley version (2026.10.70) discarded converged tail answers and returned wrong intervals for 7,019 of 34,967 sampled (k, n ≤ 5000), e.g. `rateInterval(4999, 5000).high` 0.90889 instead of 0.999965. Every interval now sits on its 5% / 95% quantile to within 1e-9 in I_x, matching 200-step bisection to 1e-6 on the n ≤ 5000 diagonals. `coChangePartners` computes intervals only for the kept partners (memoised per call) and counts pairs in a typed open-addressing table.
- Minor, non-breaking: new names only.

### Changed

- EXTRACTOR_VERSION 5 → 6, so cached code extracts re-run after the agent-tools code-pack changes (PRD 43 P0c).

## 2026.10.70

### Added — truncated analyses and the binary rule (`./knowledge`, PRD 41 WS2/WS6)

- **`CodeAnalysisSnapshot.truncated?: true`**: set when the analyzed scope was cut short. `analysisMeta` carries it (only when set), so `comparable()` gives reason `truncated` and `compareAnalysis` reports no `added` or `resolved` issues for a truncated side — an issue missing from a partial scan is never a fix. Hosts no longer wrap `compareAnalysis` for it.
- **`isBinaryBytes(bytes)`** is exported: THE binary rule (a NUL among the first 8192 bytes) `contentHash` uses, so a host hashing natively or sniffing raw bytes calls it instead of restating it.
- Minor, non-breaking: new names and an optional field only.

## 2026.10.69

### Added — symbol-level edges (`./knowledge`, `./knowledge/pack`, PRD 43 P0b)

- **Edge kinds `extends`, `implements`, `composes`, `uses`, `calls`** (`SYMBOL_EDGE_KINDS`, `isSymbolEdgeKind`, type `SymbolEdgeKind`): what a code pack says between two symbols. A target outside the root is a `src:` ref that `buildGraph` closes as a `missing` placeholder, as an unresolved import is.
- **`KEdge.sites?: CallSite[]`** (`{ path, line, column, order }`): a `calls` edge carries every call of its callee in the caller's body; `order` is the call's position among ALL the caller's calls, so a sequence shows the real call order. One deduped edge per (from, to, kind) — a repeated call is another site.
- **`dedupeSymbolEdges(edges, { maxPerSymbol, maxSites })`**, `MAX_SYMBOL_EDGES_PER_SYMBOL` (64), `MAX_CALL_SITES_PER_EDGE` (16), `compareCallSites`: the one dedupe/cap an extractor runs.
- **BFKP: a new `SITE` section (tag 13)** carries call sites; EXTR is unchanged. **Compatible both ways:** the five kinds are KNOWN to the reader as codes 12–16 but WRITTEN as strings (`WRITTEN_EDGE_KINDS` = the 11 legacy codes) until no older reader remains, and an older reader skips the unknown SITE tag (it decodes every edge without its sites). A pack written before this release decodes to the same pack and re-encodes to the same bytes.

## 2026.10.68

### Added — finding identity v2, lineage and comparability (`./knowledge`, PRD 41 §C3/§C9)

- **`Finding.anchor?` / `Finding.anchorKind?`** (`'symbol' | 'route' | 'package' | 'file' | 'group'`): the PRIMARY anchor the emitting rule names — never a line number, never the member set. `finding(…, idRefs?, anchor?)` takes a `FindingAnchor` (`{ anchor, anchorKind, snippet? }`; the snippet is hashed after `normalizeSnippet`), `refAnchor(ref)` builds one from a ref. Every generic finding sets one: cycle → its sorted-first member; clone-group → `cloneGroup:<badge>` (group), else its sorted-first member; near-duplicate → the sorted-first of the pair; partition-violation → `<from layer>-><to layer>@<from file ref>`; duplicate-copy → the copy's file; unpartitioned → `pack:<pack>`; stale → the stale source; god-module → the file; possible-same → the sorted-first entity. `id` is unchanged.
- **`identityOf(f, ruleVersion = 1): FindingIdentity`** (`{ id, anchor, anchorKind, members }`): anchor = `<kind>@<ruleVersion>:<anchorKind>:<primary>[~<snippet hash>]`. A rule that sets no anchor gets `defaultAnchorOf`: the sorted-first node ref, else `pack:<pack>`. Helpers: `composeAnchor`, `anchorWithSnippet`, `anchorKindOfRef`, `kindOfAnchor`, `normalizeSnippet`, `DEFAULT_RULE_VERSION`.
- **`matchLineage(before, after): LineageEntry[]`** — the ONE deterministic lineage matcher: exact id → `persisting`; same anchor → `changed` (`grew` / `shrank` / `moved`); same kind + anchor kind, ≥ 2 members both sides, Jaccard ≥ 0.5 (`LINEAGE_MIN_JACCARD`, `LINEAGE_MIN_MEMBERS`), greedy → `changed`; the rest `introduced` / `resolved`. Sorted by (after ?? before) id; input order never matters. A clone group gaining a member is now `changed (grew)`, not resolved + introduced.
- **`comparable(before, after): { comparable, reasons }`** over `ComparabilityMeta` (extractor, extractorVersion, contentHashRule, tools, scope, repo, branch, truncated, failed), with stable reason codes `extractor-mismatch`, `extractor-version-changed`, `content-hash-rule-changed`, `tool-missing:<tool>`, `tool-version-changed:<tool>`, `analyzer-failed:<tool>`, `scope-changed`, `repo-changed`, `branch-changed`, `truncated`.
- **Moved from Builderforce.ai agent-tools `codeAnalysis.ts`:** `AnalysisLocation`, `AnalysisIssue`, `CodeAnalysisSnapshot`, `CodeAnalysisComparison`, `normalizeAnalysis`, `compareAnalysis` (now also returning `reasons`; it delegates to `comparable()` via `analysisMeta`, plus `no-baseline` / `tools-incomplete`, behaviour unchanged). jscpd and knip issues gain an `anchor` (`ANALYSIS_ANCHOR_VERSION`); ids are unchanged. `identityOfIssue(issue)` gives an issue's `FindingIdentity`. `ANALYSIS_TOOL_COUNT` (3).
- Minor, non-breaking: new names and optional fields only.

### Added — EOL-normalized content identity and directory Merkle hashes (`./knowledge`, PRD 41 WS2 §B2 steps 0 and 4)

- **`contentHash(bytes)`**: lowercase hex sha256 of a file's bytes with CRLF and lone CR converted to LF when the bytes are text; binary bytes (a NUL in the first 8 KB, the rule of `isBinaryText`) are hashed raw. A Windows `core.autocrlf` checkout and a Linux checkout of one commit now hash identically; an LF-only file hashes exactly as its raw sha256 did. Byte-level, never decoded. **`CONTENT_HASH_RULE`** (`'eol-lf@1'`) names the rule.
- **The rule is recorded beside the extractor version, independent of it**: `ExtractorState.contentHashRule` (optional) is read from a `contentHash` record of the extractor entry. The rule never owes a file on its own: a host switching to `contentHash` already moves the hash of every CRLF file, so those re-extract by content (`contentChanged`, an extract-cache miss) and LF files keep their hash. An entry with no record owes nothing by it; only a RECORDED rule that differs from today's owes every file one re-extraction. `eol-lf@1` writes no record (the entry keeps its shape); a later rule will.
- **`merkleTree(files)`** → `{ root, dirs, entries }`: a directory's hash is sha256 over the canonical JSON of its `{ name, kind: 'file' | 'dir', hash }` entries, sorted bytewise; paths POSIX and NFC; the root's key is `""`. Any input order gives the same tree. **`changedPaths(before, after)`**: the changed, added and removed file paths, never descending into a directory whose hash is equal on both sides (O(changed)). Types `MerkleTree`, `MerkleEntry`, `MerkleKind`.
- **`ReviewPlan.merkleRoot?`** and **`PlanDetail.merkle?`**: `planDetail` hashes the planned files' tree; *Continue* (`refreshPlan`) keeps it.
- **ONE path normalization, and the plan never throws over a listing**: `normalizePath` (refs.ts: POSIX, no leading `./`, NFC) is used by the plan and by `merkleTree`, so plan paths are Merkle paths. `canonicalListing(entries)` skips a path that is empty or has an empty, `.` or `..` segment (`invalid-path`) and, of paths that collide once normalized (an NFC/NFD pair, a path listed twice, a path below a listed file), keeps the first by (normalized, listed) path bytewise and skips the rest (`duplicate-path`); `ReviewPlan.skipped[].reason` gains both. `isRootFilePath`. A re-spelled file keeps the source's spelling as **`PlannedFile.listedPath?`**: reads (`read`, `readTimeoutMs`, `hashOf`, a pool's `readByPool.path`) use it, identity uses `path`. `merkleTree` also rejects `.` and `..` segments.
- **`BASELINE_CONTENT_HASH_RULE`** (`'eol-lf@1'`) and **`contentHashRuleOf(recorded)`**: the ONE meaning of an absent rule record, used by the extractor entry and by `comparable()` — an absent `contentHashRule` and `'eol-lf@1'` are comparable.
- **ONE binary rule**: a NUL among the first 8192 BYTES. `isBinaryText` now applies it to the text's UTF-8 bytes (it counted 8192 UTF-16 code units, so a NUL after 5000 × `é` was binary to the text sniff and text to the byte sniff).
- Review fixes to finding identity: `anchorKindOfRef` splits refs with `fileRefOf`; `identityOfIssue` recomposes a legacy issue's anchor exactly as `normalizeAnalysis` writes it (the stored, 4000-character snippet's hash; `<file>#<name>` for symbols and packages); `matchLineage` keeps the minimum identity by (anchor, members) when an id repeats, so input order never matters; `compareAnalysis` reasons are sorted and deduped.
- Hosts: a `DirectorySource` that lists content hashes (and its `hashOf`) should hash with `contentHash`, so its listing agrees across OSes.
- Upgrading: plan paths are now NFC-normalized, so a root indexed by 2026.10.67 with NFD-named files re-keys those files once (the old NFD key is removed and the NFC key re-extracted), and their findings re-anchor once.
- Minor, non-breaking: new names and optional fields only.

## 2026.10.63

### Advisory folds of the working transcript (`./working`, PRD 45 §4.7 B6/B7)

- **`foldSupersededReads`** (B6): when an edit/write to path P succeeds at step N, every earlier `read_file` result for P in the model transcript becomes `"(read of P superseded by the edit at step N — re-read if needed)"`. A failed edit supersedes nothing; a read re-made after the edit is not superseded by it.
- **`foldRepeatedNarration`** (B7): assistant text preceding tool calls whose word-set overlap (Jaccard) with the previous such narration is ≥ 0.8 (`NARRATION_OVERLAP`) becomes `"(progress note)"`. A final answer and a turn without tool calls are never folded; a folded note does not become the comparison text.
- **`foldWorkingTranscript`** runs both folds over the working transcript a host sends the model. The folds are compression only: the run's own transcript and its trace keep every message at full fidelity.
- New exports on `./working`: `foldSupersededReads`, `foldRepeatedNarration`, `foldWorkingTranscript`, `supersededReadMarker`, `PROGRESS_NOTE`, `NARRATION_OVERLAP`.

## 2026.10.48

### Fixed (`./knowledge/pack`)

- **A relation leaf key no longer contains NUL.** `relationLeafKey` built `r:from␀to␀kind`, and Postgres rejects NUL in `text` and `jsonb`, so every pack with a relation failed. The key is now `r:` + `JSON.stringify([from, to, kind])`: printable and unambiguous. **The old key form is gone**; a leaf digest tree built from it differs, so both sides must run 2026.10.48.
- **The default decode caps fit a 128 MB Worker:** 24 MB decoded, 400,000 strings, 200,000 rows (were 64 MB, 2,000,000, 1,000,000). A client cuts its packs below them.

### Added (`./knowledge/pack`)

- **`parseRelationLeafKey(key)`**: the one inverse of `relationLeafKey` (null for anything else). The hand-written `split('\0')` parsers are removed.
- **`bucketDigest(leaves)` and `rootDigestOfBuckets(digests)`**: the two halves of `manifestTree`, exported so a server can keep bucket digests and recompute only the buckets a pack touched. `manifestTree` is now built from them (same digests as before for equal keys).
- **`decodePack` and `peekPack` accept the ordered parts of a split pack.** The parts are inflated as ONE stream instead of being joined first, so the compressed pack is never copied.

## 2026.10.47

### Added

- **Contract tests** (`src/contract/`):
  - an export snapshot per subpath (removing or renaming an export fails as a breaking change);
  - compile-time type contracts for the shapes consumers read (`check:contract-types`, in CI);
  - a check that every `exports` path exists in `dist/`;
  - pack determinism.
- **One `regression:` test per bug fixed in 2026.10.46.**
- **`encodePack({ stats })`** reports encoded size and its share of the equivalent JSON.
- **`DirectorySource.hashOf`** (optional).

### Fixed

- The in-thread read now checks the planned hash. A source that gives `DirectorySource.hashOf` (the function behind its `list()` hashes) has the text it read hashed once; a mismatch fails the file with "changed since plan", stores nothing under the planned hash and leaves the file owed. A pool result whose own hash differs from the plan fails the same way. 2026.10.46 claimed this check, but only the host worker made it.

## 2026.10.46

### The map is complete in minutes; judging follows (`/knowledge`, PRD 41 A10)

- **Two passes.**
  - `ReviewArgs.mechanical` reads, extracts and writes every owed file with no model call.
  - `ReviewArgs.only` judges a given list of files in the order given.
  - `ReviewReport.awaitingJudge`, `awaitingJudgeCount` and `judgedFiles` say what is still owed.
  - `orderForJudging` ranks entry points and manifests, then inbound references, then recently changed files.
- **Per-job deadlines.**
  - A read times out after 5 s and an extraction after 10 s (`timeouts`); a timed-out file is a `failed` step and the run goes on.
  - A file whose bytes no longer match the planned hash fails with "changed since plan" and stays owed.
  - `deferRead` and `readAhead` let a pool read its own files, 32–64 ahead.
- **A judge failure is not a judgement.** A timeout or network error leaves the section owed. Five failures in a row stop judging for the run (`judgeStopped`).
- **The extract cache leaves the snapshot.**
  - `LocalSnapshotStore` takes an `extractedLog` port: an append-only, content-addressed log, compacted on load.
  - An old snapshot's cache migrates into the log on load, and old snapshots still read.
  - Reads no longer wait behind a checkpoint.
  - `write` and `markStale` are incremental.
  - New `extractedEntries()`.
- **Run model.**
  - `ReviewPhase` gains `mapped`.
  - `ReviewStep.parts` splits read, cache lookup and pool time.
  - `slowestSteps` lists the slowest steps.
  - Run failures are kept in chunks (`failuresOf`, `failureTotal`).

### Sync is one compact pack (`/knowledge/pack`, PRD 41 A11)

- **The BFKP v1 codec**, `encodePack` and `decodePack`. It uses a front-coded string table, raw 32-byte hashes, columnar sections, varints, deflate-raw and a CRC-32 trailer. The bytes are deterministic and lossless. Pack size is at most 10% of the same rows as JSON.
- **Large packs.** `splitParts` and `joinParts` split a pack into 8 MB parts. `decodePack` limits stop a decompression bomb.
- **Delta helpers.**
  - `manifestTree`, `differingBuckets` and `leafDelta` reconcile through 4,096 buckets.
  - `diffEntry` and `applyEntryDiff` send only the nodes and edges that changed in a re-extracted file.

### Systems

- An edge from a file to a missing node in another system gets no partition verdict.
- Ignore globs honour backslash escapes.

## 2026.10.45

### A folder of many systems reads as many systems (`/knowledge`, PRD 41 A9)

- **Systems on file nodes.** `BuildGraphInput.systems` (`GraphSystem[]`) places each file in its deepest system and app: `KNode.system`, `KNode.app`, and `KNode.copyOf` for a stray copy of another system. The graph carries `meta.systems` with per-system file counts and repo-graph read/skip tallies; the outline carries the same per file, so a truncated graph keeps them.
- **No layer verdict across systems.** An edge between files of two different systems gets no partition verdict, so `partition-violation` is computed within a system only.
- **`duplicate-copy` finding.** A system recorded as a copy of another yields a `warn` finding `{ copy, original, similarity, reason }`.
- New exports: `GraphSystem`, `GraphApp`, `GraphReadSkip`, `placeInSystem`, `crossesSystems`.

## 2026.10.44

### A review is a run you can watch, and it runs in parallel (`/knowledge`)

- **No single file ends a review.** An extraction that throws is counted in `ReviewReport.extractErrors` and listed in `ReviewReport.failures` (`{ path, phase, reason }`, read errors and judge errors too); the file keeps its prior knowledge and manifest hash, so the next review retries it. A judge that throws leaves that file's sections unjudged. Only an abort or a local store write error ends a run.
- **`ReviewArgs.onStep`** emits one `ReviewStep` (`{ path, phase, ms, outcome: ok | cached | skipped | failed, reason?, lane? }`) per file per phase.
- **A file whose judging fails as a whole is left exactly as it was.** When every model call for a file fails (401, 429, network) or judging throws, nothing of it is written or marked stale: prior facts, manifest hash and owed state stay, and the next review retries it. The failure carries the error (`judge: 429 rate limited`). A budget cut, or some calls failing, keeps the older *Continue* behaviour.
- **`ReviewArgs.extractPool(job, signal)`** hands extraction to a host pool (e.g. `worker_threads`). The `ExtractJob` is plain data; `extractJob(job, packs)` is the pure worker-side entry (returns `{ extracted, sections? }`) and runs the same extractor as the in-thread default. With a pool the main thread never parses: the worker returns the sections when the file will be judged. A result without the file's exact `sourceRef` fails that file's extraction. With a pool, `concurrency` is the pool width.
- **`ReviewArgs.judgeConcurrency`** (default 4, never wider than `concurrency`) judges that many files at once. Each file's model calls are reserved from the budget in plan order and ONE writer applies results in plan order (and writes the extract cache), so the stored bytes equal those of a one-file-at-a-time review.
- **`ReviewArgs.maxModelCalls`** (default 6) caps model calls in flight across ALL files with one shared AIMD limiter (`createModelLimiter`): a rate limit halves it, 20 successes grow it by 1.
- **Abort.** Files whose work completed are still applied; an abort that lands while the writer is mid-file ends the run at that file. A thrown error stops every queued and in-flight call before it propagates.
- **`startRun` / `applyRunEvent` / `runRate` / `runEta`** (`reviewRun.ts`): a pure recorder that folds progress, step and end events into a `ReviewRun` (phases, lanes, the last 200 steps, every failure) for a host to persist and draw.

## 2026.10.43

### One list of external service ids (`/knowledge`)

- **`EXTERNAL_SERVICE_IDS` and `ExternalServiceId`** name the outside services a file can import: postgres, kv, llm, github, stripe, mail, slack and oauth. Extractors and viewers type against this one list, so the `externals` badge and the names a viewer shows for it cannot drift apart.

## 2026.10.42

### Outlines carry the externals badge; extractor version 3 re-extracts stored reviews (`/knowledge`)

- **A truncated graph's outline keeps the `externals` badge** (`OUTLINE_BADGE_KEYS`), so consumers of the outline see a file's external dependencies. Unlisted badges are still dropped.
- **`EXTRACTOR_VERSION` is 3**: stored reviews re-extract unchanged files once (no model calls); only sections whose text changed are re-judged.

## 2026.10.40

### Shell scripts and code samples are no longer read as wiki links (`/knowledge`)

- **Script files (`.sh`, `.bash`, `.zsh`, `.ps1`, `.bat`, `.cmd`) are never documents** in a knowledge review, so `[[ -f x ]]` tests produce no bogus missing-link nodes.
- **`[[Name]]` inside fenced blocks, indented code blocks and inline code is ignored** in markdown.
- **One fence scanner** (`ingest/fences.ts`) serves the markdown parser and the link scanners: `~~~` and longer fences, fences in list items and blockquotes, and unclosed fences are handled the same in both. Markdown and reference links inside code are no longer edges either.
- **Indented code** counts only in markdown, not after a list item, and not in plain text.
- **`EXTRACTOR_VERSION` is 2**: stored reviews re-extract unchanged files once (no model calls) and drop the bogus link nodes; only sections whose text changed are re-judged.
- **The markdown parser keeps a section's first-line indentation** (leading blank lines and trailing whitespace are still trimmed). This also affects the non-knowledge ingest pipeline: a chunk that opens with an indented line re-embeds once.

## 2026.10.39

### A knowledge review reads and extracts files in parallel (`/knowledge`)

- **`reviewDirectory` prefetches each changed file's read and extraction with bounded, order-preserving concurrency** (`ReviewArgs.concurrency`, default 8, clamped 1–32). Judging, store writes, stale marks and the manifest stay in file order, so the stored result is identical at any concurrency.
- **Files with identical content extract in file order**, so the content-hash extract cache keeps every file's entry.
- **An abort ends the run at a file boundary.** Read-ahead is discarded, and started work settles before the review returns. `concurrency: 1` reads strictly one file at a time.
- **Only a failed read counts as a read error.** A throw while checking or extracting a file surfaces from the review, as before.

## 2026.10.38

### Parallel judging that honours an abort (`/knowledge`, `/ingest`)

- **`judgeSections` and `compareClaims` run on a bounded pool**, and knowledge analyzers run in parallel with per-pack failure tolerance: one pack failing no longer fails the others.
- **`docsExtract` caps the section ref slug length**, so a long heading no longer yields a ref over the 512-character bound.
- **`runPool` surfaces the first error and stops scheduling.** A failed or aborted worker no longer leaves the pool starting new items; the first error is rethrown once in-flight work settles.
- **`compareClaims` marks unfinished pairs skipped.** Pairs still in flight or not yet started when an abort lands are marked skipped, and `aborted` is set when the abort arrives before the last pair completes.

## 2026.10.37

### A reloaded chat's model reads its fold note (`/working`)

- **`windowed` keeps a leading consolidation marker.** A reloaded seed starts with the saved fold marker, an assistant row, and the window used to walk forward to the first user row and drop it. The model then never read the fold note after a reload: the durable marker only saved a re-summary. The marker now stays in front of the window.

## 2026.10.36

### A review of a large folder no longer rewrites its snapshot per file (`/knowledge`)

- **`KnowledgeStore.batched(work)`** runs many mutations as one unit: inside it a store may hold persistence back, `flush` makes everything so far durable, and the store is durable when `work` settles. `LocalSnapshotStore` implements it with a checkpoint at most every `checkpointMs` (default 30 s). `reviewDirectory` runs as one batch, so a review writes its snapshot a bounded number of times. Before this, each file rewrote the whole snapshot: 16.4 GB written over 2,340 files for a 5.8 MB snapshot. Outside a batch, each mutation still persists before it resolves.
- **`KnowledgeStore.preloadExtracted(hashes)`** is a hint that lets a remote store fetch the extracted cache in bulk. `reviewDirectory` calls it once, with the distinct changed hashes, before the per-file loop.
- **A review stores everything it produced, durably, before it saves any manifest.** The contradiction relations from the compare phase used to reach the store only after the manifests, so a crash there lost them while the manifests said the files were done. This also applies to a store without `batched`.

## 2026.10.35

### Index membership and partition severity (`/knowledge`)

- **`KnowledgeIndex.has(id)`** says whether a doc with that id is indexed. A host that adds provisional cards for a compare can now skip claims it already holds. Before this, a host could only see whether `size()` grew, and re-upserting an indexed claim replaced its card (label, sections and entity keys) with the raw claim text.
- **`partitionSeverity(verdict)`** is exported from `/knowledge`: an `upward` edge is an `error`, a skipped layer a `warn`. `genericFindings` uses it, so a host that re-judges a partition violation from its own edge applies the same rule.

## 2026.10.34

### Claim text is bounded to 3 lines again (`/knowledge`)

- **`clampClaim` bounds the claim text to `MAX_QUOTE_LINES` lines as well as `MAX_CLAIM_TEXT_CHARS` characters.** 2026.10.33 bounded the text by length only, so a model claim whose terms spanned many lines passed whole. The privacy rule is that fact text and quote spans are each at most 3 lines. Single-line judge output still passes unchanged. The quote keeps its own 3-line, 300-character bound, and `clampClaim` stays idempotent.
- **README.** The `/knowledge` section documents the review flow:
  - planning, including `PlanOptions.compare` and `force`;
  - the manifest keys a host persists;
  - `similar(ids, texts?)`;
  - single-file re-extraction;
  - the privacy bounds;
  - a minimal host example.

## 2026.10.33

### Directory review closes the PRD 38 audit (`/knowledge`)

2026.10.32 already carried part of this work without a changelog entry. This entry covers every `/knowledge` change since 2026.10.31.

**What a review now does**

- **Extractor upgrades are tracked per file.** The `knowledge:<root>:extractor` manifest entry holds the extractor version plus the files still owed a re-extraction (`owed:<path>`, with the calls each still costs).
  - Every run records it, including runs that abort.
  - A file a run leaves unfinished stays owed until it completes: cut by the budget, a failed judge call, a read error, or an abort before it.
  - A permanently unreadable file is the only file re-read; the rest of the root is not re-processed.
  - On upgrade from 2026.10.32 or earlier every file is owed once. Unchanged sections re-anchor for free; only new sections, such as the text before the first heading, cost a call.
- **The estimate tracks what a run spends.** `planDetail` and `planReview` count owed files at their recorded cost, plus one call per file while an upgrade is pending. They also count the compare calls for claims waiting to be compared.
  - New `PlanOptions.compare` (default `true`). A host that will not pass `similar` must pass `compare: false`, or the estimate counts compare calls that never happen.
  - New `PlanOptions.force` re-reviews every file, unlike a fake empty manifest, which loses removed files and the compare backlog.
  - `refreshPlan` lets *Continue* reuse a confirmed plan without re-walking the directory.
- **Claims awaiting comparison keep their order and survive runs without the ports.** The backlog is trimmed only when a pack asks for contradictions and the compare actually ran. It is capped at 2000, oldest dropped first, and the order survives a manifest that comes back sorted by id. Only packs with `judgement.contradictions` queue claims.
- **`similar(ids, texts?)`.** The port now receives each pending claim's text, so a host need not re-read facts to embed them. A port that takes only `ids` keeps working.
- **Read errors never retire a file.** `ReviewReport.readErrors` counts them; the file's knowledge and manifest hash stay as they were.
- **Docs pack.**
  - Reference-style links are followed.
  - `[[wiki]]` links resolve by name anywhere in the root; links outside the root stay as broken links.
  - The text before the first heading is a section (`#~preamble`).
  - Nested `.gitignore` files apply to their own folder.
  - PDF and Office files are reviewed only when a host parser claims their media type.
- **Stale and contradiction findings.** Relations and model facts on a deleted or changed source show as stale. A new `contradictionRule` finding names both sections.

**New exports**

- `extractFile`, `extractFileDetailed`, `knownKeysOfEntities` (with the types `ExtractFileArgs` and `FileExtraction`) re-extract one file the way a review does. They are for a host watching edits.
- `refreshPlan`, the `ExtractorState` type, `EXTRACTOR_VERSION`, `extractorManifestKey`, `compareManifestKey`.
- `clampQuote`, `clampClaim`, `MAX_QUOTE_LINES`, `MAX_QUOTE_CHARS`, `MAX_CLAIM_TEXT_CHARS`: the privacy bounds a server applies to stored claims. `clampClaim` is idempotent and passes judge output through unchanged.
- `oneLineLabel`, `finding`, `nameResolver`, `aboutTargets`, `ignoreSourcesOf`, `isDocumentPath`, `resolveDocument`, `contradictionRule`.

**Breaking for hosts**

- `IGNORE_FILES` is no longer exported.
- `ReviewReport.readErrors` and `PlannedFile.owed` are required fields. `PlanDetail` gains `extractor` and `carried`.

## 2026.10.32

### Extra headers on the chat client (`/wire`)

- **`ChatClient.headers`.** An optional map of extra request headers a gateway needs, such as tenant or run attribution. `chatComplete` and `chatStream` send them on every request. They are spread first, so they cannot override `Content-Type`, `Authorization` or `Accept`.

## 2026.10.31

### Fact authors are user ids (`/knowledge`)

- **`StoredFact.createdBy` is a `string`.** It used to be a number, but platform user ids are 36-character strings, so a store could not carry the author through. Hosts can now show Delete to a fact's author as well as to admins.

## 2026.10.30

### Small knowledge indexes skip the graph (`/retrieval`, `/knowledge`)

- **Below `exactBelow` cards there is no graph.** The new `IndexParams.exactBelow` defaults to 20000. Below it, `compact()` only quantizes, fixes the scale and names the state; it builds no HNSW. The dense tier then scans every int8 code with an exact integer dot product, keeps the best `k × rerankFactor` (ties by id) and runs the same float rerank as before. An index of a few thousand cards now compacts in a fraction of the CPU, which is what a Worker needs, and its dense results are exact rather than approximate. At or above the threshold nothing changes; `exactBelow: 0` always builds the graph.
- **`similarPairs` without a graph.** In exact mode every pair is scored while there are at most `EXACT_PAIR_BUDGET` (250,000, about 700 cards). Above that, cards are bucketed by the signs of fixed code dimensions in 8 tables, each card is paired with the next 32 members of its bucket, and scoring stops at the budget. The cost is bounded whatever the size, and the result is still a pure function of the state.
- **`lean` applies to graph mode only.** It saves nothing without a graph to walk, so exact mode keeps every code and ignores it.
- **The mode is part of the state.** `digest()` covers `exactBelow` and the mode, so digests differ from 2026.10.29's for the same cards. `storage()` reports `mode` (`'exact'`, `'graph'` or `null`). `IndexMode` is exported.
- **Serialized format version 2.** It adds `exactBelow` and the mode. Version-1 blobs still load: as a graph, with `exactBelow` 0, so a later `compact()` keeps building the graph they were made with.

## 2026.10.29

### The knowledge graph (`/knowledge`)

- **Knowledge is a structure of memory.** A fact alone is a memory. A fact connected to other facts, and to the sources it came from, is knowledge.
- **`buildGraph`** joins a directory's extracted structure (computed, cached by content hash, never stored) with stored `model`, `human` and `agent` facts and relations.
  - Importance is a deterministic PageRank.
  - Partitions give each edge a verdict: `same`, `allowed`, `upward` or `skips`.
  - Output is bounded: by default 400 nodes are kept, plus every focus node, finding node and changed node with their neighbours, and the rest fold into areas.
  - The output is byte-identical for equal input in any order, and is named by `graphDigest`.
- **`reviewDirectory` / `planReview`** review any directory through the one `ingest` pipeline.
  - **Planning:** a walk plus a manifest diff; unchanged files are never read.
  - **Extraction** goes through each pack's parsers, and analyzers run where the pack has them.
  - **Entity resolution:** deterministic keys merge entities; near-duplicates become `possible-same` findings, never merges.
  - **Judgement:** model judgement (claims, entities, contradicts/agrees) is budgeted, at 200 calls by default, and schema-validated.
  - **Staleness:** model facts are anchored to the content hash they were judged from and go stale when it changes. On re-review, sections whose hash is unchanged keep their claims.
- **`traceGraph` / `toMermaid`**: deterministic path traces, such as a layer sequence. **`computeFindings`**: cycles, clone groups, near-duplicates, partition violations, the unpartitioned share and stale facts, each with a `requestKey` for a fixed fix request. **`toCards`**: one index card per node.
- **The docs pack (`DOCS_PACK`).** The builtin markdown, HTML and plain-text parsers gain an optional `extract` on `DocumentParser`. It produces sections, `contains` edges, relative and `[[wiki]]` links, mentions and front-matter keys, plus broken-link and orphan findings. `hashText` moved to `ingest/hash.ts` and is re-exported unchanged.
- **`localSnapshotStore`**: a `KnowledgeStore` over a pluggable persistence port, for folders that are not linked to a project.

### One total order for every ranker, and an insertion-order-free HNSW (`/retrieval`)

- **`compareHits`** (`retrieval/order.ts`) orders by score descending, then id ascending. Scores are quantized first: cosine to 1e-4 (`DENSE_SCORE_QUANTUM`), BM25 / RRF to 1e-6 (`LEXICAL_SCORE_QUANTUM`), so float noise cannot flip a near-tie. `compareDenseHits`, `compareScoreThenId`, `compareIds` and `quantizeScore` are exported.
- **Every ranker uses it.** This covers `bm25Search`, `reciprocalRankFusion`, `fuseHybridArms`, `mmrSelect` and both MMR wrappers, `denseSearch`, `HnswIndex`, `hybridRetrieve`, `SemanticRecall` and `MemoryVectorStore`. Equal scores used to keep input order; they now come back in id order. `mmrSelect` takes `idOf` and `quantum`; without `idOf`, the candidate's position is its id.
- **HNSW levels come from the id.** `levelOf(id, M)` is a geometric draw from the id's sha-256, not from a sequential RNG. `HnswIndex.canonicalBuild` inserts in id order, so the same (id, vector) set gives the same graph in any input order, and `denseSearch` uses it. `HnswOptions.seed` is kept but no longer affects anything. Vectors may be any numeric array, including int8 codes. `graph()` exposes the built layers.

### The deterministic knowledge index (`/retrieval`, `/knowledge`)

- **`createKnowledgeIndex` / `loadKnowledgeIndex`** build a LEANN-style index over cards. Search is a cascade: exact keys first, then BM25 with a code-aware tokenizer (`codeTokenize`), then a dense graph search over int8 codes with integer dot products. The dense candidates are reranked with float vectors recomputed through a hash-keyed LRU, and the tiers are fused with RRF.
- **Storage.** The graph's base layer is pruned with high-degree-preserving pruning and every layer is stored as CSR. Codes are int8, with one symmetric scale fixed at the first compaction. Lean mode drops non-hub codes and recomputes them on the search path.
- **Determinism.** `compact()` is a canonical rebuild, and `digest()` (sha-256) names the state, so a result is a pure function of `(digest, query, opts)`. `serialize()` gives a versioned binary form.
- **No silent fallback.** With no embedder, or one of another fingerprint, the dense tier is absent from `tiers()` and from hits. `compact()` then re-embeds every card under the new fingerprint.
- **More.** `expand` adds graph neighbours to the hits, and `similarPairs` returns deterministic near-duplicate pairs. `sha256` / `sha256Hex` / `Sha256` are a synchronous, pure-TS SHA-256.
- **`./knowledge` subpath export.**

## 2026.10.27

### Code maps: one size floor, a recallable root area, and the read side (`/working`)

- **The 3-file minimum applies to every run.** `consolidate` returns `null` unless the run touched at least `MIN_CONSOLIDATED_FILES` distinct files, counting changed, deleted and read together. Before, a run that changed even one file always wrote a map, so a one-line fix kept re-dating its directory's map. `ConsolidateOptions.minFiles` overrides the floor.
- **Root files have a recallable area.** A root-only run is still filed under `codemap:(root)`, now exported as `CONSOLIDATED_ROOT_AREA`. The new `areaOfPath` resolves a root-level file to that same area, so a reader can match it.
- **Reading a stored map.** `consolidatedFiles(content)` lists the files a map names, never a deletion. `withoutConsolidatedFiles(content, paths)` drops changed files and keeps the rest, returning `null` once nothing is left. With these a store can anchor a map to its files and, when one changes, drop just that file instead of hiding the whole map.
- **Choosing maps for a run.** `consolidatedFactsFor(facts, areas, max?)` keeps the maps whose area overlaps the run's areas, most specific first. `areasOfPathsIn(text)` gives the areas of the repo-relative paths a text names. `areasNamedIn(text, areas)` finds the stored areas a request names by word, for a request that names no path. `consolidatedArea(key)` and `MAX_RECALLED_MAPS` are exported. `CONSOLIDATED_KEY_PREFIX` moved to `codemapAreas.ts`, and its export from `/working` is unchanged.

### Chunked history windowing keeps the cache prefix stable

- **`windowed` drops old turns in chunks.** Over `HISTORY_WINDOW` messages or `HISTORY_TOKEN_BUDGET` tokens, the front of the history jumps to the next fixed cut point (a quarter of the limit, counted from the transcript's start), so it stays put for many turns instead of sliding by a message or two on every one. Before, every turn changed the request's prefix and no provider prompt cache could hit past the system prompt. The window still starts on a user turn and never drops the latest one. `HISTORY_WINDOW_DROP_CHUNK` and `HISTORY_TOKEN_DROP_CHUNK` are exported from the module.

### Docs: the volatile-tail rule

The README now documents where `WorkingMemory.brief()` goes: in the `<run-state>`-tagged `volatileTail`, last, built per request, never persisted into the transcript, and never in the system prompt. A host whose SDK only offers transcript-persisting insertion points, such as the Claude Agent SDK's `PostToolUse` `additionalContext`, should not use `brief()`. It also covers code maps: `consolidate`, `mergeConsolidated`, the size floor and the root area.

## 2026.10.26

### Consolidation markers never drop history; long transcripts summarize in chunks

- **An auto marker whose `keepFromId` cannot be found is ignored.** This happens when the row was deleted, or when a merge or sync re-inserted the history under new ids. `scopeToConsolidation` then falls back to the previous marker, or the whole list. Before, it read the marker as manual and dropped every message between `keepFromId` and the marker that the note never held.
- **`summarizeTranscript` reads at most `TRANSCRIPT_SUMMARY_INPUT_TOKENS` (48k) per call.** A longer transcript, such as a whole long chat on "Consolidate" or a first fold over a big seeded history, is summarized chunk by chunk, each chunk folded onto the note so far. A single message larger than the budget is clipped. A failed chunk returns null, never a note that skipped part of the history. `summaryChunks` is exported.
- **The fold note says "the earlier part of this conversation"** instead of a transcript-entry count, which was misleading once the note is persisted and shown to a reader.
- `CONSOLIDATION_META` is frozen. The fold's internal helpers (`compactTailStart`, `pinnedDirectiveIndex`, `assembleCompacted`, `verbatimStart`, `compactTailStartForBudget`, `COMPACT_TAIL_TURNS`) are no longer exported from `/working`.

## 2026.10.25

### The working transcript, its fold and consolidation markers (`/working`)

The context logic every Builderforce run uses moved here from `brain-embedded`. This is now its one tested home.

- **`buildWorkingTranscript` and the window.** This covers the drop-oldest window anchored on a user turn and the incremental auto-compaction fold: the token-sized tail, the active directive re-pinned, and nothing left in neither the note nor the tail. The budgets ship alongside (`HISTORY_TOKEN_BUDGET`, `COMPACT_TAIL_TOKEN_BUDGET`). It is generic over any OpenAI-style message. `onFolded` now receives `{ droppedMessages, note, coveredEnd }`, so a host can persist the fold.
- **`summarizeTranscript`.** This is the one summarizer prompt that every fold, hand-off note and "Consolidate" compresses with. The model call is injected (`SummaryCompletion`).
- **Consolidation markers.** Manual markers cover everything before them. **Auto** markers, a persisted fold, carry `keepFromId`. `scopeToConsolidation` seeds the next run from the note plus every message from that id on. `foldKeepFromId` maps a fold's transcript cut to that id: exact at a persisted message, conservative otherwise, so nothing is lost.

## Unreleased

### Durable memory storage (schema v3)

`MemoryStore` no longer talks to IndexedDB directly. Every operation goes through a
`ScopedFactRepository` unit of work over a narrow `FactStoragePort`, with two durable
adapters: `IdbStorageAdapter` (browser / webview) and `NodeFileStorageAdapter` (Node:
one fsynced JSON document replaced atomically).

- **Writes resolve on commit.** A `remember` resolves only once its transaction has
  committed, and rejects on abort. Nothing is half-written.
- **Scopes.** Each fact lives in an explicit `MemoryScope` partition
  (account / tenant / project / repository / chat). The default is the unbound scope.
  Unbound record ids are the bare key, so existing data stays where it was.
- **Durable identity.** Every record carries `revision`, `opId` and `provenance`.
  `rememberCas` does a compare-and-set write.
- **Opt-in sync (`sync: true`).** Turns on a sync journal, tombstones and
  `pendingSync`. Facts that have not synced are never evicted by `maxEntries`.
  `markSynced` acknowledges operations and removes them from the journal. With sync
  off (the default) nothing is journaled.
- **Cross-context invalidation.** The IndexedDB adapter uses a BroadcastChannel and
  the node-file adapter polls the file. Either way, a write from another tab or
  process clears this instance's recall caches. The channel opens on the store's
  first operation and is `unref()`ed where the runtime supports it (Node), so a store
  that is never closed does not keep the process running.
- **Concurrent writers on one node-file database.** A read-write transaction holds
  an exclusive `<file>.lock` across refresh, mutate and persist. A lock is reclaimed
  only when its owner's PID is dead, never because it is old. The wait is bounded and
  rejects with `MEMORY_UNAVAILABLE` if the lock does not free up.

#### Migration

IndexedDB databases are upgraded on first open (`DB_VERSION` 3). No action is needed.

- **v1 → v3:** facts keyed by the raw key are copied into the scoped record shape
  in the unbound partition. Their `seq` is kept, they get `revision` 1 and
  `opId` `migrated:v1`, and none are pending sync. The copy runs inside the upgrade
  transaction, so a crash rolls it back.
- **v2 → v3:** adds the `byKey` and `byTag` (multi-entry) indexes on `facts`.
  Stored records are not rewritten, so timestamps, TTLs, tags and content are kept
  exactly. Until a database has the indexes, lookups fall back to a scan.
- The `tombstones`, `journal` and `meta` stores are created if missing.
- Journals written by pre-release builds of this storage layer hold rows already
  marked synced. The next `markSynced` call removes them.

### Error contract

- `MemoryStorageError` now **extends `SSMError`**. Its `code` is the public SSMError
  code, and the storage-specific code moves to the new `storageCode` field:

  | `storageCode`         | `code`               |
  |-----------------------|----------------------|
  | `STORAGE_UNAVAILABLE` | `MEMORY_UNAVAILABLE` |
  | `STORAGE_ABORTED`     | `MEMORY_UNAVAILABLE` |
  | `STORAGE_CONFLICT`    | `MEMORY_CONFLICT` (new `SSMErrorCode`) |

  This means a store with no IndexedDB still rejects with
  `SSMError { code: 'MEMORY_UNAVAILABLE' }`, as it always did. Use
  `isStorageConflict(err)` to detect a CAS conflict.

### Recall

- New option `minRelevance` (0..1, **default 0 — off**). Hits that score below it
  are dropped by `recallSimilar*` and `recallRanked` / `recallHybrid`. The default
  keeps the existing behaviour: recall returns its best-ranked hits even when the
  overlap is weak. Opt in (e.g. `minRelevance: 0.05`) if you would rather get
  nothing back than a zero-overlap match. Note that hybrid scores use the
  reciprocal-rank-fusion scale (about `1/60` per list), so pick a hybrid floor with
  that in mind.

### Hand-off packing (`/handoff`)

- `packHandoff` keeps only the **leading** run of system messages as the prefix.
  System messages later in the conversation stay where they were (in the tail, or
  dropped with the middle) instead of all being moved to the front.
- The note that replaces the dropped middle is sent as a `user` turn when an
  assistant turn follows it, and as an `assistant` turn before a user turn. Roles
  always alternate, so the result never has two assistant turns in a row.
