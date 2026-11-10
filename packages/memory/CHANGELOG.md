# Changelog — @seanhogg/builderforce-memory

## Unreleased

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
