# Changelog — @seanhogg/builderforce-memory

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
