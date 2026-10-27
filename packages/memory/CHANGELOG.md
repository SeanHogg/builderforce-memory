# Changelog — @seanhogg/builderforce-memory

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
