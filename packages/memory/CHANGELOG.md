# Changelog — @seanhogg/builderforce-memory

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
