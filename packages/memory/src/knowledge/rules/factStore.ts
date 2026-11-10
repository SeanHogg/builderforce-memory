/**
 * knowledge/rules/factStore.ts — the FACT STORE PORT (PRD 41 §B4) and the pure functions that persist, restore and sweep
 * a {@link FactGraph} through it. No I/O here; {@link memoryFactStore} is the in-memory reference adapter.
 *
 * The graph maps onto the port 1:1:
 *   - OBJECTS: each `FileFacts` as its canonical JSON (`factsJson`) under its content address `factsHash`. Write-once.
 *   - MANIFEST: path → {@link FactManifestEntry} (`factsHash`, `contentHash`, kind hashes), stamped with a
 *     {@link FactStoreVersion}: an `epoch` (a random id minted when the store is created or reset) and a `generation`
 *     bumped by every manifest write and every sweep.
 *   - DERIVED: each {@link DerivedRow} under {@link derivedRowKey}, carrying its input-hash trace (`inputs`).
 *
 * Commit order: objects first, then the manifest delta and the derived puts/deletes in ONE transaction (`writeManifest`,
 * compare-and-swap on the version). A write interrupted between the two leaves the previous manifest valid — it references
 * only objects that already existed — and at worst unreferenced objects, which {@link sweepFactStore} removes.
 *
 * ADAPTERS:
 *   - Node (VSIX, on-prem): the G03 SQLite store (`localRecordTransaction`): an `extracted(hash PRIMARY KEY, json)` table
 *     plus manifest, derived and version rows in one `index.sqlite`. `writeManifest` and `sweepObjects` run in ONE
 *     `BEGIN IMMEDIATE` transaction each (the write lock is taken before the version is compared, so two processes never
 *     both pass the check). Adapter-private columns (stat, git blob id) never cross the port and never enter the dump (§B8.3).
 *   - Worker: the port needs a transactional compare-and-swap over manifest + derived. Put it behind a Durable Object
 *     (transactional storage), or keep manifest + derived as ONE R2 object written with an `onlyIf` etag precondition and
 *     the objects as R2 objects under their hash. Plain KV (eventually consistent, no CAS) cannot implement this port.
 *
 * HOST CONTRACT (recovery):
 *   - Saves to one store are SERIALIZED within a process (one writer queue per store); the version CAS guards only against
 *     other processes.
 *   - `fact_store_generation_conflict` (another writer, or a sweep, moved the store): `loadFactGraph`, re-apply the pending
 *     `FactChange`s with `updateFactGraph`, then `saveFactGraph` with the loaded {@link StoredFactGraph}; retry a bounded
 *     number of times (e.g. 3), then give up until the next change.
 *   - `fact_store_object_missing` / `fact_store_object_corrupt` / `fact_store_manifest_corrupt` / `fact_store_derived_corrupt`:
 *     the store is a disposable cache (§B1: it rebuilds to the same bytes) — call `reset()` and rebuild from the files.
 */

import { canonicalJson, compareStr } from '../refs.js';
import { readFileFacts, type FileFacts } from '../facts/fileFacts.js';
import { factKindHashes, factsHash, factsJson } from '../facts/factsHash.js';
import { derivedRowKey, type DerivedRow } from './evaluateRules.js';
import { factGraphDump, sameManifestEntry, type FactGraph, type FactManifestEntry } from './factGraph.js';

/** A value or a promise of it: the SQLite adapter is synchronous, the Durable Object / R2 one is not. */
export type Awaitable<T> = T | Promise<T>;

/** Which store (`epoch`, minted on create and on `reset()`) and which state of it (`generation`, 0 when empty). */
export interface FactStoreVersion {
    epoch: string;
    generation: number;
}

/** The stored manifest: path → entry, at a version. */
export interface FactStoreManifest extends FactStoreVersion {
    files: ReadonlyMap<string, FactManifestEntry>;
}

/** One manifest write: the entries put and the paths deleted, committed as `generation` (stored + 1) of `epoch`. */
export interface FactManifestDelta extends FactStoreVersion {
    puts: ReadonlyMap<string, FactManifestEntry>;
    deletes: readonly string[];
}

export interface FactStorePort {
    /** The stored JSON of each hash present (absent hashes are left out). */
    getObjects(hashes: readonly string[]): Awaitable<ReadonlyMap<string, string>>;
    /** Stores objects (hash → JSON). WRITE-ONCE: an existing hash is left as it is. */
    putObjects(objects: ReadonlyMap<string, string>): Awaitable<void>;
    /** The version alone (cheap): what a load re-reads to detect a write that landed under it. */
    readVersion(): Awaitable<FactStoreVersion>;
    readManifest(): Awaitable<FactStoreManifest>;
    /**
     * ONE transaction: compares the version (`manifest.epoch` must be the stored epoch and `manifest.generation` the stored
     * generation + 1, else `fact_store_generation_conflict`), checks that every put's object exists (else
     * `fact_store_object_missing`), applies the manifest delta, puts the derived rows (by {@link derivedRowKey}) and
     * deletes the derived keys.
     */
    writeManifest(manifest: FactManifestDelta, derivedPuts: readonly DerivedRow[], derivedDeletes: readonly string[]): Awaitable<void>;
    /** Every stored derived row by its key (bulk; the load path). */
    derivedRows(): Awaitable<ReadonlyMap<string, DerivedRow>>;
    /**
     * ONE transaction: compares the version with `expected` (else `fact_store_generation_conflict`) and deletes every object
     * the manifest does not reference; when it deleted any, it bumps the generation (a no-op sweep leaves the version as it
     * was). Returns how many it deleted. The bump makes a writer that put objects before the sweep conflict on its
     * `writeManifest`, so it re-puts them on retry.
     */
    sweepObjects(expected: FactStoreVersion): Awaitable<number>;
    /** Drops everything and mints a new epoch at generation 0 (the recovery for a corrupt store). */
    reset(): Awaitable<void>;
}

/** A graph and the store version it was loaded or saved at: what a later save diffs against. */
export interface StoredFactGraph extends FactStoreVersion {
    graph: FactGraph;
}

/** What a save wrote, and the stored graph to pass as `previous` to the next save. */
export interface FactStoreSaveResult extends StoredFactGraph {
    /** Objects put (hashes, sorted). */
    objects: string[];
    /** Manifest paths put and deleted (sorted). */
    puts: string[];
    deletes: string[];
    /** Derived keys put and deleted (sorted). */
    derivedPuts: string[];
    derivedDeletes: string[];
}

const rowJson = (row: DerivedRow) => canonicalJson(row);
const sameVersion = (a: FactStoreVersion, b: FactStoreVersion) => a.epoch === b.epoch && a.generation === b.generation;
const conflict = (why: string) => new Error(`fact_store_generation_conflict:${why}`);

/**
 * Persists `graph`, writing ONLY the delta against what the store holds: the objects of new or changed manifest entries,
 * the manifest entries that moved, the derived rows that moved and the keys that went away.
 *
 * FAST PATH: with `previous`, the save reads only the version (`readVersion`). When it is `previous`'s epoch and
 * generation, the store holds exactly `previous.graph`, so the manifest and the derived rows are diffed against it in
 * memory (a reused row is the same object) and neither `readManifest` nor `derivedRows` is called — O(changes), not
 * O(files). Otherwise (no `previous`, or a stale one) the save diffs against the stored manifest and rows. Either way the
 * commit is the version CAS of `writeManifest`, so a writer that lands in between makes this save reject
 * `fact_store_generation_conflict` (see the recovery contract in the header).
 *
 * `previous` must be EXACTLY a result of `saveFactGraph` or `loadFactGraph` on this port — never hand-built or patched: the
 * fast path trusts that its `graph` is what the store holds at its `{ epoch, generation }`.
 */
export async function saveFactGraph(port: FactStorePort, graph: FactGraph, previous?: StoredFactGraph): Promise<FactStoreSaveResult> {
    if (previous) {
        const version = await port.readVersion();
        if (sameVersion(previous, version)) return commitDelta(port, graph, version, previous.graph.manifest, previous.graph.derived);
    }
    const stored = await port.readManifest();
    return commitDelta(port, graph, stored, stored.files, await port.derivedRows());
}

/**
 * Writes the delta of `graph` against `base` (the manifest and derived rows the store holds at `version`): objects first,
 * then one `writeManifest`. A base row that is the same object as the graph's row is unchanged without serializing it.
 */
async function commitDelta(
    port: FactStorePort,
    graph: FactGraph,
    version: FactStoreVersion,
    baseFiles: ReadonlyMap<string, FactManifestEntry>,
    baseRows: ReadonlyMap<string, DerivedRow>,
): Promise<FactStoreSaveResult> {
    const puts = new Map<string, FactManifestEntry>();
    for (const [path, e] of graph.manifest) if (!sameManifestEntry(e, baseFiles.get(path))) puts.set(path, e);
    const deletes = [...baseFiles.keys()].filter((p) => !graph.manifest.has(p)).sort(compareStr);

    const objects = [...new Set([...puts.values()].map((e) => e.factsHash))].sort(compareStr);
    if (objects.length > 0) {
        const jsons = new Map<string, string>();
        for (const hash of objects) {
            const facts = graph.objects.get(hash);
            if (!facts) throw new Error(`fact_store_object_missing:${hash}`);
            jsons.set(hash, factsJson(facts));
        }
        await port.putObjects(jsons);
    }

    const derivedPuts: DerivedRow[] = [];
    for (const [key, row] of graph.derived) {
        const prior = baseRows.get(key);
        if (prior !== row && (!prior || rowJson(prior) !== rowJson(row))) derivedPuts.push(row);
    }
    const derivedDeletes = [...baseRows.keys()].filter((k) => !graph.derived.has(k));
    derivedPuts.sort((a, b) => compareStr(derivedRowKey(a.rule, a.focus), derivedRowKey(b.rule, b.focus)));
    derivedDeletes.sort(compareStr);

    const generation = version.generation + 1;
    await port.writeManifest({ epoch: version.epoch, generation, puts, deletes }, derivedPuts, derivedDeletes);
    return {
        epoch: version.epoch,
        generation,
        graph,
        objects,
        puts: [...puts.keys()].sort(compareStr),
        deletes,
        derivedPuts: derivedPuts.map((r) => derivedRowKey(r.rule, r.focus)),
        derivedDeletes,
    };
}

export interface LoadFactGraphOptions {
    /**
     * `full` (default; tests, the oracle): every object must hash to its address, and every manifest entry must match its
     * object. `manifest`: trusts the stored address and checks only each entry's content hash and kind hashes against its
     * object (cheaper on a large store; a corruption that leaves those intact goes unnoticed).
     */
    verify?: 'full' | 'manifest';
}

/** How many times a load retries when a write lands under it before rejecting `fact_store_generation_conflict:load`. */
export const FACT_STORE_LOAD_ATTEMPTS = 3;

async function loadOnce(port: FactStorePort, stored: FactStoreManifest, verify: 'full' | 'manifest'): Promise<FactGraph> {
    const hashes = [...new Set([...stored.files.values()].map((e) => e.factsHash))].sort(compareStr);
    const raw = await port.getObjects(hashes);
    const objects = new Map<string, FileFacts>();
    for (const hash of hashes) {
        const json = raw.get(hash);
        if (json === undefined) throw new Error(`fact_store_object_missing:${hash}`);
        let facts: FileFacts;
        try {
            facts = readFileFacts(JSON.parse(json) as FileFacts);
        } catch {
            throw new Error(`fact_store_object_corrupt:${hash}`);
        }
        if (verify === 'full' && factsHash(facts) !== hash) throw new Error(`fact_store_object_corrupt:${hash}`);
        objects.set(hash, facts);
    }
    const manifest = new Map<string, FactManifestEntry>();
    for (const path of [...stored.files.keys()].sort(compareStr)) {
        const e = stored.files.get(path)!;
        const facts = objects.get(e.factsHash)!;
        const entry: FactManifestEntry = { contentHash: facts.contentHash, factsHash: e.factsHash, kinds: factKindHashes(facts) };
        if (!sameManifestEntry(e, entry)) throw new Error(`fact_store_manifest_corrupt:${path}`);
        manifest.set(path, entry);
    }
    const derived = new Map<string, DerivedRow>();
    const rows = await port.derivedRows();
    for (const key of [...rows.keys()].sort(compareStr)) {
        const row = rows.get(key)!;
        if (derivedRowKey(row.rule, row.focus) !== key) throw new Error(`fact_store_derived_corrupt:${key}`);
        derived.set(key, row);
    }
    return { manifest, objects, derived };
}

/**
 * Restores the graph the store holds, with the version it was read at: the manifest, every referenced object (read through
 * `readFileFacts`, so an object written before an additive field existed reads with that field empty) and every derived
 * row with its trace. The result continues under `updateFactGraph` with early cutoff intact: a row whose trace still holds
 * is reused, not rerun. The version is re-read after the rows: a write that landed mid-load (or a sweep that removed an
 * object it was about to read) retries, up to {@link FACT_STORE_LOAD_ATTEMPTS}, so a graph never mixes generations (§B8.4).
 * Throws the corruption errors of the header when the store itself is bad.
 */
export async function loadFactGraph(port: FactStorePort, opts: LoadFactGraphOptions = {}): Promise<StoredFactGraph> {
    const verify = opts.verify ?? 'full';
    for (let attempt = 0; attempt < FACT_STORE_LOAD_ATTEMPTS; attempt++) {
        const stored = await port.readManifest();
        let graph: FactGraph;
        try {
            graph = await loadOnce(port, stored, verify);
        } catch (err) {
            if (!sameVersion(stored, await port.readVersion())) continue;
            throw err;
        }
        if (sameVersion(stored, await port.readVersion())) return { epoch: stored.epoch, generation: stored.generation, graph };
    }
    throw conflict('load');
}

/**
 * The store's canonical dump (§B8.6, the port's `canonicalDump()`): sorted canonical JSON of the manifest, the referenced
 * objects and the derived rows — exactly `factGraphDump` of the loaded graph, so a store and the graph it was saved from
 * dump to the same bytes. Versions, storage layout and unreferenced objects never take part.
 */
export async function factStoreDump(port: FactStorePort): Promise<string> {
    return factGraphDump((await loadFactGraph(port)).graph);
}

/**
 * Removes the objects the current manifest does not reference (§B4 sweep; run when idle). Rejects
 * `fact_store_generation_conflict` when a write lands between reading the version and the sweep; the next idle sweep
 * catches up. Returns how many objects it deleted.
 */
export async function sweepFactStore(port: FactStorePort): Promise<number> {
    return port.sweepObjects(await port.readVersion());
}

/** The in-memory reference adapter (tests, a Worker's per-request graph). Stores JSON, so a load is a real round trip. */
export interface MemoryFactStore extends FactStorePort {
    /** Every stored object hash, referenced or not (sorted). */
    objectHashes(): string[];
}

let epochCounter = 0;
function newEpoch(): string {
    const uuid = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto?.randomUUID?.();
    return uuid ?? `epoch-${Date.now().toString(36)}-${(epochCounter++).toString(36)}-${Math.random().toString(36).slice(2)}`;
}

export function memoryFactStore(): MemoryFactStore {
    const objects = new Map<string, string>();
    let manifest = new Map<string, string>();
    let derived = new Map<string, string>();
    let epoch = newEpoch();
    let generation = 0;
    const check = (expected: FactStoreVersion, nextGeneration: number) => {
        if (expected.epoch !== epoch || nextGeneration !== generation + 1) {
            throw conflict(`${epoch}@${generation}->${expected.epoch}@${expected.generation}`);
        }
    };
    return {
        getObjects: (hashes) => new Map(hashes.filter((h) => objects.has(h)).map((h) => [h, objects.get(h)!] as const)),
        putObjects: (puts) => { for (const [h, json] of puts) if (!objects.has(h)) objects.set(h, json); },
        readVersion: () => ({ epoch, generation }),
        readManifest: () => ({
            epoch,
            generation,
            files: new Map([...manifest].map(([p, json]) => [p, JSON.parse(json) as FactManifestEntry] as const)),
        }),
        writeManifest: (delta, derivedPuts, derivedDeletes) => {
            check(delta, delta.generation);
            for (const e of delta.puts.values()) if (!objects.has(e.factsHash)) throw new Error(`fact_store_object_missing:${e.factsHash}`);
            // Stage, then swap: all or nothing.
            const next = new Map(manifest);
            for (const [p, e] of delta.puts) next.set(p, canonicalJson(e));
            for (const p of delta.deletes) next.delete(p);
            const rows = new Map(derived);
            for (const row of derivedPuts) rows.set(derivedRowKey(row.rule, row.focus), rowJson(row));
            for (const key of derivedDeletes) rows.delete(key);
            manifest = next;
            derived = rows;
            generation = delta.generation;
        },
        derivedRows: () => new Map([...derived].map(([k, json]) => [k, JSON.parse(json) as DerivedRow] as const)),
        sweepObjects: (expected) => {
            check(expected, expected.generation + 1);
            const live = new Set([...manifest.values()].map((json) => (JSON.parse(json) as FactManifestEntry).factsHash));
            let removed = 0;
            for (const h of [...objects.keys()]) if (!live.has(h)) { objects.delete(h); removed++; }
            if (removed > 0) generation += 1;
            return removed;
        },
        reset: () => {
            objects.clear();
            manifest = new Map();
            derived = new Map();
            epoch = newEpoch();
            generation = 0;
        },
        objectHashes: () => [...objects.keys()].sort(compareStr),
    };
}
