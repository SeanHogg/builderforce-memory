/**
 * src/contract/regression.store.test.ts — bugs fixed in LocalSnapshotStore and its extract log, each pinned by
 * one test named `regression: <what broke>`.
 */

import type { Extracted } from '../knowledge/model.js';
import { LocalSnapshotStore } from '../knowledge/stores/localSnapshotStore.js';
import { memLog, memPersistence } from '../knowledge/testkit.test-util.js';

const SCOPE = { rootId: 'r', title: 'R', packs: ['docs'] };
const ex = (name: string) => [{ name }] as unknown as Extracted[];
const line = (h: string, name: string) => `${JSON.stringify({ h, e: ex(name) })}\n`;
const legacySnapshot = (extracted: Record<string, Extracted[]>) => JSON.stringify({ version: 1, scopes: [], facts: {}, relations: {}, extracted });
const countOf = (text: string | undefined, hash: string) => (text?.match(new RegExp(`"h":"${hash}"`, 'g')) ?? []).length;

describe('regression: extract log', () => {
    it('regression: checkpoints append to the extract log and never rewrite it or put it back in the snapshot', async () => {
        const snap = memPersistence();
        let snapshotWrites = 0;
        const write = snap.write;
        snap.write = async (s) => { snapshotWrites += 1; await write(s); };
        const log = memLog();
        const store = new LocalSnapshotStore(snap, { extractedLog: log, checkpointMs: 1_000 });

        const N = 6;
        await store.batched(async (flush) => {
            for (let i = 0; i < N; i++) {
                await store.putExtracted(`h${i}`, ex(`e${i}`));
                await store.write({ scope: SCOPE, facts: [{ key: `claim:${i}`, content: 'c', source: 's', importance: 0.5, origin: 'model', fileAnchors: [] }], relations: [] });
                await flush(); // a checkpoint
            }
        });

        expect(log.replaces).toBe(0); // the whole log is never rewritten
        expect(log.appends).toHaveLength(N); // one append per checkpoint…
        expect(log.appends.every((a) => a.trim().split('\n').length === 1)).toBe(true); // …carrying only the entry that was new
        expect(log.text).toBe(log.appends.join('')); // the log is exactly its appends, in order
        expect(snapshotWrites).toBeGreaterThanOrEqual(N);
        expect(snap.value).not.toContain('"extracted"'); // the cache never rides in the snapshot
    });
});

describe('regression: snapshot migration', () => {
    it('regression: after an old snapshot is migrated, extractedEntries still returns every entry, and a second store loads it without migrating again', async () => {
        const snap = memPersistence();
        snap.value = legacySnapshot({ old1: ex('o1'), old2: ex('o2') });
        const log = memLog();

        const first = new LocalSnapshotStore(snap, { extractedLog: log });
        expect(await first.extractedEntries()).toEqual({ old1: ex('o1'), old2: ex('o2') }); // readable before anything persisted
        await first.putExtracted('new', ex('n')); // persists: the legacy entries move into the log
        expect(await first.extractedEntries()).toEqual({ old1: ex('o1'), old2: ex('o2'), new: ex('n') });
        expect(snap.value).not.toContain('"extracted"');
        const appended = log.appends.length;

        const second = new LocalSnapshotStore(snap, { extractedLog: log });
        expect(await second.extractedEntries()).toEqual({ old1: ex('o1'), old2: ex('o2'), new: ex('n') });
        await second.putExtracted('later', ex('l'));
        expect(log.appends).toHaveLength(appended + 1); // only `later`
        expect(countOf(log.text, 'old1')).toBe(1);
        expect(countOf(log.text, 'old2')).toBe(1);
    });

    it('regression: an old snapshot whose entries the log already holds is migrated once, not appended twice', async () => {
        const snap = memPersistence();
        snap.value = legacySnapshot({ old: ex('stale-copy') });
        const log = memLog();
        log.text = line('old', 'logged-copy');

        const store = new LocalSnapshotStore(snap, { extractedLog: log });
        await store.putExtracted('new', ex('n'));
        expect(countOf(log.text, 'old')).toBe(1);
        expect((await store.extractedEntries()).old).toEqual(ex('logged-copy')); // the log's line wins
    });

    it('regression: a store without a log keeps reading and writing the snapshot\'s own extracted', async () => {
        const snap = memPersistence();
        snap.value = legacySnapshot({ old: ex('o') });
        const store = new LocalSnapshotStore(snap);
        await store.putExtracted('new', ex('n'));
        expect(await store.extractedEntries()).toEqual({ old: ex('o'), new: ex('n') });
        expect(Object.keys(JSON.parse(snap.value!).extracted).sort()).toEqual(['new', 'old']);
    });
});

describe('regression: re-check channel', () => {
    it('regression: hashes marked for re-check survive a reload, and clearing them removes them from the snapshot', async () => {
        const snap = memPersistence();
        await new LocalSnapshotStore(snap).markRecheck(['b', 'a']);
        const again = new LocalSnapshotStore(snap);
        expect(await again.recheckHashes()).toEqual(['a', 'b']);
        await again.clearRecheck(['a']);
        expect(await new LocalSnapshotStore(snap).recheckHashes()).toEqual(['b']);
        await again.clearRecheck(['b']);
        expect(JSON.parse(snap.value!).recheck).toBeUndefined();
    });
});
