/**
 * src/knowledge/stores/localSnapshotStore.extractedLog.test.ts
 * PRD 41 A10 rules 6-7: the extract cache lives in an append-only log, reads never wait on a pending
 * snapshot write, and `markStale` touches only the relations of its path.
 */

import type { Extracted, StoredFact, StoredRelation } from '../model.js';
import type { ExtractedLogPersistence, SnapshotPersistence } from './localSnapshotStore.js';
import { LocalSnapshotStore } from './localSnapshotStore.js';

interface Rig {
    snap: SnapshotPersistence & { value?: string; writes: number; gate?: Promise<void> };
    log: ExtractedLogPersistence & { text?: string; appends: string[] };
}

function rig(): Rig {
    const snap: Rig['snap'] = {
        writes: 0,
        async read() { return snap.value; },
        async write(s: string) { if (snap.gate) await snap.gate; snap.writes += 1; snap.value = s; },
    };
    const log: Rig['log'] = {
        appends: [],
        async read() { return log.text; },
        async append(t: string) { log.appends.push(t); log.text = (log.text ?? '') + t; },
    };
    return { snap, log };
}

const SCOPE = { rootId: 'r', title: 'R', packs: ['docs'] };
const ex = (n: string) => [{ name: n }] as unknown as Extracted[];
const mk = (r: Rig, now = 'T') => new LocalSnapshotStore(r.snap, { extractedLog: r.log, now: () => now });

describe('LocalSnapshotStore extractedLog', () => {
    it('keeps extracted out of the snapshot and appends only the new entries', async () => {
        const r = rig();
        const store = mk(r);
        await store.putExtracted('h1', ex('a'));
        await store.putExtracted('h2', ex('b'));
        expect(r.log.appends).toHaveLength(2);
        expect(r.log.appends[0]).toBe(`${JSON.stringify({ h: 'h1', e: ex('a') })}\n`);
        expect(r.log.appends[1]).not.toContain('h1');
        expect(r.snap.writes).toBe(0);
        await store.write({ scope: SCOPE, facts: [], relations: [] });
        expect(JSON.parse(r.snap.value!)).not.toHaveProperty('extracted');
        expect(r.log.appends).toHaveLength(2);
        expect(await mk(r).getExtracted('h2')).toEqual(ex('b'));
    });

    it('batches appends to the batch end', async () => {
        const r = rig();
        const store = mk(r);
        await store.batched(async () => {
            await store.putExtracted('h1', ex('a'));
            await store.putExtracted('h2', ex('b'));
            expect(r.log.appends).toHaveLength(0);
            expect(await store.getExtracted('h1')).toEqual(ex('a'));
        });
        expect(r.log.appends).toHaveLength(1);
        expect(r.log.appends[0]!.trim().split('\n')).toHaveLength(2);
    });

    it('migrates an old snapshot carrying extracted into the log once and drops it from the next write', async () => {
        const r = rig();
        r.snap.value = JSON.stringify({ version: 1, scopes: [], facts: {}, relations: {}, extracted: { old: ex('o') } });
        const store = mk(r);
        expect(await store.getExtracted('old')).toEqual(ex('o'));
        await store.putExtracted('new', ex('n'));
        expect(r.log.appends).toHaveLength(1);
        expect(r.log.text).toContain('"h":"old"');
        expect(r.log.text).toContain('"h":"new"');
        expect(JSON.parse(r.snap.value!)).not.toHaveProperty('extracted');
        await store.putExtracted('new2', ex('n2'));
        expect(r.log.text!.match(/"h":"old"/g)).toHaveLength(1);
    });

    it('ignores a torn trailing line and appends on a fresh line', async () => {
        const r = rig();
        r.log.text = `${JSON.stringify({ h: 'a', e: ex('a') })}\n{"h":"b","e":[{"na`;
        const store = mk(r);
        expect(await store.getExtracted('a')).toEqual(ex('a'));
        expect(await store.getExtracted('b')).toBeUndefined();
        await store.putExtracted('c', ex('c'));
        const again = mk(r);
        expect(await again.getExtracted('c')).toEqual(ex('c'));
        expect(await again.getExtracted('a')).toEqual(ex('a'));
    });

    it('lets the last line for a hash win', async () => {
        const r = rig();
        r.log.text = `${JSON.stringify({ h: 'a', e: ex('first') })}\n${JSON.stringify({ h: 'a', e: ex('second') })}\n`;
        expect(await mk(r).getExtracted('a')).toEqual(ex('second'));
    });

    it('does not block getExtracted while a snapshot write is pending', async () => {
        const r = rig();
        let release!: () => void;
        const store = mk(r);
        await store.putExtracted('h1', ex('a'));
        r.snap.gate = new Promise<void>((res) => { release = res; });
        const pending = store.write({ scope: SCOPE, facts: [], relations: [] });
        await new Promise((res) => setTimeout(res, 0));
        expect(await store.getExtracted('h1')).toEqual(ex('a'));
        expect(r.snap.writes).toBe(0);
        release();
        await pending;
        expect(r.snap.writes).toBe(1);
    });

    it('markStale touches only the relations of that path and matches the old full-scan result', async () => {
        const fact = (key: string): StoredFact => ({ key, origin: 'mechanical' } as unknown as StoredFact);
        const rel = (from: string, to: string, origin: string, sourceHash: string, staleAt?: string): StoredRelation =>
            ({ from, to, kind: 'refs', origin, sourceHash, ...(staleAt ? { staleAt } : {}) } as unknown as StoredRelation);
        const rels = [
            rel('src:r::a.md#1', 'src:r::b.md', 'model', 'old'),
            rel('src:r::c.md', 'src:r::a.md', 'model', 'old'),
            rel('src:r::a.md', 'src:r::d.md', 'model', 'cur'),
            rel('src:r::a.md', 'src:r::e.md', 'mechanical', 'old'),
            rel('src:r::a.md', 'src:r::f.md', 'model', 'old', 'EARLIER'),
            rel('src:r::x.md', 'src:r::y.md', 'model', 'old'),
            rel('doc:1', 'src:r::a.md#9', 'model', 'old'),
        ];
        const expected = rels.map((x) => {
            const on = (ref: string) => ref.startsWith('src:r::a.md');
            return x.origin === 'model' && !x.staleAt && x.sourceHash !== 'cur' && (on(x.from) || on(x.to))
                ? { ...x, staleAt: 'T' } : x;
        });
        const r = rig();
        const store = mk(r);
        await store.write({ scope: SCOPE, facts: [fact('k2'), fact('k1')], relations: rels });
        await store.markStale({ rootId: 'r' }, 'src:r::a.md#2' as never, 'cur');
        const byKey = (a: StoredRelation, b: StoredRelation) => `${a.from}${a.to}`.localeCompare(`${b.from}${b.to}`);
        expect([...(await store.relations({ rootId: 'r' }))].sort(byKey)).toEqual([...expected].sort(byKey));
        expect((await store.facts({ rootId: 'r' })).map((f) => f.key)).toEqual(['k2', 'k1']);
        const saved = JSON.parse(r.snap.value!) as { facts: { r: StoredFact[] } };
        expect(saved.facts.r.map((f) => f.key)).toEqual(['k1', 'k2']);
    });
});
