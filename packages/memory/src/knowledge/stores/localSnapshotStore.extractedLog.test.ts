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

    it('writes a batch as ONE append of complete lines', async () => {
        const r = rig();
        const store = mk(r);
        await store.batched(async () => {
            await store.putExtracted('h1', ex('a'));
            await store.putExtracted('h2', ex('b'));
        });
        expect(r.log.appends).toHaveLength(1);
        expect(r.log.appends[0]!.endsWith('\n')).toBe(true);
        for (const line of r.log.appends[0]!.trim().split('\n')) expect(() => JSON.parse(line)).not.toThrow();
    });

    it('drops a torn or unparseable line without losing the lines around it', async () => {
        const r = rig();
        r.log.text = `${JSON.stringify({ h: 'a', e: ex('a') })}\n{"h":"x","e":[{"na\n${JSON.stringify({ h: 'c', e: ex('c') })}\nnot json\n`;
        const store = mk(r);
        expect(await store.getExtracted('a')).toEqual(ex('a'));
        expect(await store.getExtracted('c')).toEqual(ex('c'));
        expect(await store.getExtracted('x')).toBeUndefined();
        expect(Object.keys(await store.extractedEntries())).toEqual(['a', 'c']);
    });

    it('a failed append marks the log torn: the retry starts on a fresh line and nothing is lost', async () => {
        const r = rig();
        let fail = true;
        r.log.append = async (t: string) => {
            r.log.appends.push(t);
            if (fail) { r.log.text = (r.log.text ?? '') + t.slice(0, 10); throw new Error('disk full'); } // a partial line landed
            r.log.text = (r.log.text ?? '') + t;
        };
        const store = mk(r);
        await expect(store.putExtracted('h1', ex('a'))).rejects.toThrow('disk full');
        fail = false;
        await store.putExtracted('h2', ex('b'));
        expect(r.log.appends[1]!.startsWith('\n')).toBe(true);
        const again = mk(r);
        expect(await again.getExtracted('h1')).toEqual(ex('a'));
        expect(await again.getExtracted('h2')).toEqual(ex('b'));
    });

    it('does not re-append an identical entry, but appends a changed one', async () => {
        const r = rig();
        const store = mk(r);
        await store.putExtracted('h1', ex('a'));
        await store.putExtracted('h1', ex('a'));
        expect(r.log.appends).toHaveLength(1);
        await store.putExtracted('h1', ex('changed'));
        expect(r.log.appends).toHaveLength(2);
        // An entry the log already held at load is not re-appended either.
        const reloaded = mk(r);
        await reloaded.putExtracted('h1', ex('changed'));
        expect(r.log.appends).toHaveLength(2);
    });

    it('does not rewrite the snapshot for an identical entry when there is no log', async () => {
        const snap = rig().snap;
        const store = new LocalSnapshotStore(snap, { now: () => 'T' });
        await store.putExtracted('h1', ex('a'));
        await store.putExtracted('h1', ex('a'));
        expect(snap.writes).toBe(1);
    });

    it('compacts the log on load when dead bytes exceed live bytes, and only with a `replace` port', async () => {
        const r = rig();
        const lines = [1, 2, 3, 4].map((n) => `${JSON.stringify({ h: 'a', e: ex(`v${n}`) })}\n`);
        r.log.text = `${lines.join('')}${JSON.stringify({ h: 'b', e: ex('b') })}\ntorn{"h"\n`;
        const withoutReplace = mk(r);
        expect(await withoutReplace.getExtracted('a')).toEqual(ex('v4'));
        expect(r.log.text).toContain('v1');

        const replaced: string[] = [];
        r.log.replace = async (t: string) => { replaced.push(t); r.log.text = t; };
        const store = mk(r);
        expect(await store.getExtracted('a')).toEqual(ex('v4'));
        expect(replaced).toHaveLength(1);
        expect(replaced[0]).toBe(`${lines[3]}${JSON.stringify({ h: 'b', e: ex('b') })}\n`);
        expect(await mk(r).getExtracted('b')).toEqual(ex('b'));
        expect(replaced).toHaveLength(1); // now all live: nothing more to compact
    });

    it('does not compact a log that is mostly live', async () => {
        const r = rig();
        r.log.text = `${JSON.stringify({ h: 'a', e: ex('a') })}\n${JSON.stringify({ h: 'b', e: ex('b') })}\n`;
        const replaced: string[] = [];
        r.log.replace = async (t: string) => { replaced.push(t); };
        await mk(r).getExtracted('a');
        expect(replaced).toHaveLength(0);
    });

    it('a failed background checkpoint backs off by checkpointMs instead of retrying on every mutation', async () => {
        const r = rig();
        let attempts = 0;
        r.snap.write = async () => { attempts += 1; throw new Error('disk full'); };
        const clock = { ms: 0 };
        const store = new LocalSnapshotStore(r.snap, { extractedLog: r.log, now: () => 'T', checkpointMs: 1000, nowMs: () => clock.ms });
        const settle = () => new Promise((res) => setTimeout(res, 0));
        const touch = (n: number) => store.write({ scope: { ...SCOPE, title: `t${n}` }, facts: [], relations: [] });
        await expect(store.batched(async () => {
            clock.ms = 1000;
            await touch(1);
            await settle();
            expect(attempts).toBe(1);
            clock.ms = 1500;
            await touch(2);
            await settle();
            expect(attempts).toBe(1); // inside the backoff window
            clock.ms = 2000;
            await touch(3);
            await settle();
            expect(attempts).toBe(2);
        })).rejects.toThrow('disk full'); // the batch end still reports the unwritten snapshot
    });

    it('keeps a snapshot that does not parse aside, then starts empty rather than silently dropping it', async () => {
        const r = rig();
        r.snap.value = '{"version":1,"scopes":[';
        const kept: string[] = [];
        r.snap.keepAside = async (t: string) => { kept.push(t); };
        const store = mk(r);
        expect(await store.scopes()).toEqual([]);
        expect(kept).toEqual(['{"version":1,"scopes":[']);
        await store.write({ scope: SCOPE, facts: [], relations: [] });
        expect(await mk(r).scopes()).toHaveLength(1);
    });

    it('still starts empty when keepAside itself fails', async () => {
        const r = rig();
        r.snap.value = 'garbage';
        r.snap.keepAside = async () => { throw new Error('read-only'); };
        expect(await mk(r).scopes()).toEqual([]);
    });

    it('extractedEntries reads the log, else the snapshot\'s legacy extracted', async () => {
        const r = rig();
        r.log.text = `${JSON.stringify({ h: 'a', e: ex('a') })}\n`;
        r.snap.value = JSON.stringify({ version: 1, scopes: [], facts: {}, relations: {}, extracted: { old: ex('o') } });
        expect(await mk(r).extractedEntries()).toEqual({ a: ex('a'), old: ex('o') });
        const plain = rig();
        plain.snap.value = r.snap.value;
        expect(await new LocalSnapshotStore(plain.snap, {}).extractedEntries()).toEqual({ old: ex('o') });
    });
});
