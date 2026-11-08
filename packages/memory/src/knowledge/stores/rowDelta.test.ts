/**
 * src/knowledge/stores/rowDelta.test.ts — the dirty tracking of a row-persisted store: last mark wins per key, `take` swaps
 * the marks out, `restore` merges them back UNDER newer marks, `overlay` lays unwritten rows over rows read from the host.
 */

import type { Extracted, ScopeRecord, StoredFact } from '../model.js';
import { RowDelta } from './rowDelta.js';
import { RowMemory } from './rowMemory.js';

const fact = (key: string, content: string): StoredFact => ({ key, content, source: 's', importance: 0.5, origin: 'model', fileAnchors: [] });

function memory(): RowMemory {
    const m = new RowMemory();
    m.factsOf('r').set('k', fact('k', 'v1'));
    m.scopes = [{ rootId: 'r', title: 'R', packs: ['docs'] } as ScopeRecord];
    m.extracted.set('h', [{ name: 'e' }] as unknown as Extracted[]);
    m.recheck.add('c');
    return m;
}

describe('RowDelta', () => {
    it('an upsert after a delete is an upsert, a delete after an upsert is a delete (last mark wins)', () => {
        const d = new RowDelta();
        const m = memory();
        d.mark('facts', ['r', 'k'], true);
        d.mark('facts', ['r', 'k']);
        expect(d.toChange(d.take(), m).upserts.facts.map(([, f]) => f.content)).toEqual(['v1']);
        d.mark('facts', ['r', 'k']);
        d.mark('facts', ['r', 'k'], true);
        const c = d.toChange(d.take(), m);
        expect(c.upserts.facts).toEqual([]);
        expect(c.deletes.facts).toEqual([['r', 'k']]);
    });

    it('take swaps the marks out; restore merges them back under newer marks; nothing is lost or reordered', () => {
        const d = new RowDelta();
        const m = memory();
        d.mark('facts', ['r', 'k']);
        d.mark('recheck', ['c']);
        const taken = d.take();
        expect(d.pending).toBe(false);
        d.mark('facts', ['r', 'k'], true); // newer: a delete
        d.mark('extracted', ['h']);
        d.restore(taken);
        const c = d.toChange(d.take(), m);
        expect(c.deletes.facts).toEqual([['r', 'k']]); // the newer delete beats the restored upsert
        expect(c.upserts.recheck).toEqual(['c']); // the restored mark survives
        expect(c.upserts.extracted.map(([h]) => h)).toEqual(['h']); // the newer mark survives
    });

    it('overlay skips an upsert mark whose row is gone from memory: it does not delete the host\'s row', () => {
        const d = new RowDelta();
        d.mark('facts', ['r', 'k']);
        d.mark('scopes', ['r']);
        d.mark('extracted', ['h']);
        const fresh = new RowMemory();
        fresh.factsOf('r').set('k', fact('k', 'theirs'));
        fresh.scopes = [{ rootId: 'r', title: 'T', packs: [] } as ScopeRecord];
        fresh.extracted.set('h', [{ name: 'x' }] as unknown as Extracted[]);
        d.overlay(new RowMemory(), fresh);
        expect(fresh.facts.get('r')!.get('k')!.content).toBe('theirs');
        expect(fresh.scopes).toHaveLength(1);
        expect(fresh.extracted.has('h')).toBe(true);
    });

    it('an empty take is empty; an upsert whose row is gone from memory is skipped', () => {
        const d = new RowDelta();
        expect(d.take().empty).toBe(true);
        d.mark('facts', ['r', 'missing']);
        const t = d.take();
        expect(t.empty).toBe(false);
        expect(d.toChange(t, memory()).upserts.facts).toEqual([]);
    });

    it('overlay keeps unwritten upserts and deletes (in-flight and current) over rows read from the host', () => {
        const d = new RowDelta();
        const old = memory();
        old.factsOf('r').set('k2', fact('k2', 'mine'));
        old.factsOf('r').set('k3', fact('k3', 'mine'));
        d.mark('facts', ['r', 'k2']); // will be in flight
        d.take();
        d.mark('facts', ['r', 'k3'], true); // newer: delete
        d.mark('facts', ['r', 'k'], true);
        const fresh = new RowMemory();
        fresh.factsOf('r').set('k', fact('k', 'theirs'));
        fresh.factsOf('r').set('k3', fact('k3', 'theirs'));
        fresh.factsOf('r').set('k4', fact('k4', 'theirs'));
        d.overlay(old, fresh);
        expect([...fresh.facts.get('r')!.entries()].map(([k, v]) => [k, v.content]).sort()).toEqual([['k2', 'mine'], ['k4', 'theirs']]);
    });
});
