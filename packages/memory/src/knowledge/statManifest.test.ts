/**
 * src/knowledge/statManifest.test.ts — the ONE stat model (PRD 41 §B2): a size+mtimeNs+fileId row lets a warm walk reuse a
 * stored hash without reading the file, git's racy-file rule re-hashes a file whose mtime is within 2 s of the reference
 * time, a git blob hint survives a checkout that moved the stat, and the HashCache port has two backings here (memory,
 * deferred) that agree on delta semantics.
 */

import { jest } from '@jest/globals';
import {
    EMPTY_STAT_SET, RACY_WINDOW_MS, StatSetMemo, deferredHashCache, isRacy, memoryHashCache, msToNs, reusedHash, sameRow,
    type FileStat, type GitHints, type HashCache, type StatDelta, type StatRow, type StatSet,
} from './statManifest.js';

const NOW = 1_000_000; // ms
const NOW_NS = msToNs(NOW);
const WINDOW_NS = BigInt(RACY_WINDOW_MS) * BigInt(1_000_000);
// An mtime far from the reference time so the racy rule does not fire unless a test moves it.
const OLD_MTIME = msToNs(NOW - 60_000);
const RULE = 'sha256-eol';

const row = (path: string, size: number, mtimeNs: bigint, hash: string, extra: Partial<StatRow> = {}): StatRow =>
    ({ path, size, mtimeNs, hash, ...extra });
const set = (rows: StatRow[], writtenAtNs?: bigint): StatSet =>
    ({ rows: new Map(rows.map((r) => [r.path, r] as const)), ...(writtenAtNs !== undefined ? { writtenAtNs } : {}) });
const stat = (size: number, mtimeNs: bigint, fileId?: bigint): FileStat => ({ size, mtimeNs, ...(fileId !== undefined ? { fileId } : {}) });

describe('contract: the unified stat model types', () => {
    it('a HashCache port takes a StatDelta and returns a StatSet, in every backing', async () => {
        const backings: HashCache[] = [memoryHashCache(), deferredHashCache(() => memoryHashCache())];
        const delta: StatDelta = { rule: RULE, upserts: [row('a.ts', 1, OLD_MTIME, 'h-a', { fileId: 7n, gitBlob: 'b1' })], deletes: [], writtenAtNs: NOW_NS };
        for (const cache of backings) {
            await cache.save('r', delta);
            const loaded: StatSet = await cache.load('r', RULE);
            expect(loaded.rows.get('a.ts')).toEqual(delta.upserts[0]);
            expect(loaded.writtenAtNs).toBe(NOW_NS);
            expect(cache.sizeOf?.('r', 'a.ts')).toBe(1);
        }
    });

    it('RACY_WINDOW_MS is the git window (2 s) and msToNs scales ms to ns', () => {
        expect(RACY_WINDOW_MS).toBe(2000);
        expect(msToNs(1.5)).toBe(BigInt(1_500_000));
        expect(msToNs(1_700_000_000_123)).toBe(BigInt('1700000000123000000'));
    });

    it('regression: msToNs is exact for a current epoch time (ms * 1e6 exceeds 2^53) and keeps a fractional ms', () => {
        // 0.5 is exactly representable at this magnitude (double spacing is 2^-12 ms), so the expected ns is exact.
        expect(msToNs(1_700_000_000_123.5)).toBe(BigInt('1700000000123500000'));
        expect(msToNs(1_700_000_000_123)).toBe(BigInt('1700000000123000000'));
        expect(msToNs(-1.5)).toBe(BigInt(-1_500_000));
    });
    it('regression: msToNs is unchanged below 2^53 / 1e6 ms', () => {
        expect(msToNs(0)).toBe(BigInt(0));
        expect(msToNs(1_000_000)).toBe(BigInt('1000000000000'));
        expect(msToNs(1_234.567)).toBe(BigInt(Math.round(1_234.567 * 1_000_000)));
    });
});

describe('isRacy (git racy-file rule, ns against a reference time)', () => {
    it('is racy within 2 s before the reference time, at it, and after it (future mtimes)', () => {
        expect(isRacy(NOW_NS - WINDOW_NS + BigInt(1), NOW_NS)).toBe(true);
        expect(isRacy(NOW_NS - WINDOW_NS, NOW_NS)).toBe(true);
        expect(isRacy(NOW_NS, NOW_NS)).toBe(true);
        expect(isRacy(NOW_NS + WINDOW_NS, NOW_NS)).toBe(true);
        expect(isRacy(NOW_NS - WINDOW_NS - BigInt(1), NOW_NS)).toBe(false);
    });
    it('is never racy against an absent reference time (rows never written)', () => {
        expect(isRacy(NOW_NS, undefined)).toBe(false);
    });
});

describe('reusedHash (the prefilter)', () => {
    const prior = set([row('a.ts', 10, OLD_MTIME, 'h-a', { fileId: 5n }), row('n.ts', 10, OLD_MTIME, 'h-n')], NOW_NS);

    it('reuses the stored hash on a full stat match', () => {
        expect(reusedHash(stat(10, OLD_MTIME, 5n), prior.rows.get('a.ts'), prior, undefined)).toBe('h-a');
        expect(reusedHash(stat(10, OLD_MTIME), prior.rows.get('n.ts'), prior, undefined)).toBe('h-n');
    });
    it('needs size, full-precision mtimeNs and fileId to match', () => {
        const a = prior.rows.get('a.ts');
        expect(reusedHash(stat(11, OLD_MTIME, 5n), a, prior, undefined)).toBeUndefined(); // size moved
        expect(reusedHash(stat(10, OLD_MTIME + BigInt(1), 5n), a, prior, undefined)).toBeUndefined(); // ns mtime moved
        expect(reusedHash(stat(10, OLD_MTIME, 6n), a, prior, undefined)).toBeUndefined(); // fileId moved
        expect(reusedHash(stat(10, OLD_MTIME), a, prior, undefined)).toBeUndefined(); // fileId vanished
        expect(reusedHash(stat(10, OLD_MTIME, 5n), prior.rows.get('n.ts'), prior, undefined)).toBeUndefined(); // fileId appeared
    });
    it('has nothing to reuse for a new file', () => {
        expect(reusedHash(stat(10, OLD_MTIME), undefined, prior, undefined)).toBeUndefined();
    });
    it('re-hashes a racy file whatever its stat says', () => {
        const racy = row('r.ts', 10, NOW_NS, 'h-r');
        expect(reusedHash(stat(10, NOW_NS), racy, set([racy], NOW_NS), undefined)).toBeUndefined();
    });
    it('reuses a row whose set was never written (nothing is racy against no reference time)', () => {
        const fresh = row('r.ts', 10, NOW_NS, 'h-r');
        expect(reusedHash(stat(10, NOW_NS), fresh, set([fresh]), undefined)).toBe('h-r');
    });
    it('reuses a row on a clean git blob hint even when the stat moved (a checkout)', () => {
        const hinted = row('g.ts', 10, OLD_MTIME, 'h-g', { gitBlob: 'blob-1' });
        const moved = stat(10, OLD_MTIME + BigInt(5_000_000_000));
        expect(reusedHash(moved, hinted, set([hinted], NOW_NS), 'blob-1', NOW_NS)).toBe('h-g');
        expect(reusedHash(moved, hinted, set([hinted], NOW_NS), 'blob-2', NOW_NS)).toBeUndefined(); // blob changed
        expect(reusedHash(stat(11, moved.mtimeNs), hinted, set([hinted], NOW_NS), 'blob-1', NOW_NS)).toBeUndefined(); // size changed
        expect(reusedHash(moved, hinted, set([hinted], NOW_NS), 'blob-1', undefined)).toBeUndefined(); // no hint timestamp
        expect(reusedHash(stat(10, NOW_NS + BigInt(1)), hinted, set([hinted], NOW_NS), 'blob-1', NOW_NS)).toBeUndefined(); // modified after git looked
    });
    it('regression: a GitHints hint is used only while the file mtime is strictly before the ns time git looked', () => {
        const hints: GitHints = { blobs: new Map([['g.ts', 'blob-1']]), takenAtNs: NOW_NS };
        const hinted = row('g.ts', 10, OLD_MTIME, 'h-g', { fileId: 3n, gitBlob: 'blob-1' });
        const hint = hints.blobs.get('g.ts');
        // mtime and fileId both moved (a checkout), size unchanged: reused one ns before git looked, not at it.
        expect(reusedHash(stat(10, hints.takenAtNs - BigInt(1), 9n), hinted, set([hinted], NOW_NS), hint, hints.takenAtNs)).toBe('h-g');
        expect(reusedHash(stat(10, hints.takenAtNs, 9n), hinted, set([hinted], NOW_NS), hint, hints.takenAtNs)).toBeUndefined();
    });
});

describe('sameRow (an unchanged file writes nothing)', () => {
    it('compares size, mtimeNs, fileId, hash and gitBlob', () => {
        const base = row('a.ts', 10, OLD_MTIME, 'h', { fileId: 1n, gitBlob: 'b' });
        expect(sameRow(base, { ...base })).toBe(true);
        expect(sameRow(undefined, base)).toBe(false);
        expect(sameRow(base, { ...base, size: 11 })).toBe(false);
        expect(sameRow(base, { ...base, mtimeNs: OLD_MTIME + BigInt(1) })).toBe(false);
        expect(sameRow(base, { ...base, fileId: 2n })).toBe(false);
        expect(sameRow(base, { ...base, hash: 'h2' })).toBe(false);
        expect(sameRow(base, { ...base, gitBlob: 'b2' })).toBe(false);
    });
});

describe('StatSetMemo', () => {
    it('regression: apply merges upserts and deletes onto the last rows and keeps the reference time unless the delta moves it', () => {
        const memo = new StatSetMemo();
        memo.set('r', set([row('a.ts', 1, OLD_MTIME, 'h-a'), row('b.ts', 2, OLD_MTIME, 'h-b')], NOW_NS));
        memo.apply('r', { rule: RULE, upserts: [row('c.ts', 3, OLD_MTIME, 'h-c')], deletes: ['a.ts'] });
        expect([...memo.get('r')!.rows.keys()].sort()).toEqual(['b.ts', 'c.ts']);
        expect(memo.get('r')!.writtenAtNs).toBe(NOW_NS);
        memo.apply('r', { rule: RULE, upserts: [], deletes: [], writtenAtNs: NOW_NS + BigInt(1) });
        expect(memo.get('r')!.writtenAtNs).toBe(NOW_NS + BigInt(1));
        expect(memo.sizeOf('r', 'c.ts')).toBe(3);
        expect(memo.sizeOf('r', 'a.ts')).toBeUndefined();
    });
    it('regression: a replace delta drops every earlier row, and drop forgets the root', () => {
        const memo = new StatSetMemo();
        memo.apply('r', { rule: RULE, upserts: [row('a.ts', 1, OLD_MTIME, 'h-a')], deletes: [] });
        memo.apply('r', { rule: RULE, upserts: [row('b.ts', 2, OLD_MTIME, 'h-b')], deletes: [], replace: true });
        expect([...memo.get('r')!.rows.keys()]).toEqual(['b.ts']);
        memo.drop('r');
        expect(memo.get('r')).toBeUndefined();
    });
});

describe('memoryHashCache', () => {
    it('regression: loads empty for an unknown root and for a root recorded under another rule (stale)', async () => {
        const cache = memoryHashCache();
        expect(await cache.load('r', RULE)).toBe(EMPTY_STAT_SET);
        await cache.save('r', { rule: RULE, upserts: [row('a.ts', 1, OLD_MTIME, 'h-a')], deletes: [] });
        expect((await cache.load('r', RULE)).rows.size).toBe(1);
        const stale = await cache.load('r', 'another-rule');
        expect(stale.rows.size).toBe(0);
        expect(stale.stale).toBe(true);
    });
    it('regression: drop forgets the root (rows and rule)', async () => {
        const cache = memoryHashCache();
        await cache.save('r', { rule: RULE, upserts: [row('a.ts', 1, OLD_MTIME, 'h-a')], deletes: [] });
        await cache.drop?.('r');
        expect(await cache.load('r', RULE)).toBe(EMPTY_STAT_SET);
        expect(cache.sizeOf?.('r', 'a.ts')).toBeUndefined();
    });
});

describe('deferredHashCache', () => {
    it('regression: resolves the port on first use, once, and never at build time', async () => {
        const resolve = jest.fn(() => memoryHashCache());
        const cache = deferredHashCache(resolve);
        expect(resolve).not.toHaveBeenCalled();
        await cache.save('r', { rule: RULE, upserts: [row('a.ts', 1, OLD_MTIME, 'h-a')], deletes: [] });
        await cache.load('r', RULE);
        expect(resolve).toHaveBeenCalledTimes(1);
    });
    it('regression: a store that throws while opening rejects the load, and sizeOf answers undefined', async () => {
        const cache = deferredHashCache(() => { throw new Error('locked'); });
        await expect(cache.load('r', RULE)).rejects.toThrow('locked');
        expect(cache.sizeOf?.('r', 'a.ts')).toBeUndefined();
    });
    it('regression: drop and sizeOf are no-ops on a port that has neither', async () => {
        const bare: HashCache = { load: async () => EMPTY_STAT_SET, save: async () => undefined };
        const cache = deferredHashCache(() => bare);
        await expect(cache.drop?.('r')).resolves.toBeUndefined();
        expect(cache.sizeOf?.('r', 'a.ts')).toBeUndefined();
    });
});
