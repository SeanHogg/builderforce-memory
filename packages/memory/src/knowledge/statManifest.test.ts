/**
 * src/knowledge/statManifest.test.ts — the stat prefilter (PRD 41 §B2 steps 1–2): a size+mtimeNs+fileId
 * manifest lets a warm plan reuse a stored hash without reading the file, and git's racy-file rule
 * re-hashes a file whose mtime is within 2 s of the manifest write or of now.
 */

import { jest } from '@jest/globals';
import {
    createStatPrefilter, isRacy, loadStatManifest, prefilterStats, RACY_WINDOW_MS, saveStatManifest, statMatches,
    type StatManifestEntry, type StatRecord,
} from './statManifest.js';
import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';

const stat = (size: number, mtimeNs: bigint, fileId: bigint | null = null): StatRecord => ({ size, mtimeNs, fileId });
const entry = (size: number, mtimeNs: bigint, hash: string, recordedAtMs: number, fileId: bigint | null = null): StatManifestEntry =>
    ({ size, mtimeNs, fileId, hash, recordedAtMs });

const NOW = 1_000_000;
// An mtime far from both `recordedAtMs` and NOW so the racy rule does not fire unless a test moves it.
const OLD_MTIME = BigInt((NOW - 60_000) * 1_000_000);

describe('statMatches', () => {
    it('requires size, full-precision mtimeNs and fileId', () => {
        expect(statMatches(stat(10, OLD_MTIME, 5n), stat(10, OLD_MTIME, 5n))).toBe(true);
        expect(statMatches(stat(10, OLD_MTIME, 5n), stat(11, OLD_MTIME, 5n))).toBe(false); // size moved
        expect(statMatches(stat(10, OLD_MTIME, 5n), stat(10, OLD_MTIME + 1n, 5n))).toBe(false); // ns mtime moved
        expect(statMatches(stat(10, OLD_MTIME, 5n), stat(10, OLD_MTIME, 6n))).toBe(false); // fileId moved
        expect(statMatches(stat(10, OLD_MTIME, null), stat(10, OLD_MTIME, 6n))).toBe(false); // null matches only null
        expect(statMatches(stat(10, OLD_MTIME, null), stat(10, OLD_MTIME, null))).toBe(true);
    });
});

describe('isRacy (git racy-file rule, §B2 step 2)', () => {
    it('is racy within 2 s of the manifest write time', () => {
        expect(isRacy(stat(1, BigInt((NOW - RACY_WINDOW_MS + 1) * 1_000_000)), NOW, NOW)).toBe(true);
        expect(isRacy(stat(1, BigInt((NOW + RACY_WINDOW_MS - 1) * 1_000_000)), NOW, NOW)).toBe(true);
        expect(isRacy(stat(1, BigInt((NOW - RACY_WINDOW_MS - 1) * 1_000_000)), NOW, NOW)).toBe(false);
    });
    it('is racy within 2 s of now, whatever the manifest says', () => {
        expect(isRacy(stat(1, BigInt(NOW * 1_000_000)), NOW - 60_000, NOW)).toBe(true);
        expect(isRacy(stat(1, BigInt((NOW - 10_000) * 1_000_000)), NOW - 60_000, NOW)).toBe(false);
    });
});

describe('prefilterStats', () => {
    const prior = new Map<string, StatManifestEntry>([
        ['a.ts', entry(10, OLD_MTIME, 'h-a', NOW)],
        ['b.ts', entry(20, OLD_MTIME, 'h-b', NOW)],
        ['c.ts', entry(30, OLD_MTIME, 'h-c', NOW)],
    ]);

    it('reuses the stored hash on a stat match and hashes everything else, both sorted', () => {
        const decision = prefilterStats(
            [
                { path: 'c.ts', stat: stat(30, OLD_MTIME) },
                { path: 'a.ts', stat: stat(10, OLD_MTIME) },
                { path: 'b.ts', stat: stat(21, OLD_MTIME) }, // size moved → hash
                { path: 'd.ts', stat: stat(40, OLD_MTIME) }, // new → hash
            ],
            prior, NOW,
        );
        expect(decision.reused).toEqual(['a.ts', 'c.ts']);
        expect(decision.hash).toEqual(['b.ts', 'd.ts']);
    });

    it('re-hashes a racy file whatever its stat says', () => {
        const racyMtime = BigInt(NOW * 1_000_000); // within 2 s of now
        const decision = prefilterStats([{ path: 'a.ts', stat: stat(10, racyMtime) }], prior, NOW);
        expect(decision.reused).toEqual([]);
        expect(decision.hash).toEqual(['a.ts']);
    });

    it('is deterministic: input order never changes the output', () => {
        const listed = [
            { path: 'a.ts', stat: stat(10, OLD_MTIME) },
            { path: 'b.ts', stat: stat(20, OLD_MTIME) },
            { path: 'c.ts', stat: stat(30, OLD_MTIME) },
        ];
        expect(prefilterStats([...listed].reverse(), prior, NOW)).toEqual(prefilterStats(listed, prior, NOW));
    });
});

describe('createStatPrefilter', () => {
    it('returns the stored hash without reading, and hashes a moved file through hashPath', async () => {
        const prior = new Map<string, StatManifestEntry>([['a.ts', entry(10, OLD_MTIME, 'h-a', NOW)]]);
        const hashPath = jest.fn(async (_: string) => 'h-fresh');
        const prefilter = createStatPrefilter(prior, hashPath, () => NOW);

        await expect(prefilter({ path: 'a.ts', stat: stat(10, OLD_MTIME) })).resolves.toEqual({ path: 'a.ts', hash: 'h-a' });
        expect(hashPath).not.toHaveBeenCalled();

        await expect(prefilter({ path: 'a.ts', stat: stat(11, OLD_MTIME) })).resolves.toEqual({ path: 'a.ts', hash: 'h-fresh' });
        expect(hashPath).toHaveBeenCalledTimes(1);
        expect(hashPath).toHaveBeenCalledWith('a.ts');
    });
});

describe('stat manifest persistence', () => {
    it('round-trips BigInt stat tuples through the ingest manifest and reloads them', async () => {
        const manifest = new InMemoryIngestManifest();
        const fileId = 12345678901234567890n; // exceeds a double: must survive as a BigInt
        const mtimeNs = 1700000000123456789n;
        const files = [
            { path: 'src/x.ts', hash: 'h-x', stat: stat(100, mtimeNs, fileId) },
            { path: 'src/y.ts', hash: 'h-y', stat: stat(200, OLD_MTIME, null) },
        ];
        await saveStatManifest(manifest, 'root', files, NOW);
        const loaded = await loadStatManifest(manifest, 'root');
        expect(loaded.size).toBe(2);
        expect(loaded.get('src/x.ts')).toEqual({ size: 100, mtimeNs, fileId, hash: 'h-x', recordedAtMs: NOW });
        expect(loaded.get('src/y.ts')).toEqual({ size: 200, mtimeNs: OLD_MTIME, fileId: null, hash: 'h-y', recordedAtMs: NOW });
        // A warm prefilter over the reloaded manifest reuses the stored hash.
        expect(prefilterStats([{ path: 'src/x.ts', stat: stat(100, mtimeNs, fileId) }], loaded, NOW).reused).toEqual(['src/x.ts']);
    });

    it('returns empty on a missing manifest (the first run hashes everything) and on a corrupt one', async () => {
        const manifest = new InMemoryIngestManifest();
        expect((await loadStatManifest(manifest, 'root')).size).toBe(0);
        await manifest.set('stats:root', [{ id: 'entries', hash: '{not json' }]);
        expect((await loadStatManifest(manifest, 'root')).size).toBe(0);
    });
});
