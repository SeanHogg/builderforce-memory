/**
 * src/contract/regression.scopeMeasured.test.ts — `ScopeRecord.measured` (PRD 41 §C9): what a review was measured with is
 * recorded WHEN THE REVIEW RUNS and carried by the BFKP pack in its own MEAS section, never re-derived when the pack is
 * built. Before, a host stated its versions at pack-build time, so a review queued offline across an upgrade was stored
 * with the newer versions and compared as "the same" against a review it was not measured like.
 *
 * Compatibility, both directions:
 *   - BACKWARD: a pack without MEAS (every pack written before this release) decodes with no `measured` — unknown;
 *   - FORWARD: MEAS is a new tag, so a reader of before this release skips it and decodes the same scope, unmeasured.
 */

import { readFrame, writeFrame, writeSection } from '../knowledge/pack/frame.js';
import { TAG } from '../knowledge/pack/sections/index.js';
import { ByteReader, ByteWriter } from '../knowledge/pack/varint.js';
import { decodePack, encodePack, PackFormatError, ZERO_DIGEST, type Pack } from '../knowledge/pack/index.js';
import type { MeasuredWith } from '../knowledge/index.js';
import { reviewDirectory } from '../knowledge/review.js';
import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { args, FILES, memSource, memStore, ROOT } from '../knowledge/testkit.test-util.js';

const HEAD = { rootId: ROOT, baseGeneration: 0, targetDigest: ZERO_DIGEST };
const V5: MeasuredWith = { extractorVersion: '5', toolVersions: { knip: '5.30.0', jscpd: '4.0.5' } };
const V6: MeasuredWith = { extractorVersion: '6', toolVersions: { knip: '5.31.0', jscpd: '4.0.5' } };
const scoped = (measured?: MeasuredWith): Pack => ({
    ...HEAD, scope: { rootId: ROOT, title: 'R', packs: ['code'], lastDigest: 'abcdef12', ...(measured ? { measured } : {}) },
});

function sectionsOf(bytes: Uint8Array): Map<number, Uint8Array> {
    const r = new ByteReader(readFrame(bytes).stored);
    const out = new Map<number, Uint8Array>();
    while (!r.done) { const tag = r.u8(); out.set(tag, r.bytes(r.varint())); }
    return out;
}
function frameOf(sections: Map<number, Uint8Array>): Uint8Array {
    const body = new ByteWriter();
    for (const [tag, b] of sections) writeSection(body, tag, b);
    const plain = body.finish();
    return writeFrame({ flags: 0, ...HEAD }, plain, plain);
}

describe('regression: a review is stored with the versions it was measured with (PRD 41 §C9)', () => {
    it('regression: the review records `measured` on its scope when it runs, and a pack carries THAT record — not the versions of the host that later syncs it', async () => {
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(memSource(FILES), store, manifest, { measured: V5 }));
        const [scope] = await store.scopes();
        expect(scope).toMatchObject({ rootId: ROOT, lastDigest: expect.any(String), measured: V5 });

        // The host upgraded while the review sat in its queue: the pack still says what the review was measured with.
        const back = await decodePack(await encodePack({ ...HEAD, scope: scope! }));
        expect(back.scope!.measured).toEqual(V5);

        // A later review that states nothing moves the digest: the old versions do not describe it (unknown, never V5).
        await reviewDirectory(args(memSource({ ...FILES, 'd.md': '# D\ndelta\n' }), store, manifest));
        const [again] = await store.scopes();
        expect(again!.lastDigest).not.toBe(scope!.lastDigest);
        expect(again).not.toHaveProperty('measured');

        // And one that states V6 records V6.
        await reviewDirectory(args(memSource(FILES), store, manifest, { measured: V6 }));
        expect((await store.scopes())[0]!.measured).toEqual(V6);
    });

    it('the store: a digest write replaces `measured` (even the same digest), a write without a digest keeps it', async () => {
        const store = memStore();
        const base = { rootId: ROOT, title: 'R', packs: ['code'] };
        const scope = async () => (await store.scopes())[0]!;
        await store.write({ scope: { ...base, lastDigest: 'd1', measured: V5 }, facts: [], relations: [] });
        await store.write({ scope: base, facts: [], relations: [] }); // mid-run / aborted: no digest
        expect((await scope()).measured).toEqual(V5);
        await store.write({ scope: { ...base, lastDigest: 'd1' }, facts: [], relations: [] }); // re-review, same digest, unmeasured
        expect(await scope()).not.toHaveProperty('measured');
        await store.write({ scope: { ...base, lastDigest: 'd2', measured: V6 }, facts: [], relations: [] });
        expect((await scope()).measured).toEqual(V6);
    });

    it('encode refuses a `measured` its own decoder would refuse (an unvalidated JSON snapshot)', async () => {
        const bad = [
            { extractorVersion: 5, toolVersions: {} },
            { extractorVersion: '5', toolVersions: { knip: 5 } },
            { extractorVersion: '5', toolVersions: JSON.parse('{"__proto__":"x"}') },
        ] as unknown as MeasuredWith[];
        for (const m of bad) await expect(encodePack(scoped(m))).rejects.toBeInstanceOf(PackFormatError);
    });

    it('encode → decode keeps `measured`, deflated or plain, and re-encodes to the same bytes whatever the tool order', async () => {
        const reordered: MeasuredWith = { extractorVersion: '5', toolVersions: { jscpd: '4.0.5', knip: '5.30.0' } };
        for (const deflate of [true, false]) {
            const bytes = await encodePack(scoped(V5), { deflate });
            const back = await decodePack(bytes);
            expect(back.scope!.measured).toEqual(V5);
            expect(Buffer.from(await encodePack(scoped(reordered), { deflate })).equals(Buffer.from(bytes))).toBe(true);
        }
        expect((await decodePack(await encodePack(scoped({ extractorVersion: '5', toolVersions: {} })))).scope!.measured).toEqual({ extractorVersion: '5', toolVersions: {} });
    });

    it('BACKWARD: a pack without MEAS decodes with no `measured` (unknown), and writes no MEAS section', async () => {
        const bytes = await encodePack(scoped(), { deflate: false });
        expect(sectionsOf(bytes).has(TAG.MEAS)).toBe(false);
        expect((await decodePack(bytes)).scope).not.toHaveProperty('measured');
    });

    it('FORWARD: SCOPE is unchanged and MEAS is a new tag, so an older reader (which skips it) decodes the same scope, unmeasured', async () => {
        const withMeas = sectionsOf(await encodePack(scoped(V5), { deflate: false }));
        withMeas.delete(TAG.MEAS); // what a reader that does not know the tag does (its strings stay in STR, unreferenced)
        expect(await decodePack(frameOf(withMeas))).toEqual(scoped());
    });

    it('a MEAS section without a SCOPE, a tool named twice, or trailing bytes is a malformed pack', async () => {
        const sections = sectionsOf(await encodePack(scoped(V5), { deflate: false }));
        const noScope = new Map(sections); noScope.delete(TAG.SCOPE);
        await expect(decodePack(frameOf(noScope))).rejects.toBeInstanceOf(PackFormatError);
        const trailing = new Map(sections); trailing.set(TAG.MEAS, new Uint8Array([...sections.get(TAG.MEAS)!, 0]));
        await expect(decodePack(frameOf(trailing))).rejects.toBeInstanceOf(PackFormatError);
        const meas = new ByteReader(sections.get(TAG.MEAS)!);
        const extractor = meas.varint(); meas.varint(); const tool = meas.varint(); const version = meas.varint();
        const twice = new ByteWriter(); twice.varint(extractor); twice.varint(2); for (let i = 0; i < 2; i++) { twice.varint(tool); twice.varint(version); }
        const dup = new Map(sections); dup.set(TAG.MEAS, twice.finish());
        await expect(decodePack(frameOf(dup))).rejects.toBeInstanceOf(PackFormatError);
    });
});
