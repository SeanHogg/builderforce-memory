/**
 * tests/evermind_module.test.ts — the engine's Evermind module: the shared learning
 * pipeline every learner runs (adapt → delta wire → FedAvg merge → eval).
 *
 * Ported from the Builderforce.ai gateway, where this logic used to live, plus the
 * equivalences the gateway's migration onto these exports depends on.
 */
import {
    adaptAndDiff,
    EVERMIND_ADAPT_WINDOW_TOKENS,
    parseDeltaLearnPayload,
    buildDeltaLearnPayload,
    decodeDeltaPayload,
    deltaUnusableReason,
    MAX_DELTA_B64_CHARS,
    DELTA_LABEL_MAX_CHARS,
    mergeCheckpointDiffs,
    meanEvalLoss,
    sequenceLoss,
    EvermindModelPackage,
    type EvalExample,
} from '../src/evermind/index';
import { EvermindLM, EvermindLMTrainer } from '../src/lm/evermind_lm';
import { BPETokenizer } from '../src/tokenizer/bpe';
import { appendCrcTrailer, verifyCrcTrailer } from '../src/utils/crc32';
import { diffCheckpoints, serializeRowDelta } from '../src/utils/delta';
import { bytesToBase64, base64ToBytes } from '../src/utils/base64';
import { tokenWindows } from '../src/bench/adaptation';

/** Build a CRC-trailed checkpoint from raw f32 values (mirrors exportWeights). */
function ckpt(values: number[]): ArrayBuffer {
    return appendCrcTrailer(Float32Array.from(values).buffer);
}

/** Read a checkpoint back into a plain number[] (CRC stripped). */
function readback(buf: ArrayBuffer): number[] {
    return Array.from(new Float32Array(verifyCrcTrailer(buf).body));
}

describe('base64 codec', () => {
    it('round-trips arbitrary bytes, including buffers larger than one chunk', () => {
        const bytes = new Uint8Array(70_000);
        for (let i = 0; i < bytes.length; i++) bytes[i] = (i * 31) & 0xff;
        const back = new Uint8Array(base64ToBytes(bytesToBase64(bytes)));
        expect(back.length).toBe(bytes.length);
        expect(Array.from(back.subarray(0, 64))).toEqual(Array.from(bytes.subarray(0, 64)));
        expect(back[69_999]).toBe(bytes[69_999]);
    });
});

describe('mergeCheckpointDiffs (FedAvg over contributors)', () => {
    const base = ckpt([0, 0, 0, 0, 0, 0]);

    it('averages overlapping elements and keeps single-toucher elements', () => {
        const diffA = diffCheckpoints(base, ckpt([1, 1, 0, 0, 0, 0]));
        const diffB = diffCheckpoints(base, ckpt([0, 3, 3, 0, 0, 0]));
        const { checkpoint, mergedRows, contributors } = mergeCheckpointDiffs(base, [diffA, diffB]);
        const out = readback(checkpoint);
        expect(contributors).toBe(2);
        expect(mergedRows).toBe(3);
        expect(out[0]).toBeCloseTo(1, 6);
        expect(out[1]).toBeCloseTo(2, 6);
        expect(out[2]).toBeCloseTo(3, 6);
        expect(out[3]).toBeCloseTo(0, 6);
    });

    it('honors per-contributor sample weights', () => {
        const diffA = diffCheckpoints(base, ckpt([1, 1, 0, 0, 0, 0]));
        const diffB = diffCheckpoints(base, ckpt([0, 3, 3, 0, 0, 0]));
        const out = readback(mergeCheckpointDiffs(base, [diffA, diffB], [1, 3]).checkpoint);
        expect(out[0]).toBeCloseTo(1, 6);
        expect(out[1]).toBeCloseTo((1 * 1 + 3 * 3) / 4, 6);
        expect(out[2]).toBeCloseTo(3, 6);
    });

    it('returns the base unchanged for an empty contributor set', () => {
        const res = mergeCheckpointDiffs(base, []);
        expect(res.contributors).toBe(0);
        expect(res.mergedRows).toBe(0);
        expect(res.deltaNorm).toBe(0);
        expect(readback(res.checkpoint)).toEqual([0, 0, 0, 0, 0, 0]);
    });

    it('ignores zero/negative-weight contributors', () => {
        const diffA = diffCheckpoints(base, ckpt([2, 0, 0, 0, 0, 0]));
        const diffB = diffCheckpoints(base, ckpt([0, 9, 0, 0, 0, 0]));
        const out = readback(mergeCheckpointDiffs(base, [diffA, diffB], [1, 0]).checkpoint);
        expect(out[0]).toBeCloseTo(2, 6);
        expect(out[1]).toBeCloseTo(0, 6);
    });

    it('reports deltaNorm = L2 of the actual weight movement base→merged', () => {
        const diffA = diffCheckpoints(base, ckpt([1, 1, 0, 0, 0, 0]));
        const diffB = diffCheckpoints(base, ckpt([0, 3, 3, 0, 0, 0]));
        expect(mergeCheckpointDiffs(base, [diffA, diffB]).deltaNorm).toBeCloseTo(Math.sqrt(1 + 4 + 9), 5);
    });

    it('rejects a weights array of the wrong length and a non-element-granular delta', () => {
        const d = diffCheckpoints(base, ckpt([1, 0, 0, 0, 0, 0]));
        expect(() => mergeCheckpointDiffs(base, [d], [1, 2])).toThrow(/weights length/);
        const coarse = serializeRowDelta({ rowSize: 2, rows: [0], data: Float32Array.from([1, 1]) });
        expect(() => mergeCheckpointDiffs(base, [coarse])).toThrow(/rowSize 1/);
    });
});

describe('delta wire contract', () => {
    it('accepts the producer payload shape', () => {
        const r = parseDeltaLearnPayload({ diff: 'AAAA', baseVersion: 7, weight: 0.7, label: '  ticket 12  ' });
        expect(r).toEqual({ ok: true, payload: { diff: 'AAAA', baseVersion: 7, weight: 0.7, label: 'ticket 12' } });
    });

    it('requires a diff and an INTEGER baseVersion', () => {
        expect(parseDeltaLearnPayload({ diff: 'AAAA' })).toMatchObject({ ok: false, reason: 'invalid' });
        expect(parseDeltaLearnPayload({ diff: 'AAAA', baseVersion: 7.5 })).toMatchObject({ ok: false, reason: 'invalid' });
        expect(parseDeltaLearnPayload({ baseVersion: 7 })).toMatchObject({ ok: false, reason: 'invalid' });
        expect(parseDeltaLearnPayload(null)).toMatchObject({ ok: false, reason: 'invalid' });
    });

    it('refuses non-base64, so a bad body can never throw inside a merge', () => {
        expect(parseDeltaLearnPayload({ diff: 'not base64!', baseVersion: 1 })).toMatchObject({ ok: false, reason: 'invalid' });
        expect(parseDeltaLearnPayload({ diff: 'AAA', baseVersion: 1 })).toMatchObject({ ok: false, reason: 'invalid' });
    });

    it('bounds the payload', () => {
        const huge = 'A'.repeat(MAX_DELTA_B64_CHARS + 4);
        expect(parseDeltaLearnPayload({ diff: huge, baseVersion: 1 })).toMatchObject({ ok: false, reason: 'too-large' });
    });

    it('drops a non-positive weight rather than letting it void the contribution', () => {
        const r = parseDeltaLearnPayload({ diff: 'AAAA', baseVersion: 1, weight: -2 });
        expect(r.ok && r.payload.weight).toBeUndefined();
    });

    it('builds a payload the parser accepts, and decodes back to the same diff', () => {
        const base = ckpt([0, 0, 0, 0]);
        const diff = diffCheckpoints(base, ckpt([0, 5, 0, 0]));
        const payload = buildDeltaLearnPayload(diff, 3, 0.6, 'x'.repeat(DELTA_LABEL_MAX_CHARS + 50));
        expect(payload.label!.length).toBe(DELTA_LABEL_MAX_CHARS);
        const parsed = parseDeltaLearnPayload(payload);
        expect(parsed.ok).toBe(true);
        const bytes = new Uint8Array(decodeDeltaPayload(payload));
        expect(Array.from(bytes)).toEqual(Array.from(new Uint8Array(diff)));
        expect(deltaUnusableReason(decodeDeltaPayload(payload), base)).toBeNull();
    });

    it('names what is wrong with a delta that is not a diff of this base', () => {
        const base = ckpt([0, 0, 0, 0]);
        expect(deltaUnusableReason(serializeRowDelta({ rowSize: 2, rows: [0], data: Float32Array.from([1, 1]) }), base))
            .toMatch(/rowSize 1/);
        expect(deltaUnusableReason(serializeRowDelta({ rowSize: 1, rows: [4], data: Float32Array.from([1]) }), base))
            .toMatch(/outside the base/);
        expect(deltaUnusableReason(serializeRowDelta({ rowSize: 1, rows: [1], data: Float32Array.from([Number.NaN]) }), base))
            .toMatch(/non-finite/);
        expect(deltaUnusableReason(new ArrayBuffer(3), base)).toMatch(/not a serialized RowDelta/);
    });
});

describe('adaptAndDiff + eval', () => {
    const corpus = 'alpha beta gamma delta epsilon zeta eta theta the agent fixes the bug and ships';
    const tok = new BPETokenizer();
    tok.train(corpus);
    const base = new EvermindLM({ vocabSize: tok.vocabSize, seed: 3 });
    const pkg = EvermindModelPackage.fromLM(base, { name: 'test', version: '1', card: { description: 'adapt fixture' } });
    const examples: EvalExample[] = [
        { prompt: 'greek letters', text: 'alpha beta gamma delta epsilon' },
        { text: 'the agent fixes the bug and ships' },
    ];

    it('returns null when the text yields no trainable window', () => {
        expect(adaptAndDiff(pkg, tok, '')).toBeNull();
    });

    it('produces a mergeable diff and a finite loss, leaving the package untouched', () => {
        const before = new Uint8Array(pkg.checkpoint.slice(0));
        const r = adaptAndDiff(pkg, tok, corpus)!;
        expect(r.sequences).toBe(tokenWindows(tok.encode(corpus), EVERMIND_ADAPT_WINDOW_TOKENS).length);
        expect(Number.isFinite(r.loss)).toBe(true);
        expect(deltaUnusableReason(r.diff, pkg.checkpoint)).toBeNull();
        expect(new Uint8Array(pkg.checkpoint)).toEqual(before);
        expect(mergeCheckpointDiffs(pkg.checkpoint, [r.diff]).mergedRows).toBeGreaterThan(0);
    });

    it('sequenceLoss equals the loss the trainer computes (forward-only, no grads)', () => {
        const ids = tok.encode(corpus).slice(0, 24);
        const lm = new EvermindLM({ vocabSize: tok.vocabSize, seed: 3 });
        const fwd = sequenceLoss(lm, ids)!;
        const trained = lm.lossAndBackward(ids);
        expect(fwd).toBeCloseTo(trained, 5);
        expect(sequenceLoss(lm, [1])).toBeNull();
    });

    it('meanEvalLoss: null for an empty or unscorable set, finite otherwise', () => {
        expect(meanEvalLoss(base, tok, [])).toBeNull();
        expect(meanEvalLoss(base, tok, [{ text: 'a' }])).toBeNull();
        const loss = meanEvalLoss(base, tok, examples)!;
        expect(Number.isFinite(loss)).toBe(true);
        expect(loss).toBeGreaterThan(0);
    });

    it('a model adapted on the eval text scores it as well or better', () => {
        const baseLoss = meanEvalLoss(base, tok, examples)!;
        const adapted = new EvermindLM({ vocabSize: tok.vocabSize, seed: 3 });
        adapted.loadWeights(base.exportWeights());
        const seqs = examples.flatMap((ex) => tokenWindows(tok.encode((ex.prompt ? `${ex.prompt}\n` : '') + ex.text), 32));
        new EvermindLMTrainer(adapted, { epochs: 12 }).fit(seqs);
        expect(meanEvalLoss(adapted, tok, examples)!).toBeLessThanOrEqual(baseLoss + 1e-6);
    }, 60_000);
});

describe('EvermindLM.embed equals the mean-pooled, L2-normalised final hidden state', () => {
    it('matches forward().cache.finalX pooled the way hosts used to compute it', () => {
        const lm = new EvermindLM({ vocabSize: 48, seed: 7 });
        const ids = [1, 2, 3, 4, 5, 6, 7];
        const { cache } = lm.forward(ids);
        const d = lm.config.dModel;
        const pooled = new Float32Array(d);
        for (const row of cache.finalX) for (let c = 0; c < d; c++) pooled[c] = pooled[c]! + row[c]!;
        for (let c = 0; c < d; c++) pooled[c] = pooled[c]! / cache.finalX.length;
        let norm = 0;
        for (let c = 0; c < d; c++) norm += pooled[c]! * pooled[c]!;
        norm = Math.sqrt(norm) || 1;
        const embedded = lm.embed(ids);
        for (let c = 0; c < d; c++) expect(embedded[c]).toBeCloseTo(pooled[c]! / norm, 5);
    });
});
