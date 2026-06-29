/**
 * utils/delta.ts — sparse, row-granular weight deltas for online checkpoints.
 *
 * Online adaptation (WSLA) restricts the trainable set to a few
 * selective-projection ROWS, yet each `adapt()` previously rewrote the WHOLE
 * model to disk. That is wasteful in both size and I/O: 99% of the bytes are
 * identical to the base checkpoint. A delta persists only the rows that actually
 * changed, as a diff against a base, so an online update writes kilobytes
 * instead of megabytes.
 *
 * Format (little-endian), CRC-trailed via {@link appendCrcTrailer}:
 *   magic 'EVD0' u32 | version u32 | rowSize u32 | nRows u32 |
 *   rowIndex[nRows] u32 | float32 row data[nRows * rowSize]
 *
 * The base + delta reconstruct the exact current weights; a row counts as
 * "changed" if any element differs beyond an optional epsilon.
 */

import { appendCrcTrailer, verifyCrcTrailer } from './crc32.js';

const DELTA_MAGIC = 0x30445645; // 'EVD0'
const DELTA_VERSION = 1;

export interface RowDelta {
    /** Width of one row (elements). `base.length` must be a multiple of it. */
    rowSize: number;
    /** Indices (row-major) of the rows that changed. */
    rows: number[];
    /** Concatenated new values for the changed rows (rows.length * rowSize). */
    data: Float32Array;
}

/**
 * Compute the row-sparse delta of `current` vs `base` (same length + layout).
 * A row is included when any element differs by more than `eps` (default 0).
 */
export function computeRowDelta(
    base: Float32Array,
    current: Float32Array,
    rowSize: number,
    eps = 0,
): RowDelta {
    if (base.length !== current.length) {
        throw new Error(`computeRowDelta: length mismatch (base ${base.length}, current ${current.length})`);
    }
    if (rowSize <= 0 || base.length % rowSize !== 0) {
        throw new Error(`computeRowDelta: base length ${base.length} is not a multiple of rowSize ${rowSize}`);
    }
    const nRowsTotal = base.length / rowSize;
    const rows: number[] = [];
    for (let r = 0; r < nRowsTotal; r++) {
        const off = r * rowSize;
        let changed = false;
        for (let c = 0; c < rowSize; c++) {
            if (Math.abs(current[off + c]! - base[off + c]!) > eps) { changed = true; break; }
        }
        if (changed) rows.push(r);
    }
    const data = new Float32Array(rows.length * rowSize);
    rows.forEach((r, i) => data.set(current.subarray(r * rowSize, r * rowSize + rowSize), i * rowSize));
    return { rowSize, rows, data };
}

/** Apply a delta onto a copy of `base`, returning the reconstructed weights. */
export function applyRowDelta(base: Float32Array, delta: RowDelta): Float32Array {
    if (base.length % delta.rowSize !== 0) {
        throw new Error(`applyRowDelta: base length ${base.length} is not a multiple of rowSize ${delta.rowSize}`);
    }
    const out = Float32Array.from(base);
    delta.rows.forEach((r, i) => {
        const dst = r * delta.rowSize;
        if (dst + delta.rowSize > out.length) {
            throw new Error(`applyRowDelta: row ${r} out of range for base of ${out.length} elements`);
        }
        out.set(delta.data.subarray(i * delta.rowSize, i * delta.rowSize + delta.rowSize), dst);
    });
    return out;
}

/** Serialize a delta to a CRC-trailed binary. */
export function serializeRowDelta(delta: RowDelta): ArrayBuffer {
    const nRows = delta.rows.length;
    const headerEls = 4; // magic, version, rowSize, nRows
    const headerBytes = headerEls * 4;
    const rowIdxBytes = nRows * 4;
    const dataBytes = delta.data.length * 4;
    const buf = new ArrayBuffer(headerBytes + rowIdxBytes + dataBytes);
    const view = new DataView(buf);
    view.setUint32(0, DELTA_MAGIC, true);
    view.setUint32(4, DELTA_VERSION, true);
    view.setUint32(8, delta.rowSize, true);
    view.setUint32(12, nRows, true);
    let off = headerBytes;
    for (const r of delta.rows) { view.setUint32(off, r, true); off += 4; }
    new Float32Array(buf, off, delta.data.length).set(delta.data);
    return appendCrcTrailer(buf);
}

/** Parse a binary produced by {@link serializeRowDelta}; verifies CRC + magic. */
export function deserializeRowDelta(buffer: ArrayBuffer): RowDelta {
    const crc = verifyCrcTrailer(buffer);
    if (crc.hasTrailer && !crc.ok) {
        throw new Error('deserializeRowDelta: failed CRC integrity check (corrupt or truncated)');
    }
    const body = crc.body;
    const view = new DataView(body);
    if (view.getUint32(0, true) !== DELTA_MAGIC) {
        throw new Error('deserializeRowDelta: bad magic (not an EVD0 delta)');
    }
    const rowSize = view.getUint32(8, true);
    const nRows = view.getUint32(12, true);
    let off = 16;
    const rows: number[] = [];
    for (let i = 0; i < nRows; i++) { rows.push(view.getUint32(off, true)); off += 4; }
    const data = new Float32Array(body.slice(off, off + nRows * rowSize * 4));
    return { rowSize, rows, data };
}
