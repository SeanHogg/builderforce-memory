/**
 * utils/crc32.ts — checkpoint integrity (CRC-32) + a backward-compatible trailer.
 *
 * Checkpoints had no integrity check: a truncated or bit-rotted `.bin` would
 * either throw deep in the parser or, worse, load corrupt weights silently. This
 * adds a standard IEEE CRC-32 and a self-describing trailer that callers append
 * on export and verify on load.
 *
 * The trailer is appended AFTER the existing payload as `[magic u32][crc u32]`
 * (8 bytes, little-endian). Because every checkpoint reader consumes a
 * length-bounded payload (header declares element counts) and ignores trailing
 * bytes, an old reader still loads a trailer-bearing file — the trailer is
 * purely additive. The CRC covers the payload bytes only (everything before the
 * trailer).
 */

/** Trailer sentinel: 'EVCR' (Evermind CRc), little-endian uint32. */
export const CRC_TRAILER_MAGIC = 0x52435645; // bytes 45 56 43 52 = "EVCR"
const TRAILER_BYTES = 8;

/** Precomputed CRC-32 (IEEE 802.3, reflected) lookup table. */
const CRC_TABLE: Uint32Array = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        t[n] = c >>> 0;
    }
    return t;
})();

/** IEEE CRC-32 of a byte buffer, returned as an unsigned 32-bit number. */
export function crc32(bytes: Uint8Array): number {
    let crc = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) {
        crc = CRC_TABLE[(crc ^ bytes[i]!) & 0xff]! ^ (crc >>> 8);
    }
    return (crc ^ 0xffffffff) >>> 0;
}

/** Return a new ArrayBuffer = `payload` followed by the CRC trailer. */
export function appendCrcTrailer(payload: ArrayBuffer): ArrayBuffer {
    const body = new Uint8Array(payload);
    const out = new ArrayBuffer(body.length + TRAILER_BYTES);
    const u8 = new Uint8Array(out);
    u8.set(body, 0);
    const view = new DataView(out);
    view.setUint32(body.length, CRC_TRAILER_MAGIC, true);
    view.setUint32(body.length + 4, crc32(body), true);
    return out;
}

export interface CrcCheck {
    /** True when a recognised CRC trailer is present. */
    hasTrailer: boolean;
    /** True when no trailer is present, or the trailer's CRC matches the payload. */
    ok: boolean;
    /** The payload with any trailer stripped (safe to parse). */
    body: ArrayBuffer;
}

/**
 * Inspect a buffer for a CRC trailer. When present, verifies the CRC over the
 * preceding payload. When absent (legacy file), reports `ok: true` and returns
 * the buffer unchanged — so verification is opt-in by the writer, never a
 * hard requirement that would reject pre-CRC checkpoints.
 */
export function verifyCrcTrailer(buffer: ArrayBuffer): CrcCheck {
    if (buffer.byteLength < TRAILER_BYTES) {
        return { hasTrailer: false, ok: true, body: buffer };
    }
    const view = new DataView(buffer);
    const bodyLen = buffer.byteLength - TRAILER_BYTES;
    const magic = view.getUint32(bodyLen, true);
    if (magic !== CRC_TRAILER_MAGIC) {
        return { hasTrailer: false, ok: true, body: buffer };
    }
    const stored = view.getUint32(bodyLen + 4, true);
    const body = buffer.slice(0, bodyLen);
    const ok = crc32(new Uint8Array(body)) === stored;
    return { hasTrailer: true, ok, body };
}
