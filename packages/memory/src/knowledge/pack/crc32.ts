/**
 * knowledge/pack/crc32.ts — CRC-32 (IEEE 802.3) of a byte range; the trailer of a BFKP frame (PRD 41 A11).
 * `crc32(b, crc32(a))` equals `crc32(a ++ b)`, so a frame's CRC covers its header and its body without a copy.
 */

const TABLE: Uint32Array = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        t[n] = c >>> 0;
    }
    return t;
})();

/** The CRC-32 of `data`, continuing from `previous` (the CRC of what came before; 0 to start). */
export function crc32(data: Uint8Array, previous = 0): number {
    let c = (previous ^ 0xffffffff) >>> 0;
    for (let i = 0; i < data.length; i++) c = TABLE[(c ^ data[i]!) & 0xff]! ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
}
