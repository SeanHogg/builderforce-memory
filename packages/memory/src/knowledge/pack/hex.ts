/** knowledge/pack/hex.ts — the ONE hex helper of the codec (PRD 41 A11): 64-hex digests and hashes as raw 32 bytes and back. */

export const HEX64 = /^[0-9a-f]{64}$/;

/** Lowercase hex of `bytes`. */
export function toHex(bytes: Uint8Array): string {
    let s = '';
    for (const b of bytes) s += (b < 16 ? '0' : '') + b.toString(16);
    return s;
}

/** The 32 raw bytes of a 64-character lowercase hex string. */
export function fromHex64(hex: string): Uint8Array {
    const raw = new Uint8Array(32);
    for (let i = 0; i < 32; i++) raw[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
    return raw;
}
