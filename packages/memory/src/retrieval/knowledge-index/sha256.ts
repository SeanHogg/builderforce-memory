/**
 * sha256.ts — a synchronous, pure-TypeScript SHA-256 (FIPS 180-4).
 *
 * The knowledge index names its state with a digest that must be identical on
 * every host. WebCrypto is async and `node:crypto` is Node-only, so this is the
 * one portable, synchronous hasher in the package: it runs in a Worker, a browser
 * and Node, and it is incremental so a large state is hashed without first
 * concatenating it.
 */

const K = new Uint32Array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const encoder = new TextEncoder();

/** Incremental SHA-256: `update(...)` any number of times, then `digest()` once. */
export class Sha256 {
    private readonly _h = new Uint32Array([
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]);
    private readonly _block = new Uint8Array(64);
    private readonly _w = new Uint32Array(64);
    private _blockLen = 0;
    private _bytes = 0;
    private _done = false;

    /** Feeds bytes, or a string as UTF-8. */
    update(data: Uint8Array | string): this {
        if (this._done) throw new Error('Sha256: update after digest');
        const bytes = typeof data === 'string' ? encoder.encode(data) : data;
        this._bytes += bytes.length;
        let i = 0;
        while (i < bytes.length) {
            const take = Math.min(64 - this._blockLen, bytes.length - i);
            this._block.set(bytes.subarray(i, i + take), this._blockLen);
            this._blockLen += take;
            i += take;
            if (this._blockLen === 64) {
                this._compress();
                this._blockLen = 0;
            }
        }
        return this;
    }

    /** The 32-byte digest. The hasher is finished afterwards. */
    digest(): Uint8Array {
        if (this._done) throw new Error('Sha256: digest called twice');
        const bitsHi = Math.floor(this._bytes / 0x20000000);
        const bitsLo = (this._bytes * 8) >>> 0;
        this.update(new Uint8Array([0x80]));
        while (this._blockLen !== 56) this.update(new Uint8Array(1));
        const len = new Uint8Array(8);
        const view = new DataView(len.buffer);
        view.setUint32(0, bitsHi);
        view.setUint32(4, bitsLo);
        this.update(len);
        this._done = true;
        const out = new Uint8Array(32);
        const outView = new DataView(out.buffer);
        for (let j = 0; j < 8; j++) outView.setUint32(j * 4, this._h[j]!);
        return out;
    }

    /** The digest as lowercase hex. */
    hex(): string {
        return toHex(this.digest());
    }

    private _compress(): void {
        const w = this._w;
        const b = this._block;
        for (let t = 0; t < 16; t++) {
            w[t] = ((b[t * 4]! << 24) | (b[t * 4 + 1]! << 16) | (b[t * 4 + 2]! << 8) | b[t * 4 + 3]!) >>> 0;
        }
        for (let t = 16; t < 64; t++) {
            const x = w[t - 15]!;
            const y = w[t - 2]!;
            const s0 = ((x >>> 7) | (x << 25)) ^ ((x >>> 18) | (x << 14)) ^ (x >>> 3);
            const s1 = ((y >>> 17) | (y << 15)) ^ ((y >>> 19) | (y << 13)) ^ (y >>> 10);
            w[t] = (w[t - 16]! + s0 + w[t - 7]! + s1) >>> 0;
        }
        const h = this._h;
        let a = h[0]!, bb = h[1]!, c = h[2]!, d = h[3]!, e = h[4]!, f = h[5]!, g = h[6]!, hh = h[7]!;
        for (let t = 0; t < 64; t++) {
            const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
            const ch = (e & f) ^ (~e & g);
            const t1 = (hh + S1 + ch + K[t]! + w[t]!) >>> 0;
            const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
            const maj = (a & bb) ^ (a & c) ^ (bb & c);
            const t2 = (S0 + maj) >>> 0;
            hh = g; g = f; f = e; e = (d + t1) >>> 0;
            d = c; c = bb; bb = a; a = (t1 + t2) >>> 0;
        }
        h[0] = (h[0]! + a) >>> 0; h[1] = (h[1]! + bb) >>> 0; h[2] = (h[2]! + c) >>> 0; h[3] = (h[3]! + d) >>> 0;
        h[4] = (h[4]! + e) >>> 0; h[5] = (h[5]! + f) >>> 0; h[6] = (h[6]! + g) >>> 0; h[7] = (h[7]! + hh) >>> 0;
    }
}

function toHex(bytes: Uint8Array): string {
    let s = '';
    for (let i = 0; i < bytes.length; i++) s += bytes[i]!.toString(16).padStart(2, '0');
    return s;
}

/** SHA-256 of a string (UTF-8) or bytes. */
export function sha256(data: Uint8Array | string): Uint8Array {
    return new Sha256().update(data).digest();
}

/** SHA-256 hex of a string (UTF-8) or bytes. */
export function sha256Hex(data: Uint8Array | string): string {
    return new Sha256().update(data).hex();
}
