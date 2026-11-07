/**
 * knowledge/pack/varint.ts — the byte-level primitives of BFKP v1 (PRD 41 A11): a growable byte
 * writer and a bounds-checked reader over unsigned LEB128 varints, raw bytes, UTF-8 strings, and
 * a compact number codec. Pure TypeScript — no Buffer, no node: imports.
 */

/** A malformed pack: the server answers 422 `bad_pack`. */
export class PackFormatError extends Error {
    constructor(message: string) { super(message); this.name = 'PackFormatError'; }
}

const ENCODER = new TextEncoder();
const DECODER = new TextDecoder('utf-8', { fatal: true });

export const utf8 = (s: string): Uint8Array => ENCODER.encode(s);

/** Largest integer a varint carries (2^53 - 1, JS's safe integer). */
export const MAX_VARINT = Number.MAX_SAFE_INTEGER;

export class ByteWriter {
    private buf = new Uint8Array(1024);
    private len = 0;

    private room(n: number): void {
        if (this.len + n <= this.buf.length) return;
        let cap = this.buf.length;
        while (cap < this.len + n) cap *= 2;
        const next = new Uint8Array(cap);
        next.set(this.buf.subarray(0, this.len));
        this.buf = next;
    }

    get length(): number { return this.len; }

    u8(v: number): void { this.room(1); this.buf[this.len++] = v & 0xff; }

    /** Unsigned LEB128. */
    varint(v: number): void {
        if (!Number.isSafeInteger(v) || v < 0) throw new PackFormatError(`not an unsigned varint: ${v}`);
        this.room(8);
        while (v >= 0x80) { this.buf[this.len++] = (v % 0x80) | 0x80; v = Math.floor(v / 0x80); }
        this.buf[this.len++] = v;
    }

    bytes(b: Uint8Array): void { this.room(b.length); this.buf.set(b, this.len); this.len += b.length; }

    /** A varint length, then the UTF-8 bytes. */
    str(s: string): void { const b = utf8(s); this.varint(b.length); this.bytes(b); }

    u32le(v: number): void { this.room(4); for (let i = 0; i < 4; i++) this.buf[this.len++] = (v >>> (8 * i)) & 0xff; }

    f64(v: number): void {
        this.room(8);
        new DataView(this.buf.buffer, this.buf.byteOffset + this.len, 8).setFloat64(0, v, true);
        this.len += 8;
    }

    /**
     * A finite number, as small as it can be and always exactly recoverable: a non-negative integer
     * (tag 0), a non-negative decimal of at most 6 places (tag 1, scaled), or a raw float64 (tag 2).
     */
    num(v: number): void {
        if (Number.isSafeInteger(v) && v >= 0 && v < 2 ** 50) { this.varint(v * 4); return; }
        if (v >= 0 && v < 2 ** 30) {
            const scaled = Math.round(v * 1e6);
            if (scaled / 1e6 === v) { this.varint(scaled * 4 + 1); return; }
        }
        this.varint(2);
        this.f64(v);
    }

    finish(): Uint8Array { return this.buf.slice(0, this.len); }
}

export class ByteReader {
    private pos = 0;
    constructor(private readonly buf: Uint8Array) {}

    get offset(): number { return this.pos; }
    get done(): boolean { return this.pos >= this.buf.length; }
    get remaining(): number { return this.buf.length - this.pos; }

    u8(): number {
        if (this.pos >= this.buf.length) throw new PackFormatError('unexpected end of data');
        return this.buf[this.pos++]!;
    }

    varint(): number {
        let result = 0, scale = 1;
        for (let i = 0; i < 8; i++) {
            const b = this.u8();
            result += (b & 0x7f) * scale;
            if ((b & 0x80) === 0) return result;
            scale *= 0x80;
        }
        throw new PackFormatError('varint too long');
    }

    bytes(n: number): Uint8Array {
        if (!Number.isSafeInteger(n) || n < 0 || this.pos + n > this.buf.length) throw new PackFormatError('unexpected end of data');
        const out = this.buf.subarray(this.pos, this.pos + n);
        this.pos += n;
        return out;
    }

    str(): string {
        try { return DECODER.decode(this.bytes(this.varint())); } catch (e) {
            if (e instanceof PackFormatError) throw e;
            throw new PackFormatError('invalid UTF-8');
        }
    }

    u32le(): number {
        const b = this.bytes(4);
        return (b[0]! | (b[1]! << 8) | (b[2]! << 16) | (b[3]! << 24)) >>> 0;
    }

    f64(): number {
        const b = this.bytes(8);
        return new DataView(b.buffer, b.byteOffset, 8).getFloat64(0, true);
    }

    num(): number {
        const v = this.varint();
        const tag = v % 4;
        if (tag === 0) return v / 4;
        if (tag === 1) return (v - 1) / 4 / 1e6;
        if (v === 2) return this.f64();
        throw new PackFormatError('bad number tag');
    }
}

/** What a whole pack may still hold: every list a decode reads draws its count from ONE budget. */
export interface Budget { left: number }

/**
 * A count that is bounded before anything is allocated for it, and drawn from the pack's cumulative budget, so
 * many lists that are each small cannot add up to a memory bomb.
 */
export function boundedCount(r: ByteReader, budget: Budget, what: string): number {
    const n = r.varint();
    if (n > budget.left) throw new PackFormatError(`${what}: ${n} exceeds the pack's row budget (${budget.left} left)`);
    budget.left -= n;
    return n;
}
