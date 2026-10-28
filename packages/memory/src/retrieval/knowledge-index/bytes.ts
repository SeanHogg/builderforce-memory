/**
 * bytes.ts — the little-endian byte writer/reader that `serialize` and `digest`
 * share, so the persisted format and the hashed canonical state encode every
 * field the same way. DataView only: no Node `Buffer`, no host endianness.
 */

const encoder = new TextEncoder();
const decoder = new TextDecoder();

export class ByteWriter {
    private _buf = new Uint8Array(1024);
    private _view = new DataView(this._buf.buffer);
    private _len = 0;

    private _reserve(n: number): void {
        if (this._len + n <= this._buf.length) return;
        let size = this._buf.length * 2;
        while (size < this._len + n) size *= 2;
        const next = new Uint8Array(size);
        next.set(this._buf.subarray(0, this._len));
        this._buf = next;
        this._view = new DataView(next.buffer);
    }

    u8(v: number): this {
        this._reserve(1);
        this._view.setUint8(this._len, v);
        this._len += 1;
        return this;
    }

    u32(v: number): this {
        this._reserve(4);
        this._view.setUint32(this._len, v >>> 0, true);
        this._len += 4;
        return this;
    }

    i32(v: number): this {
        this._reserve(4);
        this._view.setInt32(this._len, v, true);
        this._len += 4;
        return this;
    }

    f64(v: number): this {
        this._reserve(8);
        this._view.setFloat64(this._len, v, true);
        this._len += 8;
        return this;
    }

    /** Raw bytes, no length prefix. */
    raw(bytes: Uint8Array): this {
        this._reserve(bytes.length);
        this._buf.set(bytes, this._len);
        this._len += bytes.length;
        return this;
    }

    /** Length-prefixed UTF-8. */
    str(s: string): this {
        const b = encoder.encode(s);
        return this.u32(b.length).raw(b);
    }

    /** Count-prefixed u32 values. */
    u32s(arr: ArrayLike<number>): this {
        this.u32(arr.length);
        this._reserve(arr.length * 4);
        for (let i = 0; i < arr.length; i++) this._view.setUint32(this._len + i * 4, arr[i]! >>> 0, true);
        this._len += arr.length * 4;
        return this;
    }

    /** Count-prefixed f32 values. */
    f32s(arr: ArrayLike<number>): this {
        this.u32(arr.length);
        this._reserve(arr.length * 4);
        for (let i = 0; i < arr.length; i++) this._view.setFloat32(this._len + i * 4, arr[i]!, true);
        this._len += arr.length * 4;
        return this;
    }

    /** Count-prefixed bytes (an Int8Array's bytes are written as-is). */
    bytes(arr: Uint8Array | Int8Array): this {
        this.u32(arr.length);
        return this.raw(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
    }

    finish(): Uint8Array {
        return this._buf.slice(0, this._len);
    }
}

export class ByteReader {
    private readonly _view: DataView;
    private _pos = 0;

    constructor(private readonly _buf: Uint8Array) {
        this._view = new DataView(_buf.buffer, _buf.byteOffset, _buf.byteLength);
    }

    private _need(n: number): void {
        if (this._pos + n > this._buf.length) throw new Error('knowledge-index: truncated bytes');
    }

    u8(): number {
        this._need(1);
        return this._view.getUint8(this._pos++);
    }

    u32(): number {
        this._need(4);
        const v = this._view.getUint32(this._pos, true);
        this._pos += 4;
        return v;
    }

    i32(): number {
        this._need(4);
        const v = this._view.getInt32(this._pos, true);
        this._pos += 4;
        return v;
    }

    f64(): number {
        this._need(8);
        const v = this._view.getFloat64(this._pos, true);
        this._pos += 8;
        return v;
    }

    raw(n: number): Uint8Array {
        this._need(n);
        const out = this._buf.slice(this._pos, this._pos + n);
        this._pos += n;
        return out;
    }

    str(): string {
        return decoder.decode(this.raw(this.u32()));
    }

    u32s(): Uint32Array {
        const n = this.u32();
        this._need(n * 4);
        const out = new Uint32Array(n);
        for (let i = 0; i < n; i++) out[i] = this._view.getUint32(this._pos + i * 4, true);
        this._pos += n * 4;
        return out;
    }

    f32s(): Float32Array {
        const n = this.u32();
        this._need(n * 4);
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) out[i] = this._view.getFloat32(this._pos + i * 4, true);
        this._pos += n * 4;
        return out;
    }

    bytes(): Uint8Array {
        return this.raw(this.u32());
    }
}
