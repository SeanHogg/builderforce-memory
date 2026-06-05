/**
 * bpe.ts – Browser-side Byte Pair Encoding (BPE) tokenizer.
 */

export interface BPEEncodeOptions {
  addBos?: boolean;
  addEos?: boolean;
}

export type PadSide = 'right' | 'left';

function buildByteEncoder(): Map<number, string> {
    const enc = new Map<number, string>();
    const ranges: [number, number][] = [
        [0x21, 0x7E],
        [0xA1, 0xAC],
        [0xAE, 0xFF],
    ];
    let n = 0;
    for (const [lo, hi] of ranges) {
        for (let b = lo; b <= hi; b++) {
            enc.set(b, String.fromCodePoint(b));
        }
    }
    for (let b = 0; b < 256; b++) {
        if (!enc.has(b)) {
            enc.set(b, String.fromCodePoint(256 + n));
            n++;
        }
    }
    return enc;
}

const BYTE_ENCODER = buildByteEncoder();
const BYTE_DECODER = new Map([...BYTE_ENCODER].map(([k, v]) => [v, k]));

const PRE_TOKENIZE_RE =
    /(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

export class BPETokenizer {
    vocab: Map<string, number>;
    idToToken: Map<number, string>;
    merges: Map<string, number>;
    bosToken: string;
    eosToken: string;
    padToken: string;
    unkToken: string;
    bosId: number | null;
    eosId: number | null;
    padId: number | null;

    constructor() {
        this.vocab      = new Map();
        this.idToToken  = new Map();
        this.merges     = new Map();
        this.bosToken   = '<|im_start|>';
        this.eosToken   = '<|im_end|>';
        this.padToken   = '<|endoftext|>';
        this.unkToken   = '<unk>';
        this.bosId      = null;
        this.eosId      = null;
        this.padId      = null;
    }

    async load(vocab: string | Record<string, number>, merges: string | string[]): Promise<void> {
        let vocabObj: Record<string, number>;
        if (typeof vocab === 'string') {
            const res = await fetch(vocab);
            vocabObj = await res.json() as Record<string, number>;
        } else {
            vocabObj = vocab;
        }
        this.vocab     = new Map(Object.entries(vocabObj).map(([k, v]) => [k, Number(v)]));
        this.idToToken = new Map([...this.vocab].map(([k, v]) => [v, k]));

        let mergeLines: string[];
        if (typeof merges === 'string') {
            const res = await fetch(merges);
            const txt = await res.text();
            mergeLines = txt.split('\n').filter(l => l && !l.startsWith('#'));
        } else {
            mergeLines = merges;
        }
        this.merges = new Map();
        mergeLines.forEach((line, rank) => {
            this.merges.set(line.trim(), rank);
        });

        this.bosId = this.vocab.get(this.bosToken) ?? null;
        this.eosId = this.vocab.get(this.eosToken) ?? null;
        this.padId = this.vocab.get(this.padToken) ?? null;
    }

    loadFromObjects(vocabObj: Record<string, number>, mergeArr: string[]): void {
        this.vocab     = new Map(Object.entries(vocabObj).map(([k, v]) => [k, Number(v)]));
        this.idToToken = new Map([...this.vocab].map(([k, v]) => [v, k]));
        this.merges    = new Map(mergeArr.map((m, i) => [m, i]));
        this.bosId = this.vocab.get(this.bosToken) ?? null;
        this.eosId = this.vocab.get(this.eosToken) ?? null;
        this.padId = this.vocab.get(this.padToken) ?? null;
    }

    encode(text: string, opts: BPEEncodeOptions = {}): number[] {
        const words = text.match(PRE_TOKENIZE_RE) ?? [];
        const ids: number[]   = [];

        if (opts.addBos && this.bosId !== null) ids.push(this.bosId);

        for (const word of words) {
            const bytes    = new TextEncoder().encode(word);
            const byteStr  = Array.from(bytes).map(b => BYTE_ENCODER.get(b) ?? '?').join('');
            const bpeTokens = this._bpe(byteStr);

            for (const tok of bpeTokens) {
                const id = this.vocab.get(tok);
                if (id !== undefined) {
                    ids.push(id);
                } else {
                    for (const ch of tok) {
                        const cid = this.vocab.get(ch);
                        if (cid !== undefined) ids.push(cid);
                    }
                }
            }
        }

        if (opts.addEos && this.eosId !== null) ids.push(this.eosId);
        return ids;
    }

    decode(ids: number[]): string {
        let byteStr = '';
        for (const id of ids) {
            const tok = this.idToToken.get(id);
            if (tok !== undefined) byteStr += tok;
        }
        const bytes = new Uint8Array(
            [...byteStr].map(ch => BYTE_DECODER.get(ch) ?? ch.codePointAt(0) ?? 0)
        );
        try {
            return new TextDecoder('utf-8').decode(bytes);
        } catch {
            return byteStr;
        }
    }

    _bpe(word: string): string[] {
        if (this.vocab.has(word)) return [word];

        let symbols = [...word];

        while (symbols.length > 1) {
            let bestRank = Infinity;
            let bestIdx  = -1;

            for (let i = 0; i < symbols.length - 1; i++) {
                const pair = symbols[i] + ' ' + symbols[i + 1];
                const rank = this.merges.get(pair);
                if (rank !== undefined && rank < bestRank) {
                    bestRank = rank;
                    bestIdx  = i;
                }
            }

            if (bestIdx === -1) break;

            const merged = symbols[bestIdx]! + symbols[bestIdx + 1]!;
            symbols = [
                ...symbols.slice(0, bestIdx),
                merged,
                ...symbols.slice(bestIdx + 2),
            ];
        }

        return symbols;
    }

    padOrTruncate(ids: number[], maxLen: number, side: PadSide = 'right'): number[] {
        if (ids.length >= maxLen) return ids.slice(0, maxLen);
        const padId = this.padId ?? 0;
        const pad   = new Array<number>(maxLen - ids.length).fill(padId);
        return side === 'right' ? [...ids, ...pad] : [...pad, ...ids];
    }

    get vocabSize(): number { return this.vocab.size; }
}
