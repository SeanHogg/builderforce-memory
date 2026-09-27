/**
 * Load a local `.evermind` text model and the tokenizer behind its token ids.
 *
 * Two things on a headless host read the person's private model: recall (which
 * embeds with it) and experience training (which adapts it). They must agree on
 * which vocabulary a checkpoint speaks, so the resolution lives here once.
 */

import { dynamicImport } from "../dynamic-import.js";

/** The tokenizer surface both embedding and adaptation need. */
export interface TextCodec {
    encode(text: string): number[];
}

/** Structural view of the engine's package — enough to embed with or adapt. */
export interface EvermindPackageLike {
    manifest: { modelType?: string; version?: string; name?: string };
    checkpoint: ArrayBuffer;
    loadLM(): unknown;
    /** Present on engines that support the embedded-tokenizer section. */
    loadTokenizer?(): TextCodec | null;
    toBlob(): ArrayBuffer;
}

/** Minimal structural view of the optional engine package. */
interface PackageEngine {
    EvermindModelPackage: {
        fromBlob(blob: ArrayBuffer): EvermindPackageLike;
        manifestEnd(header: ArrayBuffer): number;
        readManifest(prefix: ArrayBuffer): EvermindPackageLike["manifest"];
    };
    PACKAGE_HEADER_BYTES: number;
    BPETokenizer: new () => { loadFromSpec(spec: unknown): void; loadHuggingFace(spec: unknown): void } & TextCodec;
}

export type ModelFs = {
    readFileSync(path: string, enc: "utf8"): string;
    readFileSync(path: string): Uint8Array;
    writeFileSync(path: string, data: string | Uint8Array): void;
    existsSync(path: string): boolean;
    copyFileSync(src: string, dest: string): void;
    renameSync(from: string, to: string): void;
};

export interface LoadedEvermind {
    pkg: EvermindPackageLike;
    codec: TextCodec;
    fs: ModelFs;
}

type PrefixFs = {
    openSync(path: string, flags: "r"): number;
    readSync(fd: number, buffer: Uint8Array, offset: number, length: number, position: number): number;
    closeSync(fd: number): void;
};

/**
 * The version in a `.evermind` file's manifest, reading only the header and manifest —
 * never the checkpoint — so a caller can ask on every poll, and always sees the file
 * as it is now (a model restored from `.prev` keeps the size and, on Windows, the
 * modification time of the one it replaces). Null when there is no readable package.
 */
export async function readEvermindVersion(modelFile: string): Promise<string | null> {
    let fd: number | undefined;
    const fs = (await dynamicImport("node:fs")) as PrefixFs;
    try {
        const engine = (await dynamicImport("@seanhogg/builderforce-memory-engine")) as Partial<PackageEngine>;
        if (!engine?.EvermindModelPackage || !engine.PACKAGE_HEADER_BYTES) return null;
        fd = fs.openSync(modelFile, "r");
        const read = (n: number) => {
            const buf = new Uint8Array(n);
            const got = fs.readSync(fd!, buf, 0, n, 0);
            return buf.buffer.slice(0, got) as ArrayBuffer;
        };
        const end = engine.EvermindModelPackage.manifestEnd(read(engine.PACKAGE_HEADER_BYTES));
        return engine.EvermindModelPackage.readManifest(read(end)).version ?? "0";
    } catch {
        return null;
    } finally {
        if (fd !== undefined) fs.closeSync(fd);
    }
}

/**
 * Load the tokenizer spec from `file`, accepting either `BPETokenizer.toObject()`
 * output or a Hugging Face `tokenizer.json`.
 */
function loadCodec(engine: PackageEngine, fs: ModelFs, file: string): TextCodec {
    const spec = JSON.parse(fs.readFileSync(file, "utf8")) as { vocab?: unknown; merges?: unknown };
    const tok = new engine.BPETokenizer();
    if (spec.vocab && spec.merges) tok.loadFromSpec(spec);
    else tok.loadHuggingFace(spec);
    return tok;
}

/**
 * The `evermind-lm` package at `modelFile` with its tokenizer, or `null` — never a
 * throw — when the engine peer is absent, a file is missing, or the package is not a
 * text LM. Both callers treat a missing model as a degrade, not a fault.
 *
 * Tokenizer resolution: the package's OWN tokenizer (checksummed and vocab-checked
 * at package time, so it cannot be the wrong vocabulary — which a separate file
 * silently can be), then `tokenizerFile`, then a sibling `<modelFile>.tokenizer.json`.
 */
export async function loadEvermindPackage(modelFile: string, tokenizerFile?: string): Promise<LoadedEvermind | null> {
    try {
        const engine = (await dynamicImport("@seanhogg/builderforce-memory-engine")) as Partial<PackageEngine>;
        if (!engine?.EvermindModelPackage || !engine.BPETokenizer) return null;

        const fs = (await dynamicImport("node:fs")) as ModelFs;
        if (!fs.existsSync(modelFile)) return null;

        const bytes = fs.readFileSync(modelFile);
        const blob = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
        const pkg = engine.EvermindModelPackage.fromBlob(blob);
        if (pkg.manifest.modelType !== "evermind-lm") return null;

        const embedded = pkg.loadTokenizer?.() ?? null;
        if (embedded) return { pkg, codec: embedded, fs };
        const file = tokenizerFile ?? `${modelFile}.tokenizer.json`;
        if (!fs.existsSync(file)) return null;
        return { pkg, codec: loadCodec(engine as PackageEngine, fs, file), fs };
    } catch {
        return null;
    }
}
