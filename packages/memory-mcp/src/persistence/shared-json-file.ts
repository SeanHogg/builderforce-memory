/**
 * A JSON file several processes share — the durable tier of every local store here.
 *
 * Each MCP client spawns its own server, and other tools (the BuilderForce VS Code
 * extension's compactor, Synapse) rewrite the same files directly. So a store never
 * trusts its in-memory copy blindly: before each operation it asks
 * {@link SharedJsonFile.readIfChanged}, which re-reads the file only when it changed
 * underneath this process.
 *
 * The loop guard is the stamp recorded immediately AFTER each of our own writes: the
 * next check stats the file, sees the same mtime+size, and returns without reading a
 * byte. A stat per call is the entire steady-state cost.
 */

export type SharedFileFs = {
    readFileSync(path: string, enc: "utf8"): string;
    writeFileSync(path: string, data: string): void;
    /** When present, writes land atomically (temp file + rename). */
    renameSync?(from: string, to: string): void;
    unlinkSync?(path: string): void;
    existsSync(path: string): boolean;
    statSync(path: string): { mtimeMs: number; size: number };
};

/** Identity of the file as this process last left it. */
interface FileStamp {
    mtimeMs: number;
    size: number;
    /** Digest of the exact bytes — settles the case where a file is touched but unchanged. */
    hash: string;
}

/**
 * FNV-1a over the file text. Non-cryptographic on purpose: this only has to separate
 * "someone rewrote the file" from "the mtime moved but the bytes are ours", and
 * node:crypto is deliberately not imported (these modules stay bundleable for the
 * browser, where the disk path is never taken).
 */
export function contentHash(text: string): string {
    let h = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
        h ^= text.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
    }
    return (h >>> 0).toString(16);
}

/** A changed file's text, and the call that marks it as absorbed. */
export interface ChangedFile {
    text: string;
    /**
     * Call once the text was parsed and applied. Not calling it (a half-written file
     * that failed to parse) leaves the stamp alone, so the next check reads again.
     */
    accept(): void;
}

export class SharedJsonFile {
    /** How the file looked when this process last wrote or absorbed it. */
    private stamp: FileStamp | null = null;

    constructor(
        readonly path: string,
        private readonly fs: SharedFileFs,
    ) {}

    /**
     * The file's text when it changed since we last touched it — and on first use,
     * which IS the boot hydration (one routine, so disk and memory never diverge by
     * taking two paths). Null when unchanged, absent or unreadable.
     */
    readIfChanged(): ChangedFile | null {
        if (!this.fs.existsSync(this.path)) return null;
        let st: { mtimeMs: number; size: number };
        try {
            st = this.fs.statSync(this.path);
        } catch {
            return null;
        }
        // Fast path: byte-for-byte what we last wrote. No read, no parse.
        if (this.stamp && st.mtimeMs === this.stamp.mtimeMs && st.size === this.stamp.size) return null;
        let text: string;
        try {
            text = this.fs.readFileSync(this.path, "utf8");
        } catch {
            return null;
        }
        const hash = contentHash(text);
        // Touched (mtime moved) but identical content — re-stamp, nothing to absorb.
        if (this.stamp && hash === this.stamp.hash) {
            this.stamp = { mtimeMs: st.mtimeMs, size: st.size, hash };
            return null;
        }
        return { text, accept: () => (this.stamp = { mtimeMs: st.mtimeMs, size: st.size, hash }) };
    }

    /**
     * Write, and stamp our own write so the next check does not read it back as foreign.
     *
     * Atomic where the filesystem allows: the text goes to a sibling temp file that is
     * then renamed over the target, so a crash mid-write leaves the previous complete
     * file, never a truncated one (which would read as empty and be overwritten by the
     * next write). If the rename is refused (Windows, while another process holds the
     * file open), the write falls back to writing in place.
     */
    write(text: string): void {
        const rename = this.fs.renameSync?.bind(this.fs);
        let written = false;
        if (rename) {
            const tmp = `${this.path}.${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}.tmp`;
            try {
                this.fs.writeFileSync(tmp, text);
                rename(tmp, this.path);
                written = true;
            } catch {
                try {
                    if (this.fs.existsSync(tmp)) this.fs.unlinkSync?.(tmp);
                } catch {
                    /* best effort: a stray temp file is harmless */
                }
            }
        }
        if (!written) this.fs.writeFileSync(this.path, text);
        try {
            const st = this.fs.statSync(this.path);
            this.stamp = { mtimeMs: st.mtimeMs, size: st.size, hash: contentHash(text) };
        } catch {
            this.stamp = null;
        }
    }
}
