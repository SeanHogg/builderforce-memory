/**
 * knowledge/sources.ts — what a review needs to know about a file BEFORE (or without)
 * reading it: its pack, its media type, whether it is binary or ignored. Shared by
 * `planReview` and `reviewDirectory`, so the plan's numbers and the run agree.
 */

import { ParserRegistry } from '../ingest/parsers.js';
import { CONTENT_HASH_FIELD, type ChunkFingerprint, type IngestManifest, type SourceDocument } from '../ingest/types.js';
import type { KnowledgePack, KnowledgeParser } from './model.js';
import { matchGlob } from './partitions.js';
import { compareStr } from './refs.js';

/** Default per-file size cap (bytes). */
export const DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024;
/** Ignore files read through the source, when present (`.gitignore` also in any sub-folder). */
export const IGNORE_FILES = ['.gitignore', '.builderforce/ignore'] as const;
/** Always ignored: VCS internals, installed dependencies, builderforce config. */
const DEFAULT_IGNORES = ['.git/', 'node_modules/', '.builderforce/', '.gitignore'];

const MEDIA_TYPES: Record<string, string> = {
    md: 'text/markdown', markdown: 'text/markdown', mdx: 'text/markdown',
    html: 'text/html', htm: 'text/html', xhtml: 'application/xhtml+xml', xml: 'application/xml',
    json: 'application/json', csv: 'text/csv', tsv: 'text/tab-separated-values',
    pdf: 'application/pdf',
    doc: 'application/msword',
    docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    xls: 'application/vnd.ms-excel',
    xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    ppt: 'application/vnd.ms-powerpoint',
    pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    odt: 'application/vnd.oasis.opendocument.text',
    ods: 'application/vnd.oasis.opendocument.spreadsheet',
    odp: 'application/vnd.oasis.opendocument.presentation',
};

/** Binary documents whose text a HOST extractor may supply (§4.5); otherwise skipped as binary. */
const DOCUMENT_EXT = new Set(['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'odt', 'ods', 'odp']);

const BINARY_EXT = new Set([
    'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'ico', 'icns', 'tif', 'tiff', 'psd', 'avif', 'heic',
    'mp3', 'mp4', 'mov', 'avi', 'mkv', 'webm', 'wav', 'flac', 'ogg', 'm4a',
    'zip', 'gz', 'tgz', 'bz2', 'xz', '7z', 'rar', 'tar', 'jar', 'war',
    'exe', 'dll', 'so', 'dylib', 'bin', 'o', 'a', 'class', 'wasm', 'node', 'pyc',
    'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'odt', 'ods', 'odp',
    'ttf', 'otf', 'woff', 'woff2', 'eot', 'sqlite', 'db', 'vsix', 'lock',
]);

const extOf = (path: string): string => {
    const m = /\.([A-Za-z0-9]+)$/.exec(path);
    return m ? m[1]!.toLowerCase() : '';
};

export function mediaTypeOfPath(path: string): string {
    return MEDIA_TYPES[extOf(path)] ?? 'text/plain';
}

/** By extension — the plan never reads a file to decide. (pdf/office text needs a host extractor.) */
export function isBinaryPath(path: string): boolean {
    return BINARY_EXT.has(extOf(path));
}

export function isDocumentPath(path: string): boolean {
    return DOCUMENT_EXT.has(extOf(path));
}

/** The builtin catch-all text parser; it never claims a binary document. */
const CATCH_ALL_PARSER = 'text';

/**
 * The pack whose parser claims a pdf/office file's media type — a host-registered
 * extractor. The builtin catch-all text parser never counts, so without one it stays binary.
 */
export function resolveDocument(composed: ComposedPacks, path: string): ReturnType<ComposedPacks['resolve']> {
    const hit = composed.resolve(stubSource(path));
    return hit && hit.parser.id !== CATCH_ALL_PARSER ? hit : undefined;
}

/** A NUL in the first 8 KB means binary, whatever the extension said. */
export function isBinaryText(text: string): boolean {
    return text.slice(0, 8192).includes('\u0000');
}

/** A read source with its id set to the root-relative path and a media type filled in. */
export function asPathSource(doc: SourceDocument, path: string, hash: string): SourceDocument {
    const content = doc.content.kind === 'text' && !doc.content.mediaType
        ? { ...doc.content, mediaType: mediaTypeOfPath(path) }
        : doc.content;
    return { ...doc, id: path, content, metadata: { ...(doc.metadata ?? {}), [CONTENT_HASH_FIELD]: hash } };
}

/** A content-free stand-in, so `accepts` can pick a pack without a read. */
export function stubSource(path: string): SourceDocument {
    return { id: path, tenantId: '', content: { kind: 'text', text: '', mediaType: mediaTypeOfPath(path) } };
}

export interface ComposedPacks {
    /** The ONE registry, composed from the packs' parsers in pack order. */
    registry: ParserRegistry;
    resolve(source: SourceDocument): { parser: KnowledgeParser; pack: KnowledgePack } | undefined;
}

export function composePacks(packs: readonly KnowledgePack[]): ComposedPacks {
    const owner = new Map<KnowledgeParser, KnowledgePack>();
    for (const pack of packs) for (const p of pack.parsers) if (!owner.has(p)) owner.set(p, pack);
    const registry = new ParserRegistry([...owner.keys()]);
    return {
        registry,
        resolve(source) {
            const parser = registry.resolve(source) as KnowledgeParser | undefined;
            const pack = parser && owner.get(parser);
            return parser && pack ? { parser, pack } : undefined;
        },
    };
}

/**
 * A `.gitignore`-style matcher: `#` comments, `!` negation (last match wins), a trailing
 * `/` for directories only, a `/` elsewhere anchors to the root, otherwise any depth.
 */
export function ignoreMatcher(texts: ReadonlyArray<string | { text: string; base: string }>): (path: string) => boolean {
    const rules: Array<{ glob: string; negate: boolean; dirOnly: boolean }> = [];
    const lines = [
        ...DEFAULT_IGNORES.map((line) => ({ line, base: '' })),
        ...texts.flatMap((t) => {
            const { text, base } = typeof t === 'string' ? { text: t, base: '' } : t;
            return text.split(/\r?\n/).map((line) => ({ line, base }));
        }),
    ];
    for (const { line, base } of lines) {
        let p = line.replace(/\s+$/, '');
        if (!p || p.startsWith('#')) continue;
        const negate = p.startsWith('!');
        if (negate) p = p.slice(1);
        const dirOnly = p.endsWith('/');
        if (dirOnly) p = p.slice(0, -1);
        const anchored = p.includes('/');
        p = p.replace(/^\//, '');
        // A nested `.gitignore` scopes its patterns to its own folder.
        const prefix = base ? `${base}/` : '';
        if (p) rules.push({ glob: anchored ? `${prefix}${p}` : `${prefix}**/${p}`, negate, dirOnly });
    }
    return (path) => {
        const segments = path.split('/');
        const dirs = segments.slice(0, -1).map((_, i) => segments.slice(0, i + 1).join('/'));
        let ignored = false;
        for (const r of rules) {
            const hit = (r.dirOnly ? dirs : [...dirs, path]).some((c) => matchGlob(c, r.glob));
            if (hit) ignored = !r.negate;
        }
        return ignored;
    };
}

/** The ignore files present in a listing, each with the folder its patterns are relative to. */
export function ignoreSourcesOf(paths: readonly string[]): Array<{ path: string; base: string }> {
    const out: Array<{ path: string; base: string }> = [];
    for (const path of [...paths].sort(compareStr)) {
        if (path === '.builderforce/ignore') out.push({ path, base: '' });
        else if (path === '.gitignore' || path.endsWith('/.gitignore')) {
            if (path.split('/').some((seg) => seg === 'node_modules' || seg === '.git')) continue;
            out.push({ path, base: path.slice(0, -'.gitignore'.length).replace(/\/$/, '') });
        }
    }
    return out;
}

/** Manifest entry holding `{ id: path, hash }` for every reviewed file of a root. */
export const filesManifestKey = (rootId: string): string => `knowledge:${rootId}`;
/** Manifest entry holding `{ id: sectionRef, hash: textHash }` for every judged section. */
export const judgedManifestKey = (rootId: string): string => `knowledge:${rootId}:judged`;

export async function manifestMap(manifest: IngestManifest, key: string): Promise<Map<string, string>> {
    const prior: ChunkFingerprint[] = (await manifest.get(key)) ?? [];
    return new Map(prior.map((f) => [f.id, f.hash] as const));
}

export async function saveManifestMap(manifest: IngestManifest, key: string, map: ReadonlyMap<string, string>): Promise<void> {
    const entries = [...map].sort((a, b) => compareStr(a[0], b[0])).map(([id, hash]) => ({ id, hash }));
    await manifest.set(key, entries);
}

/**
 * Resolves a link/import target against the linking file's folder (a leading `/` is the
 * root). Pure path arithmetic: undefined only when the target escapes the root.
 */
export function resolveRelativePath(fromPath: string, target: string): string | undefined {
    const t = target.replace(/\\/g, '/');
    const out = t.startsWith('/') ? [] : fromPath.split('/').slice(0, -1);
    for (const seg of t.split('/')) {
        if (seg === '' || seg === '.') continue;
        if (seg === '..') {
            if (out.length === 0) return undefined;
            out.pop();
        } else out.push(seg);
    }
    return out.length ? out.join('/') : undefined;
}
