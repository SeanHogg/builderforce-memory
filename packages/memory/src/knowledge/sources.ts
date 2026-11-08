/**
 * knowledge/sources.ts — what a review needs to know about a file BEFORE (or without)
 * reading it: its pack, its media type, whether it is binary or ignored. Shared by
 * `planReview` and `reviewDirectory`, so the plan's numbers and the run agree.
 */

import { htmlParser, markdownParser, ParserRegistry, plainTextParser } from '../ingest/parsers.js';
import { isBinaryDocumentType, mediaTypeOf } from '../ingest/parserSupport.js';
import { CONTENT_HASH_FIELD, type ChunkFingerprint, type IngestManifest, type SourceDocument } from '../ingest/types.js';
import type { Extracted, KnowledgePack, KnowledgeParser } from './model.js';
import { matchGlob } from './partitions.js';
import { compareStr, walkRelative } from './refs.js';

/** Default per-file size cap (bytes). */
export const DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024;
/**
 * Version of the extractors' output. Bump it whenever an extractor emits different nodes,
 * edges or keys for the same text: cached `Extracted` of another version is ignored, and a
 * review that finds a different version in {@link extractorManifestKey} owes every file one
 * re-extraction (judged sections with unchanged text are re-anchored, not judged again; see
 * owedFiles.ts for how an unfinished one is carried to the next review).
 */
export const EXTRACTOR_VERSION = 4;
/**
 * Always ignored: VCS internals, installed dependencies, builderforce config, and lockfiles (`yarn.lock`, `Cargo.lock` …): generated,
 * huge, and plain text — extracted they would cost judgement calls for nothing; they were never stored (once skipped as binary).
 * A `.builderforce/ignore` rule can still re-include one (`!yarn.lock`).
 */
const DEFAULT_IGNORES = ['.git/', 'node_modules/', '.builderforce/', '.gitignore', '*.lock'];

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
    // Assets (BINARY_EXT below): standard IANA types. A binary extension with none here is `application/octet-stream`.
    png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', gif: 'image/gif', webp: 'image/webp', bmp: 'image/bmp',
    ico: 'image/vnd.microsoft.icon', icns: 'image/icns', tif: 'image/tiff', tiff: 'image/tiff', psd: 'image/vnd.adobe.photoshop',
    avif: 'image/avif', heic: 'image/heic', svg: 'image/svg+xml',
    mp3: 'audio/mpeg', wav: 'audio/wav', flac: 'audio/flac', ogg: 'audio/ogg', m4a: 'audio/mp4',
    mp4: 'video/mp4', mov: 'video/quicktime', avi: 'video/x-msvideo', mkv: 'video/x-matroska', webm: 'video/webm',
    zip: 'application/zip', gz: 'application/gzip', tgz: 'application/gzip', bz2: 'application/x-bzip2', xz: 'application/x-xz',
    '7z': 'application/x-7z-compressed', rar: 'application/vnd.rar', tar: 'application/x-tar',
    jar: 'application/java-archive', war: 'application/java-archive', wasm: 'application/wasm',
    ttf: 'font/ttf', otf: 'font/otf', woff: 'font/woff', woff2: 'font/woff2', eot: 'application/vnd.ms-fontobject',
    sqlite: 'application/vnd.sqlite3', exe: 'application/vnd.microsoft.portable-executable', dll: 'application/vnd.microsoft.portable-executable',
};

/** Binary documents whose text a HOST extractor may supply (§4.5); otherwise stored as an asset (never read). */
const DOCUMENT_EXT = new Set(['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'odt', 'ods', 'odp']);

const BINARY_EXT = new Set([
    'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'ico', 'icns', 'tif', 'tiff', 'psd', 'avif', 'heic',
    'mp3', 'mp4', 'mov', 'avi', 'mkv', 'webm', 'wav', 'flac', 'ogg', 'm4a',
    'zip', 'gz', 'tgz', 'bz2', 'xz', '7z', 'rar', 'tar', 'jar', 'war',
    'exe', 'dll', 'so', 'dylib', 'bin', 'o', 'a', 'class', 'wasm', 'node', 'pyc',
    'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'odt', 'ods', 'odp',
    'ttf', 'otf', 'woff', 'woff2', 'eot', 'sqlite', 'db', 'vsix',
]);

/** Shell and batch scripts are code, never documents (their `[[ … ]]` tests are not links). */
const SCRIPT_EXT = new Set(['sh', 'bash', 'zsh', 'ps1', 'bat', 'cmd']);

export const extOf = (path: string): string => {
    const m = /\.([A-Za-z0-9]+)$/.exec(path);
    return m ? m[1]!.toLowerCase() : '';
};

/** The ONE media-type lookup: the table, else `application/octet-stream` for a binary extension, else `text/plain`. */
export function mediaTypeOfPath(path: string): string {
    const ext = extOf(path);
    return MEDIA_TYPES[ext] ?? (BINARY_EXT.has(ext) ? 'application/octet-stream' : 'text/plain');
}

/** Whether a cached extract is an asset's (assetFile.ts): one file node marked `badges.asset`, no text was extracted — never a hit for a file a parser now extracts. */
export const isAssetExtract = (e: Extracted): boolean => e.nodes.some((n) => n.badges?.['asset'] !== undefined);

/** By extension — the plan never reads a file to decide. (pdf/office text needs a host extractor.) */
export function isBinaryPath(path: string): boolean {
    return BINARY_EXT.has(extOf(path));
}

/**
 * The ONE decision "this listed file is an asset": binary by extension and not claimed as a document by a host parser.
 * An asset is planned, hashed and diffed like any file but never read, extracted or judged (assetFile.ts writes its node).
 */
export function isAssetPath(path: string, claimedAsDocument: boolean): boolean {
    return isBinaryPath(path) && !claimedAsDocument;
}

export function isDocumentPath(path: string): boolean {
    return DOCUMENT_EXT.has(extOf(path));
}

/**
 * The pack whose parser claims a pdf/office file's media type — a host-registered
 * extractor, wherever its pack sits: {@link composePacks} makes the builtin text parsers
 * refuse binary document types ({@link isBinaryDocumentType}), so without a host parser
 * nothing claims it and it stays binary.
 */
export function resolveDocument(composed: ComposedPacks, path: string): ReturnType<ComposedPacks['resolve']> {
    return composed.resolve(stubSource(path));
}

/**
 * The document extensions (pdf/office) some composed pack's parser claims, sorted and comma-joined (`-` for none). A document is an
 * asset exactly when its extension is not in this list, so this ONE string is all a root must remember about asset-ness (owedFiles.ts).
 */
export function claimedDocumentExts(composed: ComposedPacks): string {
    return [...DOCUMENT_EXT].filter((ext) => resolveDocument(composed, `x.${ext}`)).sort(compareStr).join(',') || '-';
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

/** The builtin text parsers, which would otherwise claim an office file's XML/plain-text media type. */
const BUILTIN_TEXT_PARSERS: ReadonlySet<KnowledgeParser> = new Set<KnowledgeParser>([markdownParser, htmlParser, plainTextParser]);

/**
 * Within knowledge composition the builtin docs parsers do not accept binary document media
 * types; a host-registered parser (any pack, any order) is untouched. The ingest pipeline
 * keeps the builtins as they are.
 */
const guardBuiltin = (p: KnowledgeParser): KnowledgeParser =>
    BUILTIN_TEXT_PARSERS.has(p)
        ? { ...p, accepts: (s) => !SCRIPT_EXT.has(extOf(s.id)) && !isBinaryDocumentType(mediaTypeOf(s)) && p.accepts(s) }
        : p;

export function composePacks(packs: readonly KnowledgePack[]): ComposedPacks {
    const owner = new Map<KnowledgeParser, KnowledgePack>();
    const guarded = new Map<KnowledgeParser, KnowledgeParser>();
    for (const pack of packs) {
        for (const p of pack.parsers) {
            if (guarded.has(p)) continue;
            const g = guardBuiltin(p);
            guarded.set(p, g);
            owner.set(g, pack);
        }
    }
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

/**
 * The ignore files present in a listing, each with the folder its patterns are relative to,
 * in precedence order (the last match wins): every `.gitignore` — the root's first, then
 * nested ones by path — and `.builderforce/ignore` LAST, so it can override them.
 */
export function ignoreSourcesOf(paths: readonly string[]): Array<{ path: string; base: string }> {
    const git: Array<{ path: string; base: string }> = [];
    let builderforce = false;
    for (const path of [...paths].sort(compareStr)) {
        if (path === '.builderforce/ignore') builderforce = true;
        else if (path === '.gitignore' || path.endsWith('/.gitignore')) {
            if (path.split('/').some((seg) => seg === 'node_modules' || seg === '.git')) continue;
            git.push({ path, base: path.slice(0, -'.gitignore'.length).replace(/\/$/, '') });
        }
    }
    git.sort((a, b) => Number(a.base !== '') - Number(b.base !== '') || compareStr(a.path, b.path));
    return builderforce ? [...git, { path: '.builderforce/ignore', base: '' }] : git;
}

/** Manifest entry holding `{ id: path, hash }` for every reviewed file of a root. */
export const filesManifestKey = (rootId: string): string => `knowledge:${rootId}`;
/** Manifest entry holding the extractor version the files were last reviewed with, and the files still owed a re-extraction (owedFiles.ts). */
export const extractorManifestKey = (rootId: string): string => `knowledge:${rootId}:extractor`;
/** Manifest entry holding `{ id: sectionRef, hash: textHash }` for every judged section. */
export const judgedManifestKey = (rootId: string): string => `knowledge:${rootId}:judged`;
/** Manifest entry holding `{ id: claimRef, hash: '<age>:<fileHash>' }` for every claim still awaiting comparison (age 0 = oldest). */
export const compareManifestKey = (rootId: string): string => `knowledge:${rootId}:compare`;

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
 * root). Pure path arithmetic: undefined when the target escapes the root or is the root itself.
 */
export function resolveRelativePath(fromPath: string, target: string): string | undefined {
    const { ups, segments } = walkRelative(fromPath, target);
    return ups > 0 || segments.length === 0 ? undefined : segments.join('/');
}
