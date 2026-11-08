/**
 * knowledge/assetFile.ts — the file node of an ASSET: a binary file (image, font, archive …) that is in the
 * root but has no text to extract. It is planned, hashed and diffed like any file and never read, extracted or
 * judged; this module writes the ONE node that stands for it, so a reference to its path (an `<img src>`, an
 * import, a link) resolves to a real node instead of a `missing` placeholder.
 *
 * The node is the SAME file node a text file gets (kind `file`, ref `srcRef(rootId, path)`, key = path, the
 * content hash as `sourceHash`), stored in the same extract cache under the file's hash. What marks it is its
 * `badges` (the node's pack-data field): `asset: 1`, `mediaType` (string) and `size` (bytes), plus `oversized: 1` for a host-claimed document left an asset by the size cap. A host that finds a
 * text-extension file binary on read gets the same node, with the path-derived media type.
 */

import type { Extracted, KnowledgeStore } from './model.js';
import { basenameOf, FALLBACK_PACK_ID, srcRef } from './refs.js';
import { EXTRACTOR_VERSION, mediaTypeOfPath } from './sources.js';
import { putExtract } from './extractFile.js';

/** The `Extracted` of one asset: one file node, no edges. Pure: the same inputs give the same bytes. */
export function assetExtracted(rootId: string, file: { path: string; hash: string; size: number; oversized?: true }): Extracted {
    const ref = srcRef(rootId, file.path);
    return {
        sourceRef: ref, sourceHash: file.hash, extractorVersion: EXTRACTOR_VERSION, edges: [], keys: [{ ref, key: file.path }],
        nodes: [{
            // No pack extracts an asset: it belongs to the generic pack, as a `missing` placeholder does.
            ref, kind: 'file', label: basenameOf(file.path), pack: FALLBACK_PACK_ID, importance: 0, origin: 'extracted', sourceHash: file.hash,
            badges: { asset: 1, mediaType: mediaTypeOfPath(file.path), size: file.size, ...(file.oversized ? { oversized: 1 } : {}) },
        }],
    };
}

/** Stores the asset node in the extract cache under the file's hash (the entries of other files sharing the hash are kept). */
export async function putAsset(store: KnowledgeStore, rootId: string, file: { path: string; hash: string; size: number; oversized?: true }): Promise<void> {
    await putExtract(store, file.hash, assetExtracted(rootId, file));
}
