/**
 * `@seanhogg/builderforce-memory/knowledge/pack` — BFKP v1, "Builderforce Knowledge Pack" (PRD 41 A11):
 * the one compact, lossless wire format a client syncs to the server in, and the server expands.
 *
 * Pure TypeScript with no Node APIs (`CompressionStream` is the only runtime dependency): the VS Code
 * host and the API Worker run this same module. There is no second copy of the codec.
 */

export { encodePack, normalizePack, type EncodeOptions } from './encode.js';
export { decodePack, peekPack } from './decode.js';
export { KPACK_CONTENT_TYPE, MAX_PART_BYTES, splitParts, joinParts } from './frame.js';
export { PackFormatError } from './varint.js';
export { crc32 } from './crc32.js';
export {
    ZERO_DIGEST, DEFAULT_PACK_LIMITS, PackTooLargeError,
} from './types.js';
export type {
    Pack, PackBody, PackFileRow, PackStaleRow, PackDeletes, RelationId, PackExtr, ExtrEntryDiff, ListDiff, ManifestLeaf, PackLimits,
} from './types.js';
export { diffEntries, applyEntryDiffs, diffEntry, applyEntryDiff, deepEqual } from './extrDiff.js';
export {
    MANIFEST_BUCKETS, manifestTree, bucketOf, keyHash, differingBuckets, leafDelta, rootLeaves, rootDigestOf,
    fileLeafKey, factLeafKey, relationLeafKey, parseRelationLeafKey, factDigest, relationDigest, bucketDigest, rootDigestOfBuckets,
} from './manifest.js';
export type { ManifestTree, LeafDelta, RootRows } from './manifest.js';
export { jsonBytesOf } from './jsonSize.js';
