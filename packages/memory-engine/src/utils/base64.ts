/**
 * utils/base64.ts — binary ⇄ standard base64, isomorphic (browser, Node ≥ 16, Workers).
 *
 * Built on the global `btoa`/`atob` every supported runtime provides, so the engine
 * keeps zero runtime dependencies (no `Buffer`). One implementation for every wire
 * encoding the Evermind contract uses — weight deltas and packed embeddings alike.
 */

/** Chunk size for `String.fromCharCode.apply` — well under every engine's arg limit. */
const CHUNK = 0x8000;

/** Encode bytes as standard (padded) base64. */
export function bytesToBase64(bytes: ArrayBuffer | Uint8Array): string {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let bin = "";
  for (let i = 0; i < u8.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, Array.from(u8.subarray(i, i + CHUNK)));
  }
  return btoa(bin);
}

/** Decode standard base64 to a fresh `ArrayBuffer`. Throws on malformed input (as `atob` does). */
export function base64ToBytes(b64: string): ArrayBuffer {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out.buffer;
}
