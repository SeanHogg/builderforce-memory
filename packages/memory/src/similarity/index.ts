/**
 * Text + vector similarity primitives.
 *
 * Shared by the MemoryStore (semantic recall) and the SemanticCache
 * (embedding-keyed response cache) so the cosine/Jaccard maths live in exactly
 * one place rather than being duplicated per consumer.
 */

/** Splits text into lowercase word tokens, removing punctuation. */
export function tokenize(text: string): string[] {
    return text.toLowerCase().split(/[\s\W]+/).filter(Boolean);
}

/** Jaccard similarity between two token sets: |A ∩ B| / |A ∪ B|. */
export function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
    if (a.size === 0 && b.size === 0) return 1;
    let intersection = 0;
    for (const token of a) {
        if (b.has(token)) intersection++;
    }
    // Both-empty is handled above, so `union` is always > 0 here.
    const union = a.size + b.size - intersection;
    return intersection / union;
}

/**
 * Cosine similarity between two vectors in the range [-1, 1] (0 when either is a
 * zero vector). Compares over the shorter length when lengths differ. Vectors
 * from MambaSession.embed() are already L2-normalised — this reduces to a dot
 * product — but we normalise defensively for vectors from other sources.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
    const n = Math.min(a.length, b.length);
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < n; i++) {
        dot += a[i]! * b[i]!;
        na  += a[i]! * a[i]!;
        nb  += b[i]! * b[i]!;
    }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom === 0 ? 0 : dot / denom;
}
