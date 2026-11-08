/**
 * Information-theoretic primitives for the stats module (PRD 41 Part D / D13):
 * Shannon entropy over empirical probabilities.
 */

/** Shannon entropy in nats (natural log) of an empirical distribution. */
export function shannonEntropy(probabilities: readonly number[]): number {
    let h = 0;
    for (const p of probabilities) {
        if (p > 0) h -= p * Math.log(p);
    }
    return h;
}

/** Entropy of a count vector; zero counts contribute nothing. */
export function entropyOfCounts(counts: readonly number[]): number {
    const total = counts.reduce((a, b) => a + b, 0);
    if (total === 0) return 0;
    return shannonEntropy(counts.map((c) => c / total));
}
