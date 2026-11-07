/**
 * knowledge/judgePriority.ts — the order pass 2 judges owed files in (PRD 41 A10 rule 1).
 *
 *   1. entry points and manifests,
 *   2. files with the most inbound references,
 *   3. recently changed files,
 *   4. the rest.
 *
 * Pure and deterministic: ties break by path, so equal input is the same order on every host. The caller
 * supplies the signals (it owns the graph and the history); a missing signal simply ranks nobody.
 */

import { compareStr } from './refs.js';

export interface JudgeSignals {
    /** Entry points and manifests (a package.json, a README, a declared `main`). */
    entry?: (path: string) => boolean;
    /** Inbound reference count per path. */
    inbound?: ReadonlyMap<string, number>;
    /** Files changed since the last review (or by the latest commits). */
    recent?: ReadonlySet<string>;
}

/** `paths` in judging order. The input is not mutated. */
export function orderForJudging(paths: readonly string[], signals: JudgeSignals = {}): string[] {
    const tier = (p: string): number => (signals.entry?.(p) ? 0 : (signals.inbound?.get(p) ?? 0) > 0 ? 1 : signals.recent?.has(p) ? 2 : 3);
    return [...paths].sort((a, b) => {
        const ta = tier(a), tb = tier(b);
        if (ta !== tb) return ta - tb;
        if (ta === 1) {
            const d = (signals.inbound!.get(b) ?? 0) - (signals.inbound!.get(a) ?? 0);
            if (d !== 0) return d;
        }
        return compareStr(a, b);
    });
}
