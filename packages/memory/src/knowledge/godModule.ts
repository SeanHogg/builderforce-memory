/**
 * knowledge/godModule.ts — the `god-module` finding: one per file node whose `patterns` badge
 * names `god-module`, carrying the measures that tripped it next to the limits they are held to.
 *
 * Memory owns the limits ({@link GOD_MODULE_LIMITS}); the code pack's detector imports them, so the
 * badge and the finding can never disagree. The measures are read from the file's badges
 * (`lines`, `exports`, `maxComplexity`); a measure is omitted when its badge is absent (a graph
 * reviewed before that badge existed). Heuristic by nature: it is a size/shape guess, never a model.
 */

import type { Finding, KnowledgeGraph } from './model.js';
import { finding, refAnchor } from './findingBuilder.js';
import { compareStr } from './refs.js';

/** What a file may grow to before it is a god module: more lines, exports or worst-function complexity than this. */
export const GOD_MODULE_LIMITS = Object.freeze({ lines: 1_000, exports: 40, complexity: 60 } as const);
export type GodModuleLimits = typeof GOD_MODULE_LIMITS;

/** The finding kind (also the pattern id in the `patterns` badge). */
export const GOD_MODULE_KIND = 'god-module';

const PATTERNS_BADGE = 'patterns';
const MEASURE_BADGES = { lines: 'lines', exports: 'exports', complexity: 'maxComplexity' } as const;

const numberOf = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);

export function godModuleFindings(g: KnowledgeGraph): Finding[] {
    const out: Finding[] = [];
    for (const n of [...g.nodes].sort((a, b) => compareStr(a.ref, b.ref))) {
        if (n.kind !== 'file') continue;
        const patterns = n.badges?.[PATTERNS_BADGE];
        if (typeof patterns !== 'string' || !patterns.split(',').includes(GOD_MODULE_KIND)) continue;
        const params: Record<string, number> = {};
        for (const [key, badge] of Object.entries(MEASURE_BADGES)) {
            const value = numberOf(n.badges?.[badge]);
            if (value !== undefined) params[key] = value;
        }
        out.push(finding(GOD_MODULE_KIND, n.pack, [n.ref], 'warn', {
            ...params,
            linesLimit: GOD_MODULE_LIMITS.lines,
            exportsLimit: GOD_MODULE_LIMITS.exports,
            complexityLimit: GOD_MODULE_LIMITS.complexity,
        }, undefined, refAnchor(n.ref)));
    }
    return out;
}
