/**
 * knowledge/ioInLoop.ts — the `io-in-loop` finding (the N+1 problem): one per file node whose
 * `patterns` badge names `io-in-loop`, i.e. a `for`/`while` body awaits a network, file or query call.
 *
 * Heuristic by nature: the code pack's detector reads the source text, never a model, and a loop
 * that must run in order (pagination, rate limits) is flagged too. Read from the badge only.
 */

import type { Finding, KnowledgeGraph } from './model.js';
import { finding, refAnchor } from './findingBuilder.js';
import { filesWithPattern } from './patternFindings.js';

/** The finding kind (also the pattern id in the `patterns` badge). */
export const IO_IN_LOOP_KIND = 'io-in-loop';

export function ioInLoopFindings(g: KnowledgeGraph): Finding[] {
    return filesWithPattern(g, IO_IN_LOOP_KIND)
        .map((n) => finding(IO_IN_LOOP_KIND, n.pack, [n.ref], 'warn', {}, undefined, refAnchor(n.ref)));
}
