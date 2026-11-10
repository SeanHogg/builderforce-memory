/**
 * src/knowledge/rules/rules.testkit.test-util.ts — fixture facts and fixture rules for the evaluator's trace
 * tests and the determinism oracle. The rules mirror real ones by what they READ: `io-in-loop` (body, file),
 * `export-count` (structure, file), `unresolved-import` (structure, neighbourhood), `route-guard` (structure,
 * neighbourhood, hops 2 + guard chain), `content-tag` (content, file), `mutual-import` (structure, global),
 * `call-total` (body, global).
 */

import type { Finding } from '../findingModel.js';
import { FACTS_EXTRACTOR_VERSION, factsExtractorTag } from '../facts/extractorVersion.js';
import type { FileFacts } from '../facts/fileFacts.js';
import { compareStr } from '../refs.js';
import { defineRule, type FactRule } from './factRule.js';

export const TAG = factsExtractorTag({ family: 'regex' }, 'fixture');

export const emptyFacts = (contentHash: string): FileFacts => ({
    factsVersion: FACTS_EXTRACTOR_VERSION, extractor: TAG, language: 'fixture', contentHash,
    symbols: [], imports: [], exports: [], entryPoints: [], calls: [], complexity: [], capabilityImports: [],
});

const f = (rule: string, key: string, params: Finding['params'] = {}): Finding => ({
    id: `${rule}:${key}`, kind: rule, pack: 'fixture', nodes: [], severity: 'warn', params,
});

export const ioInLoop = defineRule({
    id: 'io-in-loop', version: 1, reads: ['body'], scope: 'file',
    derive: ({ focus, files }) => files.get(focus!)!.calls.filter((c) => c.inLoop && c.awaited).map((c) => f('io-in-loop', `${focus}#${c.enclosing}#${c.callee}`)),
});

export const exportCount = defineRule({
    id: 'export-count', version: 1, reads: ['structure'], scope: 'file',
    derive: ({ focus, files }) => {
        const n = files.get(focus!)!.exports.length;
        return n > 2 ? [f('export-count', focus!, { exports: n })] : [];
    },
});

export const contentTag = defineRule({
    id: 'content-tag', version: 1, reads: ['content'], scope: 'file',
    derive: ({ focus, files }) => [f('content-tag', focus!, { hash: files.get(focus!)!.contentHash.slice(0, 8) })],
});

/** An import whose resolved target is not among the files the neighbourhood sees. */
export const unresolvedImport = defineRule({
    id: 'unresolved-import', version: 1, reads: ['structure'], scope: 'neighbourhood',
    derive: ({ focus, files }) => files.get(focus!)!.imports
        .filter((i) => i.resolved !== undefined && !files.has(i.resolved))
        .map((i) => f('unresolved-import', `${focus}->${i.resolved}`)),
});

/** A route whose guard chain holds no file exporting `requireAuth`. */
export const routeGuard = defineRule({
    id: 'route-guard', version: 1, reads: ['structure'], scope: 'neighbourhood', hops: 2, readsGuardChain: true,
    derive: ({ focus, files }) => files.get(focus!)!.entryPoints
        .filter((e) => !e.chain.some((c) => files.get(c)?.exports.some((x) => x.name === 'requireAuth')))
        .map((e) => f('route-guard', `${focus}#${e.key}`)),
});

export const mutualImport = defineRule({
    id: 'mutual-import', version: 1, reads: ['structure'], scope: 'global',
    derive: ({ files }) => {
        const out: Finding[] = [];
        for (const [a, fa] of files) for (const i of fa.imports) {
            const b = i.resolved;
            if (b === undefined || compareStr(a, b) >= 0) continue;
            if (files.get(b)?.imports.some((j) => j.resolved === a)) out.push(f('mutual-import', `${a}<->${b}`));
        }
        return out;
    },
});

export const callTotal = defineRule({
    id: 'call-total', version: 1, reads: ['body'], scope: 'global',
    derive: ({ files }) => [f('call-total', 'all', { calls: [...files.values()].reduce((s, x) => s + x.calls.reduce((t, c) => t + c.count, 0), 0) })],
});

export const FIXTURE_RULES: readonly FactRule[] = [ioInLoop, exportCount, contentTag, unresolvedImport, routeGuard, mutualImport, callTotal];
