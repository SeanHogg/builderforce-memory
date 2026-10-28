/**
 * knowledge/partitions.ts — ordered groups of sources and who may use whom (PRD 38 §4.4).
 *
 * Generic: code layers and doc folders are both a {@link PartitionRegistry}. An edge's
 * verdict is `same`, `allowed`, `upward` (to a lower order, i.e. a layer above) or
 * `skips` (anything else), and is set only when the registry defines `allow` and both
 * ends are partitioned.
 *
 * A rule whose id is {@link TOP_LEVEL_FOLDER} names its partition after the path's first
 * folder — the docs pack's "one plane per top-level folder" with no per-folder rows.
 */

import type { EdgeVerdict, KnowledgePack, PartitionRegistry, PartitionRule } from './model.js';

/** Rule id meaning "the matched path's top-level folder". */
export const TOP_LEVEL_FOLDER = '*';

const globCache = new Map<string, RegExp>();

/**
 * A glob as a RegExp over `/`-separated paths: `**` spans folders (`**\/` may match none),
 * `*` and `?` stay within one segment, `{a,b}` is alternation.
 */
export function globToRegExp(glob: string): RegExp {
    const cached = globCache.get(glob);
    if (cached) return cached;
    let re = '';
    for (let i = 0; i < glob.length; i++) {
        const ch = glob[i]!;
        if (ch === '*') {
            if (glob[i + 1] === '*') {
                const slashAfter = glob[i + 2] === '/';
                re += slashAfter ? '(?:.*/)?' : '.*';
                i += slashAfter ? 2 : 1;
            } else re += '[^/]*';
        } else if (ch === '?') re += '[^/]';
        else if (ch === '{') {
            const close = glob.indexOf('}', i);
            if (close < 0) { re += '\\{'; continue; }
            re += `(?:${glob.slice(i + 1, close).split(',').map(escapeRe).join('|')})`;
            i = close;
        } else re += escapeRe(ch);
    }
    const compiled = new RegExp(`^${re}$`);
    globCache.set(glob, compiled);
    return compiled;
}

function escapeRe(s: string): string {
    return s.replace(/[.+^${}()|[\]\\*?]/g, '\\$&');
}

export function matchGlob(path: string, glob: string): boolean {
    return globToRegExp(glob).test(path);
}

/** The partition a root-relative path belongs to: the first matching rule, in registry order. */
export function partitionOf(path: string, reg: PartitionRegistry): string | undefined {
    for (const rule of reg.partitions) {
        if (!rule.match.some((g) => matchGlob(path, g))) continue;
        if (rule.id !== TOP_LEVEL_FOLDER) return rule.id;
        const slash = path.indexOf('/');
        return slash > 0 ? path.slice(0, slash) : undefined;
    }
    return undefined;
}

/** The order of a partition (folder partitions from {@link TOP_LEVEL_FOLDER} share that rule's). */
export function partitionOrder(id: string, reg: PartitionRegistry): number | undefined {
    const rule = reg.partitions.find((r) => r.id === id) ?? reg.partitions.find((r) => r.id === TOP_LEVEL_FOLDER);
    return rule?.order;
}

export function verdictOf(from: string | undefined, to: string | undefined, reg: PartitionRegistry): EdgeVerdict | undefined {
    if (!reg.allow || from === undefined || to === undefined) return undefined;
    if (from === to) return 'same';
    if (reg.allow[from]?.includes(to)) return 'allowed';
    const fromOrder = partitionOrder(from, reg);
    const toOrder = partitionOrder(to, reg);
    if (fromOrder !== undefined && toOrder !== undefined && toOrder < fromOrder) return 'upward';
    return 'skips';
}

/**
 * A pack's registry, or a validated `.builderforce/partitions.json` override. Throws a
 * `TypeError` naming the first problem when the override is not a {@link PartitionRegistry}.
 */
export function loadPartitions(pack: KnowledgePack, override?: unknown): PartitionRegistry {
    if (override === undefined || override === null) return pack.partitions ?? { partitions: [] };
    const fail = (why: string): never => { throw new TypeError(`partitions.json: ${why}`); };
    if (typeof override !== 'object' || Array.isArray(override)) fail('expected an object');
    const o = override as Record<string, unknown>;
    if (!Array.isArray(o['partitions'])) fail('`partitions` must be an array');
    const partitions: PartitionRule[] = (o['partitions'] as unknown[]).map((raw, i) => {
        const r = raw as Record<string, unknown> | null;
        if (!r || typeof r !== 'object') fail(`partitions[${i}] must be an object`);
        if (typeof r!['id'] !== 'string' || !r!['id']) fail(`partitions[${i}].id must be a non-empty string`);
        if (typeof r!['order'] !== 'number' || !Number.isFinite(r!['order'])) fail(`partitions[${i}].order must be a number`);
        const match = r!['match'];
        if (!Array.isArray(match) || !match.every((g) => typeof g === 'string' && g)) {
            fail(`partitions[${i}].match must be an array of globs`);
        }
        return { id: r!['id'] as string, order: r!['order'] as number, match: [...(match as string[])] };
    });
    const ids = new Set(partitions.map((p) => p.id));
    if (ids.size !== partitions.length) fail('partition ids must be unique');
    if (o['allow'] === undefined) return { partitions };
    const allowRaw = o['allow'];
    if (!allowRaw || typeof allowRaw !== 'object' || Array.isArray(allowRaw)) fail('`allow` must be an object');
    const allow: Record<string, string[]> = {};
    for (const [from, to] of Object.entries(allowRaw as Record<string, unknown>)) {
        if (!ids.has(from)) fail(`allow.${from} names an unknown partition`);
        if (!Array.isArray(to) || !to.every((t) => typeof t === 'string' && ids.has(t))) {
            fail(`allow.${from} must list known partition ids`);
        }
        allow[from] = [...(to as string[])];
    }
    return { partitions, allow };
}
