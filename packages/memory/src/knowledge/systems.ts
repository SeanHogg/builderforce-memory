/**
 * knowledge/systems.ts — which system (and app) a root-relative path belongs to (PRD 41 A9).
 *
 * Memory does not DETECT systems (that reads manifests and `.git` folders: the host's job); it
 * is handed the detected {@link GraphSystem}s and stamps every file node with its placement, so
 * edges between two systems can be told from edges inside one. Pure and deterministic.
 */

import type { GraphSystem, KNode } from './model.js';

const within = (path: string, folder: string): boolean => folder === '' || path === folder || path.startsWith(`${folder}/`);

export interface SystemPlacement { system: string; app?: string; copyOf?: string }

/** The deepest system containing `path` (a system at '' contains every path), with its deepest app and copy mark; undefined when none does. */
export function placeInSystem(systems: readonly GraphSystem[], path: string): SystemPlacement | undefined {
    let best: GraphSystem | undefined;
    for (const s of systems) {
        if (within(path, s.path) && (!best || s.path.length > best.path.length)) best = s;
    }
    if (!best) return undefined;
    let app: string | undefined;
    for (const a of best.apps) {
        if (a.path !== '' && within(path, a.path) && (app === undefined || a.path.length > app.length)) app = a.path;
    }
    return { system: best.path, ...(app !== undefined ? { app } : {}), ...(best.copyOf !== undefined ? { copyOf: best.copyOf } : {}) };
}

/** Whether an edge between two nodes crosses systems: both are placed, and in different ones. */
export function crossesSystems(from: Pick<KNode, 'system'>, to: Pick<KNode, 'system'>): boolean {
    return from.system !== undefined && to.system !== undefined && from.system !== to.system;
}
