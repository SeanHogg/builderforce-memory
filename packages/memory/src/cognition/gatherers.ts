/**
 * Evermind — reusable evidence gatherers.
 *
 * Concrete, surface-agnostic evidence rules. The presence rule is the one the
 * IDE self-correction loop uses (ground a claim by listing the workspace), kept
 * here so the IDE, the proof harness, and tests share one implementation.
 */

import type { EvidenceGatherer } from './types.js';

export interface WorkspacePresenceRule {
    /** Lists the workspace (e.g. the IDE `list_files('.')` control tool). */
    list: () => Promise<string[]>;
    /** Entries that MUST be present for the new claim to hold. */
    mustExist?: string[];
    /** Entries that MUST be absent for the new claim to hold. */
    mustBeAbsent?: string[];
}

/**
 * Evidence rule: the new claim is supported iff every `mustExist` entry is
 * present and every `mustBeAbsent` entry is gone from the listing.
 */
export function workspacePresenceGatherer(rule: WorkspacePresenceRule): EvidenceGatherer {
    const mustExist = rule.mustExist ?? [];
    const mustBeAbsent = rule.mustBeAbsent ?? [];
    return async () => {
        const listing = await rule.list();
        const present = mustExist.filter((d) => listing.includes(d));
        const absent = mustBeAbsent.filter((d) => !listing.includes(d));
        const supportsNew = present.length === mustExist.length && absent.length === mustBeAbsent.length;
        return {
            supportsNew,
            notes: [
                `present: ${present.join(', ') || '—'}`,
                `absent (as expected): ${absent.join(', ') || '—'}`,
            ],
        };
    };
}
