/**
 * evermind/contract.ts — what a project Evermind recall returns, and when a turn teaches it.
 *
 * The gateway produces these shapes and every chat client (web, VS Code, canvas)
 * consumes them. Defining them once keeps the producer and the consumers from drifting.
 */

/** One learned memory recalled for the current turn. */
export interface EvermindRecallItem {
    /** `chat` = contributed by THIS conversation, `project` = the wider project. Absent from an older gateway. */
    tier?: 'chat' | 'project';
    /** Stable id of the learned memory (targets a specific contribution). */
    id: number;
    /** Readable snippet of the learned exemplar (or the task it answered). */
    text: string;
    /** Lexical relevance to the query, 0..1. */
    score: number;
}

/** A recall: the project's learning posture plus the recalled memories. */
export interface EvermindRecallResult {
    /** How the returned memories split between THIS chat and the wider project. Absent from an older gateway. */
    tiers?: { fromChat: number; fromProject: number };
    /** True once the project has a base Evermind (version >= 1). */
    seeded: boolean;
    /** Current head version the recall ran against. */
    version: number;
    /** `connected` = turns contribute back; `offline-frozen` = pinned, read-only. */
    mode: 'connected' | 'offline-frozen';
    /** Recalled memories, best-first. Empty when nothing matched. */
    items: EvermindRecallItem[];
}

/**
 * Assistant text shorter than this is not a teaching signal, so it is never contributed.
 * The gateway gates on it and the client uses it to show the "contributed" step only
 * when the gateway actually contributes the turn.
 */
export const EVERMIND_MIN_TEACH_CHARS = 40;
