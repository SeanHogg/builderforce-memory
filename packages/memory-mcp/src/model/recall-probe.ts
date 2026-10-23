/**
 * The recall probe compaction runs before it stubs a fact: the person's local
 * Evermind model — the same `.evermind` package recall embeds with and experience
 * training adapts (`BUILDERFORCE_MEMORY_MODEL`) — is asked the question generated
 * from the fact (`probeQuestionFor`), and the fact is stubbed only when the answer
 * carries it. No model configured, or one that cannot generate, means no probe —
 * and no probe means nothing is compacted.
 */

import type { RecallProbe } from "../compaction.js";
import { MODEL_FILE_ENV, TOKENIZER_FILE_ENV } from "../embedding/index.js";
import { loadEvermindPackage } from "./evermind-package.js";

/** Structural view of an `EvermindLM` that can generate text. */
interface GeneratingLM {
    generateText(prompt: string, codec: unknown, opts: { maxNewTokens: number; temperature?: number }): string;
}

/** Tokens the model may spend answering one probe — a cloze answer is a word or two. */
export const PROBE_MAX_NEW_TOKENS = 16;

/** Wrap any text generator as a probe. Greedy, short, never throws. */
export function generatorProbe(generate: (prompt: string) => string | Promise<string>): RecallProbe {
    return {
        async answer(question: string): Promise<string | null> {
            try {
                const out = await generate(question);
                return typeof out === "string" ? out : null;
            } catch {
                return null;
            }
        },
    };
}

export interface RecallProbeEnv {
    env?: Record<string, string | undefined>;
}

/**
 * The probe for this server, or null when no model is configured, the package is
 * not a text LM, or its tokenizer cannot decode (no answer text to check).
 */
export async function createRecallProbe(opts: RecallProbeEnv = {}): Promise<RecallProbe | null> {
    const env = opts.env ?? process.env;
    const modelFile = env[MODEL_FILE_ENV];
    if (!modelFile) return null;
    const loaded = await loadEvermindPackage(modelFile, env[TOKENIZER_FILE_ENV] || undefined);
    if (!loaded) return null;
    const codec = loaded.codec as unknown as { decode?: unknown };
    if (typeof codec.decode !== "function") return null;
    let lm: GeneratingLM;
    try {
        lm = loaded.pkg.loadLM() as GeneratingLM;
    } catch {
        return null;
    }
    if (typeof lm?.generateText !== "function") return null;
    return generatorProbe((prompt) =>
        lm.generateText(prompt, loaded.codec, { maxNewTokens: PROBE_MAX_NEW_TOKENS, temperature: 0 }),
    );
}

/**
 * {@link createRecallProbe}, deferred to the first compaction — a server should not
 * load a checkpoint at boot for a tool most sessions never call. Null when no model
 * is configured; a model that then fails to load answers nothing (every probe fails,
 * so nothing is stubbed).
 */
export function lazyRecallProbe(opts: RecallProbeEnv = {}): RecallProbe | null {
    const env = opts.env ?? process.env;
    if (!env[MODEL_FILE_ENV]) return null;
    let probe: Promise<RecallProbe | null> | null = null;
    return {
        async answer(question: string): Promise<string | null> {
            probe ??= createRecallProbe({ env });
            const p = await probe;
            return p ? p.answer(question) : null;
        },
    };
}
