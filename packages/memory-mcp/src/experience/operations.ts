/**
 * The experience operations, as plain functions.
 *
 * ONE implementation of what each experience tool does (save an episode, compile a
 * skill, schedule it, run it, audit it). The MCP tools in `tools.ts` and in-process
 * callers (the `@seanhogg/builderforce-memory-mcp/experience` export) both call these,
 * so the rules live once. Each returns an {@link OpResult}: the value, or the message a
 * tool reports as its error.
 */

import type { Episode, ReviewEdits, Run, RunStatus, SkillRoutine, StepOutcome } from "@seanhogg/builderforce-memory-engine";
import { isSafeExperienceId } from "./file-store.js";
import type { ExperienceHost } from "./host.js";

export type OpResult<T> = { ok: true; value: T } | { ok: false; error: string };

const ok = <T>(value: T): OpResult<T> => ({ ok: true, value });
const err = (error: string): OpResult<never> => ({ ok: false, error });

/** The slice of the host the operations need. */
export type OperationsHost = Pick<ExperienceHost, "store" | "engine">;

const ACTION_KINDS = new Set(["click", "setValue", "keys"]);

const isTarget = (t: unknown): boolean =>
    !!t && typeof t === "object" && typeof (t as { name?: unknown }).name === "string";

/**
 * Why one recorded action is unusable, or null. Every field the corpus and replay read
 * is checked here, so one malformed step is refused at save time instead of making
 * every later `experience_train` throw.
 */
function actionProblem(a: Record<string, unknown>): string | null {
    switch (a["kind"]) {
        case "click":
            return isTarget(a["target"]) ? null : "a click needs a target element";
        case "setValue":
            if (!isTarget(a["target"])) return "a setValue needs a target element";
            return typeof a["value"] === "string" ? null : "a setValue needs a string value (empty for a secret)";
        case "keys":
            if (typeof a["keys"] !== "string" || !a["keys"]) return "a keys action needs its keys";
            return a["target"] == null || isTarget(a["target"]) ? null : "a keys target must be an element";
        default:
            return "the action kind must be click, setValue or keys";
    }
}

/** Accept a recorder's episode, or say what is wrong with it. */
export function asEpisode(raw: unknown, newId: () => string): Episode | string {
    const e = raw as Partial<Episode> | null;
    if (!e || typeof e !== "object") return "episode must be an object.";
    if (typeof e.name !== "string" || !e.name.trim()) return "episode.name is required.";
    if (typeof e.program !== "string" || !e.program) return "episode.program is required.";
    if (!Array.isArray(e.steps)) return "episode.steps must be an array.";
    for (const [i, s] of e.steps.entries()) {
        if (!s || typeof s.id !== "string" || !s.action || !ACTION_KINDS.has(s.action.kind)) {
            return `episode.steps[${i}] needs an id and an action of kind click, setValue or keys.`;
        }
        const problem = actionProblem(s.action as unknown as Record<string, unknown>);
        if (problem) return `episode.steps[${i}]: ${problem}.`;
    }
    // A secret field records THAT it was filled, never the value: whatever a recorder
    // sent, a secret value is not stored (and so never returned by episode_get).
    const steps = e.steps.map((s) =>
        s.action.kind === "setValue" && s.action.secret ? { ...s, action: { ...s.action, value: "" } } : s,
    );
    const id = typeof e.id === "string" && e.id ? e.id : newId();
    if (!isSafeExperienceId(id)) return "episode.id may use letters, digits, '-' and '_' only.";
    return {
        id,
        name: e.name.trim(),
        program: e.program,
        args: Array.isArray(e.args) ? e.args.map(String) : [],
        startedAt: typeof e.startedAt === "number" ? e.startedAt : Date.now(),
        endedAt: typeof e.endedAt === "number" ? e.endedAt : null,
        steps,
    };
}

/** This machine's time zone as minutes east of UTC — what the engine's routines expect. */
export function localOffsetMinutes(): number {
    return -new Date().getTimezoneOffset();
}

/** Validate and store a recorded episode. Returns its id. */
export async function saveEpisode(host: OperationsHost, raw: unknown): Promise<OpResult<{ id: string }>> {
    const episode = asEpisode(raw, host.engine.newExperienceId);
    if (typeof episode === "string") return err(episode);
    await host.store.putEpisode(episode);
    return ok({ id: episode.id });
}

/** Train Once over a stored episode — ONE path for preview (no save) and compile (save). */
export async function compileSkill(
    host: OperationsHost,
    episodeId: string,
    edits: ReviewEdits | undefined,
    save: boolean,
) {
    const episode = await host.store.getEpisode(episodeId);
    if (!episode) return err(`No episode "${episodeId}".`);
    const skill = host.engine.trainOnce(episode, edits ?? {});
    if (save) await host.store.putSkill(skill);
    return ok(skill);
}

/** Give a skill a routine, or remove it with `null`. */
export async function scheduleSkill(host: OperationsHost, id: string, routine: Omit<SkillRoutine, "lastRunAt"> | null) {
    const skill = await host.store.getSkill(id);
    if (!skill) return err(`No skill "${id}".`);
    // Routine values are stored in plain JSON, so a secret parameter's value is
    // never kept here — secrets come from the host's vault at run time.
    const secret = new Set(skill.params.filter((p) => p.secret).map((p) => p.name));
    const values = routine ? Object.fromEntries(Object.entries(routine.values).filter(([k]) => !secret.has(k))) : {};
    skill.routine = routine ? { ...routine, values, lastRunAt: skill.routine?.lastRunAt ?? null } : null;
    await host.store.putSkill(skill);
    return ok(skill);
}

/** Skills whose routine is due now, least recently run first. */
export async function skillsDue(host: OperationsHost, tzOffsetMinutes?: number) {
    const offset = typeof tzOffsetMinutes === "number" ? tzOffsetMinutes : localOffsetMinutes();
    return host.engine.dueSkills(await host.store.listSkills(), Date.now(), offset);
}

/** Start a run of a skill; a routine run also marks the routine as run. */
export async function startRun(host: OperationsHost, skillId: string, trigger: string) {
    const skill = await host.store.getSkill(skillId);
    if (!skill) return err(`No skill "${skillId}".`);
    const now = Date.now();
    const run: Run = {
        id: host.engine.newExperienceId(),
        skillId: skill.id,
        skillName: skill.name,
        trigger,
        status: "running",
        startedAt: now,
        steps: [],
    };
    await host.store.putRun(run);
    if (run.trigger === "routine" && skill.routine) {
        skill.routine.lastRunAt = now;
        await host.store.putSkill(skill);
    }
    return ok(run);
}

export interface RunStepInput {
    runId: string;
    idx: number;
    stepId: string;
    outcome: StepOutcome;
    detail?: string | null;
}

/** Append one line to a run's audit trail. */
export async function stepRun(host: OperationsHost, input: RunStepInput) {
    const logged = await host.store.appendRunStep(input.runId, {
        idx: input.idx,
        stepId: input.stepId,
        outcome: input.outcome,
        detail: typeof input.detail === "string" ? input.detail : null,
        at: Date.now(),
    });
    return logged ? ok({ logged }) : err(`No run "${input.runId}".`);
}

/** Close a run with its final status. */
export async function finishRun(host: OperationsHost, runId: string, status: RunStatus, error?: string | null) {
    const run = await host.store.getRun(runId);
    if (!run) return err(`No run "${runId}".`);
    run.status = status;
    run.endedAt = Date.now();
    run.error = typeof error === "string" ? error : null;
    await host.store.putRun(run);
    return ok(run);
}
