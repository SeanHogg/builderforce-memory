/**
 * Experience tools — what Evermind has DONE, over MCP.
 *
 * The same framework-neutral {@link MemoryTool}s as the memory tools, registered
 * beside them when experience is on. Every result is JSON: Synapse drives these as a
 * program (record → review → compile → run → audit), and a model reads JSON as well.
 * The domain rules — Train Once, approval gates, routines, the corpus — are the
 * engine's; these handlers only validate input and move data through the store.
 */

import { z } from "zod";
import type { Episode, ReviewEdits, Run, RunStatus, SkillRoutine, StepOutcome } from "@seanhogg/builderforce-memory-engine";
import { fail, okJson, type MemoryTool, type ToolResult } from "../tool-core.js";
import { isSafeExperienceId } from "./file-store.js";
import type { ExperienceHost } from "./host.js";
import { pendingDocuments, trainExperience } from "./train.js";

const DEFAULT_RUNS = 50;
const MAX_RUNS_RETURNED = 500;

const editsShape = z
    .object({
        name: z.string().nullable().optional(),
        removed: z.array(z.string()).optional(),
        fixed: z.array(z.string()).optional(),
        approvals: z.record(z.boolean()).optional(),
        paramNames: z.record(z.string()).optional(),
    })
    .optional()
    .describe("Review edits keyed by recorded step id.");

const scheduleShape = z.union([
    z.object({ every: z.literal("minutes"), minutes: z.number().int().min(1).max(10_080) }),
    z.object({ every: z.literal("day"), hour: z.number().int().min(0).max(23), minute: z.number().int().min(0).max(59) }),
]);

const ACTION_KINDS = new Set(["click", "setValue", "keys"]);

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
    }
    const id = typeof e.id === "string" && e.id ? e.id : newId();
    if (!isSafeExperienceId(id)) return "episode.id may use letters, digits, '-' and '_' only.";
    return {
        id,
        name: e.name.trim(),
        program: e.program,
        args: Array.isArray(e.args) ? e.args.map(String) : [],
        startedAt: typeof e.startedAt === "number" ? e.startedAt : Date.now(),
        endedAt: typeof e.endedAt === "number" ? e.endedAt : null,
        steps: e.steps,
    };
}

/** A tool whose thrown errors come back as tool errors named after it. */
function tool(
    name: string,
    description: string,
    inputSchema: z.ZodRawShape,
    run: (args: Record<string, unknown>) => Promise<ToolResult>,
): MemoryTool {
    return {
        name,
        description,
        inputSchema,
        handler: async (args) => {
            try {
                return await run(args);
            } catch (err) {
                return fail(`${name} failed: ${String(err)}`);
            }
        },
    };
}

const str = (v: unknown): string => (typeof v === "string" ? v : "");

/** This machine's time zone as minutes east of UTC — what the engine's routines expect. */
function localOffsetMinutes(): number {
    return -new Date().getTimezoneOffset();
}

export function buildExperienceTools(host: ExperienceHost): MemoryTool[] {
    const { store, engine } = host;

    const episodeTools = [
        tool(
            "experience_info",
            "Where experience is kept and how much there is: the memory and experience snapshot files, the screenshot folder, the private model it trains, and counts.",
            {},
            async () => {
                const snap = await store.snapshot();
                return okJson({
                    memoryFile: host.memoryFile,
                    file: host.file,
                    episodesDir: host.episodesDir,
                    modelFile: host.modelFile ?? null,
                    episodes: snap.episodes.length,
                    skills: snap.skills.length,
                    runs: snap.runs.length,
                    learned: Object.keys(snap.learned).length,
                });
            },
        ),
        tool(
            "episode_save",
            "Save a recorded demonstration (an episode: a program, and the UI steps a person took in it). Returns its id. Screenshots belong in <episodesDir>/<id>/.",
            { episode: z.record(z.unknown()).describe("The episode, in the engine's Episode shape.") },
            async (args) => {
                const episode = asEpisode(args["episode"], engine.newExperienceId);
                if (typeof episode === "string") return fail(episode);
                await store.putEpisode(episode);
                return okJson({ id: episode.id });
            },
        ),
        tool("episode_list", "List recorded demonstrations, newest first (summaries only).", {}, async () =>
            okJson(await store.listEpisodes()),
        ),
        tool("episode_get", "Fetch one demonstration with all of its steps.", { id: z.string() }, async (args) => {
            const e = await store.getEpisode(str(args["id"]));
            return e ? okJson(e) : fail(`No episode "${str(args["id"])}".`);
        }),
        tool(
            "episode_forget",
            "Delete a demonstration and its screenshots. Skills compiled from it keep working.",
            { id: z.string() },
            async (args) => okJson({ forgotten: await store.deleteEpisode(str(args["id"])) }),
        ),
        tool(
            "experience_forget_all",
            "Delete every demonstration, screenshot, skill and run. Facts (memories) and the model are not touched. Requires confirm: true.",
            { confirm: z.literal(true) },
            async () => {
                await store.clear();
                return okJson({ forgotten: true });
            },
        ),
    ];

    /** Train Once over a stored episode — ONE path for preview (no save) and compile (save). */
    async function compile(args: Record<string, unknown>, save: boolean): Promise<ToolResult> {
        const episode = await store.getEpisode(str(args["episodeId"]));
        if (!episode) return fail(`No episode "${str(args["episodeId"])}".`);
        const skill = engine.trainOnce(episode, (args["edits"] as ReviewEdits | undefined) ?? {});
        if (save) await store.putSkill(skill);
        return okJson(skill);
    }

    const skillTools = [
        tool(
            "skill_preview",
            "Show the skill Train Once would compile from a demonstration with these review edits, without saving it.",
            { episodeId: z.string(), edits: editsShape },
            (a) => compile(a, false),
        ),
        tool(
            "skill_compile",
            "Compile a demonstration into a runnable skill (Train Once) and save it.",
            { episodeId: z.string(), edits: editsShape },
            (a) => compile(a, true),
        ),
        tool("skill_list", "List skills, newest first.", {}, async () => okJson(await store.listSkills())),
        tool("skill_get", "Fetch one skill.", { id: z.string() }, async (args) => {
            const s = await store.getSkill(str(args["id"]));
            return s ? okJson(s) : fail(`No skill "${str(args["id"])}".`);
        }),
        tool("skill_forget", "Delete a skill (its run history stays).", { id: z.string() }, async (args) =>
            okJson({ forgotten: await store.deleteSkill(str(args["id"])) }),
        ),
        tool(
            "skill_schedule",
            "Give a skill a routine (every N minutes, or daily at a local time, with values for its non-secret parameters), or pass null to remove it.",
            {
                id: z.string(),
                routine: z
                    .object({
                        schedule: scheduleShape,
                        values: z.record(z.string()).default({}),
                        enabled: z.boolean().default(true),
                    })
                    .nullable(),
            },
            async (args) => {
                const skill = await store.getSkill(str(args["id"]));
                if (!skill) return fail(`No skill "${str(args["id"])}".`);
                const r = args["routine"] as Omit<SkillRoutine, "lastRunAt"> | null;
                skill.routine = r ? { ...r, lastRunAt: skill.routine?.lastRunAt ?? null } : null;
                await store.putSkill(skill);
                return okJson(skill);
            },
        ),
        tool(
            "skills_due",
            "Skills whose routine is due now, least recently run first.",
            {
                tzOffsetMinutes: z
                    .number()
                    .int()
                    .min(-840)
                    .max(840)
                    .optional()
                    .describe("Local offset east of UTC; defaults to this machine's."),
            },
            async (args) => {
                const tz = args["tzOffsetMinutes"];
                const offset = typeof tz === "number" ? tz : localOffsetMinutes();
                return okJson(engine.dueSkills(await store.listSkills(), Date.now(), offset));
            },
        ),
    ];

    const runTools = [
        tool(
            "run_start",
            "Start a run of a skill. A routine run also marks the routine as run, so it is not due again until its next slot.",
            { skillId: z.string(), trigger: z.enum(["manual", "routine"]) },
            async (args) => {
                const skill = await store.getSkill(str(args["skillId"]));
                if (!skill) return fail(`No skill "${str(args["skillId"])}".`);
                const now = Date.now();
                const run: Run = {
                    id: engine.newExperienceId(),
                    skillId: skill.id,
                    skillName: skill.name,
                    trigger: str(args["trigger"]),
                    status: "running",
                    startedAt: now,
                    steps: [],
                };
                await store.putRun(run);
                if (run.trigger === "routine" && skill.routine) {
                    skill.routine.lastRunAt = now;
                    await store.putSkill(skill);
                }
                return okJson(run);
            },
        ),
        tool(
            "run_step",
            "Append one line to a run's audit trail.",
            {
                runId: z.string(),
                idx: z.number().int().min(0),
                stepId: z.string(),
                outcome: z.enum(["ok", "fallback", "failed", "approved", "denied"]),
                detail: z.string().nullable().optional(),
            },
            async (args) => {
                const logged = await store.appendRunStep(str(args["runId"]), {
                    idx: Number(args["idx"]),
                    stepId: str(args["stepId"]),
                    outcome: args["outcome"] as StepOutcome,
                    detail: typeof args["detail"] === "string" ? args["detail"] : null,
                    at: Date.now(),
                });
                return logged ? okJson({ logged }) : fail(`No run "${str(args["runId"])}".`);
            },
        ),
        tool(
            "run_finish",
            "Close a run with its final status.",
            { runId: z.string(), status: z.enum(["succeeded", "failed", "stopped", "denied"]), error: z.string().nullable().optional() },
            async (args) => {
                const run = await store.getRun(str(args["runId"]));
                if (!run) return fail(`No run "${str(args["runId"])}".`);
                run.status = args["status"] as RunStatus;
                run.endedAt = Date.now();
                run.error = typeof args["error"] === "string" ? args["error"] : null;
                await store.putRun(run);
                return okJson(run);
            },
        ),
        tool(
            "run_list",
            "Recent runs, newest first.",
            { limit: z.number().int().min(1).max(MAX_RUNS_RETURNED).optional() },
            async (args) => okJson(await store.listRuns(Number(args["limit"] ?? DEFAULT_RUNS))),
        ),
        tool("run_get", "Fetch one run with its audit trail.", { id: z.string() }, async (args) => {
            const r = await store.getRun(str(args["id"]));
            return r ? okJson(r) : fail(`No run "${str(args["id"])}".`);
        }),
    ];

    const learnTools = [
        tool(
            "experience_train",
            "Teach the private Evermind model the procedures it has not learned yet (each skill, and each demonstration not yet compiled into one). Pass dryRun to only count them. The previous model is kept beside the new one.",
            { dryRun: z.boolean().optional() },
            async (args) => {
                if (!host.modelFile) return fail("No private Evermind model is configured (set BUILDERFORCE_MEMORY_MODEL).");
                const model = await host.loadModel();
                if (!model) return fail(`Could not load an evermind-lm package and its tokenizer from ${host.modelFile}.`);
                if (args["dryRun"] === true) {
                    const version = model.pkg.manifest.version ?? "0";
                    const pending = (await pendingDocuments(store, engine, version)).length;
                    return okJson({ status: "dry_run", pending, version });
                }
                return okJson(await trainExperience(store, engine, model, host.modelFile));
            },
        ),
    ];

    return [...episodeTools, ...skillTools, ...runTools, ...learnTools];
}
