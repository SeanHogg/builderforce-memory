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
import type { ReviewEdits, RunStatus, SkillRoutine, StepOutcome } from "@seanhogg/builderforce-memory-engine";
import { fail, okJson, type MemoryTool, type ToolResult } from "../tool-core.js";
import type { ExperienceHost } from "./host.js";
import {
    compileSkill,
    finishRun,
    localOffsetMinutes,
    saveEpisode,
    scheduleSkill,
    skillsDue,
    startRun,
    stepRun,
    type OpResult,
} from "./operations.js";
import { CURATION_UNSUPPORTED, experienceIndex, pendingDocuments, trainExperience } from "./train.js";

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

const fromOp = <T>(r: OpResult<T>): ToolResult => (r.ok ? okJson(r.value) : fail(r.error));

/** A tool whose thrown errors come back as tool errors named after it. */
function tool(
    name: string,
    effect: NonNullable<MemoryTool["effect"]>,
    description: string,
    inputSchema: z.ZodRawShape,
    run: (args: Record<string, unknown>) => Promise<ToolResult>,
): MemoryTool {
    return {
        name,
        effect,
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

export function buildExperienceTools(host: ExperienceHost): MemoryTool[] {
    const { store, engine } = host;

    const episodeTools = [
        tool(
            "experience_info",
            "read",
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
            "experience_overview",
            "read",
            "What experience says about the brain, for drawing it: counts per region (hippocampus = demonstrations, basalGanglia = skills, amygdala = approval decisions, hypothalamus = routines, neocortex = procedures the private model has learned), learned vs pending, run outcomes, every adaptation's measured loss, activity per local day, and the newest events.",
            { days: z.number().int().min(1).max(365).optional() },
            async (args) => {
                const version = await host.modelVersion();
                const overview = engine.experienceOverview(await store.snapshot(), {
                    modelIndex: version ? experienceIndex(version) : 0,
                    now: Date.now(),
                    utcOffsetMinutes: localOffsetMinutes(),
                    days: typeof args["days"] === "number" ? args["days"] : undefined,
                });
                return okJson({ modelVersion: version, ...overview });
            },
        ),
        tool(
            "episode_save",
            "write",
            "Save a recorded demonstration (an episode: a program, and the UI steps a person took in it). Returns its id. Screenshots belong in <episodesDir>/<id>/.",
            { episode: z.record(z.unknown()).describe("The episode, in the engine's Episode shape.") },
            async (args) => fromOp(await saveEpisode(host, args["episode"])),
        ),
        tool("episode_list", "read", "List recorded demonstrations, newest first (summaries only).", {}, async () =>
            okJson(await store.listEpisodes()),
        ),
        tool("episode_get", "read", "Fetch one demonstration with all of its steps.", { id: z.string() }, async (args) => {
            const e = await store.getEpisode(str(args["id"]));
            return e ? okJson(e) : fail(`No episode "${str(args["id"])}".`);
        }),
        tool(
            "episode_forget",
            "write",
            "Delete a demonstration and its screenshots. Skills compiled from it keep working.",
            { id: z.string() },
            async (args) => okJson({ forgotten: await store.deleteEpisode(str(args["id"])) }),
        ),
        tool(
            "experience_forget_all",
            "write",
            "Delete every demonstration, screenshot, skill and run. Facts (memories) and the model are not touched. Requires confirm: true.",
            { confirm: z.literal(true) },
            async () => {
                await store.clear();
                return okJson({ forgotten: true });
            },
        ),
    ];

    const compile = async (args: Record<string, unknown>, save: boolean): Promise<ToolResult> =>
        fromOp(await compileSkill(host, str(args["episodeId"]), args["edits"] as ReviewEdits | undefined, save));

    const skillTools = [
        tool(
            "skill_preview",
            "read",
            "Show the skill Train Once would compile from a demonstration with these review edits, without saving it.",
            { episodeId: z.string(), edits: editsShape },
            (a) => compile(a, false),
        ),
        tool(
            "skill_compile",
            "write",
            "Compile a demonstration into a runnable skill (Train Once) and save it.",
            { episodeId: z.string(), edits: editsShape },
            (a) => compile(a, true),
        ),
        tool("skill_list", "read", "List skills, newest first.", {}, async () => okJson(await store.listSkills())),
        tool("skill_get", "read", "Fetch one skill.", { id: z.string() }, async (args) => {
            const s = await store.getSkill(str(args["id"]));
            return s ? okJson(s) : fail(`No skill "${str(args["id"])}".`);
        }),
        tool("skill_forget", "write", "Delete a skill (its run history stays).", { id: z.string() }, async (args) =>
            okJson({ forgotten: await store.deleteSkill(str(args["id"])) }),
        ),
        tool(
            "skill_schedule",
            "write",
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
            async (args) =>
                fromOp(
                    await scheduleSkill(host, str(args["id"]), args["routine"] as Omit<SkillRoutine, "lastRunAt"> | null),
                ),
        ),
        tool(
            "skills_due",
            "read",
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
                return okJson(await skillsDue(host, typeof tz === "number" ? tz : undefined));
            },
        ),
    ];

    const runTools = [
        tool(
            "run_start",
            "write",
            "Start a run of a skill. A routine run also marks the routine as run, so it is not due again until its next slot.",
            { skillId: z.string(), trigger: z.enum(["manual", "routine"]) },
            async (args) => fromOp(await startRun(host, str(args["skillId"]), str(args["trigger"]))),
        ),
        tool(
            "run_step",
            "write",
            "Append one line to a run's audit trail.",
            {
                runId: z.string(),
                idx: z.number().int().min(0),
                stepId: z.string(),
                outcome: z.enum(["ok", "fallback", "failed", "approved", "denied"]),
                detail: z.string().nullable().optional(),
            },
            async (args) =>
                fromOp(
                    await stepRun(host, {
                        runId: str(args["runId"]),
                        idx: Number(args["idx"]),
                        stepId: str(args["stepId"]),
                        outcome: args["outcome"] as StepOutcome,
                        detail: typeof args["detail"] === "string" ? args["detail"] : null,
                    }),
                ),
        ),
        tool(
            "run_finish",
            "write",
            "Close a run with its final status.",
            { runId: z.string(), status: z.enum(["succeeded", "failed", "stopped", "denied"]), error: z.string().nullable().optional() },
            async (args) =>
                fromOp(
                    await finishRun(
                        host,
                        str(args["runId"]),
                        args["status"] as RunStatus,
                        typeof args["error"] === "string" ? args["error"] : null,
                    ),
                ),
        ),
        tool(
            "run_list",
            "read",
            "Recent runs, newest first.",
            { limit: z.number().int().min(1).max(MAX_RUNS_RETURNED).optional() },
            async (args) => okJson(await store.listRuns(Number(args["limit"] ?? DEFAULT_RUNS))),
        ),
        tool("run_get", "read", "Fetch one run with its audit trail.", { id: z.string() }, async (args) => {
            const r = await store.getRun(str(args["id"]));
            return r ? okJson(r) : fail(`No run "${str(args["id"])}".`);
        }),
    ];

    const learnTools = [
        tool(
            "experience_train",
            "write",
            "Teach the private Evermind model the procedures it has not learned yet (each skill, and each demonstration not yet compiled into one). Pass dryRun to only count them. Pass curate to first leave out procedures whose runs mostly fail and near-copies of each other or of recently learned ones (each one left out is listed, and reconsidered next time). The previous model is kept beside the new one.",
            {
                dryRun: z.boolean().optional(),
                curate: z.boolean().optional().describe("Opt-in: filter by run quality and drop near-duplicates before training. Default false."),
            },
            async (args) => {
                if (!host.modelFile) return fail("No private Evermind model is configured (set BUILDERFORCE_MEMORY_MODEL).");
                const model = await host.loadModel();
                if (!model) return fail(`Could not load an evermind-lm package and its tokenizer from ${host.modelFile}.`);
                if (args["dryRun"] === true) {
                    const version = model.pkg.manifest.version ?? "0";
                    const pending = (await pendingDocuments(store, engine, version)).length;
                    return okJson({ status: "dry_run", pending, version });
                }
                const curate = args["curate"] === true;
                if (curate && (!engine.curatePendingExperience || !engine.EvermindTextEmbedder)) return fail(CURATION_UNSUPPORTED);
                return okJson(await trainExperience(store, engine, model, host.modelFile, { curate }));
            },
        ),
    ];

    return [...episodeTools, ...skillTools, ...runTools, ...learnTools];
}
