/**
 * `@seanhogg/builderforce-memory-mcp/experience` — experience, in-process.
 *
 * What the experience MCP tools do, without MCP and without the opt-in env: a host
 * (the engine, the durable store, the private model) and the operations over it.
 * The memory-engine peer stays optional — `openExperienceHost` rejects, naming it,
 * when it is missing or too old. (The root export's env-gated `createExperienceHost`
 * returns null instead; this one is a different function under a different name.)
 */

/**
 * The gate-free host factory: no `BUILDERFORCE_MEMORY_EXPERIENCE` check, options passed
 * directly. Rejects when the memory-engine peer is missing or too old.
 */
export { openExperienceHost } from "./host.js";
export type { OpenExperienceHostOptions, ExperienceHost, ExperienceEngine, HostFs, HostPath } from "./host.js";

export { FileExperienceStore } from "./file-store.js";
export { trainExperience, CURATION_UNSUPPORTED, pendingDocuments, experienceIndex } from "./train.js";
export type { TrainOutcome, TrainExperienceOptions } from "./train.js";

export {
    asEpisode,
    saveEpisode,
    compileSkill,
    scheduleSkill,
    skillsDue,
    startRun,
    stepRun,
    finishRun,
    localOffsetMinutes,
} from "./operations.js";
export type { OpResult, OperationsHost, RunStepInput } from "./operations.js";
