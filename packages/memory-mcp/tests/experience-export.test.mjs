/**
 * The in-process experience export (`@seanhogg/builderforce-memory-mcp/experience`):
 * a gate-free host and the operations the MCP tools share, with no env and no MCP.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  openExperienceHost,
  FileExperienceStore,
  trainExperience,
  asEpisode,
  saveEpisode,
  compileSkill,
  scheduleSkill,
  skillsDue,
  startRun,
  stepRun,
  finishRun,
} from "../dist/experience/index.js";

function tmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "bfmem-expexp-"));
}

const el = (name, controlType) => ({ name, automationId: "", controlType, className: "", windowTitle: "Invoices", processName: "ledger.exe", relX: 0, relY: 0 });

const episode = (id = "ep1") => ({
  id,
  name: "Enter an invoice",
  program: "C:/apps/ledger.exe",
  args: [],
  startedAt: 1,
  steps: [
    { id: "s1", atMs: 0, action: { kind: "click", target: el("New invoice", "Button") } },
    { id: "s2", atMs: 5, action: { kind: "setValue", target: el("Amount", "Edit"), value: "250.00" } },
    { id: "s3", atMs: 9, action: { kind: "click", target: el("Send invoice", "Button") } },
  ],
});

async function open() {
  const dir = tmpDir();
  const memoryFile = path.join(dir, "memory.json");
  const prev = process.env.BUILDERFORCE_MEMORY_EXPERIENCE;
  delete process.env.BUILDERFORCE_MEMORY_EXPERIENCE;
  try {
    const host = await openExperienceHost({ memoryFile });
    return { dir, memoryFile, host };
  } finally {
    if (prev !== undefined) process.env.BUILDERFORCE_MEMORY_EXPERIENCE = prev;
  }
}

test("the factory needs no env and returns a working host beside the memory file", async () => {
  const { dir, memoryFile, host } = await open();
  assert.ok(host);
  assert.equal(host.memoryFile, memoryFile);
  assert.equal(path.dirname(host.file), dir);
  assert.ok(host.store instanceof FileExperienceStore);
  assert.ok(fs.existsSync(host.episodesDir));
  assert.equal(host.modelFile, undefined);
  assert.equal(await host.modelVersion(), null);
  assert.equal(typeof trainExperience, "function");
});

test("an injected engine is used instead of the dynamic import", async () => {
  const real = await import("@seanhogg/builderforce-memory-engine");
  const calls = [];
  const engine = {
    trainOnce: real.trainOnce,
    dueSkills: real.dueSkills,
    newExperienceId: real.newExperienceId,
    experienceOverview: real.experienceOverview,
    adaptPackage: real.adaptPackage,
    parseSnapshot: real.parseSnapshot,
    InMemoryExperienceStore: class extends real.InMemoryExperienceStore {
      constructor(...args) {
        calls.push("InMemoryExperienceStore");
        super(...args);
      }
    },
  };
  const dir = tmpDir();
  const host = await openExperienceHost({ memoryFile: path.join(dir, "memory.json"), engine });
  assert.equal(host.engine, engine);
  assert.deepEqual(calls, ["InMemoryExperienceStore"]);
  assert.ok(host.store instanceof FileExperienceStore);
});

test("an injected engine missing capabilities is rejected like a loaded one", async () => {
  const dir = tmpDir();
  await assert.rejects(openExperienceHost({ memoryFile: path.join(dir, "memory.json"), engine: {} }), /Train Once and adaptation/);
});

test("a modelFile that does not exist still opens, and loadModel resolves null", async () => {
  const dir = tmpDir();
  const modelFile = path.join(dir, "missing.evermind");
  const host = await openExperienceHost({ memoryFile: path.join(dir, "memory.json"), modelFile });
  assert.equal(host.modelFile, modelFile);
  assert.equal(await host.loadModel(), null);
});

test("asEpisode refuses a malformed episode and a secret value is not kept", () => {
  assert.equal(typeof asEpisode({ name: "x", program: "y", steps: [{ id: "a", action: { kind: "hover" } }] }, () => "n"), "string");
  const e = asEpisode(
    { name: "x", program: "y", steps: [{ id: "a", action: { kind: "setValue", target: el("Pw", "Edit"), value: "hunter2", secret: true } }] },
    () => "gen1",
  );
  assert.equal(e.id, "gen1");
  assert.equal(e.steps[0].action.value, "");
});

test("save an episode, compile a skill from it, list both", async () => {
  const { host } = await open();
  const saved = await saveEpisode(host, episode());
  assert.deepEqual(saved, { ok: true, value: { id: "ep1" } });
  assert.equal((await saveEpisode(host, { ...episode(), id: "../escape" })).ok, false);
  assert.equal((await host.store.listEpisodes()).length, 1);

  const preview = await compileSkill(host, "ep1", { name: "Monthly invoice" }, false);
  assert.equal(preview.value.name, "Monthly invoice");
  assert.equal((await host.store.listSkills()).length, 0, "preview does not save");

  const compiled = await compileSkill(host, "ep1", undefined, true);
  assert.equal(compiled.ok, true);
  assert.deepEqual(compiled.value.params.map((p) => p.name), ["amount"]);
  assert.equal((await host.store.listSkills()).length, 1);
  assert.deepEqual(await compileSkill(host, "nope", undefined, true), { ok: false, error: 'No episode "nope".' });
});

test("run_start / run_step / run_finish round-trip, persisted to experience.json", async () => {
  const { host } = await open();
  await saveEpisode(host, episode());
  const skill = (await compileSkill(host, "ep1", undefined, true)).value;

  assert.equal((await startRun(host, "missing", "manual")).ok, false);
  const run = (await startRun(host, skill.id, "manual")).value;
  assert.equal(run.status, "running");

  assert.equal((await stepRun(host, { runId: run.id, idx: 0, stepId: "s1", outcome: "ok", detail: null })).ok, true);
  assert.equal((await stepRun(host, { runId: "nope", idx: 0, stepId: "s1", outcome: "ok" })).ok, false);

  const done = await finishRun(host, run.id, "succeeded");
  assert.equal(done.value.status, "succeeded");
  assert.equal(done.value.steps.length, 1);
  assert.equal(done.value.error, null);
  assert.equal((await finishRun(host, "nope", "failed")).ok, false);

  const onDisk = JSON.parse(fs.readFileSync(host.file, "utf8"));
  assert.equal(onDisk.runs.length, 1);
  assert.equal(onDisk.runs[0].status, "succeeded");
});

test("skills_due honours a routine, and a routine run is not due again at once", async () => {
  const { host } = await open();
  await saveEpisode(host, episode());
  const skill = (await compileSkill(host, "ep1", undefined, true)).value;

  assert.equal((await skillsDue(host)).length, 0, "no routine, nothing due");
  const scheduled = await scheduleSkill(host, skill.id, {
    schedule: { every: "minutes", minutes: 5 },
    values: { amount: "10" },
    enabled: true,
  });
  assert.equal(scheduled.ok, true);
  assert.deepEqual((await skillsDue(host)).map((s) => s.id), [skill.id]);

  await startRun(host, skill.id, "routine");
  assert.equal((await skillsDue(host)).length, 0);

  const removed = await scheduleSkill(host, skill.id, null);
  assert.equal(removed.value.routine, null);
  assert.equal((await scheduleSkill(host, "nope", null)).ok, false);
});
