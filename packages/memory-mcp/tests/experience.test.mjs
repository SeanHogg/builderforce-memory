/**
 * Experience over MCP: the durable host (experience.json + episodes/), the tools a
 * recorder drives (save → compile → schedule → run → audit), cross-process freshness,
 * and training the private model through the engine's adaptation recipe.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { buildMemoryTools, createExperienceHost, nextExperienceVersion, EXPERIENCE_ENV } from "../dist/index.js";
import { EvermindLM, EvermindModelPackage, BPETokenizer, experienceCorpus } from "@seanhogg/builderforce-memory-engine";

const ON = { [EXPERIENCE_ENV]: "1" };

function tmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "bfmem-exp-"));
}

/** A writable backend stub — experience tools register only on a writable server. */
function backend() {
  return { recall: async () => [], get: async () => undefined, recallByTag: async () => [], remember: async () => {}, forget: async () => {} };
}

const el = (name, controlType) => ({ name, automationId: "", controlType, className: "", windowTitle: "Invoices", processName: "ledger.exe", relX: 0, relY: 0 });

function episode(id = "ep1") {
  return {
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
  };
}

async function toolsFor(memoryFile, env = ON) {
  const host = await createExperienceHost({ memoryFile, env });
  const tools = buildMemoryTools(backend(), { experience: host });
  const call = async (name, args = {}) => {
    const t = tools.find((x) => x.name === name);
    assert.ok(t, `tool ${name} is registered`);
    const r = await t.handler(args);
    return { error: !!r.isError, text: r.content[0].text, json: r.isError ? null : JSON.parse(r.content[0].text) };
  };
  return { host, tools, call };
}

test("experience is off unless switched on, and never on a read-only server", async () => {
  const dir = tmpDir();
  assert.equal(await createExperienceHost({ memoryFile: path.join(dir, "memory.json"), env: {} }), null);
  const host = await createExperienceHost({ memoryFile: path.join(dir, "memory.json"), env: ON });
  assert.ok(host);
  const ro = buildMemoryTools(backend(), { experience: host, writable: false }).map((t) => t.name);
  assert.ok(!ro.includes("episode_save"));
});

test("record → compile → schedule → run → audit, persisted beside memory.json", async () => {
  const dir = tmpDir();
  const memoryFile = path.join(dir, "memory.json");
  const { call, host } = await toolsFor(memoryFile);

  assert.equal((await call("episode_save", { episode: episode() })).json.id, "ep1");
  assert.ok((await call("episode_save", { episode: { ...episode(), id: "../escape" } })).error);
  assert.ok((await call("episode_save", { episode: { name: "x", program: "y", steps: [{ id: "a", action: { kind: "hover" } }] } })).error);

  const preview = (await call("skill_preview", { episodeId: "ep1", edits: { name: "Monthly invoice" } })).json;
  assert.equal(preview.name, "Monthly invoice");
  assert.equal((await call("skill_list")).json.length, 0, "preview does not save");

  const skill = (await call("skill_compile", { episodeId: "ep1" })).json;
  assert.deepEqual(skill.params.map((p) => p.name), ["amount"]);
  assert.equal(skill.steps[2].requiresApproval, true);

  await call("skill_schedule", { id: skill.id, routine: { schedule: { every: "minutes", minutes: 5 }, values: { amount: "10" }, enabled: true } });
  assert.deepEqual((await call("skills_due")).json.map((s) => s.id), [skill.id]);

  const run = (await call("run_start", { skillId: skill.id, trigger: "routine" })).json;
  assert.equal((await call("skills_due")).json.length, 0, "a routine run is not due again at once");
  // Synapse sends explicit nulls for an absent detail/error.
  assert.ok(!(await call("run_step", { runId: run.id, idx: 0, stepId: "s1", outcome: "ok", detail: null })).error);
  assert.ok((await call("run_step", { runId: "nope", idx: 0, stepId: "s1", outcome: "ok" })).error);
  const done = (await call("run_finish", { runId: run.id, status: "succeeded" })).json;
  assert.equal(done.status, "succeeded");
  assert.equal(done.steps.length, 1);

  const onDisk = JSON.parse(fs.readFileSync(host.file, "utf8"));
  assert.equal(onDisk.schema, "evermind.experience/1");
  assert.equal(onDisk.skills.length, 1);
  assert.equal(path.dirname(host.file), dir);
  const info = (await call("experience_info")).json;
  assert.equal(info.memoryFile, memoryFile);
  assert.equal(info.skills, 1);
});

test("a second process sees the first one's writes, and forgetting removes screenshots", async () => {
  const dir = tmpDir();
  const memoryFile = path.join(dir, "memory.json");
  const a = await toolsFor(memoryFile);
  const b = await toolsFor(memoryFile);

  await a.call("episode_save", { episode: episode("ep2") });
  assert.equal((await b.call("episode_get", { id: "ep2" })).json.name, "Enter an invoice");

  const shots = path.join(a.host.episodesDir, "ep2");
  fs.mkdirSync(shots, { recursive: true });
  fs.writeFileSync(path.join(shots, "s1.png"), "png");
  assert.equal((await b.call("episode_forget", { id: "ep2" })).json.forgotten, true);
  assert.equal(fs.existsSync(shots), false);
  assert.equal((await a.call("episode_list")).json.length, 0);

  await a.call("episode_save", { episode: episode("ep3") });
  fs.mkdirSync(path.join(a.host.episodesDir, "ep3"), { recursive: true });
  assert.equal((await b.call("experience_forget_all", { confirm: true })).json.forgotten, true);
  assert.equal((await a.call("episode_list")).json.length, 0);
  assert.equal(fs.existsSync(path.join(a.host.episodesDir, "ep3")), false);
  assert.equal(fs.existsSync(a.host.episodesDir), true);
});

test("experience_train adapts the private model once per procedure and keeps the previous one", async () => {
  const dir = tmpDir();
  const memoryFile = path.join(dir, "memory.json");
  const modelFile = path.join(dir, "mine.evermind");
  const tok = new BPETokenizer();
  tok.train(experienceCorpus([episode()], []) + " the invoice amount send new");
  const lm = new EvermindLM({ vocabSize: tok.vocabSize, seed: 11 });
  fs.writeFileSync(modelFile, new Uint8Array(EvermindModelPackage.fromLM(lm, { name: "mine", version: "7", card: { description: "private" }, tokenizer: tok }).toBlob()));

  const { call } = await toolsFor(memoryFile, { ...ON, BUILDERFORCE_MEMORY_MODEL: modelFile });
  await call("episode_save", { episode: episode() });
  assert.equal((await call("experience_train", { dryRun: true })).json.pending, 1);

  const first = (await call("experience_train")).json;
  assert.equal(first.status, "trained");
  assert.equal(first.version, "7+exp1");
  assert.equal(first.learned, 1);
  assert.ok(fs.existsSync(`${modelFile}.prev`));
  const bytes = fs.readFileSync(modelFile);
  const reloaded = EvermindModelPackage.fromBlob(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  assert.equal(reloaded.manifest.version, "7+exp1");
  assert.equal(reloaded.validate().ok, true);

  assert.equal((await call("experience_train")).json.status, "nothing_new");
  await call("skill_compile", { episodeId: "ep1" });
  const second = (await call("experience_train")).json;
  assert.equal(second.version, "7+exp2", "the reviewed skill is new; its raw episode is superseded");
});

test("training without a model says how to configure one", async () => {
  const { call } = await toolsFor(path.join(tmpDir(), "memory.json"));
  const r = await call("experience_train");
  assert.ok(r.error);
  assert.match(r.text, /BUILDERFORCE_MEMORY_MODEL/);
});

test("nextExperienceVersion", () => {
  assert.equal(nextExperienceVersion("511"), "511+exp1");
  assert.equal(nextExperienceVersion("511+exp9"), "511+exp10");
});

test("a malformed step is refused at save time, and a secret value is never stored", async () => {
  const dir = tmpDir();
  const { call } = await toolsFor(path.join(dir, "memory.json"));

  const noValue = episode("bad1");
  noValue.steps[1] = { id: "s2", atMs: 5, action: { kind: "setValue", target: el("Amount", "Edit") } };
  const refused = await call("episode_save", { episode: noValue });
  assert.equal(refused.error, true);
  assert.match(refused.text, /steps\[1\]/);

  const noTarget = episode("bad2");
  noTarget.steps[0] = { id: "s1", atMs: 0, action: { kind: "click" } };
  assert.equal((await call("episode_save", { episode: noTarget })).error, true);

  const withSecret = episode("sec1");
  withSecret.steps[1] = { id: "s2", atMs: 5, action: { kind: "setValue", target: el("Password", "Edit"), value: "hunter2", secret: true } };
  assert.equal((await call("episode_save", { episode: withSecret })).error, false);
  const stored = (await call("episode_get", { id: "sec1" })).json;
  assert.equal(stored.steps[1].action.value, "");
  assert.ok(!fs.readFileSync(path.join(dir, "experience.json"), "utf8").includes("hunter2"));

  // A recorder that already sends an empty secret value is accepted as-is.
  const emptySecret = episode("sec2");
  emptySecret.steps[1] = { id: "s2", atMs: 5, action: { kind: "setValue", target: el("Password", "Edit"), value: "", secret: true } };
  assert.equal((await call("episode_save", { episode: emptySecret })).error, false);
});

test("forgetting everything removes only the known episode folders", async () => {
  const dir = tmpDir();
  const { call, host } = await toolsFor(path.join(dir, "memory.json"));
  await call("episode_save", { episode: episode("ep9") });
  fs.mkdirSync(path.join(host.episodesDir, "ep9"), { recursive: true });
  const unrelated = path.join(host.episodesDir, "not-an-episode");
  fs.mkdirSync(unrelated, { recursive: true });
  fs.writeFileSync(path.join(unrelated, "keep.txt"), "mine");

  assert.equal((await call("experience_forget_all", { confirm: true })).json.forgotten, true);
  assert.equal(fs.existsSync(path.join(host.episodesDir, "ep9")), false);
  assert.equal(fs.readFileSync(path.join(unrelated, "keep.txt"), "utf8"), "mine");
});

test("a snapshot in an unknown schema is refused, never overwritten", async () => {
  const dir = tmpDir();
  const file = path.join(dir, "experience.json");
  const future = JSON.stringify({ schema: "evermind.experience/99", episodes: [{ id: "x" }] });
  fs.writeFileSync(file, future);
  const { call } = await toolsFor(path.join(dir, "memory.json"));
  assert.equal((await call("episode_save", { episode: episode("ep1") })).error, true);
  assert.equal(fs.readFileSync(file, "utf8"), future);
});
