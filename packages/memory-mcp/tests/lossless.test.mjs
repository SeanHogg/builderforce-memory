/**
 * Lossless compaction, lost-content recovery, ranking signals, the shared memory-file
 * resolver and rotating backups.
 *
 *  - compaction stubs a fact ONLY after a recall probe passes, and only after the
 *    full original is archived beside the snapshot (memory.archive.json);
 *  - recall/get resolve a stub back to its archived original; the archive survives
 *    forget, eviction and an external rewrite of the snapshot;
 *  - a stub with no archive is `contentLost`: labelled everywhere (tools + hook),
 *    and `memory_restore` brings its text back from a [{key,content}] or {id,text} dump;
 *  - every overwrite of a shared file keeps the last 3 versions as .bak.1..3;
 *  - installer, servers and hook resolve ONE memory file.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  buildMemoryTools,
  createLocalMemoryStoreBackend,
  planLosslessCompaction,
  probeQuestionFor,
  probePasses,
  parseRestoreSource,
  planRestore,
  stubVersion,
  defaultArchiveFile,
  parseArchive,
  mergeArchive,
  CONTENT_LOST_LABEL,
  SharedJsonFile,
  backupFile,
  resolveMemoryFile,
  memoryFileCandidates,
  defaultMemoryFile,
  bfmemHookSource,
  MEMORY_FILE_ENV,
} from "../dist/index.js";

let db = 0;
const tempDir = () => fs.mkdtempSync(path.join(os.tmpdir(), "bfmem-lossless-"));
const tempSnapshot = (name = "memory") => path.join(tempDir(), `${name}.json`);
const readJson = (file) => JSON.parse(fs.readFileSync(file, "utf8"));
const toolOf = (tools, name) => tools.find((t) => t.name === name);
const textOf = (res) => res.content[0].text;
const quiet = () => {};

const FIRST = "The gateway mints tenant JWTs from the workspace key.";
const LONG = [FIRST, "", "Every adapter converts money to cents at its own edge before the ledger."].join("\n");
const knows = (text) => ({ answer: async () => text });

// ── pure rules ───────────────────────────────────────────────────────────────

test("probeQuestionFor blanks the most identifying token, deterministically", () => {
  const q = probeQuestionFor("arch.jwt", LONG);
  assert.equal(q.expected, "JWTs", "capitals/digits beat plain length");
  assert.ok(q.question.includes("tenant ____ from"));
  assert.ok(!q.question.includes("JWTs"), "the answer must not be in the question");
  assert.deepEqual(probeQuestionFor("arch.jwt", LONG), q);
  assert.equal(probeQuestionFor("k", "a b c"), null, "nothing worth blanking");
  assert.ok(probePasses("it is jwts", "JWTs"));
  assert.ok(!probePasses(null, "JWTs"));
  assert.ok(!probePasses("tokens", "JWTs"));
});

test("planLosslessCompaction stubs only what the probe proves, and carries the original", async () => {
  const cands = [{ key: "a", content: LONG }];
  const none = await planLosslessCompaction(cands, { version: 2 });
  assert.deepEqual(none.skipped, [{ key: "a", reason: "no_probe" }]);
  assert.equal(none.writes.length, 0);

  const wrong = await planLosslessCompaction(cands, { probe: knows("no idea") });
  assert.deepEqual(wrong.skipped, [{ key: "a", reason: "probe_failed" }]);

  const throws = await planLosslessCompaction(cands, { probe: { answer: async () => { throw new Error("x"); } } });
  assert.deepEqual(throws.skipped, [{ key: "a", reason: "probe_failed" }]);

  const noArchive = await planLosslessCompaction(cands, { probe: knows(LONG), canArchive: false });
  assert.deepEqual(noArchive.skipped, [{ key: "a", reason: "no_archive" }]);

  const ok = await planLosslessCompaction(cands, { probe: knows("JWTs"), version: 4, now: 123 });
  assert.equal(ok.writes.length, 1);
  assert.match(ok.writes[0].content, /^\[absorbed→Evermind v4\] The gateway/);
  assert.deepEqual(ok.writes[0].archive, { key: "a", content: LONG, stub: ok.writes[0].content, version: 4, archivedAt: 123 });
  assert.equal(ok.bytesSaved, LONG.length - ok.writes[0].content.length);
});

test("archive text merges, never prunes, and tolerates corruption", () => {
  assert.equal(defaultArchiveFile("/x/memory.json"), "/x/memory.archive.json");
  assert.deepEqual(parseArchive("not json"), {});
  const one = mergeArchive(null, [{ key: "a", content: "A", stub: "s", version: 1, archivedAt: 1 }]);
  const two = mergeArchive(one, [{ key: "b", content: "B", stub: "s", version: 1, archivedAt: 2 }]);
  assert.deepEqual(Object.keys(parseArchive(two)).sort(), ["a", "b"]);
});

// ── memory_compact through a durable backend ─────────────────────────────────

test("memory_compact with no model compacts nothing and says why", async () => {
  const file = tempSnapshot();
  const backend = await createLocalMemoryStoreBackend({ persistFile: file, dbName: `ll-${db++}`, log: quiet });
  await backend.remember({ key: "arch.money", content: LONG });
  const res = await toolOf(buildMemoryTools(backend), "memory_compact").handler({ keys: ["arch.money"] });
  assert.match(textOf(res), /Compacted 0/);
  assert.match(textOf(res), /no local Evermind model/);
  assert.equal(readJson(file)[0].content, LONG);
  assert.equal(fs.existsSync(defaultArchiveFile(file)), false);
});

test("compaction archives first; recall returns the full text; the archive survives forget, eviction and external rewrites", async () => {
  const file = tempSnapshot();
  const backend = await createLocalMemoryStoreBackend({ persistFile: file, dbName: `ll-${db++}`, maxEntries: 3, log: quiet });
  const tools = buildMemoryTools(backend, { recallProbe: knows(LONG) });
  await backend.remember({ key: "arch.money", content: LONG, importance: 0.9 });

  const res = await toolOf(tools, "memory_compact").handler({ keys: ["arch.money"], version: 9 });
  assert.match(textOf(res), /Compacted 1/);

  // On disk: the stub in the snapshot, the original in the archive.
  assert.match(readJson(file)[0].content, /^\[absorbed→Evermind v9\]/);
  assert.equal(readJson(defaultArchiveFile(file)).facts["arch.money"].content, LONG);

  // Recall matches the stub and answers with the archived original.
  const recall = await toolOf(tools, "memory_recall").handler({ query: "gateway tenant JWTs" });
  assert.ok(textOf(recall).includes("Every adapter converts money"), textOf(recall));
  assert.ok(!textOf(recall).includes(CONTENT_LOST_LABEL));

  // An external rewrite of the snapshot cannot touch the archive.
  const rows = readJson(file);
  fs.writeFileSync(file, JSON.stringify(rows, null, 2) + "\n ");
  assert.equal((await backend.get("arch.money")).content, LONG);

  // Eviction past the cap and an explicit forget leave the archive intact.
  for (let i = 0; i < 5; i++) await backend.remember({ key: `filler.${i}`, content: `filler ${i}`, importance: 1 });
  await backend.forget("arch.money");
  assert.equal(readJson(defaultArchiveFile(file)).facts["arch.money"].content, LONG);
});

// ── lost content + memory_restore ────────────────────────────────────────────

const LOST_STUB = "[absorbed→Evermind v511] SHIPPED 2026-07-12: Import from builderforce-memory in the VS Code Evermind panel";
const LOST_FULL = `${LOST_STUB.split("] ")[1]} — the full original, with every detail the stub dropped.`;

async function lostStore() {
  const file = tempSnapshot();
  fs.writeFileSync(
    file,
    JSON.stringify([
      { key: "project.import", content: LOST_STUB, tags: ["evermind"], importance: 0.8 },
      { key: "project.other", content: "[absorbed→Evermind v3] Gateway mints tenant JWTs from the workspace key" },
      { key: "user.lang", content: "The user prefers TypeScript." },
    ]),
  );
  const logs = [];
  const backend = await createLocalMemoryStoreBackend({ persistFile: file, dbName: `ll-${db++}`, log: (l) => logs.push(l) });
  return { file, backend, logs };
}

test("stubs with no archive load as contentLost, are logged once and labelled on recall", async () => {
  const { backend, logs } = await lostStore();
  assert.equal(logs.length, 1);
  assert.match(logs[0], /2 memory\(ies\) are compaction stubs with no archived original/);
  const hit = await backend.get("project.import");
  assert.equal(hit.contentLost, true);
  assert.equal((await backend.get("user.lang")).contentLost, undefined);
  const tools = buildMemoryTools(backend);
  const got = await toolOf(tools, "memory_get").handler({ key: "project.import" });
  assert.ok(textOf(got).includes(CONTENT_LOST_LABEL));
});

test("memory_restore matches by key, by mem: id and by text prefix, and clears contentLost", async () => {
  const { file, backend } = await lostStore();
  const exportFile = path.join(path.dirname(file), "export.json");
  fs.writeFileSync(exportFile, JSON.stringify([{ key: "project.import", content: LOST_FULL }]));
  const tools = buildMemoryTools(backend, { readRestoreFile: (p) => fs.promises.readFile(p, "utf8") });
  const restore = toolOf(tools, "memory_restore");
  assert.ok("file" in restore.inputSchema, "a local host may read an export file");

  const byKey = await restore.handler({ file: exportFile });
  assert.match(textOf(byKey), /Restored 1 memory\(ies\) of 2/);
  const restored = await backend.get("project.import");
  assert.equal(restored.content, LOST_FULL, "recall now returns the full text");
  assert.equal(restored.contentLost, undefined);
  assert.equal(restored.stub, LOST_STUB, "the stub stays — so does the token saving");
  assert.deepEqual(restored.tags, ["evermind"]);
  assert.equal(readJson(file).find((e) => e.key === "project.import").contentLost, undefined);
  assert.equal(readJson(file).find((e) => e.key === "project.other").contentLost, true);

  // Coordinator dump: {id, text}, matched by text prefix (no usable id).
  const dump = JSON.stringify([{ id: "mem:unrelated", text: "Gateway mints tenant JWTs from the workspace key, minted per request." }]);
  const byText = await restore.handler({ json: dump });
  assert.match(textOf(byText), /project\.other \(matched by text\)/);
  assert.equal((await backend.get("project.other")).contentLost, undefined);
});

test("memory_restore without a file reader takes inline JSON only", async () => {
  const { backend } = await lostStore();
  const restore = toolOf(buildMemoryTools(backend), "memory_restore");
  assert.ok(!("file" in restore.inputSchema), "a networked server never reads a caller-named path");
  const res = await restore.handler({ json: JSON.stringify([{ id: "mem:project.import", text: LOST_FULL }]) });
  assert.match(textOf(res), /project\.import \(matched by key\)/);
});

test("restore planning: wrappers, JSON Lines, ambiguity and stub versions", () => {
  assert.deepEqual(parseRestoreSource('{"entries":[{"key":"a","content":"A text"}]}'), [{ key: "a", content: "A text" }]);
  assert.deepEqual(parseRestoreSource('{"id":"mem:b","text":"B text"}\n{"id":"c","text":"[absorbed→Evermind v1] stub"}'), [
    { key: "b", content: "B text" },
  ]);
  const targets = [
    { key: "x", stub: "[absorbed→Evermind v2] Shared pointer line for two" },
    { key: "y", stub: "[absorbed→Evermind v2] Shared pointer line for two" },
  ];
  const plan = planRestore([{ content: "Shared pointer line for two facts" }], targets);
  assert.equal(plan.matches.length, 0);
  assert.deepEqual(plan.ambiguous.sort(), ["x", "y"]);
  assert.equal(stubVersion("[absorbed→Evermind v511] x"), 511);
  assert.equal(stubVersion("plain"), 0);
});

// ── ranking signals ON by default for local recall ───────────────────────────

test("local recall ranks by importance after fusion", async () => {
  const backend = await createLocalMemoryStoreBackend({ dbName: `ll-${db++}` });
  await backend.remember({ key: "low", content: "release checklist for the gateway", importance: 0.05 });
  await backend.remember({ key: "high", content: "release checklist for the gateway", importance: 0.95 });
  const hits = await backend.recall("release checklist gateway", 2);
  assert.equal(hits[0].key, "high");

  const plain = await createLocalMemoryStoreBackend({ dbName: `ll-${db++}`, ranking: false });
  await plain.remember({ key: "low", content: "release checklist for the gateway", importance: 0.05 });
  await plain.remember({ key: "high", content: "release checklist for the gateway", importance: 0.95 });
  assert.equal((await plain.recall("release checklist gateway", 2)).length, 2);
});

test("snapshots persist write time so recency survives a respawn", async () => {
  const file = tempSnapshot();
  const a = await createLocalMemoryStoreBackend({ persistFile: file, dbName: `ll-${db++}`, log: quiet });
  await a.remember({ key: "k", content: "dated fact" });
  const stamped = readJson(file)[0].timestamp;
  assert.equal(typeof stamped, "number");
  const b = await createLocalMemoryStoreBackend({ persistFile: file, dbName: `ll-${db++}`, log: quiet });
  assert.equal((await b.get("k")).timestamp, stamped);
});

// ── rotating backups ─────────────────────────────────────────────────────────

test("every overwrite keeps the previous 3 versions as .bak.1..3", () => {
  const file = tempSnapshot("shared");
  const shared = new SharedJsonFile(file, fs);
  for (let i = 1; i <= 5; i++) shared.write(`v${i}`);
  assert.equal(fs.readFileSync(file, "utf8"), "v5");
  assert.equal(fs.readFileSync(backupFile(file, 1), "utf8"), "v4");
  assert.equal(fs.readFileSync(backupFile(file, 2), "utf8"), "v3");
  assert.equal(fs.readFileSync(backupFile(file, 3), "utf8"), "v2");
  assert.equal(fs.existsSync(backupFile(file, 4)), false);

  const none = tempSnapshot("nobak");
  const off = new SharedJsonFile(none, fs, { backups: 0 });
  off.write("a");
  off.write("b");
  assert.equal(fs.existsSync(backupFile(none, 1)), false);
});

// ── ONE memory-file resolver (installer, servers, hook) ──────────────────────

test("resolveMemoryFile: env, else the location holding memory.json (newest if both), else canonical", () => {
  const home = "/home/u";
  const [canonical, legacy] = memoryFileCandidates(home);
  assert.equal(canonical, defaultMemoryFile(home));
  const probe = (m) => ({ fileMtime: (p) => (p in m ? m[p] : null) });
  const logs = [];
  const log = (l) => logs.push(l);

  assert.equal(resolveMemoryFile({ [MEMORY_FILE_ENV]: "/x.json" }, home, { probe: probe({ [legacy]: 1 }) }), "/x.json");
  assert.equal(resolveMemoryFile({}, home, { probe: probe({}) }), canonical);
  assert.equal(resolveMemoryFile({}, home, { probe: probe({ [legacy]: 5 }), log }), legacy);
  assert.match(logs.pop(), /using the store at/);
  assert.equal(resolveMemoryFile({}, home, { probe: probe({ [canonical]: 1, [legacy]: 9 }), log }), legacy);
  assert.match(logs.pop(), /exists in 2 locations; using .*Left untouched/);
  assert.equal(resolveMemoryFile({}, home, { probe: probe({ [canonical]: 9, [legacy]: 1 }), log }), canonical);
});

/** Run a generated hook script in `mode`, with HOME pointed at `home`. */
function runHook(source, mode, input, home, extraEnv = {}) {
  const dir = tempDir();
  const script = path.join(dir, "bfmem-hook.mjs");
  fs.writeFileSync(script, source);
  const env = { ...process.env, HOME: home, USERPROFILE: home, ...extraEnv };
  delete env[MEMORY_FILE_ENV];
  Object.assign(env, extraEnv);
  const r = spawnSync(process.execPath, [script, mode], { input: JSON.stringify(input), env, encoding: "utf8" });
  return r.stdout;
}

test("the runtime hook resolves the legacy store and labels + demotes lost content", () => {
  const home = tempDir();
  const legacyFile = path.join(home, ".claude", "builderforce-memory", "memory.json");
  fs.mkdirSync(path.dirname(legacyFile), { recursive: true });
  fs.mkdirSync(path.join(home, ".builderforce-memory"), { recursive: true }); // canonical dir, no memory.json
  fs.writeFileSync(
    legacyFile,
    JSON.stringify([
      { key: "lost.one", content: "[absorbed→Evermind v1] deploy pipeline pointer", importance: 1 },
      { key: "kept.one", content: "[absorbed→Evermind v1] deploy pipeline archived", importance: 0.9 },
      { key: "plain.one", content: "deploy pipeline plain fact", importance: 0.7 },
    ]),
  );
  fs.writeFileSync(
    defaultArchiveFile(legacyFile),
    mergeArchive(null, [{ key: "kept.one", content: "full", stub: "s", version: 1, archivedAt: 1 }]),
  );

  const out = runHook(bfmemHookSource(null), "--session", { source: "startup" }, home);
  const lines = out.split("\n").filter((l) => l.startsWith("• "));
  assert.equal(lines.length, 3, out);
  assert.ok(lines.find((l) => l.startsWith("• lost.one")).endsWith(CONTENT_LOST_LABEL));
  assert.ok(!lines.find((l) => l.startsWith("• kept.one")).includes(CONTENT_LOST_LABEL));
  assert.ok(lines.findIndex((l) => l.startsWith("• lost.one")) > lines.findIndex((l) => l.startsWith("• plain.one")),
    "lost content is importance-demoted below intact memories");

  const recall = runHook(bfmemHookSource(null), "--recall", { prompt: "the deploy pipeline pointer", session_id: `s${Date.now()}` }, home);
  assert.ok(recall.includes(`lost.one: [absorbed→Evermind v1] deploy pipeline pointer ${CONTENT_LOST_LABEL}`), recall);
});
