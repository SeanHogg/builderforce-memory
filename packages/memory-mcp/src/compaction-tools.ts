/**
 * The lossless compaction pair — `memory_compact` and `memory_restore` — split out
 * of tools.ts the same way the gateway and experience families are.
 *
 * `memory_compact` stubs a fact only after archiving its original and proving the
 * local Evermind absorbed it; `memory_restore` brings back the text of stubs that
 * predate archiving (`contentLost`). The rules are compaction.ts / restore.ts; this
 * module is only their I/O against a MemoryBackend.
 */

import { z } from "zod";
import type { MemoryBackend, RecallHit, RememberInput } from "./backend.js";
import { DEFAULT_STUB_CHARS, planLosslessCompaction, type CompactionSkip } from "./compaction.js";
import { parseRestoreSource, planRestore, restoredArchive, type RestoreTarget } from "./restore.js";
import { fail, ok, type MemoryTool } from "./tool-core.js";
import type { MemoryToolsOptions } from "./tools.js";

const SKIP_REASONS: Record<CompactionSkip["reason"], string> = {
    not_found: "not found",
    already_compacted: "already compacted",
    not_smaller: "already shorter than its stub",
    no_probe: "no local Evermind model to prove it was absorbed (set BUILDERFORCE_MEMORY_MODEL)",
    probe_failed: "the Evermind model could not answer a recall question about it — not absorbed yet",
    no_archive: "this backend cannot archive the original, so stubbing would lose it",
};

/** One-line compaction receipt: what shrank, what it saved, and what was left alone. */
function renderCompaction(compacted: number, bytesSaved: number, skipped: CompactionSkip[]): string {
    const lines = [`Compacted ${compacted} memory(ies), ${bytesSaved} character(s) reclaimed.`];
    for (const s of skipped) lines.push(`• skipped ${s.key} — ${SKIP_REASONS[s.reason]}`);
    return lines.join("\n");
}

/** `memory_restore` input: inline JSON always; a file path only where the host may read its disk. */
function restoreSchema(allowFile: boolean): z.ZodRawShape {
    const shape: z.ZodRawShape = {
        json: z.string().optional().describe("The export itself, as JSON text (array, or JSON Lines)."),
    };
    if (allowFile) shape["file"] = z.string().optional().describe("Path to a JSON export file to restore from.");
    return shape;
}

/** Write a batch through the backend's batch path when it has one. */
async function writeAll(backend: MemoryBackend, writes: RememberInput[]): Promise<void> {
    if (writes.length === 0) return;
    if (backend.rememberMany) await backend.rememberMany(writes);
    else for (const w of writes) await backend.remember!(w);
}

/** Registers `memory_compact` and `memory_restore`. Caller checks the backend is writable. */
export function appendCompactionTools(
    tools: MemoryTool[],
    backend: MemoryBackend,
    opts: Pick<MemoryToolsOptions, "recallProbe" | "readRestoreFile">,
): void {
    // memory_compact — the FIRST-CLASS compaction path. It writes through the live
    // store (get → plan → archive → remember), so a durable backend's in-memory state
    // and its on-disk snapshot agree afterwards. It is LOSSLESS: each original is
    // archived before its body is stubbed, and only facts the local Evermind answers
    // a recall probe about are stubbed at all (see planLosslessCompaction).
    tools.push({
        name: "memory_compact",
        description:
            "Shrink memories that have already been absorbed into the local Evermind model down to " +
            "a one-line pointer stub, reclaiming the context they were eating. Each fact is first " +
            "checked with a recall question put to the model; facts it cannot answer stay as they " +
            "are. The full original is archived and memory_recall / memory_get still return it, so " +
            "nothing is lost. Already-compacted keys are skipped, so repeating the call is safe.",
        inputSchema: {
            keys: z.array(z.string()).min(1).describe("Exact keys of the memories to compact."),
            stub: z
                .string()
                .optional()
                .describe(
                    "Explicit replacement body applied to every listed key. Omit to generate " +
                        "'[absorbed→Evermind vN] <first line>' from each entry's own first line.",
                ),
            version: z
                .number()
                .int()
                .min(0)
                .optional()
                .describe("Version of the model that absorbed these facts; appears in the generated stub."),
            maxChars: z
                .number()
                .int()
                .min(20)
                .max(500)
                .optional()
                .describe(`Max characters of the generated pointer line (default ${DEFAULT_STUB_CHARS}).`),
        },
        handler: async (args) => {
            try {
                const raw = args["keys"];
                const keys = Array.isArray(raw) ? raw.map((k) => String(k)).filter(Boolean) : [];
                if (keys.length === 0) return fail("keys is required.");

                // ONE read pass: the hit carries both the body the plan reasons about
                // and the tags/importance the rewrite must preserve. A stub resolved to
                // its archive reports the STORED stub in `hit.stub` — that is what the
                // plan must see, or an archived fact would be compacted again.
                const hits = new Map<string, RecallHit>();
                for (const key of keys) {
                    const hit = await backend.get(key);
                    if (hit) hits.set(key, hit);
                }

                const plan = await planLosslessCompaction(
                    keys.map((key) => {
                        const hit = hits.get(key);
                        return { key, content: hit ? (hit.stub ?? hit.content) : undefined };
                    }),
                    {
                        stub: typeof args["stub"] === "string" ? (args["stub"] as string) : undefined,
                        version: args["version"] == null ? undefined : Number(args["version"]),
                        maxChars: args["maxChars"] == null ? undefined : Number(args["maxChars"]),
                        probe: opts.recallProbe ?? null,
                        canArchive: typeof backend.archive === "function",
                    },
                );

                if (plan.writes.length > 0) {
                    // Originals FIRST, durably — only then may the bodies become stubs.
                    await backend.archive!(plan.writes.map((w) => w.archive));
                    // Preserve tags/importance — compaction shortens a body, it does not
                    // reclassify the fact.
                    await writeAll(
                        backend,
                        plan.writes.map((w) => {
                            const prev = hits.get(w.key);
                            return { key: w.key, content: w.content, tags: prev?.tags, importance: prev?.importance };
                        }),
                    );
                }

                return ok(renderCompaction(plan.writes.length, plan.bytesSaved, plan.skipped));
            } catch (err) {
                return fail(`compact failed: ${String(err)}`);
            }
        },
    });

    // memory_restore — the way back for stubs whose original was never archived
    // (`contentLost`). The text is matched to a stub by key, or by its first line
    // against the stub's pointer, then archived beside the stub (the stub stays, so
    // the token saving stays) — or written back as the body on a backend that
    // cannot archive.
    tools.push({
        name: "memory_restore",
        description:
            "Restore the full text of memories marked '(content lost)' — compaction stubs whose " +
            "original was never kept — from an export. Accepts [{key, content}] (a memory export) " +
            "or the Evermind coordinator's [{id, text}] record dump; records are matched by key, " +
            "else by their first line against the stub. Restored memories recall in full again.",
        inputSchema: restoreSchema(!!opts.readRestoreFile),
        handler: async (args) => {
            try {
                const file = typeof args["file"] === "string" ? (args["file"] as string) : "";
                let text = typeof args["json"] === "string" ? (args["json"] as string) : "";
                if (file && opts.readRestoreFile) text = await opts.readRestoreFile(file);
                if (!text.trim()) return fail(opts.readRestoreFile ? "file or json is required." : "json is required.");

                const records = parseRestoreSource(text);
                if (records.length === 0) return fail("no restorable records (need {key, content} or {id, text} rows).");

                // Targets: the lost stubs (all of them when the backend can list them,
                // else just the keys the records name).
                const keys = new Set<string>(backend.lostKeys ? await backend.lostKeys() : []);
                for (const r of records) if (r.key) keys.add(r.key);
                const targets: RestoreTarget[] = [];
                const hits = new Map<string, RecallHit>();
                for (const key of keys) {
                    const hit = await backend.get(key);
                    if (!hit?.contentLost) continue;
                    hits.set(key, hit);
                    targets.push({ key, stub: hit.content });
                }

                const plan = planRestore(records, targets);
                if (plan.matches.length > 0) {
                    if (backend.archive) {
                        await backend.archive(plan.matches.map((m) => restoredArchive(m, hits.get(m.key)!.content)));
                    } else {
                        await writeAll(
                            backend,
                            plan.matches.map((m) => {
                                const prev = hits.get(m.key);
                                return { key: m.key, content: m.content, tags: prev?.tags, importance: prev?.importance };
                            }),
                        );
                    }
                }

                const lines = [
                    `Restored ${plan.matches.length} memory(ies) of ${targets.length} with lost content.`,
                    ...plan.matches.map((m) => `• ${m.key} (matched by ${m.via})`),
                ];
                if (plan.ambiguous.length > 0) lines.push(`Ambiguous (left alone): ${plan.ambiguous.join(", ")}`);
                if (plan.unmatched > 0) lines.push(`${plan.unmatched} record(s) matched no lost memory.`);
                return ok(lines.join("\n"));
            } catch (err) {
                return fail(`restore failed: ${String(err)}`);
            }
        },
    });
}
