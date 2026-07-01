/**
 * tests/gatherers.test.ts — the workspace-presence evidence rule shared by the
 * IDE self-correction loop, the proof harness, and tests.
 */

import { workspacePresenceGatherer } from "../src/cognition/gatherers.js";
import type { EvidenceContext } from "../src/cognition/types.js";

const listing = (entries: string[]) => () => Promise.resolve(entries);
// The presence gatherer ignores the context, so a stub satisfies the signature.
const CTX = { claim: { subjectKey: "x", content: "y" } } as unknown as EvidenceContext;

test("supports the claim when required entries are present and forbidden ones are gone", async () => {
  const g = workspacePresenceGatherer({
    list: listing(["a.ts", "b.ts"]),
    mustExist: ["a.ts"],
    mustBeAbsent: ["old.ts"],
  });
  const r = await g(CTX);
  expect(r.supportsNew).toBe(true);
  expect(r.notes[0]).toContain("a.ts");
  expect(r.notes[1]).toContain("old.ts");
});

test("rejects the claim when a required entry is missing", async () => {
  const g = workspacePresenceGatherer({
    list: listing(["b.ts"]),
    mustExist: ["a.ts"],
  });
  expect((await g(CTX)).supportsNew).toBe(false);
});

test("rejects the claim when a forbidden entry is still present", async () => {
  const g = workspacePresenceGatherer({
    list: listing(["old.ts"]),
    mustBeAbsent: ["old.ts"],
  });
  expect((await g(CTX)).supportsNew).toBe(false);
});

test("no rules (both undefined) → vacuously supported, with em-dash notes", async () => {
  const g = workspacePresenceGatherer({ list: listing([]) });
  const r = await g(CTX);
  expect(r.supportsNew).toBe(true);
  expect(r.notes[0]).toContain("—");
  expect(r.notes[1]).toContain("—");
});
