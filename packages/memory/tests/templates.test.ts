/**
 * tests/templates.test.ts — workflow template registry lookup + clone-for-edit.
 */

import { WORKFLOW_TEMPLATES, getTemplate, cloneTemplate } from "../src/workflow/templates.js";

test("getTemplate returns a known template and undefined for an unknown id", () => {
  expect(getTemplate(WORKFLOW_TEMPLATES[0]!.id)?.id).toBe(WORKFLOW_TEMPLATES[0]!.id);
  expect(getTemplate("does-not-exist")).toBeUndefined();
});

test("cloneTemplate on an unknown id returns undefined", () => {
  expect(cloneTemplate("does-not-exist")).toBeUndefined();
});

test("cloneTemplate with no overrides derives a -custom id/name, no description, and deep-copies steps", () => {
  const base = WORKFLOW_TEMPLATES[0]!;
  const clone = cloneTemplate(base.id)!;
  expect(clone.id).toBe(`${base.id}-custom`);
  expect(clone.name).toBe(`${base.name} (custom)`);
  // No override → the base's description carries over via the spread (ternary false branch).
  expect(clone.description).toBe(base.description);
  // Steps are cloned, not shared: mutating the clone must not touch the template.
  expect(clone.steps).not.toBe(base.steps);
  clone.steps[0]!.params = { mutated: true };
  expect(base.steps[0]!.params).not.toEqual({ mutated: true });
});

test("cloneTemplate honors explicit id/name/description/steps overrides", () => {
  const base = WORKFLOW_TEMPLATES[0]!;
  const clone = cloneTemplate(base.id, {
    id: "my-flow",
    name: "My Flow",
    description: "custom desc",
    steps: [{ id: "only", type: "benchmark" }],
  })!;
  expect(clone.id).toBe("my-flow");
  expect(clone.name).toBe("My Flow");
  expect(clone.description).toBe("custom desc");
  expect(clone.steps).toHaveLength(1);
  expect(clone.steps[0]!.id).toBe("only");
});
