/**
 * tests/regression.checkSchedulerSubpath.test.ts — `./ml/tasks/check-scheduler` is read by Builderforce's run-delta
 * `predict` code, which the VS Code webview and the Worker bundle. The engine ROOT loads the WebGPU kernels, training
 * and the rest of the engine, so the subpath must stay light — the same hold as `./ml/classical`
 * (memory `regression.statsEngineFree`). This walks the real import graph (TypeScript's own pre-processor, type-only
 * imports included) and holds:
 *   - the subpath never reaches the engine's root `index.ts`, never leaves the engine for a package or a `node:`
 *     builtin, and never touches the WGSL kernels;
 *   - it exposes the task row, its model card and the feature helpers;
 *   - the package exports it.
 */
import ts from "typescript";

// The engine compiles without node typings (`types: ["@webgpu/types", "jest"]`): the builtins are loaded untyped.
interface Fs { existsSync(p: string): boolean; readFileSync(p: string, enc: "utf8"): string }
interface Path { dirname(p: string): string; join(...p: string[]): string; resolve(...p: string[]): string; sep: string }
const fs: Fs = await import("node:" + "fs");
const path: Path = await import("node:" + "path");
const { existsSync, readFileSync } = fs;
const { dirname, join, resolve, sep } = path;

const PACKAGE_DIR: string = (globalThis as unknown as { process: { cwd(): string } }).process.cwd();
const ENGINE_SRC = join(PACKAGE_DIR, "src");
const ENGINE_ROOT = join(ENGINE_SRC, "index.ts");
const ENTRY = join(ENGINE_SRC, "ml", "tasks", "check_scheduler.ts");

/** Every module specifier a file names (imports, re-exports, dynamic imports). */
function specifiersOf(file: string): string[] {
  return ts.preProcessFile(readFileSync(file, "utf8"), true, true).importedFiles.map((f) => f.fileName);
}

function resolveRelative(from: string, specifier: string): string {
  const base = resolve(dirname(from), specifier.replace(/\.js$/, ""));
  for (const candidate of [`${base}.ts`, join(base, "index.ts")]) if (existsSync(candidate)) return candidate;
  throw new Error(`unresolved ${specifier} from ${from}`);
}

/** The relative-import closure of `entry`, and every bare specifier met on the way. */
function closure(entry: string): { files: Set<string>; bare: string[] } {
  const files = new Set<string>();
  const bare: string[] = [];
  const pending = [entry];
  while (pending.length) {
    const file = pending.pop()!;
    if (files.has(file)) continue;
    files.add(file);
    for (const specifier of specifiersOf(file)) {
      if (specifier.startsWith(".")) pending.push(resolveRelative(file, specifier));
      else bare.push(specifier);
    }
  }
  return { files, bare };
}

describe("regression: ./ml/tasks/check-scheduler never loads the engine root", () => {
  test("regression: the subpath is light (no root index, no packages, no node:, no kernels)", () => {
    const { files, bare } = closure(ENTRY);
    expect(files.has(ENGINE_ROOT)).toBe(false);
    expect(bare).toEqual([]);
    expect([...files].filter((f) => f.startsWith(join(ENGINE_SRC, "kernels") + sep))).toEqual([]);
    expect([...files].filter((f) => f.startsWith(join(ENGINE_SRC, "lm") + sep))).toEqual([]);
  });

  test("it exposes the task row, its model card and the feature helpers", async () => {
    const m = await import("../src/ml/tasks/check_scheduler.js");
    expect(m.CHECK_SCHEDULER_TASK.id).toBe("run.check-scheduler");
    expect(m.CHECK_SCHEDULER_MODEL_CARD.taskId).toBe(m.CHECK_SCHEDULER_TASK.id);
    expect(m.checkSchedulerFeatures({ check: "benchmark", dimensionsTouched: [], blastRadiusBand: "low", changedFiles: 0 }))
      .toHaveLength(m.CHECK_SCHEDULER_FEATURES.length);
  });

  test("the engine package exports that subpath", () => {
    const pkg = JSON.parse(readFileSync(join(PACKAGE_DIR, "package.json"), "utf8")) as { exports: Record<string, { import?: string; types?: string }> };
    expect(pkg.exports["./ml/tasks/check-scheduler"]).toEqual({
      types: "./dist/ml/tasks/check_scheduler.d.ts", import: "./dist/ml/tasks/check_scheduler.js",
    });
  });
});
