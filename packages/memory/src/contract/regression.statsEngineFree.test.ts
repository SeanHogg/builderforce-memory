/**
 * src/contract/regression.statsEngineFree.test.ts — `./stats` is imported by BROWSER bundles (Builderforce's
 * knowledge-graph-view and frontend). When Ledoit–Wolf moved into memory-engine, `stats/multivariate.ts` re-exported it
 * from the engine ROOT, which loads the WebGPU kernels, training and the rest of the engine into every bundle that
 * wanted one statistic. It now comes through the engine's `./ml/classical` subpath. This walks the real import graph
 * (TypeScript's own pre-processor, not a regex) and holds:
 *   - the stats entry's only bare import is `@seanhogg/builderforce-memory-engine/ml/classical` — never the engine root;
 *   - that subpath's own import graph never reaches the engine's root `index.ts`, never leaves the engine for a package
 *     or a `node:` builtin, and never touches the WGSL kernels.
 */

import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import ts from 'typescript';

const PACKAGE_DIR = process.cwd();
const STATS_ENTRY = join(PACKAGE_DIR, 'src', 'stats', 'index.ts');
const ENGINE_SRC = resolve(PACKAGE_DIR, '..', 'memory-engine', 'src');
const ENGINE_ROOT = join(ENGINE_SRC, 'index.ts');
const ENGINE_CLASSICAL = '@seanhogg/builderforce-memory-engine/ml/classical';
const ENGINE_CLASSICAL_ENTRY = join(ENGINE_SRC, 'ml', 'classical', 'index.ts');

/** Every module specifier a file names (imports, re-exports, dynamic imports). */
function specifiersOf(file: string): string[] {
    return ts.preProcessFile(readFileSync(file, 'utf8'), true, true).importedFiles.map((f) => f.fileName);
}

function resolveRelative(from: string, specifier: string): string {
    const base = resolve(dirname(from), specifier.replace(/\.js$/, ''));
    for (const candidate of [`${base}.ts`, join(base, 'index.ts')]) if (existsSync(candidate)) return candidate;
    throw new Error(`unresolved ${specifier} from ${from}`);
}

/** The relative-import closure of `entry`, and every bare specifier met on the way (with the file naming it). */
function closure(entry: string): { files: Set<string>; bare: Array<{ file: string; specifier: string }> } {
    const files = new Set<string>();
    const bare: Array<{ file: string; specifier: string }> = [];
    const pending = [entry];
    while (pending.length) {
        const file = pending.pop()!;
        if (files.has(file)) continue;
        files.add(file);
        for (const specifier of specifiersOf(file)) {
            if (specifier.startsWith('.')) pending.push(resolveRelative(file, specifier));
            else bare.push({ file: relative(PACKAGE_DIR, file), specifier });
        }
    }
    return { files, bare };
}

describe('regression: ./stats never loads the engine root', () => {
    it('regression: the stats entry reaches the engine only through its ./ml/classical subpath', () => {
        const { bare } = closure(STATS_ENTRY);
        const outside = bare.filter((b) => b.specifier !== ENGINE_CLASSICAL);
        expect(outside).toEqual([]);
    });

    it('regression: the engine ./ml/classical subpath is light (no root index, no packages, no node:, no kernels)', () => {
        const { files, bare } = closure(ENGINE_CLASSICAL_ENTRY);
        expect(files.has(ENGINE_ROOT)).toBe(false);
        expect(bare).toEqual([]);
        const kernels = [...files].filter((f) => f.startsWith(join(ENGINE_SRC, 'kernels') + sep));
        expect(kernels).toEqual([]);
    });

    it('the engine package exports that subpath', () => {
        const pkg = JSON.parse(readFileSync(resolve(ENGINE_SRC, '..', 'package.json'), 'utf8')) as { exports: Record<string, { import?: string }> };
        expect(pkg.exports['./ml/classical']?.import).toBe('./dist/ml/classical/index.js');
    });
});
