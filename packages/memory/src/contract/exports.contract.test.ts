/**
 * src/contract/exports.contract.test.ts — the PUBLIC SURFACE contract.
 *
 * For every subpath of package.json `exports`, the exact sorted list of runtime names the entry exports is
 * checked in beside this file (`exports.snapshot.json`). Consumers (agent-tools, the VS Code host, the API)
 * import these names, so:
 *
 *   - a REMOVED or RENAMED name is a breaking change: this test fails and says so. It needs a CHANGELOG
 *     "Breaking" note and the consumers bumped in the same release.
 *   - an ADDED name fails too, until the snapshot is updated deliberately (so no export slips out unnoticed).
 *   - a subpath added to / dropped from `exports` fails the same way.
 *
 * Update the snapshot on purpose with `UPDATE_EXPORTS_SNAPSHOT=1 npm test -- src/contract/exports`. A missing
 * snapshot is written on the first LOCAL run (commit it); under CI (`CI` set) a missing snapshot is a failure.
 * Types are not runtime names: their contract is `types.contract.test.ts`.
 */

import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

interface ExportEntry { types?: string; import?: string }

const PACKAGE_DIR = process.cwd();
const SNAPSHOT_PATH = join(PACKAGE_DIR, 'src', 'contract', 'exports.snapshot.json');

const pkg = JSON.parse(readFileSync(join(PACKAGE_DIR, 'package.json'), 'utf8')) as { exports: Record<string, ExportEntry> };

/** `./dist/knowledge/pack/index.js` → `../knowledge/pack/index` (this file lives in src/contract). */
const sourceEntryOf = (entry: ExportEntry): string => `../${entry.import!.replace(/^\.\/dist\//, '').replace(/\.js$/, '')}`;

async function currentSurface(): Promise<Record<string, string[]>> {
    const out: Record<string, string[]> = {};
    for (const subpath of Object.keys(pkg.exports).sort()) {
        const ns = await import(sourceEntryOf(pkg.exports[subpath]!)) as Record<string, unknown>;
        out[subpath] = Object.keys(ns).sort();
    }
    return out;
}

const diff = (before: readonly string[], after: readonly string[]) => ({
    removed: before.filter((n) => !after.includes(n)),
    added: after.filter((n) => !before.includes(n)),
});

describe('public surface contract: package.json exports', () => {
    it('every export has an import and a types path', () => {
        for (const [subpath, entry] of Object.entries(pkg.exports)) {
            expect({ subpath, import: typeof entry.import }).toEqual({ subpath, import: 'string' });
            expect({ subpath, types: typeof entry.types }).toEqual({ subpath, types: 'string' });
        }
    });

    it('exports exactly the runtime names of the checked-in snapshot', async () => {
        const current = await currentSurface();
        const update = !!process.env.UPDATE_EXPORTS_SNAPSHOT;
        if (update || (!existsSync(SNAPSHOT_PATH) && !process.env.CI)) {
            writeFileSync(SNAPSHOT_PATH, `${JSON.stringify(current, null, 2)}\n`);
            return;
        }
        if (!existsSync(SNAPSHOT_PATH)) {
            throw new Error('src/contract/exports.snapshot.json is missing: run the contract tests locally once and commit the file it writes.');
        }
        const snapshot = JSON.parse(readFileSync(SNAPSHOT_PATH, 'utf8')) as Record<string, string[]>;

        const breaking: string[] = [];
        const changed: string[] = [];
        for (const subpath of Object.keys(snapshot)) {
            if (!(subpath in current)) { breaking.push(`${subpath}: the whole subpath was removed from package.json exports`); continue; }
            const { removed, added } = diff(snapshot[subpath]!, current[subpath]!);
            if (removed.length) breaking.push(`${subpath}: removed or renamed ${removed.join(', ')}`);
            if (added.length) changed.push(`${subpath}: added ${added.join(', ')}`);
        }
        for (const subpath of Object.keys(current)) if (!(subpath in snapshot)) changed.push(`${subpath}: new subpath`);

        if (breaking.length) {
            throw new Error(
                `BREAKING CHANGE to the public surface — consumers import these names.\n  ${breaking.join('\n  ')}\n`
                + 'This needs a CHANGELOG "Breaking" note and the consumers updated in the same release. If it is intended, '
                + 'update the snapshot with UPDATE_EXPORTS_SNAPSHOT=1.',
            );
        }
        if (changed.length) {
            throw new Error(
                `The public surface grew:\n  ${changed.join('\n  ')}\n`
                + 'If that is intended, update src/contract/exports.snapshot.json with UPDATE_EXPORTS_SNAPSHOT=1 and note it in the CHANGELOG.',
            );
        }
        expect(current).toEqual(snapshot);
    });
});
