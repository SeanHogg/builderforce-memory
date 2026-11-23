/**
 * src/contract/dist.imports.contract.test.ts — every relative import in dist/ resolves to a file that is in dist/.
 *
 * 2026.10.82 shipped `dist/working/transcript.js` importing `./runElision.js` while `runElision` was not in the
 * package, so every consumer of the working transcript failed at load. `package.json` `files` ships all of
 * `dist`, so "in dist/" is "in the tarball". Like dist.contract, this needs a build (CI has one).
 */

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const DIST = join(process.cwd(), 'dist');
const distBuilt = existsSync(DIST);

const jsFiles = (dir: string): string[] =>
    readdirSync(dir).flatMap((name) => {
        const full = join(dir, name);
        return statSync(full).isDirectory() ? jsFiles(full) : full.endsWith('.js') ? [full] : [];
    });

const RELATIVE_IMPORT = /(?:\bfrom\s*|\bimport\s*\(?\s*|\bimport\s+)['"](\.{1,2}\/[^'"]+)['"]/g;

describe('dist contract: relative imports in dist/ resolve inside dist/', () => {
    if (!distBuilt && !process.env.CI) {
        it('is skipped: dist/ is not built (run `npm run build` first)', () => {
            console.warn('dist.imports.contract: dist/ is missing, so relative imports were not checked.');
        });
        return;
    }

    it('no dist file imports a sibling that is not shipped', () => {
        expect(distBuilt).toBe(true);
        const missing: string[] = [];
        for (const file of jsFiles(DIST)) {
            const text = readFileSync(file, 'utf8');
            for (const match of text.matchAll(RELATIVE_IMPORT)) {
                const target = resolve(dirname(file), match[1]);
                if (!existsSync(target) && !existsSync(`${target}.js`) && !existsSync(join(target, 'index.js'))) {
                    missing.push(`${file.slice(DIST.length + 1)} imports ${match[1]}`);
                }
            }
        }
        expect(missing).toEqual([]);
    });
});
