/**
 * src/contract/dist.contract.test.ts — every `exports` entry of package.json points at files the build produced.
 *
 * The `./knowledge/pack` export once shipped pointing at nothing until a consumer ran. CI builds before it
 * runs the tests, so `dist/` exists there. Locally, without a build, the test says so and does nothing —
 * unless `CI` is set, where a missing `dist/` is a failure (the build step did not run).
 */

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const PACKAGE_DIR = process.cwd();
const pkg = JSON.parse(readFileSync(join(PACKAGE_DIR, 'package.json'), 'utf8')) as {
    main?: string; module?: string; types?: string; exports: Record<string, { types?: string; import?: string }>;
};
const distBuilt = existsSync(join(PACKAGE_DIR, 'dist'));
const exists = (rel: string | undefined) => !!rel && existsSync(join(PACKAGE_DIR, rel));

describe('dist contract: package.json exports resolve in dist/', () => {
    if (!distBuilt && !process.env.CI) {
        it('is skipped: dist/ is not built (run `npm run build` first)', () => {
            console.warn('dist.contract: dist/ is missing, so export paths were not checked.');
        });
        return;
    }

    it('dist/ exists', () => {
        expect(distBuilt).toBe(true);
    });

    it('main, module and types resolve', () => {
        const missing = [pkg.main, pkg.module, pkg.types].filter((p) => p && !exists(p));
        expect(missing).toEqual([]);
    });

    it('every exports entry has its `import` and `types` file', () => {
        const missing: string[] = [];
        for (const [subpath, entry] of Object.entries(pkg.exports)) {
            if (!exists(entry.import)) missing.push(`${subpath}: import ${entry.import} is not in dist/`);
            if (!exists(entry.types)) missing.push(`${subpath}: types ${entry.types} is not in dist/`);
        }
        expect(missing).toEqual([]);
    });
});
