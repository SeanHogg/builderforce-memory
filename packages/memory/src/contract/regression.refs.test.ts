/**
 * src/contract/regression.refs.test.ts — `pathOfRef` / `fileRefOf` split `src:<rootId>::<path>` on the FIRST '::', so a root id
 * ending in ':' (the linked repo's `cloud:<id>:`) read `src:cloud:5:::docs/a.md` as path `:docs/a.md`. They now follow
 * `splitSourceRef`, the same grammar as Builderforce.ai's agent-tools `rootId.ts`. Each case pinned by one `regression:` test.
 */

import { fileRefOf, pathOfRef, splitSourceRef, srcRef, type NodeRef } from '../knowledge/index.js';

describe('regression: source ref grammar', () => {
    it('regression: a root id ending in ":" keeps its colon; the path does not start with one', () => {
        const ref = srcRef('cloud:5:', 'docs/a.md');
        expect(ref).toBe('src:cloud:5:::docs/a.md');
        expect(splitSourceRef(ref)).toEqual({ rootId: 'cloud:5:', path: 'docs/a.md' });
        expect(pathOfRef(ref)).toBe('docs/a.md');
        expect(pathOfRef(srcRef('cloud:5::', '.eslintrc'))).toBe('.eslintrc');
        expect(fileRefOf(`${ref}#intro` as NodeRef)).toBe(ref);
    });

    it('regression: a path containing "::" stays whole (only the first "::" after the root separates)', () => {
        const ref = srcRef('vscode:f-1', 'src/a::b.ts');
        expect(splitSourceRef(ref)).toEqual({ rootId: 'vscode:f-1', path: 'src/a::b.ts' });
        expect(pathOfRef(ref)).toBe('src/a::b.ts');
        expect(fileRefOf(`${ref}#s` as NodeRef)).toBe(ref);
    });

    it('regression: a known root id splits by its exact prefix, so even a path starting with ":" is exact', () => {
        expect(splitSourceRef('src:cloud:5:::x', ['cloud:5'])).toEqual({ rootId: 'cloud:5', path: ':x' });
        expect(splitSourceRef('src:cloud:5:::x')).toEqual({ rootId: 'cloud:5:', path: 'x' });
    });

    it('regression: "#section" handling of src refs is unchanged (pathOfRef drops it, splitSourceRef keeps the full path)', () => {
        const ref = srcRef('r', 'docs/a.md');
        expect(pathOfRef(`${ref}#intro`)).toBe('docs/a.md');
        expect(fileRefOf(`${ref}#intro` as NodeRef)).toBe(ref);
        expect(fileRefOf(ref)).toBe(ref);
        expect(splitSourceRef(srcRef('r', 'src/C#/a.cs'))?.path).toBe('src/C#/a.cs');
        expect(pathOfRef(srcRef('r', 'src/C#/a.cs'))).toBe('src/C');
    });

    it('regression: fact: and ent: refs are not source refs and pass through unchanged', () => {
        for (const ref of ['fact:claim:1', 'ent:x::y', 'fact:a#b'] as NodeRef[]) {
            expect(splitSourceRef(ref)).toBeUndefined();
            expect(pathOfRef(ref)).toBeUndefined();
            expect(fileRefOf(ref)).toBe(ref);
        }
        expect(splitSourceRef('src:no-separator')).toBeUndefined();
        expect(splitSourceRef('src:::x')).toBeUndefined();
    });
});
