/**
 * src/contract/regression.contentHash.test.ts — the content identity is EOL-normalized (PRD 41 §B2 step 0), so the same
 * commit checked out on Windows with `core.autocrlf` and on Linux names every file, and the root, identically.
 */

import { contentHash, merkleTree } from '../knowledge/index.js';

const utf8 = (s: string) => new TextEncoder().encode(s);

/** One commit's text files, as git writes them to a Linux working tree (LF). */
const COMMIT: Record<string, string> = {
    'README.md': '# Project\n\nSome prose.\n',
    'src/index.ts': 'export const a = 1;\nexport const b = 2;\n',
    'src/lib/util.ts': 'export function f() {\n    return 42;\n}\n',
};

/** The same files as a Windows `core.autocrlf=true` checkout writes them: every LF a CRLF. */
const autocrlf = (text: string) => text.replace(/\n/g, '\r\n');

const listing = (files: Record<string, string>, eol: (s: string) => string = (s) => s) =>
    Object.entries(files).map(([path, text]) => ({ path, hash: contentHash(utf8(eol(text))) }));

describe('regression: cross-OS content identity', () => {
    it('regression: a Windows autocrlf checkout and a Linux checkout hash differently', () => {
        const linux = listing(COMMIT);
        const windows = listing(COMMIT, autocrlf);
        expect(windows).toEqual(linux);
        expect(merkleTree(windows).root).toBe(merkleTree(linux).root);
    });
});
