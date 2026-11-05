/**
 * ingest/fences.ts — the ONE CommonMark-ish fenced-code scanner. The markdown parser uses
 * it so a `#` line inside a fence is not a heading, and `withoutCode` (docsExtract.ts)
 * uses it to blank code before links are read, so the two can never disagree.
 */

/** Blockquote markers (`>`), then the fence indent; a list item's content may indent further. */
const BQ = String.raw`((?:[ \t]*>)*)`;
const OPENER = new RegExp(String.raw`^${BQ}( *)(\`{3,}|~{3,})(.*)$`);
const CLOSER = new RegExp(String.raw`^${BQ}( *)(\`{3,}|~{3,})[ \t]*\r?$`);
const LIST_ITEM = /^\s*(?:[-*+]|\d{1,9}[.)])(?:\s|$)/;

/** True when a line begins (or sits inside a list item and so may indent) — see {@link fenceMask}. */
export function isListItemLine(line: string): boolean {
    return LIST_ITEM.test(line);
}

/**
 * Per line: true when it is a fence opener, a fence closer or inside a fence. An opener
 * indents at most 3 spaces unless it sits inside a list item (then any indent); a backtick
 * opener's info string holds no backtick; a closer is the same marker, at least as long,
 * followed by whitespace only. An unclosed fence runs to the end of the lines.
 */
export function fenceMask(lines: readonly string[]): boolean[] {
    const mask: boolean[] = new Array(lines.length).fill(false);
    let open: { char: string; len: number } | undefined;
    let inList = false;
    lines.forEach((line, i) => {
        if (open) {
            mask[i] = true;
            const c = CLOSER.exec(line);
            if (c && (c[2]!.length <= 3 || inList) && c[3]![0] === open.char && c[3]!.length >= open.len) open = undefined;
            return;
        }
        const m = OPENER.exec(line);
        if (m && (m[2]!.length <= 3 || inList) && !(m[3]![0] === '`' && m[4]!.includes('`'))) {
            mask[i] = true;
            open = { char: m[3]![0]!, len: m[3]!.length };
            return;
        }
        if (isListItemLine(line)) inList = true;
        else if (line.trim() !== '' && !/^[ \t]/.test(line)) inList = false;
    });
    return mask;
}
