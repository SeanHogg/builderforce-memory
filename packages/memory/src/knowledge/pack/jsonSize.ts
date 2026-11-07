/** knowledge/pack/jsonSize.ts — the bytes the same rows would take as JSON: the baseline a pack's size is shown against (PRD 41 A11 rule 6). */

import type { Pack } from './types.js';

const ENCODER = new TextEncoder();

/** UTF-8 bytes of the pack's rows as JSON (frame fields and manifest sections excluded). */
export function jsonBytesOf(pack: Pack): number {
    const { files, extracted, facts, relations, stale, deletes, scope, analysis } = pack;
    return ENCODER.encode(JSON.stringify({ files, extracted, facts, relations, stale, deletes, scope, analysis })).length;
}
