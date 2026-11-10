/** knowledge/pack/sections/index.ts — every row section of BFKP v1, in the order they are written (STR and HASH are written first by encode.ts). */

import { deleteSection, analysisSection, bucketSection, fileSection, leafSection, measuredSection, scopeSection, staleSection } from './rows.js';
import { extrSection } from './extr.js';
import { factSection, relationSection } from './facts.js';
import { siteSection } from './sites.js';
import type { SectionCodec } from './columns.js';

/**
 * SITE comes right after EXTR: it attaches call sites to the edges EXTR decoded; MEAS right after SCOPE, whose `measured` it
 * sets (a reader runs these in this order).
 */
export const ROW_SECTIONS: readonly SectionCodec[] = [
    fileSection, extrSection, siteSection, factSection, relationSection, staleSection, deleteSection, scopeSection, measuredSection, analysisSection,
    leafSection, bucketSection,
];

export { TAG } from './rows.js';
export type { SectionCodec } from './columns.js';
