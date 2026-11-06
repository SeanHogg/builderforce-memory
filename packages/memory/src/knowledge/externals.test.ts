/**
 * src/knowledge/externals.test.ts
 * The external-service ids: one list, unique, lower-case words, and exported from the knowledge index.
 */

import { describe, expect, it } from '@jest/globals';
import { EXTERNAL_SERVICE_IDS as FROM_INDEX } from './index.js';
import { EXTERNAL_SERVICE_IDS, type ExternalServiceId } from './externals.js';

describe('EXTERNAL_SERVICE_IDS', () => {
    it('lists each service once, as a short lower-case word (they ride in a comma-joined badge)', () => {
        expect(new Set(EXTERNAL_SERVICE_IDS).size).toBe(EXTERNAL_SERVICE_IDS.length);
        for (const id of EXTERNAL_SERVICE_IDS) expect(id).toMatch(/^[a-z]+$/);
    });

    it('is the list the knowledge index re-exports, and its type is its members', () => {
        expect(FROM_INDEX).toBe(EXTERNAL_SERVICE_IDS);
        const id: ExternalServiceId = 'postgres';
        expect(EXTERNAL_SERVICE_IDS).toContain(id);
    });
});
