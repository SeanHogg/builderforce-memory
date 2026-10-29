/**
 * src/knowledge/findings.test.ts
 * partitionSeverity: the severity a partition-violation finding carries for each verdict.
 */

import { partitionSeverity } from './findings.js';

describe('partitionSeverity', () => {
    it('an upward edge is an error', () => {
        expect(partitionSeverity('upward')).toBe('error');
    });

    it('a skipped layer is a warning', () => {
        expect(partitionSeverity('skips')).toBe('warn');
    });
});
