/**
 * src/knowledge/jobDeadline.test.ts
 * A stop of the run settles a job at once, whether it landed before the job began or while it ran.
 */

import { createDeadlines } from './jobDeadline.js';

const never = () => new Promise<never>(() => undefined);

describe('createDeadlines: stop', () => {
    it('regression: a parent aborted before the call rejects at once, not at the deadline', async () => {
        const parent = new AbortController();
        parent.abort(new Error('stopped'));
        const withDeadline = createDeadlines(parent.signal);
        await expect(withDeadline(60_000, never)).rejects.toThrow('stopped');
    });

    it('regression: a parent aborted mid-job rejects at once, not at the deadline', async () => {
        const parent = new AbortController();
        const withDeadline = createDeadlines(parent.signal);
        const job = withDeadline(60_000, never);
        const settled = expect(job).rejects.toThrow('stopped');
        parent.abort(new Error('stopped'));
        await settled;
    });
});
