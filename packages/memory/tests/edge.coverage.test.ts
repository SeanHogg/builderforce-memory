/**
 * tests/edge.coverage.test.ts
 * Remaining edge branches:
 *   - SSMAgent.init() recovering from corrupted persisted history
 *   - SSMAgent.destroy() tolerating a memory write failure
 *   - InferenceRouter audit-log cap + getAuditLog copy
 */

import 'fake-indexeddb/auto';
import { jest } from '@jest/globals';
import { SSMAgent } from '../src/agent/SSMAgent.js';
import { MemoryStore } from '../src/memory/MemoryStore.js';
import { InferenceRouter } from '../src/router/InferenceRouter.js';
import type { SSMRuntime } from '../src/runtime/SSMRuntime.js';

let _db = 0;
const freshStore = () => new MemoryStore({ dbName: `edge-${_db++}` });

function fakeRuntime(): SSMRuntime {
    return {
        generate: jest.fn<any>(async () => 'reply'),
        destroy : jest.fn<any>(),
    } as unknown as SSMRuntime;
}

// ── SSMAgent.init() — corrupted history ───────────────────────────────────────

test('init() recovers to empty history when the stored history is not valid JSON', async () => {
    const store = freshStore();
    await store.remember('__history__', 'this is not json');

    const agent = new SSMAgent({ runtime: fakeRuntime(), memory: store });
    await agent.init();

    expect(agent.history).toEqual([]);
    expect(agent.turnCount).toBe(0);
});

test('init() recovers to empty history when the stored history is valid JSON but not an array', async () => {
    const store = freshStore();
    await store.remember('__history__', JSON.stringify({ not: 'an array' }));

    const agent = new SSMAgent({ runtime: fakeRuntime(), memory: store });
    await agent.init();

    expect(agent.history).toEqual([]);
});

test('init() is a no-op when persistHistory is disabled', async () => {
    const store = freshStore();
    await store.remember('__history__', JSON.stringify([{ role: 'user', content: 'x' }]));

    const agent = new SSMAgent({ runtime: fakeRuntime(), memory: store, persistHistory: false });
    await agent.init();
    expect(agent.history).toEqual([]); // not loaded
});

// ── SSMAgent.destroy() — memory write failure is non-fatal ─────────────────────

test('destroy() still tears down the runtime when persisting history fails', async () => {
    const runtime = fakeRuntime();
    const memory = {
        recall   : jest.fn<any>(async () => undefined),
        recallAll: jest.fn<any>(async () => []),
        remember : jest.fn<any>(async () => { throw new Error('disk full'); }),
    } as unknown as MemoryStore;

    const agent = new SSMAgent({ runtime, memory });
    await agent.think('hi');                 // creates history worth persisting
    await expect(agent.destroy()).resolves.toBeUndefined(); // swallows the write error
    expect(runtime.destroy).toHaveBeenCalledTimes(1);
});

// ── InferenceRouter — audit log cap + getAuditLog ─────────────────────────────

test('a router constructed with no options uses the documented defaults', async () => {
    const router = new InferenceRouter(); // no args → exercises the `= {}` default param + every `?? default`
    // No bridge by default → always SSM.
    const decision = await router.route('x'.repeat(2000)); // long input, but no bridge wins
    expect(decision.target).toBe('ssm');
    expect(decision.reason).toBe('no_bridge');
});

test('the routing audit log is capped at 500 entries (oldest dropped)', async () => {
    const router = new InferenceRouter({ strategy: 'ssm' });
    for (let i = 0; i < 505; i++) await router.route(`q${i}`);

    const log = router.getAuditLog();
    expect(log).toHaveLength(500);
    // getAuditLog returns a copy — mutating it doesn't corrupt the router's log.
    log.pop();
    expect(router.getAuditLog()).toHaveLength(500);
});
