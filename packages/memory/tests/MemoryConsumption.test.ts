/**
 * tests/MemoryConsumption.test.ts
 * End-to-end "can I actually use this?" test: a REAL MemoryStore (fake-indexeddb)
 * wired into an SSMAgent, asserting the full consumption pattern:
 *   - facts you `remember()` are recalled and injected into the cacheable `system`
 *     prefix (NOT the volatile per-turn conversation)
 *   - importance ordering, key-substring filtering, and injectAllFacts behave
 *   - conversation history persists across destroy()/init() through the store
 *
 * The runtime is a recording fake so we can inspect exactly what the agent feeds
 * to generate() — the (conversation, { system }) split is the thing under test.
 */

import 'fake-indexeddb/auto';
import { jest } from '@jest/globals';
import { SSMAgent } from '../src/agent/SSMAgent.js';
import { MemoryStore } from '../src/memory/MemoryStore.js';
import type { SSMRuntime } from '../src/runtime/SSMRuntime.js';

// ── Recording runtime ───────────────────────────────────────────────────────

interface GenerateCall { conversation: string; system: string | undefined; }

function recordingRuntime(reply = 'reply') {
    const calls: GenerateCall[] = [];
    const runtime = {
        generate: jest.fn<any>(async (conversation: string, opts: { system?: string }) => {
            calls.push({ conversation, system: opts?.system });
            return reply;
        }),
        stream: jest.fn<any>(async function* () { yield reply; }),
        adapt: jest.fn<any>(), evaluate: jest.fn<any>(), embed: jest.fn<any>(),
        save: jest.fn<any>(), load: jest.fn<any>(), destroy: jest.fn<any>(),
        get bridge() { return undefined; },
        get destroyed() { return false; },
        get internals() { return {} as never; },
    } as unknown as SSMRuntime;
    return { runtime, calls };
}

let _db = 0;
const freshStore = () => new MemoryStore({ dbName: `consume-${_db++}` });

// ── Memory → cacheable system prefix ──────────────────────────────────────────

test('a remembered fact whose key appears in the input is injected into the system prefix, not the conversation', async () => {
    const store = freshStore();
    await store.remember('stack', 'React + TypeScript', { importance: 0.9 });

    const { runtime, calls } = recordingRuntime();
    const agent = new SSMAgent({ runtime, memory: store, systemPrompt: 'You are helpful.' });

    await agent.think('what stack should I use?');

    expect(calls).toHaveLength(1);
    // Fact lands in the stable, cacheable system block …
    expect(calls[0].system).toContain('System: You are helpful.');
    expect(calls[0].system).toContain('Fact (stack): React + TypeScript');
    // … and NOT in the volatile conversation the model regenerates each turn.
    expect(calls[0].conversation).toContain('User: what stack should I use?');
    expect(calls[0].conversation).not.toContain('Fact (stack)');
});

test('facts whose key is absent from the input are not injected', async () => {
    const store = freshStore();
    await store.remember('stack', 'React', { importance: 0.5 });
    await store.remember('goal', 'ship a chat app', { importance: 0.5 });

    const { runtime, calls } = recordingRuntime();
    const agent = new SSMAgent({ runtime, memory: store });

    await agent.think('tell me about the stack');

    expect(calls[0].system).toContain('Fact (stack): React');
    expect(calls[0].system).not.toContain('Fact (goal)');
});

test('injected facts are ordered by importance (highest first) in the system prefix', async () => {
    const store = freshStore();
    await store.remember('alpha', 'low',  { importance: 0.2 });
    await store.remember('beta',  'high', { importance: 0.9 });

    const { runtime, calls } = recordingRuntime();
    const agent = new SSMAgent({ runtime, memory: store });

    await agent.think('compare alpha and beta');

    const sys = calls[0].system ?? '';
    expect(sys.indexOf('Fact (beta)')).toBeLessThan(sys.indexOf('Fact (alpha)'));
});

test('injectAllFacts puts every fact in the system prefix regardless of the input', async () => {
    const store = freshStore();
    await store.remember('one', 'first',  { importance: 0.5 });
    await store.remember('two', 'second', { importance: 0.5 });

    const { runtime, calls } = recordingRuntime();
    const agent = new SSMAgent({ runtime, memory: store });

    await agent.think('unrelated', { injectAllFacts: true });

    expect(calls[0].system).toContain('Fact (one): first');
    expect(calls[0].system).toContain('Fact (two): second');
});

test('with no memory store the system prefix is just the system prompt', async () => {
    const { runtime, calls } = recordingRuntime();
    const agent = new SSMAgent({ runtime, systemPrompt: 'Bare.' });

    await agent.think('hi');

    expect(calls[0].system).toBe('System: Bare.');
    expect(calls[0].conversation).toMatch(/User: hi\nAssistant:$/);
});

// ── Memory-backed persistence across sessions ─────────────────────────────────

test('conversation history persists across destroy()/init() through the MemoryStore', async () => {
    const store = freshStore();

    const first = recordingRuntime();
    const agent1 = new SSMAgent({ runtime: first.runtime, memory: store });
    await agent1.think('remember this turn');
    expect(agent1.turnCount).toBe(1);
    await agent1.destroy(); // serialises history into the store

    // A brand-new agent over the SAME store reloads the prior history on init().
    const second = recordingRuntime();
    const agent2 = new SSMAgent({ runtime: second.runtime, memory: store });
    await agent2.init();

    expect(agent2.turnCount).toBe(1);
    expect(agent2.history[0]).toEqual({ role: 'user', content: 'remember this turn' });
});

// ── remember / recall through the agent ───────────────────────────────────────

test('remember() then recall() round-trips through the real store', async () => {
    const store = freshStore();
    const { runtime } = recordingRuntime();
    const agent = new SSMAgent({ runtime, memory: store });

    await agent.remember('lang', 'TypeScript');
    expect(await agent.recall('lang')).toBe('TypeScript');
    // And it is durable in the underlying store, not just agent-local.
    expect((await store.recall('lang'))?.content).toBe('TypeScript');
});
