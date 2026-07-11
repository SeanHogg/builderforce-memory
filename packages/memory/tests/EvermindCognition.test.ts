/**
 * tests/EvermindCognition.test.ts
 * Write-Through Cognition — evidence-gated, replace-on-write knowledge.
 * Uses the real MemoryStore (fake-indexeddb) so the store contract is exercised end-to-end.
 */

import 'fake-indexeddb/auto';

import { jest } from '@jest/globals';
import { EvermindCognition } from '../src/cognition/EvermindCognition.js';
import { workspacePresenceGatherer } from '../src/cognition/gatherers.js';
import { MemoryStore } from '../src/memory/MemoryStore.js';
import type { EvidenceGatherer } from '../src/cognition/types.js';

let dbCounter = 0;
function freshCognition() {
    const store = new MemoryStore({ dbName: `cognition-${++dbCounter}-${Date.now()}` });
    return { store, cog: new EvermindCognition({ store }) };
}

const SUBJECT = 'pkg:ssm-stack';
const STALE = 'SSM stack = MambaKit + SSMjs';
const FRESH = 'SSM stack = builderforce-memory monorepo';

describe('EvermindCognition — write-through commit', () => {
    it('augments a brand-new subject and bumps the version', async () => {
        const { cog } = freshCognition();
        const r = await cog.commit({ subjectKey: SUBJECT, content: STALE });
        expect(r.verdict).toBe('augment');
        expect(r.content).toBe(STALE);
        expect(r.version).toBe(1);
    });

    it('confirms an identical re-assertion WITHOUT replacing or bumping version', async () => {
        const { cog } = freshCognition();
        await cog.commit({ subjectKey: SUBJECT, content: STALE });
        const r = await cog.commit({ subjectKey: SUBJECT, content: STALE });
        expect(r.verdict).toBe('confirm');
        expect(r.version).toBe(1); // unchanged — no new knowledge
    });

    it('SUPERSEDES a conflicting fact when evidence favours it — replace, not append', async () => {
        const { store, cog } = freshCognition();
        await cog.commit({ subjectKey: SUBJECT, content: STALE });

        // Evidence: the workspace no longer contains MambaKit/SSMjs, but does contain builderforce-memory.
        const gather = workspacePresenceGatherer({
            list: async () => ['builderforce-memory/', 'Builderforce.ai/'],
            mustExist: ['builderforce-memory/'],
            mustBeAbsent: ['MambaKit/', 'SSMjs/'],
        });

        const r = await cog.commit({ subjectKey: SUBJECT, content: FRESH }, gather);
        expect(r.verdict).toBe('supersede');
        expect(r.superseded).toBe(STALE);
        expect(r.version).toBe(2);

        // The crux: exactly ONE belief for the subject, and it is the fresh one.
        const held = await store.recall(SUBJECT);
        expect(held?.content).toBe(FRESH);
        const all = await store.recallAll();
        expect(all.filter((e) => e.key === SUBJECT)).toHaveLength(1);
    });

    it('REJECTS a conflicting fact when evidence does NOT favour it — incumbent retained', async () => {
        const { store, cog } = freshCognition();
        await cog.commit({ subjectKey: SUBJECT, content: STALE });

        const gatherAgainst: EvidenceGatherer = async () => ({ supportsNew: false, notes: ['evidence favours incumbent'] });
        const r = await cog.commit({ subjectKey: SUBJECT, content: FRESH }, gatherAgainst);
        expect(r.verdict).toBe('reject');
        expect(r.content).toBe(STALE);
        expect(r.version).toBe(1); // no knowledge change

        const held = await store.recall(SUBJECT);
        expect(held?.content).toBe(STALE);
    });

    it('does not store an unverified brand-new claim when requireEvidence is set', async () => {
        const { store, cog } = freshCognition();
        const gatherAgainst: EvidenceGatherer = async () => ({ supportsNew: false, notes: ['unverified'] });
        const r = await cog.commit(
            { subjectKey: SUBJECT, content: FRESH, requireEvidence: true },
            gatherAgainst,
        );
        expect(r.verdict).toBe('reject');
        expect(await store.recall(SUBJECT)).toBeUndefined();
    });
});

describe('EvermindCognition — write-through recall (version-token cache)', () => {
    it('serves recall and invalidates the cache when knowledge changes', async () => {
        const { store, cog } = freshCognition();
        await cog.commit({ subjectKey: SUBJECT, content: STALE });

        // Count store reads to prove caching: recallDetailed prefers the scored recall
        // (recallSimilarScored -> recallAll under the hood) when the store exposes it.
        const spy = jest.spyOn(store, 'recallSimilarScored');

        const first = await cog.recall('ssm stack', 3);
        const second = await cog.recall('ssm stack', 3);
        expect(first).toEqual(second);
        expect(spy).toHaveBeenCalledTimes(1); // second served from cache (same version)

        // A supersede bumps the version → cache invalidates → store hit again.
        const gather = workspacePresenceGatherer({
            list: async () => ['builderforce-memory/'],
            mustExist: ['builderforce-memory/'],
            mustBeAbsent: ['MambaKit/'],
        });
        await cog.commit({ subjectKey: SUBJECT, content: FRESH }, gather);

        await cog.recall('ssm stack', 3);
        expect(spy).toHaveBeenCalledTimes(2); // re-read after invalidation
        spy.mockRestore();
    });
});
