/**
 * tests/evermind-runtime.test.ts
 * The model-driving half of Evermind serving, against a scripted fake head: prompt
 * flattening, sliced generation under a deadline, the fitness-to-serve grade, and the
 * tool decoder's teacher-forced scoring.
 */

import {
    messagesToPrompt,
    generateEvermindText,
    assessLMCoherence,
    probeEvermindText,
    createEvermindToolDecoder,
    generateEvermindWithTools,
    COHERENCE_PROBE_PROMPTS,
    type EvermindTextModel,
    type EvermindTextTokenizer,
} from '../src/evermind/index.js';

/** One token per character — enough to count usage exactly. */
const tok: EvermindTextTokenizer = { encode: (text) => Array.from(text, (c) => c.charCodeAt(0) % 50) };

/** A head that replays `chunks` one per generateText call, then goes silent. */
function scripted(chunks: string[], logits: number[][] = []): EvermindTextModel & { prompts: string[] } {
    const prompts: string[] = [];
    let i = 0;
    return {
        prompts,
        generateText(prompt) {
            prompts.push(prompt);
            return chunks[i++] ?? '';
        },
        forward(ids) {
            return { logits: ids.map((_, k) => logits[k] ?? new Array(50).fill(0)) };
        },
    };
}

const SENTENCE = 'The team finished the login page and is now reviewing the billing flow for release.';

describe('messagesToPrompt', () => {
    it('flattens role-tagged turns, skips empty ones and primes the assistant', () => {
        const p = messagesToPrompt([{ role: 'system', content: 'be terse' }, { content: 'hi' }, { role: 'user', content: '' }]);
        expect(p).toBe('system: be terse\nuser: hi\nassistant:');
    });
});

describe('generateEvermindText', () => {
    it('generates in slices, feeding each slice back into the next prompt, and meters usage', () => {
        const lm = scripted(['Hello ', 'world']);
        const gen = generateEvermindText(lm, tok, [{ role: 'user', content: 'hi' }], { maxTokens: 64, seed: 7 });
        expect(gen.content).toBe('Hello world');
        expect(lm.prompts[1]).toBe('user: hi\nassistant:Hello ');
        expect(gen.truncated).toBe(false);
        expect(gen.usage.completion_tokens).toBe('Hello world'.length);
        expect(gen.usage.total_tokens).toBe(gen.usage.prompt_tokens + gen.usage.completion_tokens);
    });

    it('stops at the wall-clock budget and says the answer is partial', () => {
        const lm = scripted(new Array(100).fill('word '));
        const gen = generateEvermindText(lm, tok, [{ role: 'user', content: 'go' }], { maxTokens: 1000, deadlineMs: 0 });
        expect(lm.prompts).toHaveLength(1);
        expect(gen.truncated).toBe(true);
    });
});

describe('assessLMCoherence / probeEvermindText', () => {
    it('passes a head that answers every probe in sentences', () => {
        const lm = scripted(COHERENCE_PROBE_PROMPTS.map(() => SENTENCE));
        const verdict = assessLMCoherence(lm, tok);
        expect(verdict.passRate).toBe(1);
        expect(verdict.ready).toBe(true);
    });

    it('refuses a head that produces nothing, and names why', () => {
        const verdict = assessLMCoherence(scripted([]), tok);
        expect(verdict.ready).toBe(false);
        expect(verdict.samples.every((s) => !s.coherent && s.failure !== null)).toBe(true);
    });

    it('grades one operator prompt with the same serve-time bar', () => {
        const probe = probeEvermindText(scripted([SENTENCE]), tok, 'What is the status?');
        expect(probe.coherent).toBe(true);
        expect(probe.text).toBe(SENTENCE);
    });
});

describe('createEvermindToolDecoder', () => {
    it('scores a continuation by its mean log-probability under teacher forcing', () => {
        // Every position strongly predicts token 3, so a continuation of token-3 characters
        // scores higher than one made of other tokens.
        const peaked = new Array(50).fill(0);
        peaked[3] = 10;
        const lm = scripted([], new Array(20).fill(peaked));
        const decoder = createEvermindToolDecoder(lm, tok);
        const three = String.fromCharCode(3 + 50); // encodes to 3
        expect(decoder.score('p', three + three)).toBeGreaterThan(decoder.score('p', 'zz'));
        expect(decoder.score('p', '')).toBe(-Infinity);
        expect(decoder.usage().total_tokens).toBeGreaterThan(0);
    });
});

describe('generateEvermindWithTools', () => {
    it('answers in prose when no tool is planned', () => {
        const gen = generateEvermindWithTools(scripted([SENTENCE]), tok, [{ role: 'user', content: 'hi' }], [], { mode: 'none' });
        expect(gen.call).toBeNull();
        expect(gen.calls).toEqual([]);
        expect(gen.content).toBe(SENTENCE);
    });
});
