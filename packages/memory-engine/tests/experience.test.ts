/**
 * tests/experience.test.ts — demonstrations → Train Once skills → runs, the approval
 * rule, routines, the learning corpus and the snapshot store.
 */

import { isIrreversibleLabel, needsApproval } from '../src/experience/approval';
import { trainOnce, slug } from '../src/experience/trainOnce';
import { isDue, dueSkills } from '../src/experience/schedule';
import { experienceCorpus, experienceDocuments, packDocuments, skillText } from '../src/experience/corpus';
import { adaptPackage, EvermindModelPackage } from '../src/evermind/index';
import { EvermindLM } from '../src/lm/evermind_lm';
import { BPETokenizer } from '../src/tokenizer/bpe';
import { InMemoryExperienceStore, parseSnapshot, EXPERIENCE_SCHEMA, MAX_RUNS } from '../src/experience/store';
import type { ElementRef, Episode, RecordedStep, Run, Skill } from '../src/experience/types';

const el = (name: string, controlType: string): ElementRef => ({
    name, automationId: '', controlType, className: '', windowTitle: 'Invoices', processName: 'ledger.exe', relX: 0, relY: 0,
});

const step = (id: string, action: RecordedStep['action']): RecordedStep => ({ id, atMs: 0, action });

const episode = (): Episode => ({
    id: 'ep',
    name: 'Enter an invoice',
    program: 'C:/apps/ledger.exe',
    args: [],
    startedAt: 1,
    endedAt: 2,
    steps: [
        step('s1', { kind: 'click', target: el('New invoice', 'Button') }),
        step('s2', { kind: 'setValue', target: el('Amount', 'Edit'), value: '250.00' }),
        step('s3', { kind: 'setValue', target: el('Amount', 'Edit'), value: '12' }),
        step('s4', { kind: 'setValue', target: el('Password', 'Edit'), value: '', secret: true }),
        step('s5', { kind: 'setValue', target: el('Currency', 'Edit'), value: 'USD' }),
        step('s6', { kind: 'click', target: el('Send invoice', 'Button') }),
    ],
});

describe('approval rule', () => {
    it('gates irreversible labels in every product language, and nothing that merely resembles one', () => {
        for (const l of ['Send', 'Place order', 'Delete file', 'Jetzt bezahlen', 'Enviar', 'Supprimer', '立即付款', '确认']) {
            expect(isIrreversibleLabel(l)).toBe(true);
        }
        for (const l of ['Sender', 'Settings', 'Amount', 'Preview', 'Next', 'Payee name', '搜索']) {
            expect(isIrreversibleLabel(l)).toBe(false);
        }
    });

    it('gates Enter only on an element that is itself irreversible', () => {
        expect(needsApproval({ kind: 'keys', keys: 'Enter', target: el('Search', 'Edit') })).toBe(false);
        expect(needsApproval({ kind: 'keys', keys: 'Enter', target: el('Confirm', 'Button') })).toBe(true);
        expect(needsApproval({ kind: 'setValue', target: el('Send to', 'Edit'), value: 'x' })).toBe(false);
    });
});

describe('Train Once', () => {
    it('turns typed values into parameters, secrets into secret parameters, and gates sends', () => {
        const skill = trainOnce(episode(), { fixed: ['s5'] }, 42);
        expect(skill.params.map((p) => p.name)).toEqual(['amount', 'amount_2', 'password']);
        expect(skill.params[0].default).toBe('250.00');
        expect(skill.params[2]).toMatchObject({ secret: true, default: null });
        expect(skill.steps[4].action).toEqual({ kind: 'setValue', target: el('Currency', 'Edit'), value: { from: 'literal', text: 'USD' } });
        expect(skill.steps[0].requiresApproval).toBe(false);
        expect(skill.steps[5].requiresApproval).toBe(true);
        expect(skill).toMatchObject({ sourceEpisode: 'ep', createdAt: 42, routine: null });
    });

    it('applies review edits: dropped steps, parameter names, gate overrides, a new name', () => {
        const skill = trainOnce(episode(), {
            name: 'Monthly invoice',
            removed: ['s3'],
            approvals: { s6: false, s1: true },
            paramNames: { s2: 'Invoice total' },
        });
        expect(skill.name).toBe('Monthly invoice');
        expect(skill.steps).toHaveLength(5);
        expect(skill.params[0].name).toBe('invoice_total');
        expect(skill.steps[0].requiresApproval).toBe(true);
        expect(skill.steps[skill.steps.length - 1].requiresApproval).toBe(false);
    });

    it('never puts a secret into the skill even when review asks to keep it fixed', () => {
        const skill = trainOnce(episode(), { fixed: ['s4'] });
        const pw = skill.steps.find((s) => s.id === 's4')!;
        expect(pw.action).toMatchObject({ value: { from: 'param' } });
    });

    it('slugs labels in any script', () => {
        expect(slug('Invoice Amount (USD)')).toBe('invoice_amount_usd');
        expect(slug('  --  ')).toBe('');
        expect(slug('金额')).toBe('金额');
    });
});

describe('routines', () => {
    const DAY = 86_400_000;

    it('every N minutes', () => {
        const s = { every: 'minutes', minutes: 15 } as const;
        expect(isDue(s, null, 1_000_000, 0)).toBe(true);
        expect(isDue(s, 1_000_000, 1_000_000 + 14 * 60_000, 0)).toBe(false);
        expect(isDue(s, 1_000_000, 1_000_000 + 15 * 60_000, 0)).toBe(true);
    });

    it('daily at a local time, once per day', () => {
        const s = { every: 'day', hour: 9, minute: 30 } as const;
        const day = 20_000 * DAY;
        const offset = -240; // UTC-4
        const at = day + 9 * 3_600_000 + 30 * 60_000 - offset * 60_000;
        expect(isDue(s, null, at - 60_000, offset)).toBe(false);
        expect(isDue(s, null, at, offset)).toBe(true);
        expect(isDue(s, at + 5, at + 3_600_000, offset)).toBe(false);
        expect(isDue(s, at + 5, at + DAY, offset)).toBe(true);
    });

    it('lists only enabled, due routines, least recently run first', () => {
        const base = trainOnce(episode());
        const skills: Skill[] = [
            { ...base, id: 'a', routine: { schedule: { every: 'minutes', minutes: 5 }, values: {}, enabled: true, lastRunAt: 500 } },
            { ...base, id: 'b', routine: { schedule: { every: 'minutes', minutes: 5 }, values: {}, enabled: true, lastRunAt: 100 } },
            { ...base, id: 'c', routine: { schedule: { every: 'minutes', minutes: 5 }, values: {}, enabled: false, lastRunAt: null } },
            { ...base, id: 'd', routine: null },
        ];
        expect(dueSkills(skills, 10 * 60_000, 0).map((s) => s.id)).toEqual(['b', 'a']);
    });
});

describe('learning corpus', () => {
    it('writes skills as procedures with placeholders, and keeps secrets out', () => {
        const skill = trainOnce(episode());
        const text = skillText(skill);
        expect(text).toContain('Task: Enter an invoice (ledger.exe)');
        expect(text).toContain('Set “Amount” to <amount> in “Invoices”.');
        expect(text).toContain('Click “Send invoice” in “Invoices”. Ask before doing this.');
        expect(text).not.toContain('250.00');
    });

    it('prefers the reviewed skill over its raw demonstration', () => {
        const ep = episode();
        const other: Episode = { ...episode(), id: 'ep2', name: 'Other task' };
        const corpus = experienceCorpus([ep, other], [trainOnce(ep)]);
        expect(corpus.match(/Task: Enter an invoice/g)).toHaveLength(1);
        expect(corpus).toContain('Task: Other task');
        expect(corpus).toContain('Set “Amount” to “250.00”');
    });
});

describe('experience store', () => {
    const run = (id: string, startedAt: number): Run => ({
        id, skillId: 'sk', skillName: 'S', trigger: 'manual', status: 'running', startedAt, steps: [],
    });

    it('round-trips through a snapshot and reports every change', async () => {
        const changes: number[] = [];
        const store = new InMemoryExperienceStore(undefined, (s) => { changes.push(s.episodes.length); });
        await store.putEpisode(episode());
        const skill = trainOnce(episode());
        await store.putSkill(skill);
        await store.putRun(run('r1', 10));
        await store.appendRunStep('r1', { idx: 0, stepId: 's1', outcome: 'ok', at: 11 });
        expect(await store.appendRunStep('nope', { idx: 0, stepId: 's1', outcome: 'ok', at: 11 })).toBe(false);
        const snap = await store.snapshot();
        expect(snap.schema).toBe(EXPERIENCE_SCHEMA);
        const copy = new InMemoryExperienceStore(JSON.parse(JSON.stringify(snap)));
        expect((await copy.getRun('r1'))!.steps).toHaveLength(1);
        expect((await copy.listEpisodes())[0]).toMatchObject({ id: 'ep', steps: 6 });
        expect(changes.length).toBe(4);
    });

    it('deleting an episode leaves its skill runnable, and runs are capped newest-first', async () => {
        const store = new InMemoryExperienceStore();
        await store.putEpisode(episode());
        const skill = trainOnce(episode());
        await store.putSkill(skill);
        expect(await store.deleteEpisode('ep')).toBe(true);
        expect(await store.getSkill(skill.id)).toBeDefined();
        for (let i = 0; i < MAX_RUNS + 3; i++) await store.putRun(run(`r${i}`, i));
        const runs = await store.listRuns(MAX_RUNS + 10);
        expect(runs).toHaveLength(MAX_RUNS);
        expect(runs[0].id).toBe(`r${MAX_RUNS + 2}`);
    });

    it('treats a missing or foreign snapshot as empty', () => {
        expect(parseSnapshot(null).episodes).toEqual([]);
        expect(parseSnapshot({ schema: 'something-else', episodes: [1] }).episodes).toEqual([]);
        expect(parseSnapshot({ schema: EXPERIENCE_SCHEMA, episodes: [] }).learned).toEqual({});
    });

    it('keeps a learned ledger that survives a snapshot and drops forgotten items', async () => {
        const store = new InMemoryExperienceStore();
        await store.putEpisode(episode());
        const skill = trainOnce(episode());
        await store.putSkill(skill);
        await store.markLearned(['ep', skill.id], 7);
        const copy = new InMemoryExperienceStore(JSON.parse(JSON.stringify(await store.snapshot())));
        expect((await copy.snapshot()).learned).toEqual({ ep: 7, [skill.id]: 7 });
        await copy.deleteSkill(skill.id);
        expect((await copy.snapshot()).learned).toEqual({ ep: 7 });
    });
});

describe('learning from experience', () => {
    it('tags each procedure with its source and packs passes without splitting one', () => {
        const ep = episode();
        const other: Episode = { ...episode(), id: 'ep2', name: 'Other task' };
        const skill = trainOnce(ep);
        const docs = experienceDocuments([ep, other], [skill]);
        expect(docs.map((d) => d.id)).toEqual([skill.id, 'ep2']);
        const one = docs[0].text.length;
        expect(packDocuments(docs, 100_000)).toHaveLength(1);
        expect(packDocuments(docs, one)).toHaveLength(2);
        expect(packDocuments([{ id: 'x', text: 'y'.repeat(50) }], 10)).toEqual([[{ id: 'x', text: 'y'.repeat(50) }]]);
        expect(packDocuments([], 10)).toEqual([]);
    });

    it('adapts a private package in place: new checksum and version, still valid, same shape', () => {
        const corpus = experienceCorpus([episode()], []);
        const tok = new BPETokenizer();
        tok.train(corpus);
        const pkg = EvermindModelPackage.fromLM(new EvermindLM({ vocabSize: tok.vocabSize, seed: 5 }), {
            name: 'mine', version: '1', card: { description: 'private' },
        });
        expect(adaptPackage(pkg, tok, '', '2')).toBeNull();
        const r = adaptPackage(pkg, tok, corpus, '2')!;
        expect(r.pkg.manifest.version).toBe('2');
        expect(r.pkg.manifest.checksum).not.toBe(pkg.manifest.checksum);
        expect(r.pkg.checkpoint.byteLength).toBe(pkg.checkpoint.byteLength);
        const back = EvermindModelPackage.fromBlob(r.pkg.toBlob());
        expect(back.validate().ok).toBe(true);
        expect(back.loadLM().config.vocabSize).toBe(tok.vocabSize);
    });
});
