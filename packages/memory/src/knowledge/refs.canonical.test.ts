/**
 * src/knowledge/refs.canonical.test.ts
 * writeCanonicalJson: the pieces it writes join to exactly `canonicalJson`'s string, and a generator is written as an array.
 */

import { canonicalJson, writeCanonicalJson } from './refs.js';

const written = (value: unknown): string => {
    let out = '';
    writeCanonicalJson(value, (chunk) => { out += chunk; });
    return out;
};

describe('writeCanonicalJson', () => {
    const FIXTURES: Array<[string, unknown]> = [
        ['nested objects, keys out of order', { b: { d: 1, c: [2, { z: 1, a: 0 }] }, a: 'x' }],
        ['undefined, function and symbol in an array', [undefined, () => 1, Symbol('s'), null, 1]],
        ['undefined, function and symbol members', { a: undefined, b: () => 1, c: Symbol('s'), d: 0 }],
        ['false, null and empty containers', { live: false, none: null, o: {}, l: [], nested: [[], {}] }],
        ['unicode and escapes', { 'ключ': 'é😀\u0000"\\\n', lone: '\ud800' }],
        ['numbers, NaN and Infinity', [0, -0, 1.5, 1e21, NaN, Infinity, -Infinity]],
        ['Map, Set and Date are {}', { m: new Map([['k', 1]]), s: new Set([1]), d: new Date(0) }],
        ['sparse array', [1, , 3]], // eslint-disable-line no-sparse-arrays
        ['scalars', 'text'],
        ['null', null],
        ['undefined', undefined],
    ];

    it.each(FIXTURES)('equals canonicalJson: %s', (_name, value) => {
        expect(written(value)).toBe(canonicalJson(value) ?? 'null');
    });

    it('writes a generator as an array, item by item', () => {
        function* items() { yield { b: 1, a: 2 }; yield undefined; }
        expect(written({ list: items() })).toBe(canonicalJson({ list: [{ b: 1, a: 2 }, undefined] }));
    });
});
