/**
 * tests/code-eval.test.ts — structural validity + execution-reward checks.
 * Pure, dependency-free logic (the code-specific BUILD diagnostics); exercised
 * here across every branch of analyzeCode / runJsCases / deepEqual.
 */

import { analyzeCode, runJsCases } from "../src/workflow/code-eval.js";

describe("analyzeCode", () => {
  test("balanced code scores 1 and parses as JS", () => {
    const a = analyzeCode("function add(a, b) { return [a, b]; }");
    expect(a.balanced).toBe(true);
    expect(a.score).toBe(1);
    expect(a.jsParse).toBe(true);
  });

  test("ignores delimiters inside strings (incl. escapes and backticks)", () => {
    const a = analyzeCode("const s = \"a)b\"; const t = 'c]d'; const u = `e}f`;");
    expect(a.balanced).toBe(true);
    // Escaped closing quote does NOT terminate the string, so the `(` stays inside
    // it and the source is still balanced (exercises the escape branch).
    const b = analyzeCode('const s = "a\\"b(";');
    expect(b.balanced).toBe(true);
    // A backslash-escaped char inside a string is consumed (escape branch again).
    const c = analyzeCode('const s = "a\\nb";');
    expect(c.balanced).toBe(true);
  });

  test("unmatched opening delimiter lowers the score and is unbalanced", () => {
    const a = analyzeCode("function f() { return (1;");
    expect(a.balanced).toBe(false);
    expect(a.score).toBeGreaterThanOrEqual(0);
    expect(a.score).toBeLessThan(1);
  });

  test("mismatched closing delimiter counts as a fault", () => {
    const a = analyzeCode("foo(])");
    expect(a.balanced).toBe(false);
  });

  test("unterminated string is a fault", () => {
    const a = analyzeCode('const s = "open;');
    expect(a.balanced).toBe(false);
    expect(a.jsParse).toBe(false);
  });

  test("non-JS language skips the JS parse", () => {
    const a = analyzeCode("value: [1, 2]", "yaml");
    expect(a.jsParse).toBeUndefined();
    expect(a.balanced).toBe(true);
  });

  test.each(["ts", "javascript", "typescript"])("language %s runs the JS parse", (lang) => {
    expect(analyzeCode("const x = 1;", lang).jsParse).toBe(true);
  });

  test("syntactically invalid JS reports jsParse=false even when delimiters balance", () => {
    const a = analyzeCode("const = ;");
    expect(a.balanced).toBe(true);
    expect(a.jsParse).toBe(false);
  });
});

describe("runJsCases", () => {
  test("no cases → zero pass rate", () => {
    expect(runJsCases("function f(){}", [])).toEqual({ passed: 0, total: 0, passRate: 0 });
  });

  test("all cases pass → passRate 1", () => {
    const r = runJsCases("function add(a,b){return a+b;}", [
      { call: "add(2,3)", expect: 5 },
      { call: "add(-1,1)", expect: 0 },
    ]);
    expect(r).toEqual({ passed: 2, total: 2, passRate: 1 });
  });

  test("a runtime error fails only that case", () => {
    const r = runJsCases("function f(x){ if(x<0) throw new Error('neg'); return x; }", [
      { call: "f(1)", expect: 1 },
      { call: "f(-1)", expect: 0 },
    ]);
    expect(r.passed).toBe(1);
    expect(r.total).toBe(2);
    expect(r.error).toBeUndefined();
  });

  test("a syntax error fails every case and reports the error once", () => {
    const r = runJsCases("function bad( {", [{ call: "bad()", expect: 1 }]);
    expect(r.passed).toBe(0);
    expect(r.passRate).toBe(0);
    expect(typeof r.error).toBe("string");
  });

  test("deepEqual compares arrays, nested objects, and primitives", () => {
    const code = `
      function arr(){ return [1, {a: 2}]; }
      function obj(){ return { a: 1, b: [2, 3] }; }
      function nan(){ return NaN; }
      function wrongLen(){ return [1, 2]; }
      function wrongKeys(){ return { a: 1, c: 2 }; }
    `;
    const r = runJsCases(code, [
      { call: "arr()", expect: [1, { a: 2 }] },          // array + nested object equal
      { call: "obj()", expect: { a: 1, b: [2, 3] } },     // object with array value equal
      { call: "nan()", expect: NaN },                     // Object.is path
      { call: "wrongLen()", expect: [1, 2, 3] },          // array length mismatch → fail
      { call: "wrongKeys()", expect: { a: 1, b: 2 } },    // object key mismatch → fail
      { call: "obj()", expect: 5 },                        // object vs primitive → fail
      { call: "arr()", expect: { 0: 1 } },                // array vs non-array → fail
    ]);
    expect(r.passed).toBe(3);
    expect(r.total).toBe(7);
  });
});
