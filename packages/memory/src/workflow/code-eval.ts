/**
 * workflow/code-eval.ts — dependency-free code-validity + execution-reward checks.
 *
 * These power the code-specific BUILD diagnostics that grade what Evermind's
 * generator produces when we teach it to program:
 *   • {@link analyzeCode}  — structural validity (balanced delimiters/strings) +
 *                            an optional JS parse. The portable "does this look
 *                            like valid code" gate.
 *   • {@link runJsCases}   — execution-grounded reward: run generated JS against
 *                            test cases and score the pass-rate. Code's signal is
 *                            that it RUNS — this turns that into a number.
 *
 * Both are zero-dependency (no AST library) so they run in the same WebGPU/TS
 * runtime as the rest of the engine. Execution uses the `Function` constructor —
 * a heuristic reward sandbox, NOT a security boundary; only feed it code from a
 * trusted teacher / the model under training.
 */

const OPEN: Record<string, string> = { ")": "(", "]": "[", "}": "{" };

export interface CodeAnalysis {
  /** 0..1 structural-validity score (1 = delimiters and strings all balanced). */
  score: number;
  /** True when every bracket/brace/paren and string literal closes cleanly. */
  balanced: boolean;
  /** For language="js": whether the source parses (Function constructor). undefined otherwise. */
  jsParse?: boolean;
}

/**
 * Score how structurally valid `code` is: walk the source tracking string state
 * (so brackets inside strings are ignored) and a delimiter stack. A clean parse
 * scores 1; each unmatched/mismatched delimiter or unterminated string lowers it.
 */
export function analyzeCode(code: string, language = "js"): CodeAnalysis {
  const stack: string[] = [];
  let mismatches = 0;
  let delims = 0;
  let inString: string | null = null; // the opening quote char, or null
  let escaped = false;

  for (let i = 0; i < code.length; i++) {
    const ch = code[i]!;
    if (inString) {
      if (escaped) { escaped = false; continue; }
      if (ch === "\\") { escaped = true; continue; }
      if (ch === inString) inString = null;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === "`") { inString = ch; continue; }
    if (ch === "(" || ch === "[" || ch === "{") { stack.push(ch); delims++; continue; }
    if (ch === ")" || ch === "]" || ch === "}") {
      delims++;
      if (stack.pop() !== OPEN[ch]) mismatches++;
    }
  }
  // Anything left open is an unmatched delimiter; an open string is one more fault.
  mismatches += stack.length;
  const faults = mismatches + (inString ? 1 : 0);
  const denom = Math.max(1, delims + (inString ? 1 : 0));
  const balanced = faults === 0;
  const score = balanced ? 1 : Math.max(0, 1 - faults / denom);

  const analysis: CodeAnalysis = { score, balanced };
  if (language === "js" || language === "ts" || language === "javascript" || language === "typescript") {
    analysis.jsParse = jsParses(code);
  }
  return analysis;
}

/** True when `code` parses as JS (Function constructor parses but does not run). */
function jsParses(code: string): boolean {
  try {
    // eslint-disable-next-line no-new-func
    new Function(code);
    return true;
  } catch {
    return false;
  }
}

export interface CodeCase {
  /** An expression to evaluate after the code is loaded, e.g. "add(2, 3)". */
  call: string;
  /** The expected result (deep-equal compared). */
  expect: unknown;
}

export interface CodeEvalResult {
  passed: number;
  total: number;
  /** passed / total (0 when there are no cases). */
  passRate: number;
  /** A whole-program error (e.g. syntax error) that failed every case, if any. */
  error?: string;
}

/**
 * Load `code` (which defines functions) and evaluate each case's `call`
 * expression, comparing the result to `expect`. A syntax error in `code` fails
 * every case. Per-case runtime errors fail only that case.
 *
 * NOTE: heuristic reward sandbox, not a security boundary — see file header.
 */
export function runJsCases(code: string, cases: CodeCase[]): CodeEvalResult {
  const total = cases.length;
  if (total === 0) return { passed: 0, total: 0, passRate: 0 };

  let passed = 0;
  for (const c of cases) {
    try {
      // Code defines functions (hoisted); the trailing return evaluates the call.
      // eslint-disable-next-line no-new-func
      const fn = new Function(`${code}\n;return (${c.call});`);
      const actual = fn();
      if (deepEqual(actual, c.expect)) passed++;
    } catch (err) {
      // A syntax error throws at construction for ALL cases — report it once.
      if (err instanceof SyntaxError) {
        return { passed: 0, total, passRate: 0, error: err.message };
      }
      // Runtime error → this case simply fails.
    }
  }
  return { passed, total, passRate: passed / total };
}

/** Structural deep equality good enough for JSON-shaped test expectations. */
function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b || a === null || b === null) return false;
  if (typeof a !== "object") return Object.is(a, b);
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    return a.every((v, i) => deepEqual(v, b[i]));
  }
  const ao = a as Record<string, unknown>;
  const bo = b as Record<string, unknown>;
  const ak = Object.keys(ao);
  const bk = Object.keys(bo);
  if (ak.length !== bk.length) return false;
  return ak.every((k) => deepEqual(ao[k], bo[k]));
}
