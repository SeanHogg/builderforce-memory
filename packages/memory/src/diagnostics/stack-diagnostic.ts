/**
 * stack-diagnostic.ts — the seven-layer agent stack as a runnable diagnostic.
 *
 * The SAME evaluators the e2e test asserts, exposed as an ordered set of steps a
 * caller can RUN and observe. `runStackDiagnostic` executes each layer, captures
 * pass/fail + timing + error, and streams a per-step callback — so the result is
 * directly an "execution output" timeline a UI can render: click a button, watch
 * each layer light up, see exactly which step breaks.
 *
 * DRY: `tests/seven-layers.test.ts` consumes this module (it does not re-implement
 * the checks), and the BuilderForce execution-output / workflow surface consumes
 * the very same steps — injecting real agent-runtime implementations for the
 * layers it owns (L2/L5/L6) in place of the in-process defaults here.
 */

import { EvermindLM, BPETokenizer, EvermindModelPackage } from "@seanhogg/builderforce-memory-engine";
import { EvermindCognition } from "../cognition/index.js";
import { hybridRetrieve, chunkText } from "../retrieval/index.js";
import { MemoryStore } from "../memory/MemoryStore.js";

export type StepStatus = "pass" | "fail" | "skip";

/** One layer's result — the shape an execution-output timeline renders per row. */
export interface StackStepResult {
  id: string;
  /** "L1".."L7". */
  layer: string;
  label: string;
  status: StepStatus;
  ms: number;
  /** Short human detail on success (e.g. "retrieved 3 chunks"). */
  detail?: string;
  /** Failure message when status === "fail". */
  error?: string;
}

/** Shared scratch threaded across steps (e.g. the model L1 builds, L7 deploys). */
export interface StackContext {
  bag: Record<string, unknown>;
  /** Prior step results so far — this IS the observability trace (L6). */
  trace: StackStepResult[];
  /** Optional IDBFactory (Node/fake-indexeddb); browser uses global indexedDB. */
  idbFactory?: unknown;
  now: () => number;
}

export interface StackStep {
  id: string;
  layer: string;
  label: string;
  /** Throw to fail the step; optionally return a human detail string. */
  run: (ctx: StackContext) => Promise<string | void>;
}

export interface RunStackOptions {
  /** Streamed as each step settles — for live execution-output rendering. */
  onStep?: (r: StackStepResult) => void;
  /** Injectable clock for deterministic timing in tests. Default Date.now. */
  now?: () => number;
  /** IDBFactory for Node; omit in the browser. */
  idbFactory?: unknown;
}

const KNOWLEDGE =
  "BuilderForce orchestrates many agents through a planning loop. " +
  "The memory layer stores facts as SSM embeddings. " +
  "Deployment runs on Cloudflare Workers and Durable Objects. " +
  "Tools are gated by a capability registry.";

let dbSeq = 0;

/**
 * The default seven steps using the real Evermind components for the layers this
 * stack owns (L1/L3/L4/L7) and an in-process harness for the agent-runtime-owned
 * layers (L2/L5/L6). Ordered so dependencies (the model built in L1) precede use.
 */
export function buildEvermindStackSteps(): StackStep[] {
  return [
    {
      id: "l1-foundation",
      layer: "L1",
      label: "Foundation Model — EvermindLM + tokenizer",
      run: async (ctx) => {
        const tok = new BPETokenizer();
        tok.train(KNOWLEDGE.repeat(4), { numMerges: 60 });
        const model = new EvermindLM({ vocabSize: tok.vocabSize, dModel: 16, numLayers: 2, hiddenDim: 24, seed: 7 });
        const out = model.generateText("Deployment", tok, { maxNewTokens: 4, temperature: 0 });
        if (typeof out !== "string") throw new Error("model did not produce text");
        if (tok.decode(tok.encode("Cloudflare Workers")) !== "Cloudflare Workers") {
          throw new Error("tokenizer did not round-trip text");
        }
        ctx.bag.model = model;
        ctx.bag.tokenizer = tok;
        return `vocab ${tok.vocabSize}, generated ${out.length} chars`;
      },
    },
    {
      id: "l3-memory",
      layer: "L3",
      label: "Memory — write-through cognition (replace-on-write)",
      run: async (ctx) => {
        const store = new MemoryStore({ dbName: `diag-${ctx.now()}-${dbSeq++}`, idbFactory: ctx.idbFactory as never });
        const cognition = new EvermindCognition({ store });
        await cognition.commit({ subjectKey: "fact:deploy", content: "deploy target = unknown" });
        const r = await cognition.commit({ subjectKey: "fact:deploy", content: "deploy target = Cloudflare" });
        if (r.verdict !== "supersede") throw new Error(`expected supersede, got ${r.verdict}`);
        ctx.bag.cognition = cognition;
        return `verdict ${r.verdict}`;
      },
    },
    {
      id: "l4-rag",
      layer: "L4",
      label: "RAG — chunk + hybrid retrieve",
      run: async (ctx) => {
        const candidates = chunkText(KNOWLEDGE, { chunkSize: 70, chunkOverlap: 0 }).map((c, i) => ({ id: `c${i}`, text: c.text }));
        const hits = hybridRetrieve({ text: "where does deployment run" }, candidates, { topK: 2 });
        if (hits.length === 0 || !hits[0]!.text.toLowerCase().includes("cloudflare")) {
          throw new Error("retriever did not surface the deployment passage");
        }
        ctx.bag.candidates = candidates;
        return `top hit: ${hits[0]!.text.slice(0, 40)}…`;
      },
    },
    {
      id: "l5-tools",
      layer: "L5",
      label: "Tools — capability-gated registry",
      run: async (ctx) => {
        const tools = new Map<string, (a: string) => string>([["shout", (a) => a.toUpperCase()]]);
        const fn = tools.get("shout");
        if (!fn || fn("deploy") !== "DEPLOY") throw new Error("tool invocation failed");
        if (tools.get("missing")) throw new Error("ungated tool resolved");
        ctx.bag.tools = tools;
        return "1 tool registered + invoked";
      },
    },
    {
      id: "l2-orchestration",
      layer: "L2",
      label: "Orchestration — retrieve→recall→act→generate loop",
      run: async (ctx) => {
        const model = ctx.bag.model as EvermindLM | undefined;
        const tokenizer = ctx.bag.tokenizer as BPETokenizer | undefined;
        const cognition = ctx.bag.cognition as EvermindCognition | undefined;
        const candidates = ctx.bag.candidates as { id: string; text: string }[] | undefined;
        const tools = ctx.bag.tools as Map<string, (a: string) => string> | undefined;
        if (!model || !tokenizer || !cognition || !candidates || !tools) {
          throw new Error("a prerequisite layer (L1/L3/L4/L5) did not complete");
        }
        const steps: string[] = [];
        const hits = hybridRetrieve({ text: "deployment cloudflare" }, candidates, { topK: 1 });
        steps.push("retrieve");
        await cognition.recall("deployment", 3);
        steps.push("recall");
        tools.get("shout")!(hits[0]?.text ?? "deploy");
        steps.push("tool");
        model.generateText("Deployment", tokenizer, { maxNewTokens: 3, temperature: 0 });
        steps.push("model");
        if (steps.join(",") !== "retrieve,recall,tool,model") throw new Error("loop did not complete all stages");
        return steps.join(" → ");
      },
    },
    {
      id: "l6-observability",
      layer: "L6",
      label: "Observability — trace captured every prior layer",
      run: async (ctx) => {
        if (ctx.trace.length === 0) throw new Error("no spans captured");
        for (const s of ctx.trace) {
          if (typeof s.layer !== "string" || typeof s.status !== "string" || s.ms < 0) {
            throw new Error(`malformed span: ${s.id}`);
          }
        }
        return `${ctx.trace.length} spans recorded`;
      },
    },
    {
      id: "l7-deployment",
      layer: "L7",
      label: "Deployment — .evermind artifact ships + runs identically",
      run: async (ctx) => {
        const model = ctx.bag.model as EvermindLM | undefined;
        const tokenizer = ctx.bag.tokenizer as BPETokenizer | undefined;
        if (!model || !tokenizer) throw new Error("no model from L1 to deploy");
        const before = model.generateText("Deployment", tokenizer, { maxNewTokens: 4, temperature: 0 });
        const blob = EvermindModelPackage.fromLM(model, { name: "stack-demo", version: "1.0.0", card: { description: "diagnostic" } }).toBlob();
        const pkg = EvermindModelPackage.fromBlob(blob);
        if (!pkg.validate().ok) throw new Error("packaged artifact failed validation");
        const after = pkg.loadLM().generateText("Deployment", tokenizer, { maxNewTokens: 4, temperature: 0 });
        if (after !== before) throw new Error("deployed instance produced different output");
        return `artifact ${blob.byteLength} bytes, output matches`;
      },
    },
  ];
}

export interface StackDiagnosticResult {
  ok: boolean;
  steps: StackStepResult[];
  /** The first step that failed (the breaking point), if any. */
  firstFailure?: StackStepResult;
  totalMs: number;
}

/**
 * Run the steps in order, capturing each result. Never throws — a step that
 * throws becomes a `fail` row, and the run continues so the caller sees the full
 * timeline (and exactly where it broke). Pass `onStep` to stream rows live.
 */
export async function runStackDiagnostic(
  steps: StackStep[],
  opts: RunStackOptions = {},
): Promise<StackDiagnosticResult> {
  const now = opts.now ?? (() => Date.now());
  const ctx: StackContext = { bag: {}, trace: [], idbFactory: opts.idbFactory, now };
  const results: StackStepResult[] = [];
  let firstFailure: StackStepResult | undefined;
  const start = now();

  for (const step of steps) {
    const t0 = now();
    let result: StackStepResult;
    try {
      const detail = await step.run(ctx);
      result = { id: step.id, layer: step.layer, label: step.label, status: "pass", ms: now() - t0, ...(detail ? { detail } : {}) };
    } catch (e) {
      result = { id: step.id, layer: step.layer, label: step.label, status: "fail", ms: now() - t0, error: e instanceof Error ? e.message : String(e) };
      firstFailure ??= result;
    }
    results.push(result);
    ctx.trace.push(result);
    opts.onStep?.(result);
  }

  return { ok: !firstFailure, steps: results, ...(firstFailure ? { firstFailure } : {}), totalMs: now() - start };
}
