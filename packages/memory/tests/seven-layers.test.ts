/**
 * tests/seven-layers.test.ts
 *
 * End-to-end test of the seven-layer agent stack from the BuilderForce blog
 * ("The Agent Tech Stack: All Seven Layers"), exercised headlessly in ONE run.
 *
 *   L1 Foundation Model   → EvermindLM + BPETokenizer (a real local model)
 *   L2 Orchestration      → a think→retrieve→recall→act loop wiring the layers
 *   L3 Memory             → EvermindCognition (write-through, replace-on-write)
 *   L4 RAG                → chunk + hybridRetrieve (BM25 + dense + fusion)
 *   L5 Tools              → a capability-gated tool registry
 *   L6 Observability      → a span/trace collector over every layer touched
 *   L7 Deployment         → EvermindModelPackage (.evermind artifact ships + runs)
 *
 * L1/L3/L4/L7 use the REAL shipped Evermind components. L2/L5/L6 use a minimal
 * in-process harness — their production forms live in the BuilderForce agent
 * runtime and need the live runtime (vendors/DB/Workers); this proves the stack
 * composes and each layer's contract holds.
 */

import "fake-indexeddb/auto";

import { EvermindLM, BPETokenizer, EvermindModelPackage } from "@seanhogg/builderforce-memory-engine";
import { EvermindCognition } from "../src/cognition/index.js";
import { hybridRetrieve, chunkText } from "../src/retrieval/index.js";
import { MemoryStore } from "../src/memory/MemoryStore.js";

// ── L6 Observability — a span collector the loop writes to ──────────────────────
interface Span {
  layer: string;
  op: string;
  ms: number;
}
class Trace {
  readonly spans: Span[] = [];
  record(layer: string, op: string, ms = 0): void {
    this.spans.push({ layer, op, ms });
  }
  byLayer(layer: string): Span[] {
    return this.spans.filter((s) => s.layer === layer);
  }
}

// ── L5 Tools — a capability-gated registry ──────────────────────────────────────
type Tool = (arg: string) => string | Promise<string>;
class ToolRegistry {
  private readonly tools = new Map<string, Tool>();
  register(name: string, fn: Tool): void {
    this.tools.set(name, fn);
  }
  has(name: string): boolean {
    return this.tools.has(name);
  }
  async call(name: string, arg: string): Promise<string> {
    const fn = this.tools.get(name);
    if (!fn) throw new Error(`tool '${name}' not registered`);
    return fn(arg);
  }
}

const KNOWLEDGE =
  "BuilderForce orchestrates many agents through a planning loop. " +
  "The memory layer stores facts as SSM embeddings. " +
  "Deployment runs on Cloudflare Workers and Durable Objects. " +
  "Tools are gated by a capability registry.";

describe("seven-layer agent stack — end to end", () => {
  let tokenizer: BPETokenizer;
  let model: EvermindLM;
  let cognition: EvermindCognition;
  let candidates: { id: string; text: string }[];
  const trace = new Trace();
  const tools = new ToolRegistry();

  beforeAll(() => {
    // L1 — train a tokenizer + build a model over its vocabulary.
    tokenizer = new BPETokenizer();
    tokenizer.train(KNOWLEDGE.repeat(4), { numMerges: 60 });
    model = new EvermindLM({
      vocabSize: tokenizer.vocabSize,
      dModel: 16,
      numLayers: 2,
      hiddenDim: 24,
      numExperts: 4,
      topK: 2,
      seed: 7,
    });
    // L3 — write-through memory over a real store.
    cognition = new EvermindCognition({ store: new MemoryStore({ dbName: `seven-${Date.now()}` }) });
    // L4 — chunk the knowledge into retrieval candidates.
    candidates = chunkText(KNOWLEDGE, { chunkSize: 70, chunkOverlap: 0 }).map((c, i) => ({
      id: `c${i}`,
      text: c.text,
    }));
    // L5 — register a tool.
    tools.register("shout", (a) => a.toUpperCase());
  });

  test("L1 Foundation Model — the local model + tokenizer generate text", () => {
    const out = model.generateText("Deployment", tokenizer, { maxNewTokens: 4, temperature: 0 });
    trace.record("L1", "generate");
    expect(typeof out).toBe("string");
    // The tokenizer round-trips real text (the model's I/O contract).
    expect(tokenizer.decode(tokenizer.encode("Cloudflare Workers"))).toBe("Cloudflare Workers");
  });

  test("L3 Memory — a re-learned fact supersedes its incumbent (replace-on-write)", async () => {
    await cognition.commit({ subjectKey: "fact:deploy", content: "deploy target = unknown" });
    const r = await cognition.commit({ subjectKey: "fact:deploy", content: "deploy target = Cloudflare" });
    trace.record("L3", `commit:${r.verdict}`);
    expect(r.verdict).toBe("supersede");
  });

  test("L4 RAG — chunk + hybrid retrieve surfaces the relevant passage", () => {
    const hits = hybridRetrieve({ text: "where does deployment run" }, candidates, { topK: 2 });
    trace.record("L4", "retrieve");
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0]!.text.toLowerCase()).toContain("cloudflare");
  });

  test("L5 Tools — the registry gates + invokes a capability", async () => {
    expect(tools.has("shout")).toBe(true);
    expect(await tools.call("shout", "deploy")).toBe("DEPLOY");
    await expect(tools.call("missing", "x")).rejects.toThrow(/not registered/);
  });

  test("L2 Orchestration — a loop wires retrieve→recall→act→generate across the layers", async () => {
    const steps: string[] = [];

    // retrieve (L4)
    const t4 = Date.now();
    const hits = hybridRetrieve({ text: "deployment cloudflare" }, candidates, { topK: 1 });
    trace.record("L4", "retrieve", Date.now() - t4);
    steps.push("retrieve");

    // recall (L3)
    const recalled = await cognition.recall("deployment", 3);
    trace.record("L3", "recall");
    steps.push("recall");
    expect(Array.isArray(recalled)).toBe(true);

    // act with a tool (L5)
    const toolOut = await tools.call("shout", hits[0]?.text ?? "deploy");
    trace.record("L5", "tool:shout");
    steps.push("tool");
    expect(toolOut).toBe(toolOut.toUpperCase());

    // model step (L1)
    const gen = model.generateText("Deployment", tokenizer, { maxNewTokens: 3, temperature: 0 });
    trace.record("L1", "generate");
    steps.push("model");
    expect(typeof gen).toBe("string");

    expect(steps).toEqual(["retrieve", "recall", "tool", "model"]);
  });

  test("L6 Observability — the trace captured spans across the layers touched", () => {
    // The orchestration run above (plus the per-layer tests) recorded spans.
    for (const layer of ["L1", "L3", "L4", "L5"]) {
      expect(trace.byLayer(layer).length).toBeGreaterThan(0);
    }
    // Every span is well-formed (the shape an exporter would emit).
    for (const s of trace.spans) {
      expect(typeof s.layer).toBe("string");
      expect(typeof s.op).toBe("string");
      expect(s.ms).toBeGreaterThanOrEqual(0);
    }
  });

  test("L7 Deployment — package as .evermind, then a deployed instance runs identically", () => {
    const before = model.generateText("Deployment", tokenizer, { maxNewTokens: 4, temperature: 0 });

    // Publish the deployable artifact.
    const blob = EvermindModelPackage.fromLM(model, {
      name: "stack-demo",
      version: "1.0.0",
      card: { description: "seven-layer e2e model", license: "MIT" },
    }).toBlob();

    // "Deploy": a fresh instance loads the artifact and serves the same output.
    const pkg = EvermindModelPackage.fromBlob(blob);
    expect(pkg.validate().ok).toBe(true);
    const deployed = pkg.loadLM();
    const after = deployed.generateText("Deployment", tokenizer, { maxNewTokens: 4, temperature: 0 });
    trace.record("L7", "deploy");

    expect(after).toEqual(before);
  });
});
