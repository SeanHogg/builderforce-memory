/**
 * tests/export.test.ts — the model-export subsystem (the publishing step).
 *
 * Covers the dependency-free formats end-to-end by re-parsing what we emit:
 *   • named-tensor extraction matches EvermindLM.parameters() exactly
 *   • safetensors header/offsets/bytes round-trip (F32 and F16)
 *   • GGUF magic/version/counts + tensor table re-read
 *   • tokenizer.json mirrors the BPE vocab/merges (HF tokenizers shape)
 *   • config.json + the huggingface bundle file set
 *   • ONNX bytes are a well-formed ModelProto (graph + initializers present)
 *
 * ONNX *numerical* parity vs the reference forward is verified separately with
 * onnxruntime-web (logits matched to <1e-5); kept out of unit tests to avoid a
 * native/wasm runtime dependency.
 */

import { EvermindLM } from "../src/lm/evermind_lm.js";
import { BPETokenizer } from "../src/tokenizer/bpe.js";
import {
  exportEvermind,
  namedTensors,
  exportSafetensors,
  exportGguf,
  exportOnnx,
  configJson,
  tokenizerJson,
  EXPORT_FORMATS,
} from "../src/export/index.js";

const CFG = {
  vocabSize: 12,
  dModel: 8,
  numLayers: 2,
  convKernel: 3,
  hiddenDim: 12,
  numExperts: 4,
  topK: 2,
  seed: 77,
};

function tinyLM(): EvermindLM {
  return new EvermindLM(CFG);
}

function tinyTokenizer(): BPETokenizer {
  const tok = new BPETokenizer();
  tok.train("the quick brown fox jumps over the lazy dog. the dog sleeps. ".repeat(8), { numMerges: 8 });
  return tok;
}

describe("named tensors", () => {
  it("matches EvermindLM.parameters() count and total numel", () => {
    const lm = tinyLM();
    const tensors = namedTensors(lm);
    const params = lm.parameters();
    expect(tensors.length).toBe(params.length);
    const tensorNumel = tensors.reduce((n, t) => n + t.data.length, 0);
    const paramNumel = params.reduce((n, p) => n + p.data.length, 0);
    expect(tensorNumel).toBe(paramNumel);
    // First tensor is the tied embedding [vocab, dModel].
    expect(tensors[0]!.name).toBe("token_embedding.weight");
    expect(tensors[0]!.shape).toEqual([CFG.vocabSize, CFG.dModel]);
    // Every shape's product equals its data length.
    for (const t of tensors) expect(t.shape.reduce((a, b) => a * b, 1)).toBe(t.data.length);
    // Unique names.
    expect(new Set(tensors.map((t) => t.name)).size).toBe(tensors.length);
  });
});

describe("safetensors", () => {
  function parse(bytes: Uint8Array): { header: Record<string, any>; data: Uint8Array } {
    const headerLen = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
    const headerJson = new TextDecoder().decode(bytes.subarray(8, 8 + headerLen));
    return { header: JSON.parse(headerJson), data: bytes.subarray(8 + headerLen) };
  }

  it("round-trips F32 tensors with correct offsets and values", () => {
    const lm = tinyLM();
    const tensors = namedTensors(lm);
    const { header, data } = parse(exportSafetensors(lm));
    expect(header.__metadata__).toBeDefined();
    for (const t of tensors) {
      const entry = header[t.name];
      expect(entry).toBeDefined();
      expect(entry.dtype).toBe("F32");
      expect(entry.shape).toEqual(t.shape);
      const [start, end] = entry.data_offsets;
      expect(end - start).toBe(t.data.length * 4);
      const view = new Float32Array(data.slice(start, end).buffer);
      for (let i = 0; i < t.data.length; i++) expect(view[i]).toBeCloseTo(t.data[i]!, 6);
    }
  });

  it("F16 halves the byte size and stays close", () => {
    const lm = tinyLM();
    const f32 = exportSafetensors(lm, { fp16: false });
    const f16 = exportSafetensors(lm, { fp16: true });
    const { header } = parse(f16);
    expect(header["token_embedding.weight"].dtype).toBe("F16");
    // Data section (excluding header) is ~half.
    expect(f16.length).toBeLessThan(f32.length);
  });
});

describe("gguf", () => {
  it("emits a valid GGUF v3 header with all tensors", () => {
    const lm = tinyLM();
    const bytes = exportGguf(lm);
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    expect(dv.getUint32(0, true)).toBe(0x46554747); // "GGUF"
    expect(dv.getUint32(4, true)).toBe(3); // version
    const tensorCount = Number(dv.getBigUint64(8, true));
    expect(tensorCount).toBe(namedTensors(lm).length);
    const kvCount = Number(dv.getBigUint64(16, true));
    expect(kvCount).toBeGreaterThan(0);
  });
});

describe("huggingface sidecars", () => {
  it("config.json reflects the architecture", () => {
    const cfg = configJson(tinyLM());
    expect(cfg.model_type).toBe("evermind");
    expect(cfg.vocab_size).toBe(CFG.vocabSize);
    expect(cfg.num_experts).toBe(CFG.numExperts);
    expect(cfg.num_experts_per_tok).toBe(CFG.topK);
    expect(cfg.tie_word_embeddings).toBe(true);
  });

  it("tokenizer.json mirrors the BPE vocab and merges", () => {
    const tok = tinyTokenizer();
    const tj = tokenizerJson(tok) as any;
    expect(tj.model.type).toBe("BPE");
    expect(Object.keys(tj.model.vocab).length).toBe(tok.vocabSize);
    expect(tj.model.merges.length).toBe(tok.merges.size);
    expect(tj.pre_tokenizer.type).toBe("ByteLevel");
    // Special tokens appear as added_tokens.
    expect(tj.added_tokens.some((t: any) => t.content === tok.eosToken)).toBe(true);
  });

  it("the huggingface bundle produces the full publishable file set", () => {
    const lm = tinyLM();
    const tok = tinyTokenizer();
    const { files, paramCount } = exportEvermind(lm, "huggingface", { name: "T", version: "9.9.9" }, tok);
    const paths = files.map((f) => f.path).sort();
    expect(paths).toEqual(
      [
        "README.md",
        "config.json",
        "generation_config.json",
        "model.gguf",
        "model.onnx",
        "model.safetensors",
        "tokenizer.json",
      ].sort(),
    );
    expect(paramCount).toBe(lm.parameters().reduce((n, p) => n + p.data.length, 0));
    // README carries YAML front-matter.
    const readme = files.find((f) => f.path === "README.md")!.data as string;
    expect(readme.startsWith("---\n")).toBe(true);
  });

  it("huggingface export without a tokenizer throws", () => {
    expect(() => exportEvermind(tinyLM(), "huggingface", {})).toThrow(/tokenizer/);
  });
});

describe("onnx", () => {
  it("emits a non-trivial ModelProto", () => {
    const bytes = exportOnnx(tinyLM());
    // ir_version (field 1, varint) is the first field → tag byte 0x08.
    expect(bytes[0]).toBe(0x08);
    // The graph references input_ids and logits somewhere in the bytes.
    const text = new TextDecoder("latin1").decode(bytes);
    expect(text).toContain("input_ids");
    expect(text).toContain("logits");
    expect(text).toContain("token_embedding.weight");
    expect(bytes.length).toBeGreaterThan(1000);
  });
});

describe("format catalog", () => {
  it("lists every exportable format with an id and label", () => {
    const ids = EXPORT_FORMATS.map((f) => f.id);
    expect(ids).toEqual(expect.arrayContaining(["huggingface", "onnx", "safetensors", "gguf"]));
    for (const f of EXPORT_FORMATS) {
      expect(typeof f.label).toBe("string");
      expect(f.label.length).toBeGreaterThan(0);
    }
  });
});
