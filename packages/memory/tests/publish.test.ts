/**
 * tests/publish.test.ts — shipping an exported Evermind model.
 *
 * The export bytes are the engine's job (tested there); here we test the
 * *transport*: writing the repo to a local folder (credential-free) and uploading
 * to the Hugging Face Hub through an injected client (no real network/token).
 */

import { EvermindLM, BPETokenizer, exportEvermind } from "@seanhogg/builderforce-memory-engine";
import { writeExportToDir, publishToHuggingFace, type HubClient, type FsLike } from "../src/publish/index.js";

function fixtureExport() {
  const lm = new EvermindLM({ vocabSize: 12, dModel: 8, numLayers: 2, hiddenDim: 12, seed: 1 });
  const tok = new BPETokenizer();
  tok.train("hello world. the cat sat. ".repeat(8), { numMerges: 6 });
  return exportEvermind(lm, "huggingface", { name: "Evermind", version: "1.0.0" }, tok);
}

describe("writeExportToDir", () => {
  it("writes every file via the injected fs and returns their paths", async () => {
    const result = fixtureExport();
    const written: { path: string; len: number }[] = [];
    const fs: FsLike = {
      async mkdir() {},
      async writeFile(path, data) {
        written.push({ path, len: typeof data === "string" ? data.length : data.length });
      },
    };
    const paths = await writeExportToDir(result, "/out", fs);
    expect(paths).toEqual(result.files.map((f) => f.path));
    expect(written.length).toBe(result.files.length);
    expect(written.every((w) => w.len > 0)).toBe(true);
  });
});

describe("publishToHuggingFace", () => {
  function spyHub() {
    const calls = { createRepo: [] as any[], uploadFiles: [] as any[] };
    const hub: HubClient = {
      async createRepo(a) {
        calls.createRepo.push(a);
      },
      async uploadFiles(a) {
        calls.uploadFiles.push(a);
      },
    };
    return { hub, calls };
  }

  it("creates the repo and uploads every file as a Blob", async () => {
    const result = fixtureExport();
    const { hub, calls } = spyHub();
    const outcome = await publishToHuggingFace(result, { repoId: "builderforce/Evermind", token: "hf_test" }, { hub });
    expect(calls.createRepo).toHaveLength(1);
    expect(calls.createRepo[0].repo.name).toBe("builderforce/Evermind");
    expect(calls.createRepo[0].accessToken).toBe("hf_test");
    expect(calls.uploadFiles).toHaveLength(1);
    expect(calls.uploadFiles[0].files).toHaveLength(result.files.length);
    expect(calls.uploadFiles[0].files.every((f: any) => f.content instanceof Blob)).toBe(true);
    expect(outcome.url).toBe("https://huggingface.co/builderforce/Evermind");
  });

  it("swallows an 'already exists' createRepo error", async () => {
    const result = fixtureExport();
    const hub: HubClient = {
      async createRepo() {
        throw new Error("repo already exists (409)");
      },
      async uploadFiles() {},
    };
    await expect(publishToHuggingFace(result, { repoId: "a/b", token: "t" }, { hub })).resolves.toBeDefined();
  });

  it("rejects a repoId without an owner and a missing token", async () => {
    const result = fixtureExport();
    const { hub } = spyHub();
    await expect(publishToHuggingFace(result, { repoId: "noslash", token: "t" }, { hub })).rejects.toThrow(/owner\/name/);
    await expect(publishToHuggingFace(result, { repoId: "a/b", token: "" }, { hub })).rejects.toThrow(/token/);
  });
});
