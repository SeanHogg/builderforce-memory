/**
 * publish/huggingface.ts — push an exported Evermind model to the Hugging Face Hub.
 *
 * The runtime stays zero-dependency: the actual Hub I/O is delegated to an
 * injectable {@link HubClient}. In production it is dynamically imported from the
 * optional `@huggingface/hub` package (which handles LFS for the binary weight
 * files correctly); tests inject a fake. There is also a credential-free path —
 * {@link writeExportToDir} — that writes the repo to a local folder you push with
 * `git` / `huggingface-cli upload`.
 *
 * The export itself (turning a trained model into the file set) lives in the
 * engine's `export/` module; this module is only the *transport*.
 */

import type { ExportResult, ExportFile } from "@seanhogg/builderforce-memory-engine";

/** Where/how to publish. `repoId` is "owner/name" (e.g. "builderforce/Evermind"). */
export interface HuggingFaceTarget {
  repoId: string;
  /** A write-scoped HF access token (hf_…). */
  token: string;
  /** Create the repo private. Default false (public). */
  private?: boolean;
  /** Commit title. Default "Publish <repoId>". */
  commitTitle?: string;
  /** Target branch. Default "main". */
  branch?: string;
}

export interface PublishOutcome {
  repoId: string;
  url: string;
  files: string[];
}

/** The slice of `@huggingface/hub` we use — injectable for testing. */
export interface HubClient {
  createRepo(args: {
    repo: { type: "model"; name: string };
    accessToken: string;
    private?: boolean;
  }): Promise<unknown>;
  uploadFiles(args: {
    repo: { type: "model"; name: string };
    accessToken: string;
    branch?: string;
    commitTitle?: string;
    files: { path: string; content: Blob }[];
  }): Promise<unknown>;
}

/** Turn an emitted file into a Blob (binary or text) for upload. */
function toBlob(file: ExportFile): Blob {
  const part = (typeof file.data === "string" ? file.data : file.data) as unknown as BlobPart;
  return new Blob([part], { type: file.contentType });
}

async function loadHub(): Promise<HubClient> {
  try {
    // Optional peer dep — only required when actually uploading. A variable
    // specifier keeps it out of the static module graph (no build-time dep).
    const spec = "@huggingface/hub";
    const mod = (await import(spec)) as unknown as HubClient;
    if (typeof mod.createRepo !== "function" || typeof mod.uploadFiles !== "function") {
      throw new Error("module shape unexpected");
    }
    return mod;
  } catch {
    throw new Error(
      "publishToHuggingFace needs the optional '@huggingface/hub' package " +
        "(npm i @huggingface/hub) — or use writeExportToDir() + `huggingface-cli upload`.",
    );
  }
}

/**
 * Publish an export to the Hub: create the repo (idempotent) then upload every
 * file. Pass `deps.hub` to inject a client; otherwise `@huggingface/hub` is
 * loaded dynamically. Requires a write-scoped token (the credential-gated step).
 */
export async function publishToHuggingFace(
  result: ExportResult,
  target: HuggingFaceTarget,
  deps: { hub?: HubClient } = {},
): Promise<PublishOutcome> {
  if (!target.repoId.includes("/")) {
    throw new Error(`publishToHuggingFace: repoId must be "owner/name" (got "${target.repoId}")`);
  }
  if (!target.token) throw new Error("publishToHuggingFace: a write-scoped HF token is required");

  const hub = deps.hub ?? (await loadHub());
  const repo = { type: "model" as const, name: target.repoId };

  try {
    await hub.createRepo({ repo, accessToken: target.token, private: target.private ?? false });
  } catch (e) {
    // Re-creating an existing repo is fine; only rethrow unexpected failures.
    const msg = e instanceof Error ? e.message : String(e);
    if (!/exist|409|conflict/i.test(msg)) throw e;
  }

  const files = result.files.map((f) => ({ path: f.path, content: toBlob(f) }));
  await hub.uploadFiles({
    repo,
    accessToken: target.token,
    branch: target.branch ?? "main",
    commitTitle: target.commitTitle ?? `Publish ${target.repoId}`,
    files,
  });

  return { repoId: target.repoId, url: `https://huggingface.co/${target.repoId}`, files: files.map((f) => f.path) };
}

/** A minimal fs surface (Node `fs/promises`) — injectable for testing. */
export interface FsLike {
  mkdir(path: string, opts: { recursive: boolean }): Promise<unknown>;
  writeFile(path: string, data: Uint8Array | string): Promise<void>;
}

/**
 * Credential-free publish: write the export to a local directory (a ready-to-push
 * HF repo folder). Returns the relative paths written. Inject `fs` in tests;
 * otherwise Node's `fs/promises` is used.
 */
export async function writeExportToDir(result: ExportResult, dir: string, fs?: FsLike): Promise<string[]> {
  const spec = "node:fs/promises";
  const realFs = fs ?? ((await import(spec)) as unknown as FsLike);
  const sep = dir.endsWith("/") || dir.endsWith("\\") ? "" : "/";
  await realFs.mkdir(dir, { recursive: true });
  const written: string[] = [];
  for (const f of result.files) {
    const full = `${dir}${sep}${f.path}`;
    const slash = Math.max(full.lastIndexOf("/"), full.lastIndexOf("\\"));
    if (slash > dir.length) await realFs.mkdir(full.slice(0, slash), { recursive: true });
    await realFs.writeFile(full, f.data);
    written.push(f.path);
  }
  return written;
}
