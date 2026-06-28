#!/usr/bin/env node
/**
 * export-hf.js — package a trained Evermind model as a Hugging Face repo and
 * (optionally) push it with the `hf` CLI.
 *
 * The engine already knows how to turn a trained {@link EvermindLM} + tokenizer
 * into a complete HF bundle (model.safetensors + model.onnx + model.gguf +
 * config.json + generation_config.json + tokenizer.json + README.md) via
 * {@link exportEvermind}. This is the Node-side runner that makes that usable
 * from a terminal: read a model from disk → emit the bundle into a CLEAN staging
 * directory (never the repo root) → optionally `hf upload`.
 *
 * It reads two on-disk inputs, the same two artifacts the Studio publish flow
 * stores in R2:
 *   --model      a `.evermind` package blob (EvermindModelPackage.toBlob())
 *   --tokenizer  a tokenizer JSON — either the raw { vocab, merges } descriptor
 *                or a full HF tokenizer.json ({ model: { vocab, merges } })
 *
 * Usage:
 *   node tools/export-hf.js --model model.evermind --tokenizer tokenizer.json \
 *        --out ./hf-export/Evermind --name Evermind --version 2026.6.28 \
 *        --license mit --author "Sean Hogg" --fp16
 *
 *   # write the bundle AND push it to the Hub (needs HF_TOKEN or `hf auth login`):
 *   node tools/export-hf.js --model model.evermind --tokenizer tokenizer.json \
 *        --out ./hf-export/Evermind --repo builderforce/Evermind --upload
 *
 *   # no model yet? prove the whole pipeline end-to-end with a tiny trained model:
 *   node tools/export-hf.js --demo --out ./hf-export/demo
 *
 * Flags:
 *   --model <path>       .evermind package blob (required unless --demo)
 *   --tokenizer <path>   tokenizer JSON (required unless --demo)
 *   --out <dir>          staging directory to write into (default ./hf-export/<name>)
 *   --format <fmt>       huggingface | safetensors | onnx | gguf (default huggingface)
 *   --name <str>         model name (default from package manifest, else "Evermind")
 *   --version <str>      version string (default from package manifest)
 *   --license <spdx>     license id for the model card (default mit)
 *   --author <str>       author for the model card
 *   --description <str>  one-line description for the model card
 *   --tags <a,b,c>       extra model-card tags (comma-separated)
 *   --fp16               store weights as float16 where the format allows
 *   --repo <id>          Hugging Face repo id (e.g. builderforce/Evermind)
 *   --upload             after writing, run `hf upload <repo> <out> .`
 *   --demo               train a tiny model in-process and export it (no inputs)
 */

import { mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

import {
  EvermindLM,
  EvermindLMTrainer,
  EvermindModelPackage,
  BPETokenizer,
  exportEvermind,
} from '../dist/index.js';

// ── CLI parsing ────────────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
const has = (name) => argv.includes(name);
const flag = (name, fallback = null) => {
  const i = argv.indexOf(name);
  return i !== -1 && i + 1 < argv.length ? argv[i + 1] : fallback;
};

const opts = {
  model: flag('--model'),
  tokenizer: flag('--tokenizer'),
  out: flag('--out'),
  format: flag('--format', 'huggingface'),
  name: flag('--name'),
  version: flag('--version'),
  license: flag('--license', 'mit'),
  author: flag('--author'),
  description: flag('--description'),
  tags: flag('--tags'),
  fp16: has('--fp16'),
  repo: flag('--repo'),
  upload: has('--upload'),
  demo: has('--demo'),
};

const VALID_FORMATS = ['huggingface', 'safetensors', 'onnx', 'gguf'];
if (!VALID_FORMATS.includes(opts.format)) {
  fail(`Unknown --format "${opts.format}". Choose: ${VALID_FORMATS.join(', ')}`);
}

function fail(msg) {
  console.error(`\n  ✗ ${msg}\n`);
  process.exit(1);
}

// ── Load the model + tokenizer (or build a demo) ─────────────────────────────────

/** Reconstruct a BPETokenizer from either a raw { vocab, merges } or an HF tokenizer.json. */
function tokenizerFromJson(json) {
  // HF tokenizer.json nests the BPE model; the raw descriptor is flat.
  const src = json && typeof json === 'object' && json.model && typeof json.model === 'object' ? json.model : json;
  const vocab = src?.vocab;
  let merges = src?.merges;
  if (!vocab || typeof vocab !== 'object' || !Array.isArray(merges)) {
    fail('tokenizer JSON must have { vocab, merges } (raw descriptor) or { model: { vocab, merges } } (HF format)');
  }
  // HF's newer format stores merges as ["a", "b"] pairs; our loader wants "a b" strings.
  merges = merges.map((m) => (Array.isArray(m) ? m.join(' ') : String(m)));
  const tok = new BPETokenizer();
  tok.loadFromObjects(vocab, merges);
  return tok;
}

async function loadInputs() {
  if (opts.demo) return buildDemo();

  if (!opts.model) fail('--model <path to .evermind> is required (or pass --demo)');
  if (!existsSync(opts.model)) fail(`model not found: ${opts.model}`);
  const needsTokenizer = opts.format === 'huggingface';
  if (needsTokenizer && !opts.tokenizer) fail('--tokenizer <path> is required for the huggingface bundle');
  if (opts.tokenizer && !existsSync(opts.tokenizer)) fail(`tokenizer not found: ${opts.tokenizer}`);

  const blob = await readFile(opts.model);
  const ab = blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength);
  let pkg;
  try {
    pkg = EvermindModelPackage.fromBlob(ab);
  } catch (err) {
    fail(`could not parse .evermind package: ${err.message}`);
  }
  const verdict = pkg.validate();
  if (!verdict.ok) fail(`invalid .evermind package: ${verdict.errors.join('; ')}`);
  if (pkg.manifest.modelType !== 'evermind-lm') {
    fail(`package is '${pkg.manifest.modelType}', not a runnable 'evermind-lm' — cannot export to HF`);
  }
  const lm = pkg.loadLM();

  let tokenizer;
  if (opts.tokenizer) {
    const tokJson = JSON.parse(await readFile(opts.tokenizer, 'utf8'));
    tokenizer = tokenizerFromJson(tokJson);
    if (tokenizer.vocabSize !== lm.config.vocabSize) {
      fail(`tokenizer vocab (${tokenizer.vocabSize}) ≠ model vocabSize (${lm.config.vocabSize}) — mismatched pair`);
    }
  }

  return {
    lm,
    tokenizer,
    nativeBlob: new Uint8Array(ab),
    meta: {
      name: opts.name ?? pkg.manifest.name ?? 'Evermind',
      version: opts.version ?? pkg.manifest.version,
    },
  };
}

/** Train a tiny but real Evermind model so the export pipeline is verifiable with no inputs. */
function buildDemo() {
  console.log('  --demo: training a tiny Evermind model in-process…');
  const corpus = [
    'evermind is a self-updating state space model.',
    'builderforce trains custom models in the browser.',
    'the agent remembers facts through write-through cognition.',
    'a shared expert mixture routes tokens to experts.',
    'memory replaces knowledge instead of appending to it.',
  ];
  const tokenizer = new BPETokenizer();
  tokenizer.train(corpus, { numMerges: 80, minPairFreq: 1 });

  const lm = new EvermindLM({
    vocabSize: tokenizer.vocabSize,
    dModel: 32,
    numLayers: 2,
    convKernel: 3,
    hiddenDim: 64,
    numExperts: 4,
    topK: 2,
    seed: 0x45564d44,
  });
  const sequences = corpus.map((line) => tokenizer.encode(line, { addBos: true, addEos: true })).filter((s) => s.length >= 2);
  const trainer = new EvermindLMTrainer(lm, { lr: 0.01, epochs: 30 });
  const history = trainer.fit(sequences);
  console.log(`  --demo: trained ${tokenizer.vocabSize} vocab, loss ${history[0].toFixed(3)} → ${history[history.length - 1].toFixed(3)}`);

  // Round-trip through a real .evermind package so the demo also covers that path
  // (and so `model.evermind` lands in the bundle just like the non-demo flow).
  const pkg = EvermindModelPackage.fromLM(lm, {
    name: opts.name ?? 'Evermind-demo',
    version: opts.version ?? '0.0.0',
    card: { description: 'Tiny demo model trained by tools/export-hf.js --demo.', license: opts.license ?? 'mit' },
  });
  const nativeBlob = new Uint8Array(pkg.toBlob());
  const reloaded = EvermindModelPackage.fromBlob(nativeBlob.buffer.slice(0)).loadLM();

  return {
    lm: reloaded,
    tokenizer,
    nativeBlob,
    meta: { name: opts.name ?? 'Evermind-demo', version: opts.version ?? '0.0.0' },
  };
}

// ── Write the bundle ─────────────────────────────────────────────────────────────

async function main() {
  const { lm, tokenizer, nativeBlob, meta } = await loadInputs();

  const name = meta.name ?? 'Evermind';
  const outDir = path.resolve(opts.out ?? path.join('hf-export', name));

  const exportOpts = {
    fp16: opts.fp16,
    name,
    ...(meta.version ? { version: meta.version } : {}),
    ...(opts.license ? { license: opts.license } : {}),
    ...(opts.author ? { author: opts.author } : {}),
    ...(opts.description ? { description: opts.description } : {}),
    ...(opts.tags ? { tags: opts.tags.split(',').map((t) => t.trim()).filter(Boolean) } : {}),
  };

  const result = exportEvermind(lm, opts.format, exportOpts, tokenizer);

  // Clean staging dir so a re-run never mixes stale files into the upload.
  await rm(outDir, { recursive: true, force: true });
  await mkdir(outDir, { recursive: true });

  for (const file of result.files) {
    const dest = path.join(outDir, file.path);
    await mkdir(path.dirname(dest), { recursive: true });
    await writeFile(dest, typeof file.data === 'string' ? file.data : Buffer.from(file.data));
  }

  // The HF README advertises a native model.evermind — include it so the claim holds
  // and the package stays runnable in @seanhogg/builderforce-memory.
  if (opts.format === 'huggingface' && nativeBlob) {
    await writeFile(path.join(outDir, 'model.evermind'), Buffer.from(nativeBlob));
  }

  console.log(`\n  ✓ wrote ${opts.format} bundle (${result.paramCount.toLocaleString()} params) → ${outDir}`);
  for (const file of result.files) console.log(`      ${file.path}`);
  if (opts.format === 'huggingface' && nativeBlob) console.log('      model.evermind');

  if (opts.upload) await upload(outDir);
  else if (opts.repo) {
    console.log(`\n  To publish:\n      hf upload ${opts.repo} ${outDir} .\n  (add --upload to do this automatically)`);
  }
}

// ── Upload via the hf CLI ────────────────────────────────────────────────────────

function upload(outDir) {
  if (!opts.repo) fail('--upload requires --repo <id> (e.g. builderforce/Evermind)');

  const hasToken = !!(process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN);
  const hasLogin = existsSync(path.join(process.env.USERPROFILE || process.env.HOME || '', '.cache', 'huggingface', 'token'));
  if (!hasToken && !hasLogin) {
    fail('not authenticated — set HF_TOKEN or run `hf auth login` before --upload');
  }

  console.log(`\n  ↑ hf upload ${opts.repo} ${outDir} .`);
  const res = spawnSync('hf', ['upload', opts.repo, outDir, '.', '--repo-type', 'model'], {
    stdio: 'inherit',
    shell: process.platform === 'win32', // resolve hf.exe / hf.cmd on Windows
    env: process.env,
  });
  if (res.error) fail(`hf upload failed to start: ${res.error.message} (is the hf CLI installed and on PATH?)`);
  if (res.status !== 0) fail(`hf upload exited with code ${res.status}`);
  console.log(`\n  ✓ published → https://huggingface.co/${opts.repo}\n`);
}

main().catch((err) => fail(err.stack || err.message));
