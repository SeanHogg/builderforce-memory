#!/usr/bin/env node
/**
 * generate-bin.js
 *
 * Creates an MBJS v2 checkpoint compatible with mambacode.js.
 *
 * The weights are NOT pretrained — use tools/pretrain.html to run
 * language-model training in the browser and produce meaningful output.
 *
 * Usage:
 *   node tools/generate-bin.js                            # nano mamba1 → model.bin
 *   node tools/generate-bin.js --size small               # small preset
 *   node tools/generate-bin.js --type mamba2              # Mamba-2 (SSD) architecture
 *   node tools/generate-bin.js --type mamba3              # Mamba-3 (complex states)
 *   node tools/generate-bin.js --type jamba               # Jamba hybrid (every 4th = attention)
 *   node tools/generate-bin.js --type zamba               # Zamba hybrid (every 6th = attention)
 *   node tools/generate-bin.js --size medium --out my.bin # custom output path
 *
 * Output can be loaded in pretrain.html via the Checkpoint URL field.
 */

import { writeFileSync } from 'fs';

// ── CLI args ──────────────────────────────────────────────────────────────────

const args  = process.argv.slice(2);
const flag  = (name) => { const i = args.indexOf(name); return i !== -1 ? args[i + 1] : null; };

// ── Size presets ──────────────────────────────────────────────────────────────

const SIZE_PRESETS = {
  nano  : { dModel: 128,  numLayers: 4,  nHeads: 4  },
  small : { dModel: 256,  numLayers: 6,  nHeads: 8  },
  medium: { dModel: 512,  numLayers: 8,  nHeads: 8  },
  large : { dModel: 768,  numLayers: 12, nHeads: 12 },
};

const VALID_TYPES = ['mamba1', 'mamba2', 'mamba3', 'jamba', 'zamba'];

const sizeName = flag('--size') ?? 'nano';
const typeName = flag('--type') ?? 'mamba1';

if (!SIZE_PRESETS[sizeName]) {
  console.error(`Unknown size "${sizeName}". Choose: ${Object.keys(SIZE_PRESETS).join(', ')}`);
  process.exit(1);
}
if (!VALID_TYPES.includes(typeName)) {
  console.error(`Unknown type "${typeName}". Choose: ${VALID_TYPES.join(', ')}`);
  process.exit(1);
}

const { dModel, numLayers, nHeads } = SIZE_PRESETS[sizeName];
const outPath = flag('--out') ?? 'model.bin';

// Qwen2.5-Coder vocab size (matches what mambacode.js / MambaKit expect)
const VOCAB_SIZE = 151936;

// ── Architecture constants ────────────────────────────────────────────────────

const D_STATE  = 16;                         // dState (N)
const D_CONV   = 4;                          // dConv  (K)
const EXPAND   = 2;                          // expand
const D_INNER  = EXPAND * dModel;            // dInner (D)
const DT_RANK  = Math.ceil(dModel / 16);     // dtRank (R) — Mamba-1 only
const N_GROUPS = 1;                          // nGroups (G) — Mamba-2/3

// Layer type codes for MBJS v2 header
const LAYER_TYPE_CODES = { mamba1: 0, mamba2: 1, mamba3: 2, attention: 3 };

// ── Resolve per-layer type schedule ──────────────────────────────────────────

function resolveSchedule(type, numLayers) {
  if (type === 'jamba') {
    // Jamba: every 4th layer is attention, rest mamba2
    return Array.from({ length: numLayers }, (_, i) =>
      i % 4 === 3 ? 'attention' : 'mamba2'
    );
  }
  if (type === 'zamba') {
    // Zamba: every 6th layer is attention, rest mamba3
    return Array.from({ length: numLayers }, (_, i) =>
      i % 6 === 5 ? 'attention' : 'mamba3'
    );
  }
  // Uniform: fill all layers with the specified type
  return Array(numLayers).fill(type);
}

const layerSchedule = resolveSchedule(typeName, numLayers);

// ── Random number generation (Box-Muller, seeded) ─────────────────────────────

let seed = 0x12345678;

function rand() {
  seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
  return (seed >>> 0) / 0xffffffff;
}

function randn(std = 0.02) {
  const u1 = Math.max(rand(), 1e-10);
  const u2  = rand();
  return std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function gaussianFill(buf, offset, n, std) {
  for (let i = 0; i < n; i++) buf[offset + i] = randn(std);
}

function zeroFill(buf, offset, n) {
  buf.fill(0, offset, offset + n);
}

function oneFill(buf, offset, n) {
  buf.fill(1, offset, offset + n);
}

// A_log for Mamba-1: log(n+1) per (d, n) entry
function mamba1AlogFill(buf, offset) {
  for (let d = 0; d < D_INNER; d++) {
    for (let n = 0; n < D_STATE; n++) {
      buf[offset + d * D_STATE + n] = Math.log(n + 1);
    }
  }
}

// A_log for Mamba-2: scalar log(1) per head
function mamba2AlogFill(buf, offset, nH) {
  buf.fill(0, offset, offset + nH);  // log(1) = 0
}

// A_log for Mamba-3: [log|A|, arg(A)] per head — evenly-spaced phases
function mamba3AlogFill(buf, offset, nH) {
  for (let h = 0; h < nH; h++) {
    buf[offset + h * 2 + 0] = 0.0;                        // log|A| = 0 → |A| = 1
    buf[offset + h * 2 + 1] = (2 * Math.PI * h) / nH;     // arg(A)
  }
}

// ── Collect parameter shapes per layer type ───────────────────────────────────

function mamba1Params(add) {
  const D = D_INNER;
  const R = DT_RANK;
  const N = D_STATE;
  const K = D_CONV;

  add(2 * D * dModel,                 (b, o) => gaussianFill(b, o, 2 * D * dModel, 0.02));         // wInProj
  add(2 * D,                          (b, o) => zeroFill(b, o, 2 * D));                             // bInProj
  add(D * K,                          (b, o) => gaussianFill(b, o, D * K, 0.01));                   // wConv
  add(D,                              (b, o) => zeroFill(b, o, D));                                 // bConv
  add((R + 2 * N) * D,                (b, o) => gaussianFill(b, o, (R + 2 * N) * D, 0.01));        // wXProj
  add(R + 2 * N,                      (b, o) => zeroFill(b, o, R + 2 * N));                         // bXProj
  add(D * R,                          (b, o) => gaussianFill(b, o, D * R, 0.02));                   // wDtProj
  add(D,                              (b, o) => zeroFill(b, o, D));                                  // bDtProj
  add(D * N,                          (b, o) => mamba1AlogFill(b, o));                              // A_log
  add(D,                              (b, o) => oneFill(b, o, D));                                   // D_vec
  add(dModel * D,                     (b, o) => gaussianFill(b, o, dModel * D, 0.02));              // wOutProj
  add(dModel,                         (b, o) => zeroFill(b, o, dModel));                            // bOutProj
  add(dModel,                         (b, o) => oneFill(b, o, dModel));                             // normWeight
}

function mamba2Params(add, H) {
  const D = D_INNER;
  const N = D_STATE;
  const K = D_CONV;
  const G = N_GROUPS;

  const inProjRows = D + 2 * G * N + H;
  const convD      = D + 2 * G * N;

  add(inProjRows * dModel,  (b, o) => gaussianFill(b, o, inProjRows * dModel, 0.02));  // wInProj
  add(convD * K,            (b, o) => gaussianFill(b, o, convD * K, 0.01));            // wConv
  add(convD,                (b, o) => zeroFill(b, o, convD));                           // bConv
  add(H,                    (b, o) => mamba2AlogFill(b, o, H));                        // A_log
  add(H,                    (b, o) => zeroFill(b, o, H));                              // dt_bias
  add(H,                    (b, o) => oneFill(b, o, H));                               // D_vec
  add(dModel * D,           (b, o) => gaussianFill(b, o, dModel * D, 0.02));           // wOutProj
  add(D,                    (b, o) => oneFill(b, o, D));                               // normWeight
  add(dModel,               (b, o) => oneFill(b, o, dModel));                          // preNormWeight
}

function mamba3Params(add, H) {
  const D  = D_INNER;
  const Nc = D_STATE;   // complex state count per head
  const K  = D_CONV;
  const G  = N_GROUPS;

  const inProjRows = D + 4 * G * Nc + H;  // x + B_re+B_im + C_re+C_im + A_log(per head)
  const convD      = D + 4 * G * Nc;      // x + B_re+B_im + C_re+C_im

  add(inProjRows * dModel,  (b, o) => gaussianFill(b, o, inProjRows * dModel, 0.02));  // wInProj
  add(convD * K,            (b, o) => gaussianFill(b, o, convD * K, 0.01));            // wConv
  add(convD,                (b, o) => zeroFill(b, o, convD));                           // bConv
  add(H * 2,                (b, o) => mamba3AlogFill(b, o, H));                        // A_log [log|A|, arg(A)] per head
  add(H,                    (b, o) => zeroFill(b, o, H));                              // dt_bias
  add(H,                    (b, o) => oneFill(b, o, H));                               // D_vec
  add(dModel * D,           (b, o) => gaussianFill(b, o, dModel * D, 0.02));           // wOutProj
  add(D,                    (b, o) => oneFill(b, o, D));                               // normWeight
  add(dModel,               (b, o) => oneFill(b, o, dModel));                          // preNormWeight
}

function attentionParams(add, H) {
  const D    = dModel;
  const dH   = Math.floor(D / H);
  const hasFfn = false;  // blank checkpoints default to no FFN sublayer

  add(3 * H * dH * D,  (b, o) => gaussianFill(b, o, 3 * H * dH * D, 0.02));  // wQKV
  add(3 * H * dH,      (b, o) => zeroFill(b, o, 3 * H * dH));                 // bQKV
  add(D * H * dH,      (b, o) => gaussianFill(b, o, D * H * dH, 0.02));       // wO
  add(D,               (b, o) => zeroFill(b, o, D));                           // bO
  add(D,               (b, o) => oneFill(b, o, D));                            // normWeight
}

function buildParamList() {
  const params = [];
  const add = (numel, fill) => params.push({ numel, fill });

  // Embedding table
  add(VOCAB_SIZE * dModel, (b, o) => gaussianFill(b, o, VOCAB_SIZE * dModel, 0.02));

  // Per-layer params
  for (let i = 0; i < numLayers; i++) {
    const t = layerSchedule[i];
    if      (t === 'mamba1')    mamba1Params(add);
    else if (t === 'mamba2')    mamba2Params(add, nHeads);
    else if (t === 'mamba3')    mamba3Params(add, nHeads);
    else if (t === 'attention') attentionParams(add, nHeads);
  }

  // Final RMSNorm
  add(dModel, (b, o) => oneFill(b, o, dModel));

  return params;
}

// ── Write MBJS v2 binary ──────────────────────────────────────────────────────

function writeMbjsV2(params, layerTypes, path) {
  const nL          = layerTypes.length;
  const typeCodes   = layerTypes.map(t => LAYER_TYPE_CODES[t] ?? 0);
  // pad layerType bytes to 4-byte alignment
  const typeByteLen = Math.ceil(nL / 4) * 4;

  const totalFloats = params.reduce((s, p) => s + p.numel, 0);
  const nParams     = params.length;

  // Header layout:
  //   [0..3]   magic 'MBJS'
  //   [4..7]   version = 2
  //   [8..11]  nLayers
  //   [12..12+typeByteLen-1]  layerType bytes (uint8), padded to 4 bytes
  //   [12+typeByteLen..+3]    nParams (uint32)
  //   [.. + nParams*4]        numel per param (uint32)
  //   [data]   float32 values

  const headerBytes = 12 + typeByteLen + 4 + nParams * 4;
  const dataBytes   = totalFloats * 4;
  const buf         = Buffer.allocUnsafe(headerBytes + dataBytes);

  let pos = 0;
  buf.writeUInt32LE(0x4D424A53, pos); pos += 4;  // magic 'MBJS'
  buf.writeUInt32LE(2,          pos); pos += 4;  // version 2
  buf.writeUInt32LE(nL,         pos); pos += 4;  // nLayers

  for (let i = 0; i < nL; i++) buf[pos + i] = typeCodes[i];
  pos += typeByteLen;

  buf.writeUInt32LE(nParams, pos); pos += 4;
  for (const p of params) { buf.writeUInt32LE(p.numel, pos); pos += 4; }

  // Float data
  const f32 = new Float32Array(totalFloats);
  let f32Off = 0;
  for (const p of params) { p.fill(f32, f32Off); f32Off += p.numel; }

  Buffer.from(f32.buffer).copy(buf, headerBytes);

  writeFileSync(path, buf);
  return { nParams, totalFloats };
}

// ── Main ──────────────────────────────────────────────────────────────────────

const baseType = layerSchedule[0];
const attLayers = layerSchedule.filter(t => t === 'attention').length;

console.log(`\nMambaCode.js generate-bin`);
console.log(`  Size:       ${sizeName} (dModel=${dModel}, numLayers=${numLayers}, nHeads=${nHeads})`);
console.log(`  Type:       ${typeName}${attLayers ? ` (${numLayers - attLayers} SSM + ${attLayers} attention layers)` : ''}`);
console.log(`  Vocab:      ${VOCAB_SIZE.toLocaleString()} tokens (Qwen2.5-Coder)`);
console.log(`  D_INNER:    ${D_INNER}  D_STATE: ${D_STATE}  D_CONV: ${D_CONV}`);
console.log(`  Output:     ${outPath}`);
console.log('');

const params = buildParamList();
const totalFloats = params.reduce((s, p) => s + p.numel, 0);
const sizeMb = (totalFloats * 4 / 1024 / 1024).toFixed(1);
console.log(`  Parameters: ${params.length} tensors, ${totalFloats.toLocaleString()} floats (${sizeMb} MB)`);
console.log('  Writing…');

writeMbjsV2(params, layerSchedule, outPath);

console.log(`  Done → ${outPath}\n`);
console.log(`Next steps:`);
console.log(`  1. Serve it:  npm run build && npm run serve`);
console.log(`  2. Open http://localhost:3000/tools/pretrain.html and train on a text corpus`);
console.log(`  3. Or load directly via the Checkpoint URL field in pretrain.html\n`);
