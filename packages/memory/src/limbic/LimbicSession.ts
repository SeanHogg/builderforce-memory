/**
 * LimbicSession.ts – high-level facade over the limbic affect model.
 *
 * Mirrors {@link MambaSession}: collapses GPU acquisition, model construction,
 * checkpoint load, and the trainer into a single `LimbicSession.create()` call,
 * with the same WebGPU-or-CPU-fallback contract. The agent runtime consumes
 * this through `@seanhogg/builderforce-memory` to run the limbic system on a
 * self-hosted node (GPU via @webgpu/node when present, CPU otherwise).
 *
 *   const limbic = await LimbicSession.create({ gpuAdapter, checkpointBuffer });
 *   const { delta, reward } = await limbic.step(experienceEmbedding, state);
 *   await limbic.train(samples, { epochs: 30 });
 *   const bin = limbic.exportWeights({ fp16: true });
 */

import {
  LimbicModel,
  LimbicTrainer,
  LIMBIC_AFFECT_WGSL,
  createStorageBuffer,
  createEmptyStorageBuffer,
  createUniformBuffer,
  createComputePipeline,
  createBindGroup,
  dispatchKernel,
  readBuffer,
  personalitySetpoint,
  clampState,
  neutralState,
  LIMBIC_STATE_DIM,
  type LimbicModelConfig,
  type LimbicForward,
  type LimbicSample,
  type LimbicTrainOptions,
  type PersonalityTraits,
} from "@seanhogg/builderforce-memory-engine";

import { saveToIndexedDB, loadFromIndexedDB } from "../session/persistence.js";

export type LimbicGpuMode = "webgpu" | "cpu-fallback" | "cpu";

export interface LimbicSessionOptions {
  /** Pre-created GPUAdapter (e.g. from @webgpu/node). When set, navigator.gpu is not used. */
  gpuAdapter?: GPUAdapter;
  /** Attempt a software (CPU) WebGPU adapter when no GPU is available. Default false. */
  allowCpuFallback?: boolean;
  /** Pre-read checkpoint bytes (Node: read the .bin with fs and pass the ArrayBuffer). */
  checkpointBuffer?: ArrayBuffer;
  /** IndexedDB key for save()/load(). Default 'limbic-default'. */
  name?: string;
  /** Injected IDBFactory (e.g. fake-indexeddb in Node). */
  idbFactory?: IDBFactory;
  /** Model configuration overrides. */
  modelConfig?: Partial<LimbicModelConfig>;
  /** Deterministic init seed. */
  seed?: number;
  /**
   * The agent's static personality traits (0..100). Mapped to the resting affective
   * SETPOINT the learned dynamics ride on top of, so training + relaxation happen
   * around the personality baseline instead of neutral ("personality = setpoints,
   * limbic = dynamics"). Ignored when {@link personalitySetpoint} is given.
   */
  personalityTraits?: PersonalityTraits;
  /**
   * An explicit 8-dim resting setpoint (e.g. one already derived by the runtime's
   * `deriveLimbicSetpoints`). Overrides {@link personalityTraits}. Clamped to bounds.
   */
  personalitySetpoint?: ArrayLike<number>;
}

interface StepBuffers {
  win: GPUBuffer;
  ws: GPUBuffer;
  aLogit: GPUBuffer;
  woutState: GPUBuffer;
  boutState: GPUBuffer;
}

export class LimbicSession {
  readonly model: LimbicModel;
  readonly trainer: LimbicTrainer;
  readonly device: GPUDevice | null;
  readonly gpuMode: LimbicGpuMode;
  /** The personality-conditioned resting setpoint (8-dim). NEUTRAL when no
   *  personality was supplied. The homeostatic baseline the learned dynamics
   *  settle toward — see {@link baselineState}. */
  readonly setpoint: Float32Array;
  private readonly _name: string;
  private readonly _idbFactory: IDBFactory | undefined;

  // GPU step pipeline + buffers, allocated lazily on first GPU step.
  private _pipeline: GPUComputePipeline | null = null;
  private _dimsBuf: GPUBuffer | null = null;
  private _paramBufs: StepBuffers | null = null;
  private _paramsDirty = true;

  private constructor(
    model: LimbicModel,
    trainer: LimbicTrainer,
    device: GPUDevice | null,
    gpuMode: LimbicGpuMode,
    name: string,
    idbFactory: IDBFactory | undefined,
    setpoint: Float32Array,
  ) {
    this.model = model;
    this.trainer = trainer;
    this.device = device;
    this.gpuMode = gpuMode;
    this._name = name;
    this._idbFactory = idbFactory;
    this.setpoint = setpoint;
  }

  /**
   * A fresh copy of the personality-conditioned resting state. Seed training-sample
   * states and initial affect from this so the learned dynamics move AROUND the
   * agent's personality baseline rather than a fixed neutral. NEUTRAL when the
   * session was created with no personality.
   */
  baselineState(): Float32Array {
    return Float32Array.from(this.setpoint);
  }

  static async create(options: LimbicSessionOptions = {}): Promise<LimbicSession> {
    let device: GPUDevice | null = null;
    let gpuMode: LimbicGpuMode = "cpu";

    if (options.gpuAdapter != null) {
      try {
        device = await options.gpuAdapter.requestDevice();
        gpuMode = "webgpu";
      } catch {
        device = null;
        gpuMode = "cpu";
      }
    } else if (typeof navigator !== "undefined" && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
        if (adapter) {
          device = await adapter.requestDevice();
          gpuMode = "webgpu";
        }
      } catch {
        device = null;
      }
      if (!device && options.allowCpuFallback && typeof navigator !== "undefined" && navigator.gpu) {
        try {
          const fb = await navigator.gpu.requestAdapter({ forceFallbackAdapter: true });
          if (fb) {
            device = await fb.requestDevice();
            gpuMode = "cpu-fallback";
          }
        } catch {
          device = null;
        }
      }
    }

    const model = new LimbicModel({ ...options.modelConfig, seed: options.seed });
    if (options.checkpointBuffer) {
      model.loadWeights(options.checkpointBuffer);
    }
    const trainer = new LimbicTrainer(model, device);
    // Resolve the resting setpoint: an explicit 8-dim vector wins; else map the
    // personality traits; else neutral. This is where the STATIC personality enters
    // the (dynamic, trainable) limbic system.
    let setpoint: Float32Array;
    if (options.personalitySetpoint && options.personalitySetpoint.length >= LIMBIC_STATE_DIM) {
      setpoint = clampState(Float32Array.from(options.personalitySetpoint));
    } else if (options.personalityTraits) {
      setpoint = personalitySetpoint(options.personalityTraits);
    } else {
      setpoint = neutralState();
    }
    return new LimbicSession(model, trainer, device, gpuMode, options.name ?? "limbic-default", options.idbFactory, setpoint);
  }

  /** One affect step. Uses the GPU kernel when a device is available, else CPU. */
  async step(input: ArrayLike<number>, state: ArrayLike<number>, hidden?: ArrayLike<number>): Promise<LimbicForward> {
    const h = hidden ?? this.model.initHidden();
    if (!this.device) return this.model.forward(input, h, state);
    return this._stepGpu(input, h, state);
  }

  private _ensureGpuStep(): { pipeline: GPUComputePipeline; params: StepBuffers; dims: GPUBuffer } {
    const device = this.device!;
    if (!this._pipeline) {
      this._pipeline = createComputePipeline(device, LIMBIC_AFFECT_WGSL, "affect_step");
      const { inputDim, hiddenDim, stateDim } = this.model.config;
      const dims = new ArrayBuffer(16);
      new Uint32Array(dims).set([inputDim, hiddenDim, stateDim, 0]);
      this._dimsBuf = createUniformBuffer(device, dims);
    }
    if (this._paramsDirty || !this._paramBufs) {
      this._destroyParamBufs();
      this._paramBufs = {
        win: createStorageBuffer(device, this.model.win, false),
        ws: createStorageBuffer(device, this.model.ws, false),
        aLogit: createStorageBuffer(device, this.model.aLogit, false),
        woutState: createStorageBuffer(device, this.model.woutState, false),
        boutState: createStorageBuffer(device, this.model.boutState, false),
      };
      this._paramsDirty = false;
    }
    return { pipeline: this._pipeline, params: this._paramBufs, dims: this._dimsBuf! };
  }

  private async _stepGpu(input: ArrayLike<number>, hPrev: ArrayLike<number>, sPrev: ArrayLike<number>): Promise<LimbicForward> {
    const device = this.device!;
    const { inputDim, hiddenDim, stateDim } = this.model.config;
    const { pipeline, params, dims } = this._ensureGpuStep();

    const xBuf = createStorageBuffer(device, Float32Array.from({ length: inputDim }, (_, i) => input[i] ?? 0), false);
    const hBuf = createStorageBuffer(device, Float32Array.from({ length: hiddenDim }, (_, j) => hPrev[j] ?? 0), false);
    const sBuf = createStorageBuffer(device, Float32Array.from({ length: stateDim }, (_, k) => sPrev[k] ?? 0), false);
    const hOut = createEmptyStorageBuffer(device, hiddenDim * 4, true);
    const dOut = createEmptyStorageBuffer(device, stateDim * 4, true);

    const bg = createBindGroup(device, pipeline, [
      dims,
      params.win,
      params.ws,
      params.aLogit,
      params.woutState,
      params.boutState,
      xBuf,
      hBuf,
      sBuf,
      hOut,
      dOut,
    ]);
    dispatchKernel(device, pipeline, bg, [1, 1, 1]);

    const hidden = (await readBuffer(device, hOut, hiddenDim * 4)).subarray(0, hiddenDim);
    const delta = (await readBuffer(device, dOut, stateDim * 4)).subarray(0, stateDim);

    // Reward head is small — compute on CPU from the GPU-produced hidden state.
    let reward = this.model.boutReward[0]!;
    for (let j = 0; j < hiddenDim; j++) reward += this.model.woutReward[j]! * hidden[j]!;

    xBuf.destroy();
    hBuf.destroy();
    sBuf.destroy();
    hOut.destroy();
    dOut.destroy();

    return { hidden: Float32Array.from(hidden), delta: Float32Array.from(delta), reward };
  }

  /** Train the affect model on observed experiences. Marks GPU step buffers dirty. */
  async train(samples: LimbicSample[], opts?: LimbicTrainOptions): Promise<number[]> {
    const losses = await this.trainer.train(samples, opts);
    this._paramsDirty = true; // weights changed → GPU step buffers must be re-uploaded
    return losses;
  }

  evaluate(samples: LimbicSample[]): number {
    return this.trainer.evaluate(samples);
  }

  exportWeights(opts?: { fp16?: boolean }): ArrayBuffer {
    return this.model.exportWeights(opts);
  }

  /** Persist weights to IndexedDB under the session name. */
  async save(): Promise<void> {
    await saveToIndexedDB(`limbic_${this._name}`, this.model.exportWeights({ fp16: true }), this._idbFactory);
  }

  /** Load weights from IndexedDB; returns true if a checkpoint was found. */
  async load(): Promise<boolean> {
    const buf = await loadFromIndexedDB(`limbic_${this._name}`, this._idbFactory);
    if (!buf) return false;
    this.model.loadWeights(buf);
    this._paramsDirty = true;
    return true;
  }

  private _destroyParamBufs(): void {
    if (this._paramBufs) {
      this._paramBufs.win.destroy();
      this._paramBufs.ws.destroy();
      this._paramBufs.aLogit.destroy();
      this._paramBufs.woutState.destroy();
      this._paramBufs.boutState.destroy();
      this._paramBufs = null;
    }
  }

  destroy(): void {
    this._destroyParamBufs();
    this._dimsBuf?.destroy();
    this._dimsBuf = null;
    this._pipeline = null;
  }
}
