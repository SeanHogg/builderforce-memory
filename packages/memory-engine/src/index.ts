/**
 * MambaCode.js – Entry Point (v2.0.0)
 */

// ── Model classes ─────────────────────────────────────────────────────────────

export { HybridMambaModel, MambaModel } from './model/mamba_model.js';

// New block classes
export { Mamba1Block }   from './model/mamba1_block.js';
export { Mamba2Block }   from './model/mamba2_block.js';
export { Mamba3Block }   from './model/mamba3_block.js';
export { AttentionBlock } from './model/attention_block.js';

// Deprecated alias — kept until 3.0.0
export { MambaBlock } from './model/mamba1_block.js';

// ── Mixture-of-Experts (shared-expert hybrid sparsity) ─────────────────────────
export {
    SharedExpertMoE,
    LoadBalanceAccumulator,
    DEFAULT_MOE_CONFIG,
    DEFAULT_MOE_SEED,
    MoETrainer,
    EvermindModelPackage,
} from './moe/index.js';
export type {
    MoEConfig,
    MoEParam,
    RouteResult,
    MoESample,
    MoETrainOptions,
    MoEEpochResult,
    EvermindModelManifest,
    EvermindModelCard,
    PackageMeta,
    ValidationResult,
} from './moe/index.js';

// ── EvermindLM (the generative model) + AdamW ──────────────────────────────────
export { EvermindLM, EvermindLMTrainer, DEFAULT_LM_CONFIG, DEFAULT_LM_SEED } from './lm/index.js';
export type { EvermindLMConfig, LMGenerateOptions, TextCodec } from './lm/index.js';
export { AdamW } from './optim/adamw.js';
export type { AdamWOptions, OptimTarget, OptimParam } from './optim/adamw.js';

// ── Training ──────────────────────────────────────────────────────────────────

export { MambaTrainer } from './training/trainer.js';
export {
    Tensor,
    backward,
    enableGrad,
    noGrad,
    clearTape,
    recordOperation,
    crossEntropyLoss,
    crossEntropyGrad,
} from './training/autograd.js';

// ── Tokenizer ─────────────────────────────────────────────────────────────────

export { BPETokenizer } from './tokenizer/bpe.js';
export type { BPEEncodeOptions, PadSide, HuggingFaceTokenizerSpec, SpecialTokenOverrides } from './tokenizer/bpe.js';

// ── Modality codecs (media ⇄ tokens; lets EvermindLM generate video) ────────────

export {
    MultimodalVocab,
    VIDEO_BANK_INTRA,
    VIDEO_BANK_INTER,
    VideoRVQCodec,
    buildVideoSequence,
    generateVideo,
} from './codec/index.js';
export type {
    MultimodalVocabConfig,
    TokenKind,
    VideoRVQConfig,
    Frame,
    Video,
} from './codec/index.js';

// ── Model export (the publishing step: ONNX / safetensors / GGUF / HF repo) ─────

export {
    exportEvermind,
    EXPORT_FORMATS,
    exportSafetensors,
    tensorsToSafetensors,
    exportOnnx,
    exportGguf,
    configJson,
    generationConfigJson,
    tokenizerJson,
    modelCardMarkdown,
    namedTensors,
    evermindTensorSpec,
    archOf,
    paramCount,
} from './export/index.js';
export type {
    ExportFormat,
    ExportFile,
    ExportResult,
    ExportOptions,
    HfMeta,
    NamedTensor,
    EvermindArch,
    TensorSpec,
} from './export/index.js';

// ── Model import (warm-start / weight-port: safetensors → EvermindLM) ────────────

export {
    safetensorsToTensors,
    importEvermind,
    importEvermindTensors,
    inferArchFromTensors,
} from './import/index.js';
export type { ImportOptions } from './import/index.js';

// ── Benchmarking (held-out perplexity / accuracy / throughput + A/B) ──────────

export {
    benchmarkModel,
    benchmarkModelAsync,
    benchmarkText,
    compareModels,
    compareReports,
    corpusToSequences,
    trainAndBenchmark,
    argmax as benchArgmax,
    topKIndices,
    perplexity,
    bitsPerToken,
    LN2,
} from './bench/index.js';
export type {
    LogitsModel,
    AsyncLogitsModel,
    BenchmarkOptions,
    BenchmarkReport,
    ComparisonReport,
    TrainAndBenchmarkOptions,
    TrainAndBenchmarkResult,
} from './bench/index.js';

// ── Checkpoint integrity (CRC-32 + trailer) ───────────────────────────────────

export { crc32, appendCrcTrailer, verifyCrcTrailer, CRC_TRAILER_MAGIC } from './utils/crc32.js';
export type { CrcCheck } from './utils/crc32.js';
export {
    computeRowDelta,
    applyRowDelta,
    serializeRowDelta,
    deserializeRowDelta,
    diffCheckpoints,
    applyCheckpointDiff,
} from './utils/delta.js';
export type { RowDelta } from './utils/delta.js';

// ── Seeded RNG (reproducible weight init) ─────────────────────────────────────

export { SeededRng, setInitSeed, randn, gaussianArray } from './utils/rng.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export type {
    HybridMambaModelConfig,
    MambaModelConfig,
    ModelForwardResult,
    SamplingOptions,
    LayerSpec,
} from './model/mamba_model.js';

export type { SequenceLayer, LayerParam, LayerType, LayerForwardResult } from './model/sequence_layer.js';
export type { Mamba1BlockConfig, BlockParam, BlockCache, BlockForwardResult, MambaBlockConfig } from './model/mamba1_block.js';
export type { Mamba2BlockConfig, Mamba2Cache } from './model/mamba2_block.js';
export type { Mamba3BlockConfig, Mamba3Cache }  from './model/mamba3_block.js';
export type { AttentionBlockConfig, AttentionCache } from './model/attention_block.js';

// ── GPU utilities ─────────────────────────────────────────────────────────────

export {
    initWebGPU,
    createStorageBuffer,
    createEmptyStorageBuffer,
    createUniformBuffer,
    createComputePipeline,
    createBindGroup,
    dispatchKernel,
    readBuffer,
    uploadBuffer,
    cdiv,
    BufferPool,
} from './utils/gpu_utils.js';

// ── Quantization ──────────────────────────────────────────────────────────────

export {
    quantizeFp16,
    dequantizeFp16,
    floatToFp16,
    fp16ToFloat,
    quantizeInt8,
    dequantizeInt8,
    quantizeInt8PerChannel,
    dequantizeInt8PerChannel,
    estimateMemory,
} from './utils/quantization.js';

// ── WGSL kernel sources ───────────────────────────────────────────────────────

// Mamba-1 kernels (unchanged)
export { SELECTIVE_SCAN_FORWARD_WGSL, SELECTIVE_SCAN_BACKWARD_WGSL }
    from './kernels/selective_scan.js';
export { CONV1D_FORWARD_WGSL, CONV1D_BACKWARD_WGSL }
    from './kernels/conv1d.js';
export { LINEAR_FORWARD_WGSL, LINEAR_BACKWARD_WGSL }
    from './kernels/linear_projection.js';
export { WEIGHT_UPDATE_WGSL, GRAD_CLIP_WGSL }
    from './kernels/weight_update.js';
export { ACTIVATIONS_WGSL, ACTIVATIONS_BACKWARD_WGSL, SOFTMAX_FORWARD_WGSL, SOFTMAX_BACKWARD_WGSL }
    from './kernels/activations.js';

// Mamba-2 SSD kernels
export { SSD_FORWARD_WGSL, SSD_BACKWARD_WGSL }
    from './kernels/ssd.js';

// Mamba-3 complex SSD kernels
export { COMPLEX_SSD_FORWARD_WGSL, COMPLEX_SSD_BACKWARD_WGSL }
    from './kernels/complex_ssd.js';

// Attention kernels
export { ATTENTION_FORWARD_WGSL, ATTENTION_BACKWARD_WGSL, SOFTMAX_WGSL }
    from './kernels/attention.js';

// ── Limbic system (trainable affective dynamics) ──────────────────────────────

export {
    REGION,
    LIMBIC_DIM,
    LIMBIC_DIM_NAMES,
    LIMBIC_STATE_DIM,
    LIMBIC_BOUNDS,
    NEUTRAL_STATE,
    clampDim,
    clampState,
    neutralState,
    stateToRecord,
    recordToState,
    personalitySetpoint,
    LimbicModel,
    DEFAULT_LIMBIC_CONFIG,
    DEFAULT_LIMBIC_SEED,
    LimbicTrainer,
} from './limbic/index.js';
export type {
    Region,
    LimbicDimName,
    PersonalityTraits,
    LimbicModelConfig,
    LimbicForward,
    LimbicParam,
    LimbicSample,
    LimbicTrainOptions,
} from './limbic/index.js';

export { LIMBIC_AFFECT_WGSL } from './kernels/limbic_affect.js';

// ── Version ───────────────────────────────────────────────────────────────────

export const VERSION     = '2.0.0';
export const DESCRIPTION = 'MambaCode.js: WebGPU-accelerated Mamba-1/2/3 and Hybrid SSM for browser code models';
