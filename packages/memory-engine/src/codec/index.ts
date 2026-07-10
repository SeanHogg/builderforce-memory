/**
 * codec/ — modality codecs that turn non-text media into the discrete token
 * streams Evermind's generator already speaks. Today: video (temporal residual
 * VQ). Images are the single-frame case; audio/other modalities follow the same
 * "media ⇄ tokens + unified vocab" shape.
 */

export { MultimodalVocab, VIDEO_BANK_INTRA, VIDEO_BANK_INTER } from "./multimodal_vocab.js";
export type { MultimodalVocabConfig, TokenKind } from "./multimodal_vocab.js";

export { VideoRVQCodec } from "./video_rvq.js";
export type { VideoRVQConfig, Frame, Video } from "./video_rvq.js";

export { ImageRVQCodec } from "./image_rvq.js";

export { buildVideoSequence, generateVideo, generateImage } from "./evermind_video.js";
