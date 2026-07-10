/**
 * multimodal_vocab.ts — one flat token space shared by text and video.
 *
 * Evermind's generator ({@link ../lm/evermind_lm.EvermindLM}) is a pure
 * next-token predictor: it neither knows nor cares whether a token id is a BPE
 * text piece or a quantised video-patch code. So "make Evermind generate video"
 * is a *vocabulary* problem, not a model problem — lay text tokens, a handful of
 * modality control tokens, and the video codebook codes out on a single integer
 * axis, and the same model handles `(caption … <vid> … frames … </vid>)`.
 *
 * Layout (low → high):
 *
 *   [ 0 .. textVocabSize )                     text (BPE) — passthrough
 *   textVocabSize + 0                          <vid>   (BOS_VIDEO)
 *   textVocabSize + 1                          </vid>  (EOS_VIDEO)
 *   textVocabSize + 2                          <key>   (start of an intra frame)
 *   textVocabSize + 3                          <delta> (start of an inter frame)
 *   codeBase + bank·levels·K + level·K + code  residual-VQ code tokens
 *
 * where `bank ∈ {INTRA, INTER}`, `level ∈ [0, levels)`, `code ∈ [0, K)` and
 * `K = codebookSize`. A frame is therefore a marker followed by
 * `patchesPerFrame · levels` code tokens — a deterministic, self-delimiting
 * structure the codec can parse straight out of the generator's output.
 */

export const VIDEO_BANK_INTRA = 0;
export const VIDEO_BANK_INTER = 1;

export interface MultimodalVocabConfig {
  /** Size of the text (BPE) region; 0 for a pure-video model. */
  textVocabSize: number;
  /** Residual-VQ depth (codes emitted per patch). */
  levels: number;
  /** Entries per codebook per level per bank. */
  codebookSize: number;
}

/** What a token id decodes to when parsing a mixed stream. */
export type TokenKind =
  | { kind: "text"; id: number }
  | { kind: "control"; control: "bosVideo" | "eosVideo" | "frameKey" | "frameDelta" }
  | { kind: "code"; bank: number; level: number; code: number };

/**
 * Bijective map between (text ids, control tokens, video codes) and a single
 * flat token id space. Owns the offset arithmetic so nothing else has to.
 */
export class MultimodalVocab {
  readonly textVocabSize: number;
  readonly levels: number;
  readonly codebookSize: number;

  readonly bosVideo: number;
  readonly eosVideo: number;
  readonly frameKey: number;
  readonly frameDelta: number;
  readonly codeBase: number;
  /** Total vocabulary size — pass this as EvermindLM's `vocabSize`. */
  readonly size: number;

  private readonly perBank: number;

  constructor(cfg: MultimodalVocabConfig) {
    if (cfg.textVocabSize < 0) throw new Error("MultimodalVocab: textVocabSize must be ≥ 0");
    if (cfg.levels <= 0 || cfg.codebookSize <= 0) throw new Error("MultimodalVocab: levels and codebookSize must be > 0");
    this.textVocabSize = cfg.textVocabSize;
    this.levels = cfg.levels;
    this.codebookSize = cfg.codebookSize;

    this.bosVideo = cfg.textVocabSize;
    this.eosVideo = cfg.textVocabSize + 1;
    this.frameKey = cfg.textVocabSize + 2;
    this.frameDelta = cfg.textVocabSize + 3;
    this.codeBase = cfg.textVocabSize + 4;
    this.perBank = cfg.levels * cfg.codebookSize;
    this.size = this.codeBase + 2 * this.perBank;
  }

  /** Global token id for a residual-VQ code. */
  codeToken(bank: number, level: number, code: number): number {
    return this.codeBase + bank * this.perBank + level * this.codebookSize + code;
  }

  isText(tok: number): boolean {
    return tok >= 0 && tok < this.textVocabSize;
  }

  isCode(tok: number): boolean {
    return tok >= this.codeBase && tok < this.size;
  }

  isVideoMarker(tok: number): boolean {
    return tok === this.frameKey || tok === this.frameDelta;
  }

  /** Decompose a code token into (bank, level, code). Level/code are meaningful even if the bank differs. */
  decodeCode(tok: number): { bank: number; level: number; code: number } {
    const idx = tok - this.codeBase;
    const bank = Math.floor(idx / this.perBank);
    const rem = idx - bank * this.perBank;
    const level = Math.floor(rem / this.codebookSize);
    const code = rem - level * this.codebookSize;
    return { bank, level, code };
  }

  /** Classify any token id in the flat space. */
  classify(tok: number): TokenKind {
    if (this.isText(tok)) return { kind: "text", id: tok };
    if (tok === this.bosVideo) return { kind: "control", control: "bosVideo" };
    if (tok === this.eosVideo) return { kind: "control", control: "eosVideo" };
    if (tok === this.frameKey) return { kind: "control", control: "frameKey" };
    if (tok === this.frameDelta) return { kind: "control", control: "frameDelta" };
    return { kind: "code", ...this.decodeCode(tok) };
  }
}
