/**
 * The hand-off contract (PRD 34 §3.3) — the shapes a host passes in and gets back when
 * a conversation moves from one model to another whose context window may be smaller.
 * Pure data: no runtime imports, so the subpath stays Worker/browser/Node safe.
 */

/** One chat message in the OpenAI-style shape every Builderforce host already speaks. */
export interface HandoffMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: unknown;
  tool_calls?: unknown;
  tool_call_id?: string;
}

/** What {@link packHandoff} packs, and the budget it packs into. */
export interface HandoffPackInput {
  messages: readonly HandoffMessage[];
  /** The NEXT model's context window, in tokens. */
  targetContextTokens: number;
  /** Tokens held back for the next model's reply. */
  reserveOutputTokens: number;
  /** Tokens the advertised tool schemas cost. */
  toolsTokens?: number;
  /** Model refs, for the one-line hand-off note. */
  from?: string;
  to?: string;
  /** Memory block for the latest directive, at most `budgetChars` long. */
  recall?: (query: string, budgetChars: number) => Promise<string>;
  /** Summary of the dropped middle. */
  summarize?: (dropped: readonly HandoffMessage[]) => Promise<string>;
  /** Token estimate for a text. Default: chars/4, rounded up. */
  estimateTokens?: (text: string) => number;
  /**
   * Optional ceiling (ms) on each of `summarize` / `recall`. A call that outlives it is
   * treated as a failure and packing falls through to the next note source. No ceiling
   * when omitted.
   */
  timeoutMs?: number;
  /**
   * Messages that must survive packing verbatim even when they fall in the dropped
   * middle — e.g. the task brief (the FIRST user message), which the latest-directive
   * rule does not protect once a nudge or steer has become the latest user turn.
   * `index` is the position in {@link messages}. Only `user` messages and `assistant`
   * messages without `tool_calls` can be pinned (moving a tool-call turn would split it
   * from its results); leading system messages are always kept anyway. A pinned message
   * that is dropped is re-injected right after the system prefix, in original order.
   */
  pinned?: (message: HandoffMessage, index: number) => boolean;
}

/** The packed transcript and how it was made. */
export interface HandoffPack {
  messages: HandoffMessage[];
  /** Messages from the input that are in neither the verbatim tail nor re-injected. */
  dropped: number;
  /** The note carries the `summarize` result. */
  summarized: boolean;
  /** The note carries the `recall` result. */
  recalled: boolean;
  estimatedTokens: number;
  /** False only when the system message(s) + the latest directive alone exceed the budget. */
  fits: boolean;
}

/** The three numbers the usable budget is computed from. */
export interface HandoffBudgetInput {
  targetContextTokens: number;
  reserveOutputTokens: number;
  toolsTokens?: number;
}

/** A text → token-count estimator. */
export type TokenEstimator = (text: string) => number;
