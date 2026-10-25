/**
 * Chat consolidation markers — the durable form of a compressed conversation.
 *
 * A long conversation is compressed into ONE summary that becomes the new base context:
 * everything before the marker stays visible in the transcript, but the next run's model
 * seed starts from the marker instead of the full history. The marker is a normal
 * assistant message (so a reader SEES the summary) tagged in its metadata.
 *
 * Two writers:
 *   - MANUAL — "Consolidate" / "Fork": the summary covers everything before the marker.
 *   - AUTO — a run's working-transcript fold (`transcript.ts`), persisted so the next run
 *     (after a reload, on another surface) starts from the note instead of summarizing the
 *     whole history again. A fold covers only the OLDER part of the transcript; its
 *     `keepFromId` names the first persisted message it did NOT cover, and every message
 *     from there on stays verbatim in the seed. Nothing is in neither.
 */

/** The marker's metadata. `auto` / `keepFromId` / `folded` are set only by a run's fold. */
export interface ConsolidationMarkerMeta {
  consolidation: true;
  /** Written by auto-compaction, not by a person. */
  auto?: true;
  /** The first persisted message the note does NOT cover (kept verbatim after the marker). */
  keepFromId?: number;
  /** How many transcript messages the fold newly compressed. */
  folded?: number;
}

/** The metadata of a manual marker. */
export const CONSOLIDATION_META: ConsolidationMarkerMeta = { consolidation: true };

/** Serialized metadata for a marker message (ready to persist). Manual when no `auto`. */
export function consolidationMetadata(auto?: { keepFromId?: number; folded?: number }): string {
  if (!auto) return JSON.stringify(CONSOLIDATION_META);
  const meta: ConsolidationMarkerMeta = {
    consolidation: true,
    auto: true,
    ...(auto.keepFromId !== undefined ? { keepFromId: auto.keepFromId } : {}),
    ...(auto.folded !== undefined ? { folded: auto.folded } : {}),
  };
  return JSON.stringify(meta);
}

/** The marker metadata a persisted message carries, or null when it is not a marker. */
export function consolidationMarkerOf(msg: { metadata?: string | null }): ConsolidationMarkerMeta | null {
  if (!msg.metadata) return null;
  try {
    const m = JSON.parse(msg.metadata) as Partial<ConsolidationMarkerMeta> | null;
    if (m?.consolidation !== true) return null;
    return {
      consolidation: true,
      ...(m.auto === true ? { auto: true as const } : {}),
      ...(typeof m.keepFromId === 'number' && Number.isFinite(m.keepFromId) ? { keepFromId: m.keepFromId } : {}),
      ...(typeof m.folded === 'number' && Number.isFinite(m.folded) && m.folded > 0 ? { folded: m.folded } : {}),
    };
  } catch {
    return null;
  }
}

/** True when a persisted message is a consolidation marker (by its metadata flag). */
export function isConsolidationMarker(msg: { metadata?: string | null }): boolean {
  return consolidationMarkerOf(msg) !== null;
}

/** The index of the LAST consolidation marker in a message list, or -1 if none. */
export function lastConsolidationIndex(messages: ReadonlyArray<{ metadata?: string | null }>): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (isConsolidationMarker(messages[i]!)) return i;
  }
  return -1;
}

/**
 * Trim a message list to the compressed context: the last marker, then what it does not
 * cover. A manual marker covers everything before it (the list from the marker on). An
 * auto marker covers up to its `keepFromId`: the marker, then every message from that id
 * on (other markers dropped — an older note is superseded by the newer one). Unchanged
 * when there is no marker; a `keepFromId` no longer in the list reads as manual.
 */
export function scopeToConsolidation<T extends { id?: number; metadata?: string | null }>(messages: T[]): T[] {
  const idx = lastConsolidationIndex(messages);
  if (idx < 0) return messages;
  const marker = messages[idx]!;
  const keepFromId = consolidationMarkerOf(marker)?.keepFromId;
  const from = keepFromId === undefined ? -1 : messages.findIndex((m) => m.id === keepFromId);
  if (from < 0 || from > idx) return messages.slice(idx);
  return [marker, ...messages.slice(from).filter((m) => m !== marker && !isConsolidationMarker(m))];
}

/**
 * Where an auto marker's verbatim part begins: given the persisted messages the run's
 * transcript holds (`anchors`: transcript index → message id) and the index the fold now
 * covers up to (exclusive), the id of the persisted message the next seed must keep from.
 *
 * Exact when a persisted message sits AT `coveredEnd` (the fold ended inside seeded
 * history). Otherwise the nearest persisted message BEFORE it — conservative: the span
 * between it and `coveredEnd` is then both in the note and verbatim, never in neither.
 * Undefined when no persisted message precedes the cut (nothing to keep from).
 */
export function foldKeepFromId(anchors: ReadonlyArray<{ index: number; id: number }>, coveredEnd: number): number | undefined {
  let best: { index: number; id: number } | undefined;
  for (const a of anchors) {
    if (a.index <= coveredEnd && (!best || a.index > best.index)) best = a;
  }
  return best?.id;
}

/** The visible header prefixed onto a consolidation summary so a reader recognizes it. */
export const CONSOLIDATION_MARKER_PREFIX = '📌 **Consolidated summary** — context continues from here.\n\n';

/** Wrap a raw summary as the marker's visible content (prefix + summary). */
export function consolidationMarkerContent(summary: string): string {
  return `${CONSOLIDATION_MARKER_PREFIX}${summary.trim()}`;
}
