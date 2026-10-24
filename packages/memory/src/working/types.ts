/**
 * The working-memory contract — pure data, so the subpath stays Worker/browser/Node safe.
 */

/** One thing a run did that is worth remembering, in the memory's own terms. */
export type Observation =
  | { kind: 'read' | 'created' | 'edited' | 'deleted'; path: string }
  | { kind: 'searched'; query: string };

/** What a run's working memory holds. Lists keep duplicates — how often is a signal. */
export interface WorkingMemorySnapshot {
  /** What the run was asked. */
  task?: string;
  created: readonly string[];
  edited: readonly string[];
  deleted: readonly string[];
  read: readonly string[];
  searched: readonly string[];
}

/** One long-term memory, ready for a keyed store: the same key replaces (or merges). */
export interface ConsolidatedMemory {
  key: string;
  content: string;
}

export interface ConsolidateOptions {
  /** Dates the memory so a reader can judge its age. */
  at?: Date;
  /** The workspace root — absolute paths under it are kept, made relative. */
  root?: string;
}
