/**
 * knowledge/externals.ts — the ids of the outside services a code file can be said to import
 * (the `externals` badge). The module-specifier patterns that decide an id live with the code pack
 * that writes the badge; the ids live HERE so the writer (the pack) and every reader (the map that
 * names a line's end) type against one list.
 */

export const EXTERNAL_SERVICE_IDS = ['postgres', 'kv', 'llm', 'github', 'stripe', 'mail', 'slack', 'oauth'] as const;

export type ExternalServiceId = (typeof EXTERNAL_SERVICE_IDS)[number];
