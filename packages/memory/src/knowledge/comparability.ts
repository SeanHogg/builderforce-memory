/**
 * knowledge/comparability.ts — the ONE comparability module (PRD 41 §C9), and the portable analyzer
 * artifacts (`CodeAnalysisSnapshot`) whose compare it gates — moved here from Builderforce.ai's
 * agent-tools `codeAnalysis.ts` so the web analysis compare and the knowledge compare share one
 * identity (§C3) and one rule set. No filesystem or UI dependencies.
 *
 * Two sides are comparable only when NONE of the {@link ComparabilityReason} codes hold. A kind that
 * is not comparable contributes no `introduced` / `resolved` entries: missing analyzer output must
 * never be presented as a fix.
 */

import { anchorWithSnippet, composeAnchor, type FindingIdentity } from './findingIdentity.js';
import { compareStr } from './refs.js';

/* ── comparable() ───────────────────────────────────────────────────────────────────────────────────── */

/** What one side was measured with. Absent fields compare equal only to absent fields. */
export interface ComparabilityMeta {
    /** `'tree-sitter@<v>'` | `'regex@<v>'` — the family is the part before `@`. */
    extractor?: string;
    extractorVersion?: string | number;
    contentHashRule?: string;
    /** Tool (or rule) name → version. */
    tools?: Record<string, string>;
    scope?: string;
    repo?: string;
    branch?: string;
    truncated?: boolean;
    /** Tools that failed or timed out on this side. */
    failed?: string[];
}

/**
 * Stable reason codes (localized at render): `extractor-mismatch` (family differs), `extractor-version-changed`,
 * `content-hash-rule-changed`, `tool-missing:<tool>` (measured on one side only), `tool-version-changed:<tool>`,
 * `analyzer-failed:<tool>`, `scope-changed`, `repo-changed`, `branch-changed`, `truncated`.
 * `compareAnalysis` adds `no-baseline` and `tools-incomplete`.
 */
export type ComparabilityReason = string;

const familyOf = (extractor: string | undefined) => extractor?.split('@')[0];
const versionOf = (extractor: string | undefined) => {
    const at = extractor?.indexOf('@') ?? -1;
    return at < 0 ? undefined : extractor!.slice(at + 1);
};

/** Whether two sides can be compared, and every reason they cannot (sorted, deduped). */
export function comparable(before: ComparabilityMeta, after: ComparabilityMeta): { comparable: boolean; reasons: ComparabilityReason[] } {
    const reasons = new Set<string>();
    if (familyOf(before.extractor) !== familyOf(after.extractor)) reasons.add('extractor-mismatch');
    else if (versionOf(before.extractor) !== versionOf(after.extractor)
        || String(before.extractorVersion ?? '') !== String(after.extractorVersion ?? '')) reasons.add('extractor-version-changed');
    if (before.contentHashRule !== after.contentHashRule) reasons.add('content-hash-rule-changed');
    const bt = before.tools ?? {}, at = after.tools ?? {};
    for (const tool of new Set([...Object.keys(bt), ...Object.keys(at)])) {
        if (!(tool in bt) || !(tool in at)) reasons.add(`tool-missing:${tool}`);
        else if (bt[tool] !== at[tool]) reasons.add(`tool-version-changed:${tool}`);
    }
    for (const tool of [...(before.failed ?? []), ...(after.failed ?? [])]) reasons.add(`analyzer-failed:${tool}`);
    if (before.scope !== after.scope) reasons.add('scope-changed');
    if (before.repo !== after.repo) reasons.add('repo-changed');
    if (before.branch !== after.branch) reasons.add('branch-changed');
    if (before.truncated || after.truncated) reasons.add('truncated');
    const sorted = [...reasons].sort(compareStr);
    return { comparable: sorted.length === 0, reasons: sorted };
}

/* ── Code analysis snapshots (ported from agent-tools codeAnalysis.ts) ──────────────────────────────── */

export interface AnalysisLocation { path: string; startLine: number; endLine: number }
export interface AnalysisIssue {
    id: string; kind: 'duplicate' | 'unused-file' | 'unused-export' | 'unused-dependency'; locations: AnalysisLocation[]; symbol?: string; snippet?: string;
    /** The §C3 anchor ({@link composeAnchor}, version {@link ANALYSIS_ANCHOR_VERSION}): no line numbers, no location set. */
    anchor?: string;
}
export interface CodeAnalysisSnapshot {
  version: 1;
  id: string;
  capturedAt: string;
  source: { commit: string | null; branch: string | null; dirty: boolean | null; dirtyHash?: string; repository?: string; scope?: string };
  tools: Array<{ name: string; version: string; status: 'complete' | 'failed' | 'not_measured'; error?: string; reason?: 'analyzer-unavailable' }>;
  nodes: string[];
  edges: Array<{ from: string; to: string; circular: boolean }>;
  issues: AnalysisIssue[];
}
export interface CodeAnalysisComparison { current: CodeAnalysisSnapshot; baseline?: CodeAnalysisSnapshot }

/** The rule version the jscpd / knip issue anchors are composed with. */
export const ANALYSIS_ANCHOR_VERSION = 1;
/** dependency-cruiser, jscpd, knip: a snapshot missing one is never comparable. */
export const ANALYSIS_TOOL_COUNT = 3;

const record = (v: unknown): Record<string, any> => v && typeof v === 'object' ? v as Record<string, any> : {};
const list = (v: unknown): any[] => Array.isArray(v) ? v : [];
const path = (v: unknown) => String(v ?? '').replace(/\\/g, '/').replace(/^\.\//, '');
const line = (v: unknown) => Math.max(1, Number(v) || 1);
const hash = (value: string) => { let n = BigInt('14695981039346656037'); const prime = BigInt('1099511628211'); for (let i = 0; i < value.length; i++) n = BigInt.asUintN(64, (n ^ BigInt(value.charCodeAt(i))) * prime); return n.toString(16); };

export function normalizeAnalysis(dependencies: unknown, duplicates: unknown, unused: unknown, scope = '.'): Pick<CodeAnalysisSnapshot, 'nodes' | 'edges' | 'issues'> {
  const unusedPath = (value: unknown) => { const file = path(value); return scope === '.' || file.startsWith(`${scope}/`) ? file : `${scope}/${file}`; };
  const nodes = new Set<string>();
  const edges = new Map<string, { from: string; to: string; circular: boolean }>();
  for (const module of list(record(dependencies).modules)) {
    const from = path(module.source); if (!from) continue;
    nodes.add(from);
    for (const dependency of list(module.dependencies)) {
      if (dependency.couldNotResolve || dependency.coreModule || dependency.dependencyTypes?.some((t: string) => t.startsWith('npm'))) continue;
      const to = path(dependency.resolved); if (!to || to.includes('node_modules/')) continue;
      nodes.add(to); edges.set(JSON.stringify([from, to]), { from, to, circular: !!dependency.circular });
    }
  }
  const issues: AnalysisIssue[] = [];
  for (const clone of list(record(duplicates).duplicates)) {
    const locations = [clone.firstFile, clone.secondFile].map((raw) => {
      const file = record(raw); return { path: path(file.name), startLine: line(file.start), endLine: line(file.end) };
    });
    // jscpd's fragment survives line movement. Range identity is a fallback for reporters without fragments.
    locations.sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : a.startLine - b.startLine);
    const identity = typeof clone.fragment === 'string' ? clone.fragment.replace(/\r\n/g, '\n') : locations.map((l) => `${l.path}:${l.startLine}-${l.endLine}`).sort().join('|');
    // Anchor: the first location's FILE plus the normalized fragment hash — never its lines.
    const anchor = composeAnchor('duplicate', ANALYSIS_ANCHOR_VERSION, 'file',
      anchorWithSnippet(locations[0]?.path ?? '', typeof clone.fragment === 'string' ? clone.fragment : undefined));
    issues.push({ id: `duplicate:${hash(JSON.stringify([locations.map((l) => l.path).sort(), identity]))}`, kind: 'duplicate', locations,
      ...(typeof clone.fragment === 'string' ? { snippet: clone.fragment.slice(0, 4000) } : {}), anchor });
  }
  for (const file of list(record(unused).files)) issues.push({ id: `unused-file:${unusedPath(file)}`, kind: 'unused-file', locations: [{ path: unusedPath(file), startLine: 1, endLine: 1 }],
    anchor: composeAnchor('unused-file', ANALYSIS_ANCHOR_VERSION, 'file', unusedPath(file)) });
  for (const issue of list(record(unused).issues)) {
    const file = unusedPath(issue.file);
    for (const field of ['exports', 'types', 'dependencies', 'devDependencies', 'unlisted']) {
      for (const raw of list(issue[field])) {
        const item = typeof raw === 'string' ? { name: raw } : record(raw);
        const kind = field === 'exports' || field === 'types' ? 'unused-export' : 'unused-dependency';
        if (field === 'unlisted') continue; // Missing dependencies are not unused dependencies.
        // Anchor: the export as `<file>#<name>` (symbol) / the dependency as `<manifest>#<name>` (package).
        const anchor = composeAnchor(kind, ANALYSIS_ANCHOR_VERSION, kind === 'unused-export' ? 'symbol' : 'package', `${file}#${String(item.name)}`);
        issues.push({ id: `${kind}:${file}:${item.name}`, kind, symbol: String(item.name), locations: [{ path: file, startLine: line(item.line), endLine: line(item.line) }], anchor });
      }
    }
  }
  return { nodes: [...nodes].sort(), edges: [...edges.values()], issues: [...new Map(issues.map((i) => [i.id, i])).values()] };
}

/** The v2 identity of an analysis issue (members: its sorted location paths), for {@link matchLineage}. */
export function identityOfIssue(issue: AnalysisIssue): FindingIdentity {
  const members = [...new Set(issue.locations.map((l) => l.path))].sort(compareStr);
  const anchorKind = issue.kind === 'unused-export' ? 'symbol' : issue.kind === 'unused-dependency' ? 'package' : 'file';
  return { id: issue.id, anchor: issue.anchor ?? composeAnchor(issue.kind, ANALYSIS_ANCHOR_VERSION, anchorKind, members[0] ?? ''), anchorKind, members };
}

/** A snapshot's comparability meta: source scope/repo/branch, tool versions, and every tool not `complete` as failed. */
export function analysisMeta(snapshot: CodeAnalysisSnapshot): ComparabilityMeta {
  return {
    scope: snapshot.source.scope, repo: snapshot.source.repository, branch: snapshot.source.branch ?? undefined,
    tools: Object.fromEntries(snapshot.tools.map((t) => [t.name, t.version])),
    failed: snapshot.tools.filter((t) => t.status !== 'complete').map((t) => t.name),
  };
}

export function compareAnalysis(current: CodeAnalysisSnapshot, baseline?: CodeAnalysisSnapshot) {
  const before = new Set(baseline?.issues.map((i) => i.id) ?? []);
  const after = new Set(current.issues.map((i) => i.id));
  const reasons = baseline ? comparable(analysisMeta(baseline), analysisMeta(current)).reasons : ['no-baseline'];
  if (baseline && (current.tools.length !== ANALYSIS_TOOL_COUNT || baseline.tools.length !== ANALYSIS_TOOL_COUNT)) reasons.push('tools-incomplete');
  const complete = reasons.length === 0;
  return {
    comparable: complete,
    reasons,
    added: complete ? current.issues.filter((i) => !before.has(i.id)) : [],
    unchanged: current.issues.filter((i) => before.has(i.id)),
    // Missing analyzer output must never be presented as a fix.
    resolved: complete ? baseline!.issues.filter((i) => !after.has(i.id)) : [],
  };
}
