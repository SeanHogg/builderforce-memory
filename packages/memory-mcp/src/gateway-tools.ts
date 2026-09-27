/**
 * Gateway-backed cost tools (token_usage, model_efficiency). Registered only when a
 * gateway URL and a `bfk_*` key are both configured, so memory-only deployments
 * never see them.
 */

import { fail, ok, type MemoryTool } from "./tool-core.js";

/** Snapshot shape returned by GET /llm/v1/builder-insights. */
interface BuilderInsightsSnapshot {
    windowLabel?: string;
    todayTokens?: number;
    todayCostUsd?: number;
    dailyCapTokens?: number | null;
    pctOfDailyCap?: number | null;
    topModel?: { model: string; tokens: number } | null;
    costPerMergedPrUsd?: number | null;
    tip?: string | null;
}

/** Ranking shape returned by GET /llm/v1/model-analytics. */
interface ModelAnalytics {
    byAction?: Array<{
        actionType?: string;
        label?: string;
        models?: Array<{
            model: string;
            samples?: number;
            avgScore?: number;
            mergeRate?: number;
            avgCostMillicents?: number;
        }>;
    }>;
}

async function gatewayGet<T>(gatewayUrl: string, path: string, apiKey: string): Promise<T> {
    const res = await fetch(`${gatewayUrl}${path}`, {
        headers: { authorization: `Bearer ${apiKey}` },
    });
    if (!res.ok) throw new Error(`gateway ${path} → HTTP ${res.status}`);
    return (await res.json()) as T;
}

export function appendGatewayCostTools(tools: MemoryTool[], gatewayUrl: string, apiKey: string): void {
    tools.push({
        name: "token_usage",
        description: "Current token spend + budget for the workspace (today).",
        inputSchema: {},
        handler: async () => {
            try {
                const s = await gatewayGet<BuilderInsightsSnapshot>(gatewayUrl, "/llm/v1/builder-insights", apiKey);
                const lines = [
                    `Token usage (${s.windowLabel ?? "today"}):`,
                    `• Tokens: ${(s.todayTokens ?? 0).toLocaleString()}`,
                    `• Cost: $${(s.todayCostUsd ?? 0).toFixed(2)}`,
                    `• % of daily cap: ${
                        s.pctOfDailyCap == null
                            ? "no cap"
                            : `${s.pctOfDailyCap}%${s.dailyCapTokens ? ` of ${s.dailyCapTokens.toLocaleString()}` : ""}`
                    }`,
                    `• Top model: ${s.topModel ? `${s.topModel.model} (${s.topModel.tokens.toLocaleString()} tok)` : "—"}`,
                ];
                if (s.costPerMergedPrUsd != null) lines.push(`• Cost / merged PR: $${s.costPerMergedPrUsd.toFixed(2)}`);
                if (s.tip) lines.push(`• Tip: ${s.tip}`);
                return ok(lines.join("\n"));
            } catch (err) {
                return fail(`token_usage failed: ${String(err)}`);
            }
        },
    });

    tools.push({
        name: "model_efficiency",
        description: "Which models performed best/cheapest for this workspace's recent work.",
        inputSchema: {},
        handler: async () => {
            try {
                const a = await gatewayGet<ModelAnalytics>(gatewayUrl, "/llm/v1/model-analytics", apiKey);
                const groups = a.byAction ?? [];
                if (groups.length === 0) return ok("No model efficiency data yet for this workspace.");
                const out: string[] = ["Model efficiency by action type (best first):"];
                for (const g of groups) {
                    out.push(`\n${g.label ?? g.actionType ?? "action"}:`);
                    for (const m of (g.models ?? []).slice(0, 3)) {
                        const cost = m.avgCostMillicents != null ? ` $${(m.avgCostMillicents / 100_000).toFixed(4)}/call` : "";
                        const merge = m.mergeRate != null ? ` merge=${Math.round(m.mergeRate * 100)}%` : "";
                        const score = m.avgScore != null ? ` score=${m.avgScore}` : "";
                        out.push(`  • ${m.model}${score}${merge}${cost} (n=${m.samples ?? 0})`);
                    }
                }
                return ok(out.join("\n"));
            } catch (err) {
                return fail(`model_efficiency failed: ${String(err)}`);
            }
        },
    });
}
