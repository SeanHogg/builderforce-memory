import { toJsonSchemaCompat } from "@modelcontextprotocol/sdk/server/zod-json-schema-compat.js";
import { z } from "zod";
import type { MemoryBackend } from "../backend.js";
import { fail } from "../tool-core.js";
import { buildMemoryTools, type MemoryToolsOptions } from "../tools.js";

type NameCodec = (source: string, tool: string) => string;

function bindMemoryTool(tool: ReturnType<typeof buildMemoryTools>[number], advertiseName?: NameCodec) {
    const legacyName = tool.name;
    const legacyAdvertisedName = advertiseName?.("memory", legacyName.replace(/^memory_/, "")) ?? legacyName;
    const name = tool.advertisedName
        ? advertiseName?.("memory", tool.advertisedName) ?? tool.advertisedName
        : legacyAdvertisedName;
    const schema = z.object(tool.inputSchema);
    return {
        name,
        legacyName,
        legacyAdvertisedName,
        description: tool.description,
        effect: tool.effect ?? "write",
        surfaces: ["memory"],
        inputSchema: { ...toJsonSchemaCompat(schema, { strictUnions: true, pipeStrategy: "input" }), type: "object" as const },
        async invoke(args: Record<string, unknown>) {
            const parsed = await schema.safeParseAsync(args);
            if (!parsed.success) return fail(`Invalid arguments for ${name}: ${parsed.error.issues.map((issue) => issue.message).join("; ")}`);
            return tool.handler(parsed.data);
        },
    };
}

type MemoryToolEntry = ReturnType<typeof bindMemoryTool>;

/** The Memory catalog for hosts using Builderforce's shared MCP server. */
export function createMemoryMcpCatalog(
    backend: MemoryBackend,
    options: MemoryToolsOptions = {},
    advertiseName?: NameCodec,
) {
    const entries = buildMemoryTools(backend, options).map((tool) => bindMemoryTool(tool, advertiseName));
    return {
        list: () => entries,
        resolve: (tools: readonly MemoryToolEntry[], name: string) =>
            tools.find((tool) => tool.name === name || tool.legacyName === name || tool.legacyAdvertisedName === name) ?? null,
        callPrepared: (tool: MemoryToolEntry, args: Record<string, unknown>) => tool.invoke(args),
    };
}
