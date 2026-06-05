# @builderforce/memory-mcp

Expose [`@builderforce/memory`](../memory) to any MCP client. One token-saving
tool core over a pluggable `MemoryBackend`, three transports:

| Transport | Factory | Consumed as | Use when |
|---|---|---|---|
| **In-process (Claude Agent SDK)** | `createMemoryMcpServer(backend)` | the returned `type:"sdk"` object | the consuming product is TS and runs `@anthropic-ai/claude-agent-sdk` in-process — lowest latency |
| **stdio** | `runStdio(backend)` / `npx @builderforce/memory-mcp` | `{ type:"stdio", command, args }` | any language / separate process; decouples the SSM+IndexedDB deps from the consumer |
| **HTTP (Streamable)** | `createMemoryHttpHandler(backend, { authToken })` | `{ type:"http", url, headers }` | multi-tenant / networked (builderforce.ai hosting, remote claws) |

## Why this saves tokens

The point of an external memory store is **fetch-on-demand** instead of pinning
your whole memory file into every prompt. That only pays off if recall is
*selective*, so the caps are enforced server-side in [`tools.ts`](src/tools.ts):

- `memory_recall` returns a **ranked top-K** (default 5, hard-capped), never the store.
- Each entry's content is **truncated** (default 500 chars) before it hits context.
- There is **no "return everything" tool** — a dump-all is more expensive than inlining.

Tool descriptions are prescriptive ("call this *before* answering when…") because
recent Claude models reach for tools more conservatively; the trigger condition in
the description is what drives should-call rate. Pair this with prompt caching (keep
the tool set identical across tenants so the cached prefix survives) and context
editing (prune stale recalled facts) for the full win.

## The seam

Everything is written against [`MemoryBackend`](src/backend.ts):

```ts
interface MemoryBackend {
  recall(query: string, topK: number): Promise<RecallHit[]>;  // semantic
  get(key: string): Promise<RecallHit | undefined>;           // exact
  recallByTag(tag: string, limit: number): Promise<RecallHit[]>;
  remember?(input: RememberInput): Promise<void>;             // optional → read-only backends omit
  forget?(key: string): Promise<void>;
}
```

Ship the local `MemoryStoreBackend` (IndexedDB via `@builderforce/memory`) today;
drop in a networked builderforce.ai adapter later with **zero** changes to tools
or transports.

## Quick start — in-process with the Claude Agent SDK

```ts
import { query } from "@anthropic-ai/claude-agent-sdk";
import { createMemoryMcpServer, createLocalMemoryStoreBackend } from "@builderforce/memory-mcp";

const backend = await createLocalMemoryStoreBackend();          // IndexedDB (fake-indexeddb in Node)
const memory  = await createMemoryMcpServer(backend);           // type:"sdk" config

for await (const msg of query({
  prompt: "What language does this user prefer? Check memory first.",
  options: {
    mcpServers: { builderforce_memory: memory },
    allowedTools: ["mcp__builderforce_memory__*"],             // auto-approve, no prompts
  },
})) {
  if (msg.type === "result" && msg.subtype === "success") console.log(msg.result);
}
```

### SSM-embedding recall (the premium path)

The local backend defaults to **lexical (Jaccard)** recall. For SSM-embedding
cosine recall, pass an SSM runtime — e.g. reuse the agent-runtime's already-loaded
hippocampus instead of standing up a second model:

```ts
const backend = await createLocalMemoryStoreBackend({ runtime: ssmMemoryService.runtime });
```

Recall quality then improves automatically as that model is adapted/distilled.

## Quick start — stdio (any process / language)

```bash
npx -y @builderforce/memory-mcp        # serves the local MemoryStore over stdio
```

```ts
mcpServers: {
  builderforce_memory: { type: "stdio", command: "npx", args: ["-y", "@builderforce/memory-mcp"] },
}
```

Env: `BUILDERFORCE_MEMORY_DB` (db name), `BUILDERFORCE_MEMORY_READONLY=1` (drop write tools).

## Quick start — HTTP (multi-tenant)

```bash
BUILDERFORCE_MEMORY_TOKEN=secret PORT=8787 npx @builderforce/memory-mcp-http
```

```ts
mcpServers: {
  builderforce_memory: {
    type: "http",
    url: "http://localhost:8787",
    headers: { Authorization: "Bearer secret" },
  },
}
```

Or mount `createMemoryHttpHandler(backend, { authToken })` on your own Node/Express
server against any backend (including a future remote one).

## Custom / remote backend

Implement `MemoryBackend` and hand it to any transport:

```ts
import { createMemoryMcpServer, type MemoryBackend } from "@builderforce/memory-mcp";

const remote: MemoryBackend = {
  recall:      (q, k)   => bfClient.search(q, k),
  get:         (key)    => bfClient.get(key),
  recallByTag: (t, lim) => bfClient.byTag(t, lim),
  remember:    (input)  => bfClient.put(input),
  forget:      (key)    => bfClient.delete(key),
};
const server = await createMemoryMcpServer(remote);
```

## Dependencies

`@modelcontextprotocol/sdk` + `zod` are hard deps. `@anthropic-ai/claude-agent-sdk`
(SDK transport), `@builderforce/memory` + `fake-indexeddb` (local backend) are
**optional peers**, imported indirectly — install only what your transport/backend
needs.
