# BuilderForce Agent Memory

A framework for giving **any AI agent** persistent, browser-trainable SSM memory. It is provider-neutral — swappable LLM bridges (Anthropic / OpenAI / Fetch), an inference router, online distillation, and a persistent memory store — so any agent or app can depend on it, not just BuilderForce.

**[BuilderForce.ai](https://builderforce.ai) is the flagship deployment**, not the boundary: it's the *hippocampus* in BuilderForce's cortex/hippocampus split — BuilderForce trains a custom SSM in the browser, exports a model, and pushes it to an agent runtime as persistent memory while the frontier LLM stays the cortex. The same framework is reusable by other agents off the shelf.

This monorepo consolidates two packages that previously lived in separate repos (`mambacode.js` and `ssmjs`) whose names described the *technique* rather than the *role*. They are renamed and unified here so that "what they do" is legible: they are Agent Memory.

## Packages

| Package | Layer | Was | Responsibility |
|---|---|---|---|
| [`@seanhogg/builderforce-memory-engine`](packages/memory-engine) | **Engine** | `@seanhogg/mambacode.js` | WGSL/WebGPU Mamba SSM kernels, model blocks (Mamba1/2/3 + attention), autograd, trainer, BPE tokenizer, quantization. Zero runtime deps. |
| [`@seanhogg/builderforce-memory`](packages/memory) | **Runtime** | `@seanhogg/ssmjs` | SSM execution, Transformer orchestration (Anthropic/OpenAI/Fetch bridges), online distillation, inference router, sessions, and the persistent `MemoryStore`. Depends on `memory-engine`. |

The two-package split is deliberate: the engine is zero-dep and WebGPU-pure and can be consumed standalone; the runtime pulls in LLM-vendor bridges. Flattening them would force engine-only consumers to drag in vendor code and vice versa. They release in lockstep from one pipeline, which kills the publish-drift bug class that the separate-repo setup suffered (bumping one version without publishing + regenerating the consumer lockfile).

## Cutting token cost

Agent Memory ships three layers that reduce LLM spend, in increasing power. All are **portable** — the same code runs in the browser (WebGPU SSM) and in Node (the agent's `@webgpu/node` SSM) because the embedder and storage are injected, never hard-wired.

| Layer | What it does | Saves |
|---|---|---|
| **Prompt caching** ([`AnthropicBridge`](packages/memory/src/bridges/AnthropicBridge.ts) `cacheSystem`) | Marks the stable system prompt as an Anthropic `cache_control` block | ~90% on the cached **input** prefix (cost, not count) |
| **Exact-match cache** ([`CachingBridge`](packages/memory/src/bridges/CachingBridge.ts) + [`ResponseCache`](packages/memory/src/bridges/ResponseCache.ts)) | Reuses byte-identical completions | Eliminates duplicate calls (retries, identical fan-out) |
| **Semantic cache** ([`SemanticCache`](packages/memory/src/cache/SemanticCache.ts) + [`SemanticCachingBridge`](packages/memory/src/bridges/SemanticCachingBridge.ts)) | Reuses a prior answer when the new prompt is within a cosine **threshold** of one already answered — catches paraphrases | Avoids frontier calls entirely on semantically-repeated prompts |

The semantic cache is the real lever. It is **read-through with two tiers**, mirroring an L1-Map / L2-KV cache:

- **L1** — an in-process vector list, scanned locally with on-device SSM embeddings (free, offline-capable).
- **L2** — an optional shared backend ([`FetchSemanticCacheBackend`](packages/memory/src/cache/FetchSemanticCacheBackend.ts) → the BuilderForce.ai gateway), so a paraphrase answered by the **web app** is reusable by an **agent** and vice-versa.

```ts
import { SemanticCache, FetchSemanticCacheBackend, AnthropicBridge } from '@seanhogg/builderforce-memory';

const cache = new SemanticCache({
  embed: (t) => runtime.embed(t),                                   // on-device SSM (free)
  l2: new FetchSemanticCacheBackend({ baseUrl, apiKey }),           // shared via the gateway
  threshold: 0.92,
});

const { response, cached, tier } = await cache.getOrGenerate(
  prompt,
  () => bridge.generate(prompt),                                    // only runs on a miss
);
```

Memory-backed fact injection is semantic too: [`SSMAgent`](packages/memory/src/agent/SSMAgent.ts) defaults to `factSelection: 'semantic'`, injecting only the top-`maxFacts` embedding-relevant facts each turn (paraphrase-robust, smaller prompts) instead of an exact key-substring match.

## Develop

```bash
pnpm install          # links workspace packages; memory-engine auto-builds via prepare
pnpm build            # builds memory-engine, then memory (order matters — runtime needs core's .d.ts)
pnpm test             # 232 tests (127 core + 105 runtime)
pnpm lint
```

`@seanhogg/builderforce-memory` consumes `@seanhogg/builderforce-memory-engine` via the pnpm `workspace:^` protocol; on publish, pnpm rewrites it to a real `^<version>` range.

## Release

All three packages (`memory-engine`, `memory`, `memory-mcp`) version in lockstep (`YYYY.M.D[-beta.N]`).

1. Bump `version` in every `packages/*/package.json` (and root) to the same value, plus the `@seanhogg/builderforce-memory` range in `memory-mcp`'s `peerDependencies`.
2. `pnpm release:check` (build + test).
3. Commit, then `git tag vYYYY.M.D`.
4. Push the tag — `.github/workflows/release.yml` runs `pnpm -r publish`, which publishes `memory-engine`, `memory`, and `memory-mcp` to npm with provenance (dependency order is resolved automatically; `workspace:` specifiers are rewritten to real ranges at pack time).

## Migration / redirect plan for the old packages

The old packages (`@seanhogg/mambacode.js`, `@seanhogg/ssmjs`) are superseded. To migrate consumers cleanly:

1. **Publish the new scope** from this repo (`@seanhogg/builderforce-memory-engine`, `@seanhogg/builderforce-memory`).
2. **Deprecate the old names** pointing at the new scope:
   ```bash
   npm deprecate "@seanhogg/mambacode.js" "Moved to @seanhogg/builderforce-memory-engine"
   npm deprecate "@seanhogg/ssmjs"        "Moved to @seanhogg/builderforce-memory"
   ```
3. **Update consumers' imports**:
   - `@seanhogg/mambacode.js` → `@seanhogg/builderforce-memory-engine`
   - `@seanhogg/ssmjs`        → `@seanhogg/builderforce-memory`
   The primary in-repo consumer to update is the BuilderForce.ai agent-runtime (the claw that loads the SSM as hippocampus).
4. The old `SeanHogg/mambacodejs` and `SeanHogg/SSMjs` repos can be archived once consumers are cut over. Full git history from both was preserved into this repo via subtree merge — no history was lost.

## Consolidated Gap Register

Roadmap items identified during the consolidation but intentionally not closed in this pass:

- **[DONE 2026-06-18] Published under `@seanhogg/builderforce-memory*`; old names deprecated; mistakes cleaned up.** Final published names: `@seanhogg/builderforce-memory-engine` / `@seanhogg/builderforce-memory` / `@seanhogg/builderforce-memory-mcp` at `2026.6.19` (the `@builderforce` org doesn't exist on npm, so the scope is `@seanhogg` but the `builderforce-` name prefix is kept). Earlier wrong-name publishes are gone: `@builderforce/*` never landed (404), and the mistaken `@seanhogg/memory{,-engine,-mcp}@2026.6.18` were unpublished. Superseded legacy packages deprecated on npm via the `deprecate-old-packages.yml` workflow: `@seanhogg/ssmjs` → `@seanhogg/builderforce-memory`, `@seanhogg/mambakit` → `@seanhogg/builderforce-memory` (`@seanhogg/mambacode.js` was never on npm — engine was git-only).
- **[DONE 2026-06-18] BuilderForce.ai consumers repointed to the published packages.** frontend now depends on `@seanhogg/builderforce-memory-engine` (the `mambacode.js` git dep + `global.d.ts` shim removed; `tsc` clean, tests green); agent-runtime's SSM service imports `@seanhogg/builderforce-memory`. Open: agent-runtime can't add the `optionalDependencies` entry until `2026.6.19` is >48h old (its `pnpm.minimumReleaseAge: 2880` guard); add it after 2026-06-20, plus wire the frontend browser `SemanticCache` (needs the runtime dep + WebGPU bundle verify).
- **[DONE 2026-06-18] Multi-host installer — memory now extends to any MCP-capable agent, not just Claude.** New `builderforce-memory-install` bin + `src/install/` module (host registry + shared launch spec) register the stdio server into Claude Code, Claude Desktop, Cursor, Windsurf, VS Code, Cline, Gemini CLI, and Codex CLI, idempotently with `.bak` backups, all pointing at one shared `~/.builderforce-memory/memory.json` store. Auto-detects installed hosts (`--host=auto`), or `--host=all|<ids>`. 7 unit tests over an in-memory fs (`tests/install.test.mjs`). `buildServerSpec`/`installMemoryServer` exported for reuse.
- **builderforce.ai's own agents — memory wiring status.** (1) **On-prem agent hosts** run on a machine with an MCP host → covered by the multi-host installer above. (2) **[DONE 2026-06-18] Cloud Node/container agents** now load memory in-process via `ssm-memory-service.ts` — `@seanhogg/builderforce-memory`(+`-engine`) added to agent-runtime `optionalDependencies`; the 48h `minimumReleaseAge` cooldown was bypassed for first-party packages only via `minimumReleaseAgeExclude` (third-party guard intact). Verified: recall/remember loads in the agent-runtime env; `server-startup` boots `initSsmMemoryService`, the orchestrator injects `recallSimilar` into task prompts, and `KnowledgeLoopService` remembers/learns per run. (3) **[Node DONE 2026-06-18] Active recall/remember TOOLS** — instead of the MCP-into-custom-loop path, the `memory` capability in `@builderforce/agent-tools` was completed with `memory_recall`/`memory_remember` tool definitions, backed on-prem by the SSM service; on-prem + cloud-container (Node) agents now actively recall/remember mid-run. The server-side `getCachedOrGenerate` semantic cache was also wired into the cortex call. (4) **[DONE 2026-06-18] Cloud Worker/DO (V2 durable) active memory** — backed by Postgres `agent_memory` (migration 0200, tenant-scoped key→fact, lexical ILIKE recall, read-through cached) wired into the api's `buildCloudProvider` + `'memory'` in `CLOUD_SURFACE_CAPS`; graceful-degrades pre-migration, activates on deploy. The only remaining sibling is the **web (browser) `SemanticCache`** — genuinely browser-bound: needs a client WebGPU SSM embedder + deployed tokenizer/model assets + an L2 endpoint; the server-side cortex `getCachedOrGenerate` equivalent is already wired.
- **Lint not enforced in CI and likely dirty.** CI runs build + test only. The two packages had divergent ESLint setups (runtime had `@typescript-eslint`, engine did not) now unified into one root config; `pnpm lint` has not been run to green and is absent from the CI gate. Closing this makes lint a real signal instead of decoration.
- **`tsconfig` not wired for project references / incremental.** Root `tsconfig.json` lists references but the packages aren't `composite: true`, so `tsc -b` incremental builds and editor cross-package go-to-def rely on the built `dist` rather than source. Making the packages composite would speed builds and tighten editor UX.
- **README claims unverified at runtime.** The "browser-trained SSM → exported model → pushed to agent as hippocampus" loop is documented as the product strategy but there is no end-to-end test in this repo exercising export→load→recall across the two packages. An integration test would convert the narrative into a guarantee.
- **Local working-copy folder still named `SSMjs`.** This monorepo lives in the former `SSMjs` checkout (its native git history is the base); the on-disk rename to `builderforce-memory` is blocked by VS Code holding a handle on the folder (the open `vscode.git.Git.log` tab) — ruled out tooling shells, the IDE is the locker. Cosmetic only — the GitHub repo, local remote URL, repo identity, and `package.json` name are all already `builderforce-memory`. Close the git-log tab/window, then run `Rename-Item C:\code\agentic\SSMjs builderforce-memory`.
- **`mambacodejs` retired.** Local checkout removed and `SeanHogg/mambacodejs` archived on GitHub (history preserved here via subtree merge into `packages/memory-engine`). Its archive description points to `SeanHogg/builderforce-memory`, which now exists (rename landed) — the pointer resolves.
- **Coverage excludes the WebGPU session layer (`src/session/**`) and barrels (`src/**/index.ts`).** `pnpm test:coverage` enforces a 100% global threshold, but `collectCoverageFrom` skips `src/session/**` — `MambaSession`, the WGSL/WebGPU `complete`/`completeStream`/`evaluate`/`embed` orchestration, `streaming.ts` sampling, `presets.ts`, and IDB weight `persistence.ts`. These bind to `@seanhogg/builderforce-memory-engine` + a GPU and can't be unit-tested without a WebGPU/`MambaModel` mock harness; the runtime/consumption surface that wraps them is at 100%. Closing this = a GPU-or-mock integration harness that drives `MambaSession` end-to-end (also closes the unverified export→load→recall loop noted above). Until then the engine boundary is covered only by the build + the consumers' own tests.
- **Fact injection still uses exact-substring match, not semantic recall.** `SSMAgent._buildPrompt` selects facts via `input.includes(f.key)` (`packages/memory/src/agent/SSMAgent.ts`). The store already exposes `recallSimilar()` (SSM-embedding top-K), which would inject fewer, more-relevant facts and shrink the escalated prompt further. Switching injection to top-K similarity (capped to N facts) closes this — deferred from the token-cost pass that added system-prefix caching.
- **`@seanhogg/builderforce-memory-mcp` zod peer skew (v3 vs v4).** The new MCP package pins `zod@^3.23`, but `@anthropic-ai/claude-agent-sdk@0.3.163` requires `zod@^4` (peer warning at install). The SDK transport (`src/transports/sdk.ts`) dynamically imports the SDK and hands it zod-v3 raw shapes; basic shapes are cross-compatible so it compiles and runs, but this is unverified against v4-only schema behaviour and a consumer installing the Agent SDK inherits the mismatch. Closing = bump `memory-mcp` to `zod@^4`, re-verify the MCP SDK `registerTool` + tool shapes still type-check, then drop this note.
- **`memory-mcp` headless bins fall back to lexical (Jaccard) recall.** The `builderforce-memory-mcp` stdio/HTTP bins don't load the SSM checkpoint/GPU, so `recall` uses Jaccard word-overlap. Smoke-verified mis-rank: query "what language does the user like?" ranked a `project.*` entry above `user.preferred-language`. Closing = let the bins optionally load an SSM runtime (or an embedding provider) and pass it to `createLocalMemoryStoreBackend({ runtime })`; until then, production recall should use the in-process transport with the agent-runtime's hippocampus, or accept lexical recall. Unblocks trustworthy headless recall.
- **`memory-mcp` has no automated tests / not in CI.** Verified by an ad-hoc smoke script (since removed); there is no jest suite for `buildMemoryTools` caps (top-K / truncation / read-only gating) and no MCP stdio round-trip integration test, and the package has no `test` script so `pnpm -r test` skips it. Closing = add unit + stdio-handshake tests and wire into the CI gate; converts the token-saving caps from "intended" to "guaranteed".
- **`memory-mcp` HTTP transport is stateless and tenant-blind.** `createMemoryHttpHandler` builds one `McpServer` per request against a single fixed backend with only bearer-token gating — no per-tenant backend routing, no rate limiting, and no read-through cache. The multi-tenant story (builderforce.ai hosting) needs a tenant→backend resolver and `getOrSetCached` on recall. Closing = add tenant resolution + caching when the remote builderforce.ai memory store lands (see next item).
- **No remote builderforce.ai memory backend yet.** `MemoryBackend` is the seam for "coordination flows through builderforce.ai", but only the local `MemoryStoreBackend` (per-process IndexedDB) ships. Cross-product/cross-machine shared memory needs a networked adapter implementing `recall/get/recallByTag/remember/forget` against builderforce.ai. Closing = build the remote read/write API + a `RemoteMemoryBackend` adapter; the tools and all three transports already consume it unchanged.
- **agent-runtime should consume `@seanhogg/builderforce-memory-mcp` instead of hand-rolling memory exposure.** `Builderforce.ai/agent-runtime/src/infra/ssm-memory-service.ts` wraps `MemoryStore` with its own `remember`/`recallSimilar` and would separately need to surface those to its own Claude Agent SDK loop. The in-process SDK transport (`createMemoryMcpServer`, passing `ssmMemoryService.runtime` as the embedding backend) is the canonical seam for that. Closing = refactor agent-runtime to depend on this package, removing the parallel wrapper — a DRY consolidation, deferred as a cross-repo change out of scope for shipping the package.
- **Facts share the cached system block, so any fact write busts the prefix.** Prompt caching now marks `System + Facts` as one ephemeral block. A fact update invalidates the whole cached prefix on the next turn. Acceptable (facts change slowly), but splitting the system prompt and the fact block into two `cache_control` breakpoints would let the static system prompt stay cached across fact writes. Deferred.
- **OpenAI/Fetch bridges rely on provider-side auto-caching, not an explicit marker.** Only `AnthropicBridge` emits `cache_control`; OpenAI-compatible endpoints cache prefixes automatically when the stable system message comes first (it now does, via the runtime `system` split). No marker API exists to assert it — undocumented and unverifiable from this side. Note in bridge docs.
