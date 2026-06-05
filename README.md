# BuilderForce Agent Memory

A framework for giving **any AI agent** persistent, browser-trainable SSM memory. It is provider-neutral — swappable LLM bridges (Anthropic / OpenAI / Fetch), an inference router, online distillation, and a persistent memory store — so any agent or app can depend on it, not just BuilderForce.

**[BuilderForce.ai](https://builderforce.ai) is the flagship deployment**, not the boundary: it's the *hippocampus* in BuilderForce's cortex/hippocampus split — BuilderForce trains a custom SSM in the browser, exports a model, and pushes it to an agent runtime as persistent memory while the frontier LLM stays the cortex. The same framework is reusable by other agents off the shelf.

This monorepo consolidates two packages that previously lived in separate repos (`mambacode.js` and `ssmjs`) whose names described the *technique* rather than the *role*. They are renamed and unified here so that "what they do" is legible: they are Agent Memory.

## Packages

| Package | Layer | Was | Responsibility |
|---|---|---|---|
| [`@builderforce/memory-engine`](packages/memory-engine) | **Engine** | `@seanhogg/mambacode.js` | WGSL/WebGPU Mamba SSM kernels, model blocks (Mamba1/2/3 + attention), autograd, trainer, BPE tokenizer, quantization. Zero runtime deps. |
| [`@builderforce/memory`](packages/memory) | **Runtime** | `@seanhogg/ssmjs` | SSM execution, Transformer orchestration (Anthropic/OpenAI/Fetch bridges), online distillation, inference router, sessions, and the persistent `MemoryStore`. Depends on `memory-engine`. |

The two-package split is deliberate: the engine is zero-dep and WebGPU-pure and can be consumed standalone; the runtime pulls in LLM-vendor bridges. Flattening them would force engine-only consumers to drag in vendor code and vice versa. They release in lockstep from one pipeline, which kills the publish-drift bug class that the separate-repo setup suffered (bumping one version without publishing + regenerating the consumer lockfile).

## Develop

```bash
pnpm install          # links workspace packages; memory-engine auto-builds via prepare
pnpm build            # builds memory-engine, then memory (order matters — runtime needs core's .d.ts)
pnpm test             # 232 tests (127 core + 105 runtime)
pnpm lint
```

`@builderforce/memory` consumes `@builderforce/memory-engine` via the pnpm `workspace:^` protocol; on publish, pnpm rewrites it to a real `^<version>` range.

## Release

Both packages version in lockstep (`YYYY.M.D[-beta.N]`).

1. Bump `version` in both `packages/*/package.json` (and root) to the same value.
2. `pnpm release:check` (build + test).
3. Commit, then `git tag vYYYY.M.D`.
4. Push the tag — `.github/workflows/release.yml` publishes `memory-engine` then `memory` to npm with provenance.

## Migration / redirect plan for the old packages

The old packages (`@seanhogg/mambacode.js`, `@seanhogg/ssmjs`) are superseded. To migrate consumers cleanly:

1. **Publish the new scope** from this repo (`@builderforce/memory-engine`, `@builderforce/memory`).
2. **Deprecate the old names** pointing at the new scope:
   ```bash
   npm deprecate "@seanhogg/mambacode.js" "Moved to @builderforce/memory-engine"
   npm deprecate "@seanhogg/ssmjs"        "Moved to @builderforce/memory"
   ```
3. **Update consumers' imports**:
   - `@seanhogg/mambacode.js` → `@builderforce/memory-engine`
   - `@seanhogg/ssmjs`        → `@builderforce/memory`
   The primary in-repo consumer to update is the BuilderForce.ai agent-runtime (the claw that loads the SSM as hippocampus).
4. The old `SeanHogg/mambacodejs` and `SeanHogg/SSMjs` repos can be archived once consumers are cut over. Full git history from both was preserved into this repo via subtree merge — no history was lost.

## Consolidated Gap Register

Roadmap items identified during the consolidation but intentionally not closed in this pass:

- **Old packages not yet deprecated on npm.** `@seanhogg/mambacode.js` / `@seanhogg/ssmjs` remain live and undeprecated; the `npm deprecate` + new-scope publish (steps 1–2 above) is a manual registry action requiring npm auth, not doable from this migration commit. Until done, downstream installs can still resolve the stale names and re-trigger publish drift.
- **Downstream consumer imports not rewired.** The BuilderForce.ai agent-runtime (and any other repo importing `@seanhogg/ssmjs`/`mambacode.js`) still references the old specifiers. Closing this unblocks deleting the old repos. Needs a cross-repo grep + bump once the new scope is published.
- **Lint not enforced in CI and likely dirty.** CI runs build + test only. The two packages had divergent ESLint setups (runtime had `@typescript-eslint`, engine did not) now unified into one root config; `pnpm lint` has not been run to green and is absent from the CI gate. Closing this makes lint a real signal instead of decoration.
- **`tsconfig` not wired for project references / incremental.** Root `tsconfig.json` lists references but the packages aren't `composite: true`, so `tsc -b` incremental builds and editor cross-package go-to-def rely on the built `dist` rather than source. Making the packages composite would speed builds and tighten editor UX.
- **README claims unverified at runtime.** The "browser-trained SSM → exported model → pushed to agent as hippocampus" loop is documented as the product strategy but there is no end-to-end test in this repo exercising export→load→recall across the two packages. An integration test would convert the narrative into a guarantee.
- **Local working-copy folder still named `SSMjs`.** This monorepo lives in the former `SSMjs` checkout (its native git history is the base); the on-disk rename to `builderforce-memory` is blocked by VS Code holding a handle on the folder (the open `vscode.git.Git.log` tab) — ruled out tooling shells, the IDE is the locker. Cosmetic only — the GitHub repo, local remote URL, repo identity, and `package.json` name are all already `builderforce-memory`. Close the git-log tab/window, then run `Rename-Item C:\code\agentic\SSMjs builderforce-memory`.
- **`mambacodejs` retired.** Local checkout removed and `SeanHogg/mambacodejs` archived on GitHub (history preserved here via subtree merge into `packages/memory-engine`). Its archive description points to `SeanHogg/builderforce-memory`, which now exists (rename landed) — the pointer resolves.
- **Fact injection still uses exact-substring match, not semantic recall.** `SSMAgent._buildPrompt` selects facts via `input.includes(f.key)` (`packages/memory/src/agent/SSMAgent.ts`). The store already exposes `recallSimilar()` (SSM-embedding top-K), which would inject fewer, more-relevant facts and shrink the escalated prompt further. Switching injection to top-K similarity (capped to N facts) closes this — deferred from the token-cost pass that added system-prefix caching.
- **Facts share the cached system block, so any fact write busts the prefix.** Prompt caching now marks `System + Facts` as one ephemeral block. A fact update invalidates the whole cached prefix on the next turn. Acceptable (facts change slowly), but splitting the system prompt and the fact block into two `cache_control` breakpoints would let the static system prompt stay cached across fact writes. Deferred.
- **OpenAI/Fetch bridges rely on provider-side auto-caching, not an explicit marker.** Only `AnthropicBridge` emits `cache_control`; OpenAI-compatible endpoints cache prefixes automatically when the stable system message comes first (it now does, via the runtime `system` split). No marker API exists to assert it — undocumented and unverifiable from this side. Note in bridge docs.
