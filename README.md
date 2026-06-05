# BuilderForce Agent Memory

The memory subsystem of **[BuilderForce.ai](https://builderforce.ai)**. This is the *hippocampus* in the cortex/hippocampus split: BuilderForce trains a custom SSM in the browser, exports a model, and pushes it to an agent runtime as persistent memory. The agent's frontier LLM stays the cortex.

This monorepo consolidates two packages that previously lived in separate repos (`mambacode.js` and `ssmjs`) whose names described the *technique* rather than the *role*. They are renamed and unified here so that "what they do" is legible: they are Agent Memory.

## Packages

| Package | Layer | Was | Responsibility |
|---|---|---|---|
| [`@builderforce/memory-core`](packages/memory-core) | **Engine** | `@seanhogg/mambacode.js` | WGSL/WebGPU Mamba SSM kernels, model blocks (Mamba1/2/3 + attention), autograd, trainer, BPE tokenizer, quantization. Zero runtime deps. |
| [`@builderforce/memory`](packages/memory) | **Runtime** | `@seanhogg/ssmjs` | SSM execution, Transformer orchestration (Anthropic/OpenAI/Fetch bridges), online distillation, inference router, sessions, and the persistent `MemoryStore`. Depends on `memory-core`. |

The two-package split is deliberate: the engine is zero-dep and WebGPU-pure and can be consumed standalone; the runtime pulls in LLM-vendor bridges. Flattening them would force engine-only consumers to drag in vendor code and vice versa. They release in lockstep from one pipeline, which kills the publish-drift bug class that the separate-repo setup suffered (bumping one version without publishing + regenerating the consumer lockfile).

## Develop

```bash
pnpm install          # links workspace packages; memory-core auto-builds via prepare
pnpm build            # builds memory-core, then memory (order matters — runtime needs core's .d.ts)
pnpm test             # 232 tests (127 core + 105 runtime)
pnpm lint
```

`@builderforce/memory` consumes `@builderforce/memory-core` via the pnpm `workspace:^` protocol; on publish, pnpm rewrites it to a real `^<version>` range.

## Release

Both packages version in lockstep (`YYYY.M.D[-beta.N]`).

1. Bump `version` in both `packages/*/package.json` (and root) to the same value.
2. `pnpm release:check` (build + test).
3. Commit, then `git tag vYYYY.M.D`.
4. Push the tag — `.github/workflows/release.yml` publishes `memory-core` then `memory` to npm with provenance.

## Migration / redirect plan for the old packages

The old packages (`@seanhogg/mambacode.js`, `@seanhogg/ssmjs`) are superseded. To migrate consumers cleanly:

1. **Publish the new scope** from this repo (`@builderforce/memory-core`, `@builderforce/memory`).
2. **Deprecate the old names** pointing at the new scope:
   ```bash
   npm deprecate "@seanhogg/mambacode.js" "Moved to @builderforce/memory-core"
   npm deprecate "@seanhogg/ssmjs"        "Moved to @builderforce/memory"
   ```
3. **Update consumers' imports**:
   - `@seanhogg/mambacode.js` → `@builderforce/memory-core`
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
- **Local working-copy folder still named `SSMjs`.** This monorepo lives in the former `SSMjs` checkout (its native git history is the base); the on-disk folder rename to `builderforce-memory` was blocked by a Windows lock (IDE/watcher holding the dir). Cosmetic only — the repo identity and `package.json` name are already `builderforce-memory`. Rename the folder when no process holds it, and rename `SeanHogg/SSMjs → builderforce-memory` on GitHub (auto-redirects old URLs).
- **`mambacodejs` retired.** Local checkout removed and `SeanHogg/mambacodejs` archived on GitHub (history preserved here via subtree merge into `packages/memory-core`). Its archive description points to `SeanHogg/builderforce-memory` — a link that resolves only after the `SeanHogg/SSMjs → builderforce-memory` GitHub rename lands (still pending, see above).
