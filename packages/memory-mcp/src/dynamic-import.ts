/**
 * Load a module by name at run time without the bundler or tsc resolving it.
 *
 * The optional peers (the memory runtime, the engine, fake-indexeddb, the Claude
 * Agent SDK) and Node builtins are reached through this, so a consumer that never
 * takes those paths — a browser bundle, a custom backend, the HTTP thin client —
 * never has to install or ship them.
 */
export function dynamicImport(m: string): Promise<unknown> {
    // eslint-disable-next-line @typescript-eslint/no-implied-eval, no-new-func
    return new Function("m", "return import(m)")(m) as Promise<unknown>;
}
