import { dynamicImport } from "./dynamic-import.js";

const SHARED_PACKAGE = "@seanhogg/builderforce-mcp";
const modules = new Map<string, Promise<unknown | undefined>>();

/** Resolve the shared runtime when installed alongside Memory. */
export function sharedMcpModule<T>(entry = ""): Promise<T | undefined> {
    let loading = modules.get(entry);
    if (!loading) {
        loading = dynamicImport(SHARED_PACKAGE + entry).catch((error: NodeJS.ErrnoException) => {
            if (error.code === "ERR_MODULE_NOT_FOUND"
                && error.message.startsWith(`Cannot find package '${SHARED_PACKAGE}' imported from `)) {
                return undefined;
            }
            throw error;
        });
        modules.set(entry, loading);
    }
    return loading as Promise<T | undefined>;
}
