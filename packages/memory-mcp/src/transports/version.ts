import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
export const memoryMcpVersion: string = require("../../package.json").version;
