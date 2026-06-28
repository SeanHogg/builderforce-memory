import { test } from "node:test";
import assert from "node:assert/strict";
import { buildMemoryTools } from "../dist/index.js";

/** Minimal in-memory MemoryBackend (read-only) for tool-shape assertions. */
function fakeBackend() {
  return {
    recall: async () => [],
    get: async () => undefined,
    recallByTag: async () => [],
  };
}

test("gateway cost tools are absent without gateway options", () => {
  const tools = buildMemoryTools(fakeBackend(), {});
  const names = tools.map((t) => t.name);
  assert.ok(!names.includes("token_usage"));
  assert.ok(!names.includes("model_efficiency"));
});

test("gateway cost tools are absent when only one option is present", () => {
  const a = buildMemoryTools(fakeBackend(), { gatewayUrl: "https://api.builderforce.ai" });
  const b = buildMemoryTools(fakeBackend(), { gatewayApiKey: "bfk_test" });
  assert.ok(!a.map((t) => t.name).includes("token_usage"));
  assert.ok(!b.map((t) => t.name).includes("token_usage"));
});

test("gateway cost tools appear when both options are present", () => {
  const tools = buildMemoryTools(fakeBackend(), {
    gatewayUrl: "https://api.builderforce.ai",
    gatewayApiKey: "bfk_test",
  });
  const names = tools.map((t) => t.name);
  assert.ok(names.includes("token_usage"));
  assert.ok(names.includes("model_efficiency"));
});

test("token_usage formats a snapshot from the gateway", async () => {
  const origFetch = globalThis.fetch;
  let calledUrl, calledAuth;
  globalThis.fetch = async (url, init) => {
    calledUrl = url;
    calledAuth = init?.headers?.authorization;
    return {
      ok: true,
      status: 200,
      json: async () => ({
        windowLabel: "today",
        todayTokens: 12345,
        todayCostUsd: 0.42,
        dailyCapTokens: 1000000,
        pctOfDailyCap: 1.2,
        topModel: { model: "anthropic/claude-sonnet", tokens: 9000 },
        costPerMergedPrUsd: 0.13,
        tip: "Looking good",
      }),
    };
  };
  try {
    const tools = buildMemoryTools(fakeBackend(), {
      gatewayUrl: "https://api.builderforce.ai",
      gatewayApiKey: "bfk_test",
    });
    const tool = tools.find((t) => t.name === "token_usage");
    const res = await tool.handler({});
    assert.equal(res.isError, undefined);
    const text = res.content[0].text;
    assert.match(calledUrl, /\/llm\/v1\/builder-insights$/);
    assert.equal(calledAuth, "Bearer bfk_test");
    assert.match(text, /12,345/);
    assert.match(text, /\$0\.42/);
    assert.match(text, /anthropic\/claude-sonnet/);
    assert.match(text, /Tip: Looking good/);
  } finally {
    globalThis.fetch = origFetch;
  }
});

test("token_usage returns an error result on a failed gateway call", async () => {
  const origFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: false, status: 503, json: async () => ({}) });
  try {
    const tools = buildMemoryTools(fakeBackend(), {
      gatewayUrl: "https://api.builderforce.ai",
      gatewayApiKey: "bfk_test",
    });
    const res = await tools.find((t) => t.name === "token_usage").handler({});
    assert.equal(res.isError, true);
    assert.match(res.content[0].text, /503/);
  } finally {
    globalThis.fetch = origFetch;
  }
});

test("model_efficiency formats the analytics ranking", async () => {
  const origFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      byAction: [
        {
          actionType: "code_edit",
          label: "Code Edit",
          models: [
            { model: "anthropic/claude-sonnet", samples: 10, avgScore: 0.9, mergeRate: 0.8, avgCostMillicents: 500 },
          ],
        },
      ],
    }),
  });
  try {
    const tools = buildMemoryTools(fakeBackend(), {
      gatewayUrl: "https://api.builderforce.ai",
      gatewayApiKey: "bfk_test",
    });
    const res = await tools.find((t) => t.name === "model_efficiency").handler({});
    assert.equal(res.isError, undefined);
    const text = res.content[0].text;
    assert.match(text, /Code Edit/);
    assert.match(text, /anthropic\/claude-sonnet/);
    assert.match(text, /merge=80%/);
  } finally {
    globalThis.fetch = origFetch;
  }
});
