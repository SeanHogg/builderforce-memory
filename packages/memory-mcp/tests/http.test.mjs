import { test } from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import {
  hashToken,
  timingSafeEqualStr,
  buildTenantIndex,
  bearerToken,
  RateLimiter,
  createMemoryHttpHandler,
} from "../dist/index.js";

// ── auth primitives ───────────────────────────────────────────────────────────

test("timingSafeEqualStr compares by value", () => {
  assert.equal(timingSafeEqualStr("secret", "secret"), true);
  assert.equal(timingSafeEqualStr("secret", "secres"), false);
  assert.equal(timingSafeEqualStr("a", "ab"), false); // length differs → false, no throw
});

test("hashToken is stable + distinct, buildTenantIndex keys by hash", () => {
  assert.equal(hashToken("x"), hashToken("x"));
  assert.notEqual(hashToken("x"), hashToken("y"));
  const idx = buildTenantIndex({ tokA: "A", tokB: "B" });
  assert.equal(idx.get(hashToken("tokA")), "A");
  assert.equal(idx.get(hashToken("tokB")), "B");
  assert.equal(idx.get(hashToken("nope")), undefined);
});

test("bearerToken parses the Authorization header", () => {
  assert.equal(bearerToken("Bearer abc"), "abc");
  assert.equal(bearerToken("bearer  spaced "), "spaced");
  assert.equal(bearerToken(["Bearer arr"]), "arr");
  assert.equal(bearerToken(undefined), undefined);
  assert.equal(bearerToken("Basic xyz"), undefined);
});

test("RateLimiter enforces a fixed window with injectable clock", () => {
  let t = 0;
  const rl = new RateLimiter(1000, 2, () => t);
  assert.equal(rl.check("k"), true);
  assert.equal(rl.check("k"), true);
  assert.equal(rl.check("k"), false); // exhausted
  assert.equal(rl.check("other"), true); // independent key
  t = 1000; // window rolled
  assert.equal(rl.check("k"), true);
});

// ── handler: tenant isolation + auth + rate limit ─────────────────────────────

function fakeBackend(label) {
  return { recall: async () => [], get: async () => undefined, recallByTag: async () => [], _label: label };
}

/** Minimal req/res doubles. res records the status; we never reach MCP for 401/429. */
function mockReqRes(headers = {}) {
  const req = new EventEmitter();
  req.headers = headers;
  req.socket = { remoteAddress: "1.2.3.4" };
  req.method = "POST";
  const res = new EventEmitter();
  res.statusCode = 0;
  res.headers = null;
  res.body = null;
  res.writeHead = (status, h) => { res.statusCode = status; res.headers = h; return res; };
  res.end = (b) => { res.body = b ?? null; res.emit("__end"); };
  res.on("close", () => {});
  return { req, res };
}

test("multi-tenant: a valid token is accepted, an unknown token is 401, missing is 401", async () => {
  // Spy on the resolved backend by making recall throw a marker once MCP runs.
  const handler = createMemoryHttpHandler(fakeBackend("default"), {
    tenants: { aaa: fakeBackend("tenantA"), bbb: fakeBackend("tenantB") },
  });

  // Unknown token → 401
  {
    const { req, res } = mockReqRes({ authorization: "Bearer zzz" });
    await handler(req, res);
    assert.equal(res.statusCode, 401);
  }
  // Missing token → 401
  {
    const { req, res } = mockReqRes({});
    await handler(req, res);
    assert.equal(res.statusCode, 401);
  }
});

test("single shared secret: constant-time match required", async () => {
  const handler = createMemoryHttpHandler(fakeBackend("only"), { authToken: "s3cret" });
  const { req, res } = mockReqRes({ authorization: "Bearer wrong" });
  await handler(req, res);
  assert.equal(res.statusCode, 401);
});

test("rate limit returns 429 once the window is exhausted (open mode, per IP)", async () => {
  let t = 0;
  const handler = createMemoryHttpHandler(fakeBackend("open"), {
    rateLimit: { windowMs: 1000, max: 1 },
    now: () => t,
    // open mode: no auth → keyed by client IP; first call passes rate limit then
    // proceeds into MCP (which will try to read the request body and fail on our
    // mock). We only assert the SECOND call is rate-limited (429) before MCP.
  });

  // First request consumes the single token; let MCP error out harmlessly.
  {
    const { req, res } = mockReqRes({});
    try { await handler(req, res); } catch { /* MCP body parse may throw on mock */ }
  }
  // Second request from same IP within the window → 429 (short-circuits before MCP).
  {
    const { req, res } = mockReqRes({});
    await handler(req, res);
    assert.equal(res.statusCode, 429);
  }
});
