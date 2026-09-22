"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
require("./app.js");

const { frameAt, statusForCase } = globalThis.NavLabShowcaseLogic;

test("frameAt keeps the manifest start before the first post-step trace sample", () => {
  const caseData = {
    start: { x: 1, y: 2, yaw: .3, v: 0, delta: 0 },
    trace: { columns: ["t", "x", "y", "yaw", "v", "delta", "v_ref", "cte", "s"], rows: [[.05, 1.1, 2.1, .31, .2, .01, 1, .1, .01]] },
  };
  assert.deepEqual(frameAt(caseData, .01), { t: 0, x: 1, y: 2, yaw: .3, v: 0, delta: 0, v_ref: null, cte: null, s: 0, sampleIndex: -1 });
  assert.equal(frameAt(caseData, .05).sampleIndex, 0);
});

test("a nominal planning success cannot label an unsuccessful rollout as success", () => {
  assert.equal(statusForCase({ manifest: { planning: { status: "success" } }, metrics: { success: false } }), "trajectory_failed");
  assert.equal(statusForCase({ manifest: { planning: { status: "no_path" } }, metrics: { success: false } }), "no_path");
});
