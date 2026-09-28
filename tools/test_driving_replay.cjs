const assert = require("node:assert/strict");
const {
  frameIndex,
  sample,
  events,
} = require("../docs/assets/driving-replay.js");
const box = (x) => [x, 0, 1, 4.5, 1.9, 1.5, 0];
const frame = (time, x, yaw, obstacles = []) => ({
  time_s: time,
  ego: { x, y: 0, yaw, speed: 5, steering: 0 },
  obstacles,
  status: "cruise",
  signal: "none",
  command: { acceleration: 2 },
});
const frames = [
  frame(0, 0, (179 * Math.PI) / 180),
  frame(0.2, 1, (-179 * Math.PI) / 180, [box(8)]),
  frame(0.3, 1.5, (-177 * Math.PI) / 180, [box(8.5)]),
];
const source = JSON.stringify(frames);
assert.equal(
  frameIndex(
    [{ video_time_s: 0 }, { video_time_s: 0.200000003 }],
    0.2,
    "video_time_s",
    1e-5,
  ),
  1,
);
assert.equal(frameIndex(frames, -1), 0);
assert.equal(frameIndex(frames, 0.19), 0);
assert.equal(frameIndex(frames, 0.2), 1);
assert.equal(frameIndex(frames, 4), 2);
const half = sample(frames, 0.1);
assert.equal(half.ego.x, 0.5);
assert.ok(
  Math.abs(half.ego.yaw - Math.PI) < 1e-10,
  "yaw must cross ±pi by the short arc",
);
assert.equal(half.obstacles.length, 0, "do not reveal a future obstacle");
assert.equal(sample(frames, 0.2).obstacles.length, 1);
assert.equal(sample(frames, 0.25).obstacles[0][0], 8.25);
assert.equal(sample(frames, 0.1, false).ego.x, 0);
assert.equal(sample(frames, -2).ego.x, 0);
assert.equal(sample(frames, 10).ego.x, 1.5);
assert.equal(
  JSON.stringify(frames),
  source,
  "display must not change source records",
);
const multiple = [
  frame(0, 0, 0, [box(1), box(2)]),
  frame(0.2, 1, 0, [box(2), box(1)]),
];
assert.deepEqual(
  sample(multiple, 0.1).obstacles,
  multiple[0].obstacles,
  "array order is not an identity",
);
frames[1].status = "emergency_stop";
frames[2].status = "yielding";
frames[2].signal = "green";
frames[2].ego.speed = 0;
const markers = events(frames, {
  emergency_stop: "紧急制动",
  yielding: "让行",
});
assert.match(markers[1].label, /障碍出现.*紧急制动/);
assert.match(markers[2].label, /绿灯.*车辆停下/);
assert.equal(
  frameIndex(
    [{ video_time_s: 0 }, { video_time_s: 0.2 }],
    0.199,
    "video_time_s",
  ),
  0,
);
console.log(
  "PASS: angle wrap, irregular timestamps, exact samples, no future actors, no mutation, event times and video alignment.",
);
