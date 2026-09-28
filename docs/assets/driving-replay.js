"use strict";

// Display interpolation only. Recorded plans, commands and metrics stay intact.
const DrivingReplay = (() => {
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const lerp = (a, b, u) => a + (b - a) * u;
  function angle(a, b, u, period = 2 * Math.PI) {
    const difference =
      ((((b - a + period / 2) % period) + period) % period) - period / 2;
    return a + difference * u;
  }
  function frameIndex(frames, time, key = "time_s", tolerance = 1e-8) {
    let lo = 0,
      hi = frames.length - 1;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      if (frames[mid][key] <= time + tolerance) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }
  function sample(frames, time, smooth = true) {
    const index = frameIndex(frames, time),
      a = frames[index],
      b = frames[Math.min(index + 1, frames.length - 1)],
      dt = b.time_s - a.time_s,
      u = smooth && dt > 0 ? clamp((time - a.time_s) / dt, 0, 1) : 0,
      ego = { ...a.ego };
    for (const key of ["x", "y", "speed", "steering"])
      ego[key] = lerp(a.ego[key], b.ego[key], u);
    ego.yaw = angle(a.ego.yaw, b.ego.yaw, u);
    // Published synthetic scenarios have at most one ground-truth actor.
    // With no IDs in this schema, do not associate multiple actors or tracks.
    let obstacles = a.obstacles;
    if (a.obstacles.length === 1 && b.obstacles.length === 1) {
      const x = a.obstacles[0],
        y = b.obstacles[0];
      if (
        [3, 4, 5].every((k) => Math.abs(x[k] - y[k]) < 1e-4) &&
        Math.hypot(x[0] - y[0], x[1] - y[1]) <= 20 * dt
      ) {
        const box = x.slice();
        box[0] = lerp(x[0], y[0], u);
        box[1] = lerp(x[1], y[1], u);
        box[6] = angle(x[6], y[6], u, Math.PI);
        obstacles = [box];
      }
    }
    return { index, ego, obstacles };
  }
  function events(frames, statuses) {
    const result = [];
    frames.forEach((frame, index) => {
      const previous = frames[index - 1],
        labels = [];
      if (!previous) labels.push("开始");
      else {
        if (frame.obstacles.length > previous.obstacles.length)
          labels.push("障碍出现");
        if (frame.signal !== previous.signal && frame.signal !== "none")
          labels.push(frame.signal === "green" ? "绿灯" : "红灯");
        if (frame.status !== previous.status)
          labels.push(statuses[frame.status] || frame.status);
        if (frame.ego.speed < 0.15 && previous.ego.speed >= 0.15)
          labels.push("车辆停下");
      }
      if (labels.length)
        result.push({ index, time: frame.time_s, label: labels.join(" · ") });
    });
    return result;
  }
  return { frameIndex, sample, events };
})();
if (typeof module !== "undefined") module.exports = DrivingReplay;
