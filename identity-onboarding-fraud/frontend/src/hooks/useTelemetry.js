import { useCallback, useEffect, useRef } from "react";

/** Captures REAL behavioral telemetry from the actual form interaction --
 * keystroke timing, per-field focus transitions, and mouse-movement
 * entropy -- the same signals a production bot-detection layer would
 * observe. Nothing here is synthetic; it's the literal DOM events. */
export function useTelemetry() {
  const startedAt = useRef(Date.now());
  const firstInteractionAt = useRef(null);
  const keyTimestamps = useRef([]);
  const corrections = useRef(0);
  const fieldTransitions = useRef([]); // gaps between blur(prev) and focus(next)
  const lastBlurAt = useRef(null);
  const mouseSamples = useRef([]);
  const lastMouseSampleAt = useRef(0);

  const onKeyDown = useCallback((e) => {
    if (!firstInteractionAt.current) firstInteractionAt.current = Date.now();
    keyTimestamps.current.push(Date.now());
    if (e.key === "Backspace" || e.key === "Delete") corrections.current += 1;
  }, []);

  const onFieldFocus = useCallback(() => {
    if (!firstInteractionAt.current) firstInteractionAt.current = Date.now();
    if (lastBlurAt.current) {
      fieldTransitions.current.push((Date.now() - lastBlurAt.current) / 1000);
    }
  }, []);

  const onFieldBlur = useCallback(() => {
    lastBlurAt.current = Date.now();
  }, []);

  useEffect(() => {
    const onMouseMove = (e) => {
      const now = Date.now();
      if (now - lastMouseSampleAt.current < 40) return; // throttle ~25/sec
      lastMouseSampleAt.current = now;
      const samples = mouseSamples.current;
      samples.push({ x: e.clientX, y: e.clientY, t: now });
      if (samples.length > 600) samples.shift();
    };
    window.addEventListener("mousemove", onMouseMove);
    return () => window.removeEventListener("mousemove", onMouseMove);
  }, []);

  const computeMouseEntropy = () => {
    const s = mouseSamples.current;
    if (s.length < 5) return 0.05; // effectively no mouse activity -> low entropy
    const angles = [];
    for (let i = 1; i < s.length; i++) {
      const dx = s[i].x - s[i - 1].x;
      const dy = s[i].y - s[i - 1].y;
      if (dx === 0 && dy === 0) continue;
      angles.push(Math.atan2(dy, dx));
    }
    if (angles.length < 3) return 0.05;
    const mean = angles.reduce((a, b) => a + b, 0) / angles.length;
    const variance = angles.reduce((a, b) => a + (b - mean) ** 2, 0) / angles.length;
    const std = Math.sqrt(variance);
    return Math.max(0, Math.min(1, std / Math.PI));
  };

  const computeTypingStats = () => {
    const ts = keyTimestamps.current;
    if (ts.length < 3) return { typing_speed_cps: 0, typing_variance: 0.5 };
    const intervals = [];
    for (let i = 1; i < ts.length; i++) intervals.push(ts[i] - ts[i - 1]);
    const meanMs = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const speed = meanMs > 0 ? Math.min(20, 1000 / meanMs) : 0;
    const varMs = intervals.reduce((a, b) => a + (b - meanMs) ** 2, 0) / intervals.length;
    const cv = meanMs > 0 ? Math.sqrt(varMs) / meanMs : 0;
    return { typing_speed_cps: Number(speed.toFixed(2)), typing_variance: Number(Math.min(1, cv).toFixed(3)) };
  };

  /** Call once at submit time to get the final telemetry payload. */
  const snapshot = useCallback(() => {
    const now = Date.now();
    const { typing_speed_cps, typing_variance } = computeTypingStats();
    const mouse_entropy = computeMouseEntropy();
    const avgFieldGap = fieldTransitions.current.length
      ? fieldTransitions.current.reduce((a, b) => a + b, 0) / fieldTransitions.current.length
      : 2.0;
    const sessionSec = (now - startedAt.current) / 1000;
    const activeSec = firstInteractionAt.current ? (now - firstInteractionAt.current) / 1000 : sessionSec;

    // A simple, transparent heuristic -- NOT a trained model -- combining
    // three real observed signals into one score. Documented, not hidden.
    const automation_score = Math.max(
      0,
      Math.min(
        1,
        0.5 * (1 - Math.min(1, typing_variance * 3)) +
          0.3 * (1 - mouse_entropy) +
          0.2 * (activeSec < 4 && keyTimestamps.current.length > 15 ? 1 : 0)
      )
    );

    return {
      session_duration_sec: Number(sessionSec.toFixed(1)),
      form_completion_time_sec: Number(activeSec.toFixed(1)),
      typing_speed_cps,
      typing_variance,
      mouse_entropy: Number(mouse_entropy.toFixed(3)),
      num_corrections: corrections.current,
      avg_time_between_fields_sec: Number(avgFieldGap.toFixed(2)),
      automation_score: Number(automation_score.toFixed(3)),
      _raw_samples: { keystrokes: keyTimestamps.current.length, mouse_points: mouseSamples.current.length },
    };
  }, []);

  return { onKeyDown, onFieldFocus, onFieldBlur, snapshot };
}
