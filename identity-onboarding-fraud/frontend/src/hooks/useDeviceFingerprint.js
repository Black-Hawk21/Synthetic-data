import { useEffect, useState } from "react";

function parseUserAgent(ua) {
  let os = "Unknown";
  if (/Windows/i.test(ua)) os = "Windows";
  else if (/Mac OS X/i.test(ua)) os = "macOS";
  else if (/Android/i.test(ua)) os = "Android";
  else if (/iPhone|iPad|iOS/i.test(ua)) os = "iOS";
  else if (/Linux/i.test(ua)) os = "Linux";

  let browser = "Unknown";
  if (/Edg\//i.test(ua)) browser = "Edge";
  else if (/Chrome\//i.test(ua)) browser = "Chrome";
  else if (/Firefox\//i.test(ua)) browser = "Firefox";
  else if (/Safari\//i.test(ua)) browser = "Safari";

  return { os, browser };
}

function randomId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID().replace(/-/g, "").slice(0, 20);
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

/** Persists a stable per-browser device id + "first seen" timestamp in
 * localStorage -- a rough client-side approximation of the kind of device
 * fingerprinting a real fraud-detection SDK would do, so repeat visits from
 * the same browser genuinely show up as device reuse in the identity graph. */
export function useDeviceFingerprint() {
  const [fingerprint, setFingerprint] = useState(null);

  useEffect(() => {
    let deviceId = localStorage.getItem("ifdl_device_id");
    let firstSeen = localStorage.getItem("ifdl_device_first_seen");
    if (!deviceId) {
      deviceId = randomId();
      firstSeen = String(Date.now());
      localStorage.setItem("ifdl_device_id", deviceId);
      localStorage.setItem("ifdl_device_first_seen", firstSeen);
    }
    const { os, browser } = parseUserAgent(navigator.userAgent);
    const deviceAgeDays = Math.floor((Date.now() - Number(firstSeen)) / 86400000);

    setFingerprint({
      device_id: deviceId,
      device_age_days: deviceAgeDays,
      os,
      browser,
      screen_resolution: `${window.screen.width}x${window.screen.height}`,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC",
      language: navigator.language || "en-US",
    });
  }, []);

  return fingerprint;
}
