import { useEffect, useRef, useState } from "react";

/** Real webcam capture -- getUserMedia + canvas snapshot. The captured JPEG
 * blob is handed to the parent via onCapture; nothing is uploaded until the
 * form is actually submitted, and the camera stream is stopped as soon as a
 * photo is taken or the component unmounts. */
export default function WebcamCapture({ onCapture }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [active, setActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [error, setError] = useState(null);

  const stopStream = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setActive(false);
  };

  useEffect(() => () => stopStream(), []);

  const enableCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setActive(true);
    } catch (e) {
      setError("Camera access denied or unavailable. You can still submit without a selfie.");
    }
  };

  const capture = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth || 480;
    canvas.height = video.videoHeight || 360;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        setPreviewUrl(URL.createObjectURL(blob));
        onCapture(blob);
        stopStream();
      },
      "image/jpeg",
      0.9
    );
  };

  const retake = () => {
    setPreviewUrl(null);
    onCapture(null);
    enableCamera();
  };

  return (
    <div>
      {!active && !previewUrl && (
        <button type="button" className="btn secondary" onClick={enableCamera}>
          Enable Camera
        </button>
      )}
      {error && <p style={{ fontSize: 12, color: "var(--danger)", marginTop: 8 }}>{error}</p>}

      <div style={{ display: active || previewUrl ? "block" : "none", marginTop: 10 }}>
        <video
          ref={videoRef}
          style={{ display: active ? "block" : "none", width: 240, borderRadius: 10, border: "1px solid var(--panel-border)" }}
          muted
          playsInline
        />
        {previewUrl && (
          <img src={previewUrl} alt="captured selfie" style={{ width: 240, borderRadius: 10, border: "1px solid var(--ok)" }} />
        )}
      </div>
      <canvas ref={canvasRef} style={{ display: "none" }} />

      <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
        {active && (
          <button type="button" className="btn" onClick={capture}>
            Capture Selfie
          </button>
        )}
        {previewUrl && (
          <button type="button" className="btn secondary" onClick={retake}>
            Retake
          </button>
        )}
      </div>
    </div>
  );
}
