import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { toast } from "react-toastify";

export default function useTouchSimulator(
  canvasRef,
  previewCanvasRef,
  selectedGesture,
  penColor
) {
  const DEFAULT_ENDPOINT = "http://localhost:8000/api/gestures/predict";

  const pointers = useRef(new Map());
  const gestureFrames = useRef([]);
  const [payloadPreview, setPayloadPreview] = useState(null);
  const [hoverPosition, setHoverPosition] = useState(null);
  let nextId = useRef(1);

  const activePathsRef = useRef(new Map());
  const completedPathsRef = useRef([]);

  const SAMPLING_RATE = 10; // 10ms لكل فريم
  const frameTimer = useRef(null);
  const lastPointRef = useRef(null);

  const clearCanvasState = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    activePathsRef.current.clear();
    completedPathsRef.current = [];
    pointers.current.clear();
    gestureFrames.current = [];
    setPayloadPreview(null);
    setHoverPosition(null);
    nextId.current = 1;
    clearInterval(frameTimer.current);
    lastPointRef.current = null;
  };

  const computePointFeatures = (prevPoint, currPoint, deltaMs) => {
    if (!prevPoint) {
      return { dx: 0, dy: 0, vx: 0, vy: 0, angle: 0, pressure: 1.0 };
    }

    const dx = currPoint.x - prevPoint.x;
    const dy = currPoint.y - prevPoint.y;

    const vx = dx / (deltaMs || 1);
    const vy = dy / (deltaMs || 1);

    const angle = Math.atan2(dy, dx) * (180 / Math.PI);

    const speed = Math.sqrt(dx * dx + dy * dy) / (deltaMs || 1);
    const pressure = Math.max(0.1, Math.min(1.0, 1.0 - speed * 5));

    return { dx, dy, vx, vy, angle, pressure };
  };

  const createFixedFrame = (point, maxPoints = 21) => {
    const canvas = canvasRef.current;
    const ts = Date.now();
    const delta_ms = gestureFrames.current.length
      ? ts - gestureFrames.current[gestureFrames.current.length - 1].ts
      : 0;

    const prevFrame = gestureFrames.current.length
      ? gestureFrames.current[gestureFrames.current.length - 1]
      : null;

    const points = Array.from({ length: maxPoints }, (_, i) => {
      const prevPoint = prevFrame ? prevFrame.points[i] : null;
      const features = computePointFeatures(prevPoint, point, delta_ms);

      return {
        id: i + 1,
        x: parseFloat((point.x / canvas.width).toFixed(4)),
        y: parseFloat((point.y / canvas.height).toFixed(4)),
        state: "move",
        ...features,
      };
    });

    return { ts, delta_ms, frame_id: ts, points };
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    canvas.style.touchAction = "none";

    ctx.lineWidth = 6;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = penColor;

    const redraw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const pts of completedPathsRef.current) {
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        ctx.stroke();
      }

      for (const [, pts] of activePathsRef.current.entries()) {
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        ctx.stroke();
      }
    };

    const pointerDown = (e) => {
      e.target.setPointerCapture(e.pointerId);
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      const id = nextId.current++;
      pointers.current.set(e.pointerId, { id, x, y, state: "down" });
      activePathsRef.current.set(e.pointerId, [{ x, y, state: "down", id }]);

      lastPointRef.current = { x, y };

      frameTimer.current = setInterval(() => {
        if (!lastPointRef.current) return;
        const frame = createFixedFrame(lastPointRef.current);
        gestureFrames.current.push(frame);
        setPayloadPreview({ frames: gestureFrames.current });
      }, SAMPLING_RATE);

      redraw();
    };

    const pointerMove = (e) => {
      if (!pointers.current.has(e.pointerId)) return;

      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;
      setHoverPosition({ x, y });

      lastPointRef.current = { x, y };

      const path = activePathsRef.current.get(e.pointerId);
      path.push({ x, y, state: "move", id: path.length + 1 });

      redraw();
    };

    const pointerUp = (e) => {
      if (!pointers.current.has(e.pointerId)) return;

      clearInterval(frameTimer.current);
      frameTimer.current = null;
      lastPointRef.current = null;

      completedPathsRef.current.push(activePathsRef.current.get(e.pointerId));
      pointers.current.delete(e.pointerId);
      activePathsRef.current.delete(e.pointerId);

      redraw();
    };

    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointermove", pointerMove);
    canvas.addEventListener("pointerup", pointerUp);

    return () => {
      canvas.removeEventListener("pointerdown", pointerDown);
      canvas.removeEventListener("pointermove", pointerMove);
      canvas.removeEventListener("pointerup", pointerUp);
    };
  }, [penColor]);

  const sendGesture = async () => {
    if (!gestureFrames.current.length) return;

    const frames = gestureFrames.current;
    const startTime = frames[0].ts;
    const endTime = frames.at(-1).ts;

    const payload = {
      start_time: startTime,
      end_time: endTime,
      duration_ms: endTime - startTime,
      frame_count: frames.length,
      frames,
    };

    setPayloadPreview(payload);
    console.log("Final Payload:", payload);

    try {
      await axios.post(DEFAULT_ENDPOINT, payload, {
        headers: { "Content-Type": "application/json" },
      });
      toast.success("Gesture sent successfully");
    } catch (err) {
      toast.error("Failed to send gesture");
    }

    gestureFrames.current = [];
  };

  return {
    payloadPreview,
    hoverPosition,
    sendGesture,
    clearCanves: clearCanvasState,
  };
}
