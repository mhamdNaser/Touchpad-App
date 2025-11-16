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

  const SAMPLING_RATE = 10;
  const frameTimer = useRef(null);

  // ==================== createFixedFrame محسنة ====================
  const createFixedFrame = (activePoints, maxPoints = 20) => {
    const ts = Date.now();
    const prevFrame = gestureFrames.current[gestureFrames.current.length - 1];

    const delta_ms = prevFrame ? ts - prevFrame.ts : 1; // لتجنب الصفر

    const points = Array.from({ length: maxPoints }, (_, i) => {
      const p = activePoints[i];
      let dx = 0, dy = 0, vx = 0, vy = 0, angle = 0;

      if (p && prevFrame) {
        const prevPoint = prevFrame.points[i] || { x: 0, y: 0 };
        dx = p.x - prevPoint.x;
        dy = p.y - prevPoint.y;
        vx = dx / delta_ms;
        vy = dy / delta_ms;
        angle = Math.atan2(dy, dx);
      }

      return p
        ? {
            id: p.id,
            x: parseFloat(p.x.toFixed(4)),
            y: parseFloat(p.y.toFixed(4)),
            state: p.state || "move",
            pressure: p.pressure || 1.0,
            dx: parseFloat(dx.toFixed(4)),
            dy: parseFloat(dy.toFixed(4)),
            vx: parseFloat(vx.toFixed(4)),
            vy: parseFloat(vy.toFixed(4)),
            angle: parseFloat(angle.toFixed(4)),
          }
        : {
            id: i + 1,
            x: 0,
            y: 0,
            state: "none",
            pressure: 0,
            dx: 0,
            dy: 0,
            vx: 0,
            vy: 0,
            angle: 0,
          };
    });

    return { ts, delta_ms, frame_id: ts, points };
  };

  // ==================== مسح حالة اللوحة ====================
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
      pointers.current.set(e.pointerId, {
        id,
        x: x / canvas.width,
        y: y / canvas.height,
        state: "down",
      });

      activePathsRef.current.set(e.pointerId, [{ x, y, state: "down", id }]);

      frameTimer.current = setInterval(() => {
        if (pointers.current.size === 0) return;

        const activePoints = Array.from(pointers.current.values());
        const frame = createFixedFrame(activePoints);
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

      const p = pointers.current.get(e.pointerId);
      p.x = x / canvas.width;
      p.y = y / canvas.height;
      p.state = "move";

      const path = activePathsRef.current.get(e.pointerId);
      path.push({ x, y, state: "move", id: path.length + 1 });

      redraw();
    };

    const pointerUp = (e) => {
      if (!pointers.current.has(e.pointerId)) return;

      pointers.current.delete(e.pointerId);
      completedPathsRef.current.push(activePathsRef.current.get(e.pointerId));
      activePathsRef.current.delete(e.pointerId);

      if (pointers.current.size === 0) {
        clearInterval(frameTimer.current);
        frameTimer.current = null;
      }

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
