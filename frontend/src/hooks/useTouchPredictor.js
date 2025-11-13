import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { toast } from "react-toastify";
import simplify from "simplify-js";

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
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const color = penColor;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    canvas.style.touchAction = "none";

    ctx.lineWidth = 6;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = color;

    const redraw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const pts of completedPathsRef.current) {
        const smooth = simplify(pts, 10, true);
        ctx.beginPath();
        ctx.moveTo(smooth[0].x, smooth[0].y);
        for (let i = 1; i < smooth.length; i++) ctx.lineTo(smooth[i].x, smooth[i].y);
        ctx.stroke();
      }

      for (const [, pts] of activePathsRef.current.entries()) {
        const smooth = simplify(pts, 4, true);
        ctx.beginPath();
        ctx.moveTo(smooth[0].x, smooth[0].y);
        for (let i = 1; i < smooth.length; i++) ctx.lineTo(smooth[i].x, smooth[i].y);
        ctx.stroke();
      }
    };

    // نفس منطق الهوك الثاني: تحويل المسار إلى frame كامل
    const createFrameFromPath = (path) => {
      if (!path || path.length === 0) return null;
      const canvas = canvasRef.current;
      const points = path.map((p, idx) => ({
        id: idx + 1,
        x: parseFloat((p.x / canvas.width).toFixed(4)),
        y: parseFloat((p.y / canvas.height).toFixed(4)),
        state: idx === 0 ? "down" : idx === path.length - 1 ? "up" : "move",
      }));

      const ts = Date.now();
      return { ts, frame_id: ts, points };
    };

    const pointerDown = (e) => {

      e.target.setPointerCapture(e.pointerId);
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      const id = nextId.current++;
      const point = { id, x, y, state: "down" };
      pointers.current.set(e.pointerId, point);
      activePathsRef.current.set(e.pointerId, [{ x, y }]);

      redraw();
    };

    const pointerMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      setHoverPosition({ x, y });

      if (!pointers.current.has(e.pointerId)) return;

      const p = pointers.current.get(e.pointerId);
      p.x = x;
      p.y = y;
      p.state = "move";

      const path = activePathsRef.current.get(e.pointerId);
      path.push({ x, y });

      redraw();
    };

    const pointerUp = (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      if (!pointers.current.has(e.pointerId)) return;

      const path = activePathsRef.current.get(e.pointerId);
      if (path && path.length > 0) {
        completedPathsRef.current.push(path);
        const frame = createFrameFromPath(path);
        if (frame) gestureFrames.current.push(frame);
        setPayloadPreview({ frames: gestureFrames.current });
      }

      pointers.current.delete(e.pointerId);
      activePathsRef.current.delete(e.pointerId);
      redraw();
    };

    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointermove", pointerMove);
    canvas.addEventListener("pointerup", pointerUp);
    canvas.addEventListener("pointerleave", () => {
      pointers.current.clear();
      activePathsRef.current.clear();
      setHoverPosition(null);
      redraw();
    });

    return () => {
      canvas.removeEventListener("pointerdown", pointerDown);
      canvas.removeEventListener("pointermove", pointerMove);
      canvas.removeEventListener("pointerup", pointerUp);
    };
  }, []);

  const sendGesture = async () => {

    const framesToSend = gestureFrames.current;
    const startTime = framesToSend[0].ts;
    const endTime = framesToSend.at(-1).ts;

    const payload = {
      start_time: startTime,
      end_time: endTime,
      duration_ms: endTime - startTime,
      frame_count: framesToSend.length,
      frames: framesToSend,
    };

    setPayloadPreview(payload);

    console.log(payload);
    

    try {
      const res = await axios.post(DEFAULT_ENDPOINT, payload, {
        headers: { "Content-Type": "application/json" },
      });
      toast.success(res?.data?.message || "Gesture sent successfully");
      console.log("✅ Gesture sent successfully:", res.data);
    } catch (err) {
      toast.error(
        err.response?.data?.message || "Failed to send gesture to the server."
      );
      console.error("❌ Send gesture failed:", err);
    }

    gestureFrames.current = [];
    const previewCanvas = previewCanvasRef.current;
    if (previewCanvas)
      previewCanvas.getContext("2d").clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  };

  const handleClearCanvas = async () => {
    clearCanvasState();
    toast.info("تم مسح اللوحة وإعادة تصفير البيانات بنجاح.");
  };

  return { payloadPreview, hoverPosition, sendGesture, clearCanves: handleClearCanvas };
}
