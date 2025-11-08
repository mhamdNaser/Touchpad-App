import { useEffect, useRef, useState } from "react";
import axios from "axios";
import simplify from "simplify-js";
import { toast } from "react-toastify";

export default function useTouchPredictor(canvasRef, previewCanvasRef, endpoint, character, penColor) {
  const pointers = useRef(new Map());
  const gestureFrames = useRef([]);
  const [payloadPreview, setPayloadPreview] = useState(null);
  const [hoverPosition, setHoverPosition] = useState(null);
  const [prediction, setPrediction] = useState(null);
  let nextId = useRef(1);

  const activePathsRef = useRef(new Map());
  const completedPathsRef = useRef([]);

  const clearCanvasState = () => {
    const canvas = canvasRef.current;
    const previewCanvas = previewCanvasRef.current;
    
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    if (previewCanvas) {
      const previewCtx = previewCanvas.getContext("2d");
      previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    }
    
    activePathsRef.current.clear();
    completedPathsRef.current = [];
    pointers.current.clear();
    gestureFrames.current = [];
    setPayloadPreview(null);
    setHoverPosition(null);
    setPrediction(null);
    nextId.current = 1;
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

      // رسم المسارات المكتملة
      for (const pts of completedPathsRef.current) {
        if (pts.length > 0) {
          const smooth = simplify(pts, 10, true);
          ctx.beginPath();
          ctx.moveTo(smooth[0].x, smooth[0].y);
          for (let i = 1; i < smooth.length; i++) ctx.lineTo(smooth[i].x, smooth[i].y);
          ctx.stroke();
        }
      }

      // رسم المسارات النشطة
      for (const [, pts] of activePathsRef.current.entries()) {
        if (pts.length > 0) {
          const smooth = simplify(pts, 4, true);
          ctx.beginPath();
          ctx.moveTo(smooth[0].x, smooth[0].y);
          for (let i = 1; i < smooth.length; i++) ctx.lineTo(smooth[i].x, smooth[i].y);
          ctx.stroke();
        }
      }
    };

    const recordFrame = () => {
      const pts = Array.from(pointers.current.values()).map((p) => ({
        id: p.id,
        x: parseFloat((p.x / canvas.width).toFixed(4)),
        y: parseFloat((p.y / canvas.height).toFixed(4)),
        state: p.state,
      }));

      const frame = {
        ts: Date.now(),
        frame_id: Date.now(),
        points: pts,
      };

      gestureFrames.current.push(frame);
      setPayloadPreview(frame);
    };

    const pointerDown = (e) => {
      e.preventDefault();
      e.target.setPointerCapture(e.pointerId);
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (canvas.width / rect.width);
      const y = (e.clientY - rect.top) * (canvas.height / rect.height);

      const id = nextId.current++;
      const point = { id, x, y, state: "down" };
      pointers.current.set(e.pointerId, point);
      activePathsRef.current.set(e.pointerId, [{ x, y }]);

      recordFrame();
      redraw();
    };

    const pointerMove = (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (canvas.width / rect.width);
      const y = (e.clientY - rect.top) * (canvas.height / rect.height);

      setHoverPosition({ x, y });

      if (!pointers.current.has(e.pointerId)) return;

      const p = pointers.current.get(e.pointerId);
      p.x = x;
      p.y = y;
      p.state = "move";

      const path = activePathsRef.current.get(e.pointerId);
      path.push({ x, y });

      redraw();
      recordFrame();
    };

    const pointerUp = async (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (canvas.width / rect.width);
      const y = (e.clientY - rect.top) * (canvas.height / rect.height);

      if (!pointers.current.has(e.pointerId)) return;

      const p = pointers.current.get(e.pointerId);
      p.x = x;
      p.y = y;
      p.state = "up";

      const finishedPath = activePathsRef.current.get(e.pointerId);
      if (finishedPath && finishedPath.length > 0) {
        completedPathsRef.current.push(finishedPath);
      }

      recordFrame();
      pointers.current.delete(e.pointerId);
      activePathsRef.current.delete(e.pointerId);
      redraw();

      // إرسال البيانات تلقائيًا بعد انتهاء الإيماءة
      await sendGesture();
    };

    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointermove", pointerMove);
    canvas.addEventListener("pointerup", pointerUp);
    canvas.addEventListener("pointercancel", pointerUp);
    
    canvas.addEventListener("pointerleave", (e) => {
      if (pointers.current.has(e.pointerId)) {
        pointerUp(e);
      }
      setHoverPosition(null);
    });

    return () => {
      canvas.removeEventListener("pointerdown", pointerDown);
      canvas.removeEventListener("pointermove", pointerMove);
      canvas.removeEventListener("pointerup", pointerUp);
      canvas.removeEventListener("pointercancel", pointerUp);
      canvas.removeEventListener("pointerleave", pointerUp);
    };
  }, [penColor]);

  const sendGesture = async () => {
    if (gestureFrames.current.length === 0) {
      console.log("No gesture data to send");
      return;
    }

    const framesToSend = [...gestureFrames.current];
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

    try {
      console.log("Sending gesture data:", payload);
      
      const res = await axios.post(endpoint, payload, {
        headers: { "Content-Type": "application/json" },
      });
      
      console.log("✅ Prediction Response:", res.data);
      setPrediction(res.data);
      
      // عرض النتائج في الكونسول
      if (res.data.predicted_class) {
        console.log("🎯 Predicted Class:", res.data.predicted_class);
      }
      if (res.data.probabilities) {
        console.log("📊 Probabilities:", res.data.probabilities);
      }
      
    } catch (err) {
      console.error("❌ Prediction failed:", err);
      if (err.response) {
        console.error("Response data:", err.response.data);
        console.error("Response status:", err.response.status);
      }
    }

    gestureFrames.current = [];
    const previewCanvas = previewCanvasRef.current;
    if (previewCanvas) {
      previewCanvas.getContext("2d").clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    }
  };

  return { 
    sendGesture, 
    clearCanvas: clearCanvasState, 
    payloadPreview, 
    hoverPosition,
    prediction 
  };
}