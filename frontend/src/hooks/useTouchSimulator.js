import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { toast } from "react-toastify";
import simplify from "simplify-js";

export default function useTouchSimulator(
  canvasRef,
  previewCanvasRef,
  endpoint,
  selectedChar,
  penColor
) {
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
    nextId.current = 1; // 💡 ربما تريد إعادة تعيين الـ id أيضاً
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const color = penColor
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

      // نرسم المسارات النشطة
      for (const [, pts] of activePathsRef.current.entries()) {
        const smooth = simplify(pts, 4, true);
        ctx.beginPath();
        ctx.moveTo(smooth[0].x, smooth[0].y);
        for (let i = 1; i < smooth.length; i++) ctx.lineTo(smooth[i].x, smooth[i].y);
        ctx.stroke();
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
      if (!selectedChar) {
        toast.warn("يرجى اختيار الحرف أولاً ⚠️");
        return;
      }

      e.target.setPointerCapture(e.pointerId);
      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      const id = nextId.current++;
      const point = { id, x, y, state: "down" };
      pointers.current.set(e.pointerId, point);
      activePathsRef.current.set(e.pointerId, [{ x, y }]);

      recordFrame();
      redraw();
    };

    const pointerMove = (e) => {
      const canvas = canvasRef.current;
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
      recordFrame();
    };

    const pointerUp = (e) => {
      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

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
  }, [endpoint, selectedChar]);



  const sendGesture = async () => {
    if (!selectedChar || gestureFrames.current.length === 0) return;

    // استخراج الإطارات الحقيقية التي تم جمعها
    const framesToSend = gestureFrames.current;

    // التأكد من وجود إطارات
    if (framesToSend.length === 0) {
      console.log("No frames to send.");
      return;
    }

    const startTime = framesToSend[0].ts;
    const endTime = framesToSend.at(-1).ts;

    const payload = {
      character: selectedChar?.value || "",
      start_time: startTime,
      end_time: endTime,
      duration_ms: endTime - startTime, // مدة زمنية حقيقية
      frame_count: framesToSend.length, // عدد إطارات حقيقي
      frames: framesToSend, // إرسال الإطارات الأصلية
    };

    // هذا سيعرض البيانات الحقيقية في واجهتك
    setPayloadPreview(payload);

    try {
      const res = await axios.post(endpoint, payload, {
        headers: { "Content-Type": "application/json" },
      });
      toast.success(res?.data?.message || "Gesture saved successfully");
      console.log("✅ Gesture sent successfully:", res.data);
    } catch (err) {
      toast.error(
        err.response?.data?.message || "Failed to send gesture to the server."
      );
      console.error("❌ Send gesture failed:", err);
    }

    // تفريغ الإطارات بعد الإرسال
    gestureFrames.current = [];
    const previewCanvas = previewCanvasRef.current;
    if (previewCanvas)
      previewCanvas.getContext("2d").clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  };


  const handleClearCanvas = async () => {
    clearCanvasState();
    toast.info("تم مسح اللوحة وإعادة تصفير البيانات بنجاح.");
  }

  return { payloadPreview, hoverPosition, sendGesture, clearCanves: handleClearCanvas };
}