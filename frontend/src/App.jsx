import React, { useRef, useState } from "react";
import useTouchPredictor from "./hooks/useTouchPredictor"; // تأكد من المسار الصحيح
import Button from "./Components/Button";
import Header from "./components/Header";
import {
  MdFullscreenExit,
  MdSave,
  MdCleaningServices,
  MdPalette,
  MdFormatColorFill,
} from "react-icons/md";

const defaultWidth = 1280;
const defaultHeight = 800;

export default function DrowerLayout() {
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [penColor, setPenColor] = useState("#000000");
  const [bgColor, setBgColor] = useState("#ffffff");

  const canvasRef = useRef();
  const previewCanvasRef = useRef();
  const containerRef = useRef();

  const { sendGesture, clearCanves, prediction } = useTouchPredictor(
    canvasRef,
    previewCanvasRef,
    null, // لا حرف محدد - رسم حر
    penColor
  );

  const toggleFullScreen = () => {
    const elem = containerRef.current;
    if (!document.fullscreenElement) {
      elem.requestFullscreen?.();
      setIsFullScreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullScreen(false);
    }
  };

  const canvasStyle = {
    width: isFullScreen ? "100%" : `${defaultWidth}px`,
    height: isFullScreen ? "100%" : `${defaultHeight}px`,
    minHeight: isFullScreen ? "100vh" : `${defaultHeight}px`,
  };

  // دالة مسح محسنة
  const handleClearCanvas = () => {
    clearCanves();
    console.log("🧹 Canvas cleared");
  };

  return (
    <div className="bg-gradient-to-br from-indigo-50 to-white min-h-screen">
      <Header />
      <div
        ref={containerRef}
        className={`${
          isFullScreen
            ? "fixed inset-0 z-[100] flex items-center justify-center bg-gray-900 p-0 transition-all duration-300"
            : "flex flex-col lg:flex-row gap-4 p-4 items-start justify-center w-full"
        }`}
      >
        {/* Thin Sidebar */}
        {!isFullScreen && (
          <div className="flex flex-col gap-4 w-full lg:w-[80px] bg-white/95 backdrop-blur-sm p-4 rounded-2xl shadow-xl border border-indigo-100 transition-all duration-300">
            
            {/* Pen Color Section */}
            <div className="space-y-3 text-center">
              <div className="flex flex-col items-center gap-2">
                <MdPalette className="text-indigo-500 text-xl" />
                <span className="text-xs font-semibold text-gray-700">القلم</span>
              </div>
              
              <input
                type="color"
                value={penColor}
                onChange={(e) => setPenColor(e.target.value)}
                className="w-8 h-8 border-2 border-gray-300 cursor-pointer shadow-lg hover:scale-105 transition-transform"
                title="اختر لون القلم"
              />
            </div>

            {/* Background Color Section */}
            <div className="space-y-3 text-center">
              <div className="flex flex-col items-center gap-2">
                <MdFormatColorFill className="text-indigo-500 text-xl" />
                <span className="text-xs font-semibold text-gray-700">الخلفية</span>
              </div>
              
              <input
                type="color"
                value={bgColor}
                onChange={(e) => setBgColor(e.target.value)}
                className="w-8 h-8 border-2 border-gray-300 cursor-pointer shadow-lg hover:scale-105 transition-transform"
                title="اختر لون الخلفية"
              />
            </div>

            {/* Divider */}
            <div className="border-t border-gray-200 my-2"></div>

            {/* Action Buttons */}
            <div className="space-y-3 mx-auto">
              <button
                className="bg-indigo-600 p-2 flex items-center gap-1 rounded-xl text-white hover:bg-indigo-700 shadow-lg transition-all duration-300"
                onClick={sendGesture}
                title="حفظ الرسم"
              >
                <MdSave size={18} />
              </button>
              
              <button
                className="bg-gray-600 p-2 flex items-center gap-1 rounded-xl text-white hover:bg-gray-700 shadow-lg transition-all duration-300"
                onClick={handleClearCanvas} // استخدام الدالة المحسنة
                title="مسح اللوحة"
              >
                <MdCleaningServices size={18} />
              </button>

              <button
                className="bg-green-600 p-2 flex items-center gap-1 rounded-xl text-white hover:bg-green-700 shadow-lg transition-all duration-300"
                onClick={toggleFullScreen}
                title="ملء الشاشة"
              >
                <MdFullscreenExit size={18} />
              </button>
            </div>
          </div>
        )}

        {/* Canvas Area */}
        <div
          className={`${
            isFullScreen
              ? "w-full h-full p-0 transition-all duration-300 relative"
              : "w-full flex justify-center relative min-w-[300px] lg:flex-1"
          }`}
        >
          <div
            className={`${
              isFullScreen
                ? "w-full h-full bg-white shadow-none rounded-none"
                : "bg-white border border-gray-300 rounded-2xl shadow-xl overflow-hidden max-w-full"
            } relative`}
          >
            {!isFullScreen && (
              <button
                onClick={toggleFullScreen}
                className="absolute top-3 right-3 z-10 p-2 bg-gray-800 text-white rounded-full shadow-lg hover:bg-gray-700 transition duration-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                title="الدخول إلى وضع الملء الشاشة"
              >
                <MdFullscreenExit size={20} className="rotate-45" />
              </button>
            )}

            <canvas
              ref={canvasRef}
              width={defaultWidth}
              height={defaultHeight}
              style={{
                ...canvasStyle,
                backgroundColor: bgColor,
                transition: "background-color 0.3s ease",
              }}
              className="rounded-xl bg-gray-50 transition-all duration-500 cursor-crosshair"
            />

            {isFullScreen && (
              <div className="absolute top-4 right-4 z-10 flex gap-3">
                <Button
                  onClickFun={toggleFullScreen}
                  title={"خروج"}
                  color={"bg-red-600/90 backdrop-blur-md"}
                  textColor={"white"}
                  hoverStyles={"hover:bg-red-700"}
                  styles="text-sm px-4 py-2 font-semibold rounded-lg shadow-lg transition-all duration-300"
                />
                <Button
                  onClickFun={sendGesture}
                  title={"حفظ"}
                  color={"bg-indigo-600/90 backdrop-blur-md"}
                  textColor={"white"}
                  hoverStyles={"hover:bg-indigo-700"}
                  styles="text-sm px-4 py-2 font-semibold rounded-lg shadow-lg transition-all duration-300"
                />
                <Button
                  onClickFun={handleClearCanvas}
                  title={"مسح"}
                  color={"bg-gray-600/90 backdrop-blur-md"}
                  textColor={"white"}
                  hoverStyles={"hover:bg-gray-700"}
                  styles="text-sm px-4 py-2 font-semibold rounded-lg shadow-lg transition-all duration-300"
                />
              </div>
            )}
          </div>
          <canvas ref={previewCanvasRef} className="hidden" />
        </div>
      </div>

      {/* عرض النتائج في الكونسول تلقائياً */}
      {prediction && (
        <script dangerouslySetInnerHTML={{
          __html: `
            console.log('🎯 Latest Prediction:', ${JSON.stringify(prediction)});
            console.log('📊 Predicted Class:', '${prediction.predicted_class}');
            console.log('📈 Probabilities:', ${JSON.stringify(prediction.probabilities)});
          `
        }} />
      )}
    </div>
  );
}