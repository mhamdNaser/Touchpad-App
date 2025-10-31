import React, { useRef, useState } from "react";
import CreatableSelect from "react-select/creatable";
import useTouchSimulator from "./hooks/useTouchSimulator";
import Button from "./Components/Button";
import Header from "./components/Header";
import {
  MdFullscreenExit,
  MdSave,
  MdCleaningServices,
  MdKeyboardArrowDown,
  MdKeyboardArrowUp,
} from "react-icons/md";

const defaultWidth = 1280;
const defaultHeight = 800;

export default function DrowerLayout() {
  const [endpoint, setEndpoint] = useState(
    localStorage.getItem("ENDPOINT") || ""
  );
  const [selectedChar, setSelectedChar] = useState(null);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [penColor, setPenColor] = useState("#000000");
  const [bgColor, setBgColor] = useState("#ffffff");
  const [isOpenOptional, setIsOpenOtional] = useState(false);

  const canvasRef = useRef();
  const previewCanvasRef = useRef();
  const containerRef = useRef();

  const { payloadPreview, hoverPosition, sendGesture, clearCanves } =
    useTouchSimulator(
      canvasRef,
      previewCanvasRef,
      endpoint,
      selectedChar,
      penColor
    );

  const arabicChars = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
    .split("")
    .map((ch) => ({ label: ch, value: ch }));

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

  const handelEndPoint = (e) => {
    setEndpoint(e);
    localStorage.setItem("ENDPOINT", e);
  };

  const handelPenColor = (e) => {
    setPenColor(e);
  };

  const canvasStyle = {
    width: isFullScreen ? "100%" : `${defaultWidth}px`,
    height: isFullScreen ? "100%" : `${defaultHeight}px`,
    minHeight: isFullScreen ? "100vh" : `${defaultHeight}px`,
  };

  return (
    <div className="bg-gradient-to-br from-indigo-50 to-white">
      <Header />
      <div
        ref={containerRef}
        className={`${
          isFullScreen
            ? "fixed inset-0 z-[100] flex items-center justify-center bg-gray-900 p-0 transition-all duration-300"
            : "flex flex-col xl:flex-row gap-8 p-6 items-start justify-center w-full"
        }`}
      >
        {!isFullScreen && (
          <div className="flex flex-col gap-6 w-full xl:w-[450px] bg-white/95 backdrop-blur-sm p-8 rounded-3xl shadow-2xl border border-indigo-100 transition-all duration-300">
            <div className="space-y-2">
              <h2 className="text-sm font-bold text-gray-800 border-b-2 border-indigo-200 pb-1">
                Required Fields
              </h2>

              <div className="flex flex-col sm:flex-row gap-4">
                <input
                  type="text"
                  placeholder="Insert your endpoint URL"
                  value={endpoint}
                  onChange={(e) => handelEndPoint(e.target.value)}
                  className="flex-1 p-3 rounded-xl text-xs border-2 border-gray-300 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100 outline-none transition-all text-gray-800 placeholder-gray-500 shadow-inner"
                />

                <CreatableSelect
                  isClearable
                  placeholder="Select or type a character..."
                  options={arabicChars}
                  value={selectedChar}
                  onChange={setSelectedChar}
                  classNamePrefix="react-select"
                  styles={{
                    control: (base, state) => ({
                      ...base,
                      fontSize: "12px",
                      borderRadius: "0.75rem",
                      borderColor: state.isFocused ? "#6366f1" : "#d1d5db",
                      boxShadow: state.isFocused ? "0 0 0 4px #e0e7ff" : "none",
                      minWidth: "150px",
                      height: "48px",
                      "&:hover": { borderColor: "#6366f1" },
                    }),
                  }}
                />
              </div>

              {/* Optional Fields Toggle Header */}
              <h2 className="flex justify-between pe-2 items-center text-sm font-bold text-gray-800 border-b-2 border-indigo-200 pb-1 mt-3">
                Optional Settings
                <button onClick={() => setIsOpenOtional(!isOpenOptional)}>
                  {isOpenOptional ? (
                    <MdKeyboardArrowUp size={24} />
                  ) : (
                    <MdKeyboardArrowDown size={24} />
                  )}
                </button>
              </h2>

              {isOpenOptional && (
                <div className="flex flex-col gap-6 pt-2 animate-in fade-in slide-in-from-top-1">
                  <div className="flex flex-col sm:flex-row gap-6">
                    <div className="flex flex-col w-full">
                      <label className="text-gray-700 font-semibold mb-2 text-sm">
                        Pen Color ({penColor})
                      </label>
                      <input
                        type="color"
                        value={penColor}
                        onChange={(e) => handelPenColor(e.target.value)}
                        className="h-8 w-full rounded-xl border border-gray-300 cursor-pointer shadow-md transition-all duration-300 transform hover:scale-[1.02]"
                      />
                    </div>
                    <div className="flex flex-col w-full">
                      <label className="text-gray-700 font-semibold mb-2  text-sm">
                        Background Color ({bgColor})
                      </label>
                      <input
                        type="color"
                        value={bgColor}
                        onChange={(e) => setBgColor(e.target.value)}
                        className="w-full h-8 rounded-xl border border-gray-300 cursor-pointer shadow-md transition-all duration-300 transform hover:scale-[1.02]"
                      />
                    </div>
                  </div>

                  <h2 className="text-sm font-bold text-gray-800 border-b-2 border-indigo-200 pb-1">
                    Action Buttons
                  </h2>

                  <div className="flex flex-col sm:flex-row gap-4">
                    <button
                      className="bg-indigo-600 py-2 flex justify-center items-center space-x-2 w-full rounded-xl text-white hover:bg-indigo-700 shadow-lg transition-all duration-300"
                      onClick={sendGesture}
                    >
                      <span>Send & Save</span> <MdSave size={18} />
                    </button>
                    <button
                      className="bg-red-500 py-2 flex justify-center items-center space-x-2 w-full rounded-xl text-white hover:bg-red-600 shadow-lg transition-all duration-300"
                      onClick={sendGesture}
                    >
                      <span>Clear Canvas</span> <MdCleaningServices size={18} />
                    </button>
                  </div>

                  <div className="bg-gray-800 text-green-400 text-sm p-4 rounded-xl w-full max-h-[100px] overflow-auto font-mono shadow-inner border border-gray-700">
                    <h3 className="text-yellow-300 font-semibold mb-1 text-base">
                      Live Coordinates
                    </h3>
                    {hoverPosition ? (
                      <pre className="whitespace-pre-wrap">{`x: ${hoverPosition.x}, y: ${hoverPosition.y}`}</pre>
                    ) : (
                      <p className="text-gray-500 italic">
                        Move the mouse over the board to see coordinates...
                      </p>
                    )}
                  </div>

                  <div className="bg-gray-800 text-green-400 text-sm p-4 rounded-xl w-full max-h-[240px] overflow-auto font-mono shadow-inner border border-gray-700">
                    <h3 className="text-yellow-300 font-semibold mb-2 text-base">
                      Collected Gesture Data
                    </h3>
                    {payloadPreview ? (
                      <pre className="whitespace-pre-wrap">
                        {JSON.stringify(payloadPreview, null, 2)}
                      </pre>
                    ) : (
                      <p className="text-gray-500 italic">
                        No gesture data has been recorded yet.
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div
          className={`${
            isFullScreen
              ? "w-full h-full p-0 transition-all duration-300 relative"
              : "w-full flex justify-center relative min-w-[300px] xl:max-w-4xl"
          }`}
        >
          <div
            className={`${
              isFullScreen
                ? "w-full h-full bg-white shadow-none rounded-none"
                : "bg-white border border-gray-300 rounded-3xl shadow-2xl overflow-hidden max-w-full"
            } relative`}
          >
            {!isFullScreen && (
              <button
                onClick={toggleFullScreen}
                className="absolute top-4 right-4 z-10 p-3 bg-gray-800 text-white rounded-full shadow-xl hover:bg-gray-700 transition duration-300 focus:outline-none focus:ring-4 focus:ring-indigo-500/50"
                title="Enter Fullscreen"
              >
                <MdFullscreenExit size={24} className="rotate-45" />
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
              className="rounded-2xl bg-gray-50 transition-all duration-500"
            />

            {isFullScreen && (
              <div className="absolute top-4 right-4 z-10 flex gap-4">
                <Button
                  onClickFun={toggleFullScreen}
                  title={"Exit"}
                  color={"bg-red-600/90 backdrop-blur-md"}
                  textColor={"white"}
                  hoverStyles={"hover:bg-red-700"}
                  styles="text-lg px-6 py-3 font-bold rounded-xl shadow-xl transition-all duration-300"
                />
                <Button
                  onClickFun={sendGesture}
                  title={"Save"}
                  color={"bg-indigo-600/90 backdrop-blur-md"}
                  textColor={"white"}
                  hoverStyles={"hover:bg-indigo-700"}
                  styles="text-lg px-6 py-3 font-bold rounded-xl shadow-xl transition-all duration-300"
                />
              </div>
            )}
          </div>
          <canvas ref={previewCanvasRef} className="hidden" />
        </div>
      </div>
    </div>
  );
}
