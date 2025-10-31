// import { app, BrowserWindow } from "electron";
// import path from "path";
// import { fileURLToPath } from "url";

// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);

// function createWindow() {
//   const win = new BrowserWindow({
//     width: 1280,
//     height: 800,
//     webPreferences: {
//       preload: path.join(__dirname, "preload.js"),
//     },
//   });

//   if (process.env.VITE_DEV_SERVER_URL) {
//     win.loadURL(process.env.VITE_DEV_SERVER_URL);
//   } else {
//     win.loadFile(path.join(__dirname, "../dist/index.html"));
//   }
// }

// app.whenReady().then(() => {
//   createWindow();
// });
const { app, BrowserWindow } = require("electron");
const path = require("path");

const isDev = process.env.NODE_ENV === "development";

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
    },
  });

  if (isDev) {
    // أثناء التطوير: نعرض سيرفر Vite
    mainWindow.loadURL("http://localhost:5173");
  } else {
    // بعد البناء: نعرض ملفات React الجاهزة
    mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
  }
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

