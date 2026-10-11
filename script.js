/*
 * script.js - Indian Currency Note Reader (browser side)
 * -------------------------------------------------------
 * 1. Load the TensorFlow.js model from ./model/model.json (relative path, so it
 *    works on Vercel, GitHub Pages or any static host).
 * 2. Grab frames from the webcam (or an uploaded image), resize to 224x224,
 *    run the model, and show the top class + confidence.
 * 3. Speak the result with the Web Speech API (speechSynthesis), but only
 *    when confidence >= 85 %. Otherwise say "Please show the note again".
 */

// ------------------------------------------------------------- settings ---
const MODEL_URL = "./model/model.json";          // relative path (Vercel-safe)
const CLASSES_URL = "./model/class_names.json";  // written by train.py
const IMG_SIZE = 224;                            // MobileNetV2 input size
const THRESHOLD = 0.85;                          // speak only above 85 %
const AUTO_INTERVAL_MS = 2000;                   // auto-read every 2 s
const REPEAT_COOLDOWN_MS = 4000;                 // do not repeat same note too often

// Human-friendly spoken names for each class
const SPOKEN = {
  "10": "Ten rupees",
  "20": "Twenty rupees",
  "50": "Fifty rupees",
  "100": "One hundred rupees",
  "200": "Two hundred rupees",
  "500": "Five hundred rupees",
  "2000": "Two thousand rupees",
  "no_note": "No note detected",
};
const LOW_CONF_MESSAGE = "Please show the note again";

// ------------------------------------------------------------- elements ---
const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const predictionEl = $("prediction");
const confidenceEl = $("confidence");
const barFill = $("bar-fill");
const video = $("video");
const preview = $("preview");
const canvas = $("canvas");
const fileInput = $("file-input");
const btnCamera = $("btn-camera");
const btnCapture = $("btn-capture");
const btnSpeak = $("btn-speak");
const autoToggle = $("auto-toggle");

let model = null;
let classNames = [];
let stream = null;
let autoTimer = null;
let lastSpoken = { text: "", time: 0 };
let lastResultText = "";

// --------------------------------------------------------------- speech ---
function speak(text, force = false) {
  if (!("speechSynthesis" in window)) return;           // browser lacks TTS
  const now = Date.now();
  // Avoid repeating the exact same sentence every 2 s - annoying for the user.
  if (!force && text === lastSpoken.text && now - lastSpoken.time < REPEAT_COOLDOWN_MS) return;
  window.speechSynthesis.cancel();                       // stop anything queued
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = "en-IN";
  utter.rate = 0.95;
  window.speechSynthesis.speak(utter);
  lastSpoken = { text, time: now };
}

// ---------------------------------------------------------- load model ---
async function loadModel() {
  try {
    statusEl.textContent = "Loading model…";
    const [m, c] = await Promise.all([
      tf.loadGraphModel(MODEL_URL).catch(() => tf.loadLayersModel(MODEL_URL)),
      fetch(CLASSES_URL).then((r) => r.json()),
    ]);
    model = m;
    classNames = c;
    // Warm up: first prediction is slow because of shader compilation.
    tf.tidy(() => model.predict(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3])));
    statusEl.textContent = "Model ready. Start the camera or upload a photo.";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Could not load the model: " + err.message;
  }
}

// -------------------------------------------------------------- predict ---
/** Run the model on an <img>, <video> or <canvas> element. */
async function predict(source) {
  if (!model) return null;
  const probs = tf.tidy(() => {
    // pixels 0-255 -> resize -> add batch dim. The model itself scales to [-1,1].
    const img = tf.browser.fromPixels(source)
      .resizeBilinear([IMG_SIZE, IMG_SIZE])
      .toFloat()
      .expandDims(0);
    return model.predict(img);
  });
  const data = await probs.data();
  probs.dispose();
  // Find the best class
  let best = 0;
  for (let i = 1; i < data.length; i++) if (data[i] > data[best]) best = i;
  return { label: classNames[best], confidence: data[best] };
}

/** Update the UI and speak, given a prediction. */
function showResult(res, forceSpeak = false) {
  if (!res) return;
  const pct = Math.round(res.confidence * 100);
  confidenceEl.textContent = `Confidence: ${pct} %`;
  barFill.style.width = pct + "%";

  if (res.confidence >= THRESHOLD) {
    const text = SPOKEN[res.label] || res.label;
    predictionEl.textContent = res.label === "no_note" ? text : `₹${res.label} — ${text}`;
    predictionEl.classList.remove("low");
    barFill.classList.add("ok");
    lastResultText = text;
    speak(text, forceSpeak);
  } else {
    predictionEl.textContent = LOW_CONF_MESSAGE;
    predictionEl.classList.add("low");
    barFill.classList.remove("ok");
    lastResultText = LOW_CONF_MESSAGE;
    speak(LOW_CONF_MESSAGE, forceSpeak);
  }
}

// --------------------------------------------------------------- camera ---
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = stream;
    video.hidden = false;
    preview.hidden = true;
    await video.play();
    btnCapture.disabled = false;
    btnCamera.textContent = "Stop camera";
    statusEl.textContent = "Camera on. Hold the note in front of the camera.";
    speak("Camera started. Show me a note.", true);
    startAuto();
  } catch (err) {
    statusEl.textContent = "Camera error: " + err.message + ". You can still upload a photo.";
  }
}

function stopCamera() {
  stopAuto();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  btnCapture.disabled = true;
  btnCamera.textContent = "Start camera";
  statusEl.textContent = "Camera stopped.";
}

async function readFromCamera(force = false) {
  if (!stream || video.readyState < 2) return;
  const res = await predict(video);
  showResult(res, force);
}

function startAuto() {
  stopAuto();
  if (autoToggle.checked && stream) autoTimer = setInterval(() => readFromCamera(false), AUTO_INTERVAL_MS);
}
function stopAuto() {
  if (autoTimer) clearInterval(autoTimer);
  autoTimer = null;
}

// --------------------------------------------------------------- upload ---
function handleFile(file) {
  if (!file) return;
  stopCamera();
  const url = URL.createObjectURL(file);
  preview.onload = async () => {
    const res = await predict(preview);
    showResult(res, true);
    URL.revokeObjectURL(url);
  };
  preview.src = url;
  preview.hidden = false;
  video.hidden = true;
  statusEl.textContent = "Reading uploaded photo…";
}

// --------------------------------------------------------------- events ---
btnCamera.addEventListener("click", () => (stream ? stopCamera() : startCamera()));
btnCapture.addEventListener("click", () => readFromCamera(true));
btnSpeak.addEventListener("click", () => lastResultText && speak(lastResultText, true));
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));
autoToggle.addEventListener("change", startAuto);

// Keyboard shortcut: Space = read now (handy for users who cannot see the buttons)
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target === document.body) {
    e.preventDefault();
    readFromCamera(true);
  }
});

loadModel();
