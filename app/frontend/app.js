/**
 * Frontend Application Logic: Human-AI Chess Move Detection
 * ==========================================================
 * Provides interactive chessboard visualization, zero-leakage move scrubbing,
 * dynamic behavioral meters, real-time Chart.js telemetry, and offline demo games.
 */

// SVG Chess Piece Definitions
const CHESS_PIECES_SVG = {
  P: `<svg viewBox="0 0 45 45" class="piece-svg"><path d="m 22.5,9 c -2.21,0 -4,1.79 -4,4 0,0.89 0.29,1.71 0.78,2.38 C 17.33,16.5 16,18.59 16,21 c 0,2.03 0.94,3.84 2.41,5.03 C 15.41,27.09 11,31.58 11,39.5 H 34 C 34,31.58 29.59,27.09 26.59,26.03 28.06,24.84 29,23.03 29,21 29,18.59 27.67,16.5 25.72,15.38 26.21,14.71 26.5,13.89 26.5,13 c 0,-2.21 -1.79,-4 -4,-4 z" style="fill:#ffffff; stroke:#000000; stroke-width:1.5; stroke-linejoin:round;"/></svg>`,
  N: `<svg viewBox="0 0 45 45" class="piece-svg"><path d="m 22,10 c 10.5,1 16.5,8 16,29 H 15 c 0,-9 10,-6.5 8,-21 z" style="fill:#ffffff; stroke:#000000; stroke-width:1.5;"/><path d="m 24,18 c 0.38,2.91 -5.55,7.37 -8,9 -3,2 -2.82,4.34 -5,4 -1.042,-0.94 1.41,-3.04 0,-3 -1,0 0.19,1.23 -1,2 -1,0 -4.003,1 -4,-4 0,-2 6,-12 6,-12 0,0 1.89,-1.9 2,-3.5 -0.73,-0.994 -0.5,-2 -0.5,-3 1,-1 3,2.5 3,2.5 z" style="fill:#ffffff; stroke:#000000; stroke-width:1.5;"/></svg>`,
  B: `<svg viewBox="0 0 45 45" class="piece-svg"><g style="fill:#ffffff; stroke:#000000; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round;"><path d="M 9,36 C 12.39,35.03 19.11,36.43 22.5,34 C 25.89,36.43 32.61,35.03 36,36 C 36,36 37.65,36.54 39,38 C 38.32,38.97 37.35,38.99 36,38.5 C 32.61,37.53 25.89,38.96 22.5,37.5 C 19.11,38.96 12.39,37.53 9,38.5 C 7.646,38.99 6.677,38.97 6,38 C 7.354,36.54 9,36 9,36 z"/><path d="M 15,32 C 17.5,34.5 27.5,34.5 30,32 C 30.5,30.5 30,30 30,30 C 30,27.5 27.5,26 27.5,26 C 33,24.5 33.5,14.5 22.5,10.5 C 11.5,14.5 12,24.5 17.5,26 C 17.5,26 15,27.5 15,30 C 15,30 14.5,30.5 15,32 z"/><path d="m 25 8 a 2.5 2.5 0 1 1 -5 0 a 2.5 2.5 0 1 1 5 0 z"/></g></svg>`,
  R: `<svg viewBox="0 0 45 45" class="piece-svg"><g style="fill:#ffffff; stroke:#000000; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round;"><path d="M 9,39 L 36,39 L 36,36 L 9,36 z"/><path d="M 12,36 L 12,32 L 33,32 L 33,36 z"/><path d="M 11,14 L 11,9 L 15,9 L 15,11 L 20,11 L 20,9 L 25,9 L 25,11 L 30,11 L 30,9 L 34,9 L 34,14"/><path d="M 34,14 L 31,17 L 14,17 L 11,14"/><path d="M 14,17 L 14,29.5 L 31,29.5 L 31,17"/></g></svg>`,
  Q: `<svg viewBox="0 0 45 45" class="piece-svg"><g style="fill:#ffffff; stroke:#000000; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round;"><path d="M 9 26 C 17.5 24.5 30 24.5 36 26 L 38 14 L 31 25 L 22.5 10 L 14 25 L 7 14 L 9 26 z"/><path d="M 9,26 C 9,28 10.5,28 11.5,30 C 12.5,31.5 12.5,31 12,33.5 C 10.5,34.5 10.5,36 10.5,36 C 9,37.5 11,38.5 11,38.5 L 34,38.5 C 34,38.5 36,37.5 34.5,36 C 34.5,36 34.5,34.5 33,33.5 C 32.5,31 32.5,31.5 33.5,30 C 34.5,28 36,28 36,26"/><circle cx="6" cy="12" r="2"/><circle cx="14" cy="9" r="2"/><circle cx="22.5" cy="8" r="2"/><circle cx="31" cy="9" r="2"/><circle cx="39" cy="12" r="2"/></g></svg>`,
  K: `<svg viewBox="0 0 45 45" class="piece-svg"><g style="fill:#ffffff; stroke:#000000; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round;"><path d="M 22.5,11.63 L 22.5,6"/><path d="M 20,8 L 25,8"/><path d="M 22.5,25 C 22.5,25 27,17.5 25.5,14.5 C 24,11.5 21,11.5 22.5,9 C 24,11.5 21,11.5 19.5,14.5 C 18,17.5 22.5,25 22.5,25"/><path d="M 11.5,37 C 17,40.5 28,40.5 33.5,37 C 36.5,34 36.5,30 33.5,27 C 28,24 17,24 11.5,27 C 8.5,30 8.5,34 11.5,37 z"/><path d="M 12.5,30 C 17,27 28,27 32.5,30"/><path d="M 11.5,33.5 C 17,30.5 28,30.5 33.5,33.5"/><path d="M 11.5,37 C 17,34 28,34 33.5,37"/></g></svg>`,
  
  // Black pieces (filled with dark slate)
  p: `<svg viewBox="0 0 45 45" class="piece-svg"><path d="m 22.5,9 c -2.21,0 -4,1.79 -4,4 0,0.89 0.29,1.71 0.78,2.38 C 17.33,16.5 16,18.59 16,21 c 0,2.03 0.94,3.84 2.41,5.03 C 15.41,27.09 11,31.58 11,39.5 H 34 C 34,31.58 29.59,27.09 26.59,26.03 28.06,24.84 29,23.03 29,21 29,18.59 27.67,16.5 25.72,15.38 26.21,14.71 26.5,13.89 26.5,13 c 0,-2.21 -1.79,-4 -4,-4 z" style="fill:#1e293b; stroke:#94a3b8; stroke-width:1.5; stroke-linejoin:round;"/></svg>`,
  n: `<svg viewBox="0 0 45 45" class="piece-svg"><path d="m 22,10 c 10.5,1 16.5,8 16,29 H 15 c 0,-9 10,-6.5 8,-21 z" style="fill:#1e293b; stroke:#94a3b8; stroke-width:1.5;"/><path d="m 24,18 c 0.38,2.91 -5.55,7.37 -8,9 -3,2 -2.82,4.34 -5,4 -1.042,-0.94 1.41,-3.04 0,-3 -1,0 0.19,1.23 -1,2 -1,0 -4.003,1 -4,-4 0,-2 6,-12 6,-12 0,0 1.89,-1.9 2,-3.5 -0.73,-0.994 -0.5,-2 -0.5,-3 1,-1 3,2.5 3,2.5 z" style="fill:#1e293b; stroke:#94a3b8; stroke-width:1.5;"/></svg>`,
  b: `<svg viewBox="0 0 45 45" class="piece-svg"><g style="fill:#1e293b; stroke:#94a3b8; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round;"><path d="M 9,36 C 12.39,35.03 19.11,36.43 22.5,34 C 25.89,36.43 32.61,35.03 36,36 C 36,36 37.65,36.54 39,38 C 38.32,38.97 37.35,38.99 36,38.5 C 32.61,37.53 25.89,38.96 22.5,37.5 C 19.11,38.96 12.39,37.53 9,38.5 C 7.646,38.99 6.677,38.97 6,38 C 7.354,36.54 9,36 9,36 z"/><path d="M 15,32 C 17.5,34.5 27.5,34.5 30,32 C 30.5,30.5 30,30 30,30 C 30,27.5 27.5,26 27.5,26 C 33,24.5 33.5,14.5 22.5,10.5 C 11.5,14.5 12,24.5 17.5,26 C 17.5,26 15,27.5 15,30 C 15,30 14.5,30.5 15,32 z"/><path d="m 25 8 a 2.5 2.5 0 1 1 -5 0 a 2.5 2.5 0 1 1 5 0 z"/></g></svg>`,
  r: `<svg viewBox="0 0 45 45" class="piece-svg"><g style="fill:#1e293b; stroke:#94a3b8; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round;"><path d="M 9,39 L 36,39 L 36,36 L 9,36 z"/><path d="M 12,36 L 12,32 L 33,32 L 33,36 z"/><path d="M 11,14 L 11,9 L 15,9 L 15,11 L 20,11 L 20,9 L 25,9 L 25,11 L 30,11 L 30,9 L 34,9 L 34,14"/><path d="M 34,14 L 31,17 L 14,17 L 11,14"/><path d="M 14,17 L 14,29.5 L 31,29.5 L 31,17"/></g></svg>`,
  q: `<svg viewBox="0 0 45 45" class="piece-svg"><g style="fill:#1e293b; stroke:#94a3b8; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round;"><path d="M 9 26 C 17.5 24.5 30 24.5 36 26 L 38 14 L 31 25 L 22.5 10 L 14 25 L 7 14 L 9 26 z"/><path d="M 9,26 C 9,28 10.5,28 11.5,30 C 12.5,31.5 12.5,31 12,33.5 C 10.5,34.5 10.5,36 10.5,36 C 9,37.5 11,38.5 11,38.5 L 34,38.5 C 34,38.5 36,37.5 34.5,36 C 34.5,36 34.5,34.5 33,33.5 C 32.5,31 32.5,31.5 33.5,30 C 34.5,28 36,28 36,26"/><circle cx="6" cy="12" r="2"/><circle cx="14" cy="9" r="2"/><circle cx="22.5" cy="8" r="2"/><circle cx="31" cy="9" r="2"/><circle cx="39" cy="12" r="2"/></g></svg>`,
  k: `<svg viewBox="0 0 45 45" class="piece-svg"><g style="fill:#1e293b; stroke:#94a3b8; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round;"><path d="M 22.5,11.63 L 22.5,6"/><path d="M 20,8 L 25,8"/><path d="M 22.5,25 C 22.5,25 27,17.5 25.5,14.5 C 24,11.5 21,11.5 22.5,9 C 24,11.5 21,11.5 19.5,14.5 C 18,17.5 22.5,25 22.5,25"/><path d="M 11.5,37 C 17,40.5 28,40.5 33.5,37 C 36.5,34 36.5,30 33.5,27 C 28,24 17,24 11.5,27 C 8.5,30 8.5,34 11.5,37 z"/><path d="M 12.5,30 C 17,27 28,27 32.5,30"/><path d="M 11.5,33.5 C 17,30.5 28,30.5 33.5,33.5"/><path d="M 11.5,37 C 17,34 28,34 33.5,37"/></g></svg>`
};

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Application State
const state = {
  gameData: null,
  currentPlyIndex: 0,
  isPlaying: false,
  playTimer: null,
  playSpeedMs: 1000,
  threshold: 0.50,
  demoGames: [],
  charts: {
    aiProbs: null,
    moveTime: null,
    cv: null,
    clock: null
  }
};

// DOM Elements
const elements = {
  chessboard: document.getElementById("chessboard"),
  moveScrubber: document.getElementById("move-scrubber"),
  scrubberPlyLabel: document.getElementById("scrubber-ply-label"),
  scrubberSanLabel: document.getElementById("scrubber-san-label"),
  moveTableBody: document.getElementById("move-table-body"),
  btnPlayPause: document.getElementById("btn-play-pause"),
  btnFirst: document.getElementById("btn-first"),
  btnPrev: document.getElementById("btn-prev"),
  btnNext: document.getElementById("btn-next"),
  btnLast: document.getElementById("btn-last"),
  speedButtons: document.querySelectorAll(".speed-btn"),
  thresholdSlider: document.getElementById("threshold-slider"),
  thresholdVal: document.getElementById("threshold-val"),
  thresholdBadge: document.getElementById("threshold-badge"),
  btnSetStrict: document.getElementById("btn-set-strict"),
  presetBalanced: document.getElementById("preset-balanced"),
  btnTogglePaste: document.getElementById("btn-toggle-paste"),
  pasteDrawer: document.getElementById("paste-drawer"),
  btnSubmitPaste: document.getElementById("btn-submit-paste"),
  btnCancelPaste: document.getElementById("btn-cancel-paste"),
  pgnTextInput: document.getElementById("pgn-text-input"),
  pgnFileInput: document.getElementById("pgn-file-input"),
  appAlert: document.getElementById("app-alert"),
  alertMessage: document.getElementById("alert-message"),
  demoButtons: document.querySelectorAll(".btn-demo"),
  // White card
  whiteCard: document.getElementById("white-card"),
  whiteName: document.getElementById("white-name"),
  whiteRating: document.getElementById("white-rating"),
  whitePill: document.getElementById("white-prediction-pill"),
  whiteAiPct: document.getElementById("white-ai-pct"),
  whiteHumanPct: document.getElementById("white-human-pct"),
  whiteProbBar: document.getElementById("white-prob-bar"),
  whiteMovesCount: document.getElementById("white-moves-count"),
  whiteFlaggedCount: document.getElementById("white-flagged-count"),
  whitePeakAi: document.getElementById("white-peak-ai"),
  // Black card
  blackCard: document.getElementById("black-card"),
  blackName: document.getElementById("black-name"),
  blackRating: document.getElementById("black-rating"),
  blackPill: document.getElementById("black-prediction-pill"),
  blackAiPct: document.getElementById("black-ai-pct"),
  blackHumanPct: document.getElementById("black-human-pct"),
  blackProbBar: document.getElementById("black-prob-bar"),
  blackMovesCount: document.getElementById("black-moves-count"),
  blackFlaggedCount: document.getElementById("black-flagged-count"),
  blackPeakAi: document.getElementById("black-peak-ai"),
  // Telemetry
  activePlyBadge: document.getElementById("active-ply-badge"),
  currentPredictionText: document.getElementById("current-prediction-text"),
  currentConfidenceText: document.getElementById("current-confidence-text"),
  meterVal: document.getElementById("meter-val"),
  circularMeter: document.getElementById("circular-meter"),
  signalList: document.getElementById("signal-list"),
  featuresMetersGrid: document.getElementById("features-meters-grid"),
  importanceContainer: document.getElementById("importance-container"),
  // Meta tags
  metaTc: document.getElementById("meta-tc"),
  metaPhase: document.getElementById("meta-phase"),
  metaEco: document.getElementById("meta-eco"),
  clockWhite: document.getElementById("clock-white"),
  clockBlack: document.getElementById("clock-black")
};

// --------------------------------------------------------------------------
// Initialization
// --------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  initChessboard();
  initCharts();
  setupEventListeners();
  await loadFeatureImportance();
  await loadDemoGames();
});

// --------------------------------------------------------------------------
// Chessboard Grid & FEN Rendering
// --------------------------------------------------------------------------
function initChessboard() {
  elements.chessboard.innerHTML = "";
  for (let r = 8; r >= 1; r--) {
    for (let f = 1; f <= 8; f++) {
      const square = document.createElement("div");
      const isLight = (r + f) % 2 === 1;
      const fileChar = String.fromCharCode(96 + f);
      const sqName = `${fileChar}${r}`;

      square.className = `square ${isLight ? "light" : "dark"}`;
      square.id = `sq-${sqName}`;
      square.dataset.square = sqName;
      elements.chessboard.appendChild(square);
    }
  }
}

function renderFEN(fen, highlightFrom = null, highlightTo = null) {
  // Clear piece contents and highlights
  document.querySelectorAll(".square").forEach(sq => {
    sq.innerHTML = "";
    sq.classList.remove("highlight", "check");
  });

  const parts = fen.split(" ");
  const rows = parts[0].split("/");

  for (let rIdx = 0; rIdx < 8; rIdx++) {
    const r = 8 - rIdx;
    let f = 1;
    for (const char of rows[rIdx]) {
      if (char >= "1" && char <= "8") {
        f += parseInt(char, 10);
      } else {
        const fileChar = String.fromCharCode(96 + f);
        const sqName = `${fileChar}${r}`;
        const sqEl = document.getElementById(`sq-${sqName}`);
        if (sqEl && CHESS_PIECES_SVG[char]) {
          sqEl.innerHTML = CHESS_PIECES_SVG[char];
        }
        f++;
      }
    }
  }

  if (highlightFrom) {
    const elFrom = document.getElementById(`sq-${highlightFrom}`);
    if (elFrom) elFrom.classList.add("highlight");
  }
  if (highlightTo) {
    const elTo = document.getElementById(`sq-${highlightTo}`);
    if (elTo) elTo.classList.add("highlight");
  }
}

// --------------------------------------------------------------------------
// Chart.js Telemetry Initializer
// --------------------------------------------------------------------------
function initCharts() {
  const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: "#94a3b8", font: { family: "Inter", size: 11 } }
      }
    },
    scales: {
      x: {
        grid: { color: "rgba(255, 255, 255, 0.05)" },
        ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 10 } }
      },
      y: {
        grid: { color: "rgba(255, 255, 255, 0.05)" },
        ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 10 } }
      }
    }
  };

  // 1. AI Probability Trajectory
  const ctxAi = document.getElementById("chart-ai-probs").getContext("2d");
  state.charts.aiProbs = new Chart(ctxAi, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "White AI Prob (%)",
          data: [],
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.1)",
          tension: 0.25,
          pointRadius: 3,
          pointHoverRadius: 6
        },
        {
          label: "Black AI Prob (%)",
          data: [],
          borderColor: "#f87171",
          backgroundColor: "rgba(248, 113, 113, 0.1)",
          tension: 0.25,
          pointRadius: 3,
          pointHoverRadius: 6
        },
        {
          label: "Operating Threshold (50%)",
          data: [],
          borderColor: "#fbbf24",
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      ...chartDefaults,
      onClick: (e, items) => handleChartClick(items),
      scales: {
        ...chartDefaults.scales,
        y: { ...chartDefaults.scales.y, min: 0, max: 100 }
      }
    }
  });

  // 2. Move Thinking Time
  const ctxTime = document.getElementById("chart-movetime").getContext("2d");
  state.charts.moveTime = new Chart(ctxTime, {
    type: "bar",
    data: {
      labels: [],
      datasets: [
        {
          label: "Move Time (s)",
          data: [],
          backgroundColor: [],
          borderRadius: 4
        }
      ]
    },
    options: {
      ...chartDefaults,
      onClick: (e, items) => handleChartClick(items)
    }
  });

  // 3. Move Time CV
  const ctxCv = document.getElementById("chart-cv").getContext("2d");
  state.charts.cv = new Chart(ctxCv, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Move Time CV (W=10)",
          data: [],
          borderColor: "#a855f7",
          backgroundColor: "rgba(168, 85, 247, 0.1)",
          fill: true,
          tension: 0.3,
          pointRadius: 2
        }
      ]
    },
    options: {
      ...chartDefaults,
      onClick: (e, items) => handleChartClick(items)
    }
  });

  // 4. Clock Remaining
  const ctxClock = document.getElementById("chart-clock").getContext("2d");
  state.charts.clock = new Chart(ctxClock, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "White Clock (s)",
          data: [],
          borderColor: "#e2e8f0",
          stepped: true,
          pointRadius: 0
        },
        {
          label: "Black Clock (s)",
          data: [],
          borderColor: "#64748b",
          stepped: true,
          pointRadius: 0
        }
      ]
    },
    options: {
      ...chartDefaults,
      onClick: (e, items) => handleChartClick(items)
    }
  });
}

function handleChartClick(items) {
  if (items && items.length > 0) {
    const clickedIdx = items[0].index;
    goToPly(clickedIdx + 1);
  }
}

// --------------------------------------------------------------------------
// Event Listeners
// --------------------------------------------------------------------------
function setupEventListeners() {
  // Demo preset selector
  elements.demoButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      elements.demoButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const demoId = btn.dataset.demo;
      loadPresetDemo(demoId);
    });
  });

  // Scrubber slider
  elements.moveScrubber.addEventListener("input", (e) => {
    goToPly(parseInt(e.target.value, 10));
  });

  // Stepper buttons
  elements.btnFirst.addEventListener("click", () => goToPly(0));
  elements.btnPrev.addEventListener("click", () => goToPly(state.currentPlyIndex - 1));
  elements.btnNext.addEventListener("click", () => goToPly(state.currentPlyIndex + 1));
  elements.btnLast.addEventListener("click", () => {
    if (state.gameData && state.gameData.moves) {
      goToPly(state.gameData.moves.length);
    }
  });

  // Play / Pause
  elements.btnPlayPause.addEventListener("click", togglePlayPause);

  // Speed buttons
  elements.speedButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      elements.speedButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.playSpeedMs = parseInt(btn.dataset.speed, 10);
      if (state.isPlaying) {
        clearInterval(state.playTimer);
        state.playTimer = setInterval(stepPlayForward, state.playSpeedMs);
      }
    });
  });

  // Threshold slider
  elements.thresholdSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    setThreshold(val);
  });

  elements.presetBalanced.addEventListener("click", () => setThreshold(0.50));
  elements.btnSetStrict.addEventListener("click", () => setThreshold(0.85));

  // Toggle paste drawer
  elements.btnTogglePaste.addEventListener("click", () => {
    elements.pasteDrawer.classList.toggle("hidden");
  });
  elements.btnCancelPaste.addEventListener("click", () => {
    elements.pasteDrawer.classList.add("hidden");
  });
  elements.btnSubmitPaste.addEventListener("click", () => {
    const text = elements.pgnTextInput.value.trim();
    if (text) {
      analyzePGNString(text);
      elements.pasteDrawer.classList.add("hidden");
    }
  });

  // File Upload
  elements.pgnFileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        analyzePGNString(event.target.result);
      };
      reader.readAsText(file);
    }
  });

  // Keyboard navigation
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
    if (e.key === "ArrowLeft") {
      goToPly(state.currentPlyIndex - 1);
    } else if (e.key === "ArrowRight") {
      goToPly(state.currentPlyIndex + 1);
    } else if (e.key === " ") {
      e.preventDefault();
      togglePlayPause();
    }
  });
}

function setThreshold(val) {
  state.threshold = val;
  elements.thresholdSlider.value = val;
  elements.thresholdVal.textContent = val.toFixed(2);
  elements.thresholdBadge.textContent = `Threshold: τ = ${val.toFixed(2)}`;

  if (state.gameData) {
    // Re-evaluate current game under new threshold
    recalculateSummariesUnderThreshold();
    goToPly(state.currentPlyIndex);
  }
}

function togglePlayPause() {
  if (state.isPlaying) {
    state.isPlaying = false;
    clearInterval(state.playTimer);
    elements.btnPlayPause.textContent = "▶ Play";
    elements.btnPlayPause.classList.remove("btn-danger");
  } else {
    if (!state.gameData || !state.gameData.moves) return;
    if (state.currentPlyIndex >= state.gameData.moves.length) {
      state.currentPlyIndex = 0;
    }
    state.isPlaying = true;
    elements.btnPlayPause.textContent = "⏸ Pause";
    elements.btnPlayPause.classList.add("btn-danger");
    state.playTimer = setInterval(stepPlayForward, state.playSpeedMs);
  }
}

function stepPlayForward() {
  if (!state.gameData || !state.gameData.moves) {
    togglePlayPause();
    return;
  }
  if (state.currentPlyIndex < state.gameData.moves.length) {
    goToPly(state.currentPlyIndex + 1);
  } else {
    togglePlayPause();
  }
}

// --------------------------------------------------------------------------
// API Client Functions
// --------------------------------------------------------------------------
async function loadFeatureImportance() {
  try {
    const res = await fetch("/feature-importance");
    if (!res.ok) return;
    const data = await res.json();
    renderFeatureImportance(data.features || []);
  } catch (err) {
    console.warn("Could not load feature importance:", err);
  }
}

function renderFeatureImportance(features) {
  elements.importanceContainer.innerHTML = "";
  features.forEach(f => {
    const row = document.createElement("div");
    row.className = "importance-row";
    const pct = (f.normalized_gain * 100).toFixed(1);

    row.innerHTML = `
      <span class="imp-label" title="${f.feature}">${f.feature}</span>
      <div class="imp-track">
        <div class="imp-fill" style="width: ${pct}%"></div>
      </div>
      <span class="imp-val">${pct}%</span>
    `;
    elements.importanceContainer.appendChild(row);
  });
}

async function loadDemoGames() {
  try {
    const res = await fetch("/demo-games");
    if (!res.ok) return;
    const data = await res.json();
    state.demoGames = data.demo_games || [];
    if (state.demoGames.length > 0) {
      // Load first demo by default
      loadPresetDemo(state.demoGames[0].id);
    }
  } catch (err) {
    console.error("Failed to fetch demo games:", err);
  }
}

function loadPresetDemo(demoId) {
  const match = state.demoGames.find(g => g.id === demoId);
  if (match) {
    analyzePGNString(match.pgn);
  }
}

async function analyzePGNString(pgnText) {
  showAlert(null); // hide previous alerts
  try {
    const res = await fetch("/analyze-pgn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pgn: pgnText, threshold: state.threshold })
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Analysis request failed.");
    }

    const data = await res.json();
    state.gameData = data;

    if (data.warning) {
      showAlert(data.warning);
    }

    displayLoadedGame(data);
  } catch (err) {
    showAlert(`Error analyzing PGN: ${err.message}`);
    console.error(err);
  }
}

function showAlert(msg) {
  if (!msg) {
    elements.appAlert.classList.add("hidden");
    elements.alertMessage.textContent = "";
  } else {
    elements.appAlert.classList.remove("hidden");
    elements.alertMessage.textContent = msg;
  }
}

// --------------------------------------------------------------------------
// Display Loaded Game & Update Telemetry
// --------------------------------------------------------------------------
function displayLoadedGame(data) {
  const meta = data.metadata || {};
  const moves = data.moves || [];

  // Metadata pills
  elements.metaTc.textContent = meta.time_control || "Unknown TC";
  elements.metaEco.textContent = `ECO: ${meta.eco || "--"}`;

  // Update Player Cards
  updatePlayerCard(elements.whiteCard, elements.whiteName, elements.whiteRating, elements.whitePill,
    elements.whiteAiPct, elements.whiteHumanPct, elements.whiteProbBar,
    elements.whiteMovesCount, elements.whiteFlaggedCount, elements.whitePeakAi,
    data.summary.white, meta.white, meta.white_elo);

  updatePlayerCard(elements.blackCard, elements.blackName, elements.blackRating, elements.blackPill,
    elements.blackAiPct, elements.blackHumanPct, elements.blackProbBar,
    elements.blackMovesCount, elements.blackFlaggedCount, elements.blackPeakAi,
    data.summary.black, meta.black, meta.black_elo);

  // Scrubber setup
  elements.moveScrubber.max = moves.length;
  elements.moveScrubber.value = 0;

  // Populate Move Table
  populateMoveTable(moves);

  // Populate Charts
  updateChartsData(moves);

  // Jump to first move or start position
  goToPly(0);
}

function updatePlayerCard(cardEl, nameEl, ratingEl, pillEl, aiPctEl, humanPctEl, barEl, movesCountEl, flaggedCountEl, peakEl,
  summary, defaultName, defaultElo) {
  nameEl.textContent = summary.name || defaultName || "Player";
  ratingEl.textContent = `Rating: ${defaultElo || "1500"}`;

  const aiPct = summary.mean_ai_probability_pct ?? 0;
  const humanPct = summary.mean_human_probability_pct ?? (100 - aiPct);

  aiPctEl.textContent = `${aiPct}%`;
  humanPctEl.textContent = `${humanPct}%`;
  barEl.style.width = `${aiPct}%`;

  movesCountEl.textContent = summary.moves_count || 0;
  flaggedCountEl.textContent = `${summary.ai_flagged_move_pct ?? 0}%`;
  peakEl.textContent = `${summary.max_ai_probability_pct ?? 0}%`;

  cardEl.classList.remove("is-ai", "is-human");
  pillEl.classList.remove("pill-ai", "pill-human", "pill-unknown");

  if (summary.predicted_class === "UNAVAILABLE" || summary.mean_ai_probability_pct === null) {
    aiPctEl.textContent = "N/A";
    humanPctEl.textContent = "N/A";
    barEl.style.width = "0%";
    movesCountEl.textContent = summary.moves_count || 0;
    flaggedCountEl.textContent = "N/A";
    peakEl.textContent = "N/A";
    pillEl.classList.add("pill-unknown");
    pillEl.textContent = "CLOCK DATA UNAVAILABLE";
  } else {
    const rawAiPct = summary.mean_ai_probability_pct ?? 0;
    const aiPct = Number(rawAiPct).toFixed(1);
    const rawHumanPct = summary.mean_human_probability_pct ?? (100.0 - Number(aiPct));
    const humanPct = Number(rawHumanPct).toFixed(1);

    aiPctEl.textContent = `${aiPct}%`;
    humanPctEl.textContent = `${humanPct}%`;
    barEl.style.width = `${aiPct}%`;

    movesCountEl.textContent = summary.moves_count || 0;
    flaggedCountEl.textContent = `${summary.ai_flagged_move_pct ?? 0}%`;
    peakEl.textContent = `${summary.max_ai_probability_pct ?? 0}%`;

    if (summary.predicted_class === "AI") {
      cardEl.classList.add("is-ai");
      pillEl.classList.add("pill-ai");
      pillEl.textContent = "AI-like behavioral pattern";
    } else {
      cardEl.classList.add("is-human");
      pillEl.classList.add("pill-human");
      pillEl.textContent = "Predicted HUMAN";
    }
  }
}

function populateMoveTable(moves) {
  elements.moveTableBody.innerHTML = "";
  const numMoves = Math.ceil(moves.length / 2);

  for (let m = 0; m < numMoves; m++) {
    const tr = document.createElement("tr");
    const whiteIdx = m * 2;
    const blackIdx = m * 2 + 1;

    const wMove = moves[whiteIdx];
    const bMove = blackIdx < moves.length ? moves[blackIdx] : null;

    const wTimeStr = (wMove && wMove.move_time !== null && wMove.move_time !== undefined) ? `${wMove.move_time.toFixed(1)}s` : "N/A";
    const wProbStr = (wMove && wMove.ai_probability !== null && wMove.ai_probability !== undefined) ? `${(wMove.ai_probability * 100).toFixed(0)}%` : "N/A";

    const bTimeStr = (bMove && bMove.move_time !== null && bMove.move_time !== undefined) ? `${bMove.move_time.toFixed(1)}s` : "N/A";
    const bProbStr = (bMove && bMove.ai_probability !== null && bMove.ai_probability !== undefined) ? `${(bMove.ai_probability * 100).toFixed(0)}%` : "N/A";

    tr.innerHTML = `
      <td>${m + 1}</td>
      <td id="table-ply-${whiteIdx + 1}" class="${wMove && wMove.ai_probability !== null && wMove.ai_probability >= state.threshold ? "move-flag-ai" : ""}">
        ${wMove ? wMove.san : ""}
      </td>
      <td>${wTimeStr}</td>
      <td>${wProbStr}</td>
      <td id="table-ply-${blackIdx + 1}" class="${bMove && bMove.ai_probability !== null && bMove.ai_probability >= state.threshold ? "move-flag-ai" : ""}">
        ${bMove ? bMove.san : ""}
      </td>
      <td>${bTimeStr}</td>
      <td>${bProbStr}</td>
    `;

    // Click row or cell to jump
    if (wMove) {
      tr.querySelector(`#table-ply-${whiteIdx + 1}`).addEventListener("click", () => goToPly(whiteIdx + 1));
    }
    if (bMove) {
      tr.querySelector(`#table-ply-${blackIdx + 1}`).addEventListener("click", () => goToPly(blackIdx + 1));
    }

    elements.moveTableBody.appendChild(tr);
  }
}

// --------------------------------------------------------------------------
// Update Charts
// --------------------------------------------------------------------------
function updateChartsData(moves) {
  const labels = [];
  const whiteProbs = [];
  const blackProbs = [];
  const threshLine = [];
  const moveTimes = [];
  const timeBarColors = [];
  const cvList = [];
  const whiteClock = [];
  const blackClock = [];

  let lastWClk = state.gameData.metadata.base_time;
  let lastBClk = state.gameData.metadata.base_time;

  moves.forEach((m, idx) => {
    labels.push(`${m.move_number}.${m.player === "White" ? "" : ".."}${m.san}`);
    threshLine.push(state.threshold * 100);

    const hasProb = m.ai_probability !== null && m.ai_probability !== undefined;
    const probPct = hasProb ? m.ai_probability * 100 : null;

    if (m.player === "White") {
      whiteProbs.push(probPct);
      blackProbs.push(null);
      if (m.clock_after_move !== null && m.clock_after_move !== undefined) {
        lastWClk = m.clock_after_move;
      }
    } else {
      blackProbs.push(probPct);
      whiteProbs.push(null);
      if (m.clock_after_move !== null && m.clock_after_move !== undefined) {
        lastBClk = m.clock_after_move;
      }
    }

    moveTimes.push(m.move_time !== null && m.move_time !== undefined ? m.move_time : 0);
    timeBarColors.push(hasProb && m.ai_probability >= state.threshold ? "#ef4444" : "#10b981");

    cvList.push(m.features_10 && m.features_10.move_time_cv_10 !== undefined ? m.features_10.move_time_cv_10 : 0);
    whiteClock.push(lastWClk);
    blackClock.push(lastBClk);
  });

  // Chart 1: Probs
  state.charts.aiProbs.data.labels = labels;
  state.charts.aiProbs.data.datasets[0].data = whiteProbs;
  state.charts.aiProbs.data.datasets[1].data = blackProbs;
  state.charts.aiProbs.data.datasets[2].data = threshLine;
  state.charts.aiProbs.update();

  // Chart 2: Move times
  state.charts.moveTime.data.labels = labels;
  state.charts.moveTime.data.datasets[0].data = moveTimes;
  state.charts.moveTime.data.datasets[0].backgroundColor = timeBarColors;
  state.charts.moveTime.update();

  // Chart 3: CV
  state.charts.cv.data.labels = labels;
  state.charts.cv.data.datasets[0].data = cvList;
  state.charts.cv.update();

  // Chart 4: Clock
  state.charts.clock.data.labels = labels;
  state.charts.clock.data.datasets[0].data = whiteClock;
  state.charts.clock.data.datasets[1].data = blackClock;
  state.charts.clock.update();
}

// --------------------------------------------------------------------------
// Ply Navigation & Behavioral Display
// --------------------------------------------------------------------------
function goToPly(plyIndex) {
  if (!state.gameData) return;
  const moves = state.gameData.moves || [];
  const totalMoves = moves.length;

  const validPly = Math.max(0, Math.min(plyIndex, totalMoves));
  state.currentPlyIndex = validPly;
  elements.moveScrubber.value = validPly;

  // Remove previous table active highlights
  document.querySelectorAll(".move-history-table td").forEach(td => td.classList.remove("active-move"));

  if (validPly === 0) {
    // Start position
    renderFEN(START_FEN);
    elements.scrubberPlyLabel.textContent = `Move 0 / ${totalMoves}`;
    elements.scrubberSanLabel.textContent = "Start Position";
    elements.metaPhase.textContent = "Phase: Opening";
    elements.clockWhite.textContent = `W: ${formatClock(state.gameData.metadata.base_time)}`;
    elements.clockBlack.textContent = `B: ${formatClock(state.gameData.metadata.base_time)}`;
    elements.activePlyBadge.textContent = "Start Position";
    elements.currentPredictionText.textContent = "Position Before Move 1";
    elements.currentPredictionText.className = "hero-prediction";
    elements.currentConfidenceText.textContent = "No behavioral moves observed yet.";
    elements.meterVal.textContent = "0%";
    elements.circularMeter.style.background = `conic-gradient(#10b981 0deg, #1e293b 0deg)`;

    renderSignalList([]);
    renderFeatureMetersGrid({
      consecutive_fast_moves: 0,
      premove_rate_10: 0,
      cumulative_premove_rate: 0,
      current_endgame_clock_ratio: 0,
      phase_deliberation_ratio: 1.0,
      move_time_cv_10: 0,
      time_pressure_jitter: 0,
      relative_move_time: 0,
      time_spent_ratio: 0,
      tank_move_count_10: 0
    });
    return;
  }

  // Active move
  const m = moves[validPly - 1];
  renderFEN(m.fen_after, m.from_square, m.to_square);

  // Scrubber labels
  const timeDisplay = (m.move_time !== null && m.move_time !== undefined) ? `${m.move_time.toFixed(1)}s` : "N/A";
  elements.scrubberPlyLabel.textContent = `Move ${m.move_number} (${m.player}) | Ply ${validPly} / ${totalMoves}`;
  elements.scrubberSanLabel.textContent = `${m.san} (${timeDisplay})`;
  elements.metaPhase.textContent = `Phase: ${m.game_phase}`;

  // Clocks
  if (m.player === "White") {
    elements.clockWhite.textContent = `W: ${formatClock(m.clock_after_move)}`;
  } else {
    elements.clockBlack.textContent = `B: ${formatClock(m.clock_after_move)}`;
  }

  // Table highlight
  const cell = document.getElementById(`table-ply-${validPly}`);
  if (cell) {
    cell.classList.add("active-move");
    cell.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  // Current Prediction Hero
  elements.activePlyBadge.textContent = `Ply ${validPly}: ${m.move_number}.${m.player === "White" ? "" : ".."}${m.san}`;

  if (m.ai_probability === null || m.ai_probability === undefined) {
    elements.currentPredictionText.textContent = "UNAVAILABLE (No Clock Data)";
    elements.currentPredictionText.className = "hero-prediction";
    elements.currentConfidenceText.textContent = "Move clock data unavailable. Timing-based prediction cannot be performed.";
    elements.meterVal.textContent = "N/A";
    elements.circularMeter.style.background = `conic-gradient(#64748b 0deg, #1e293b 0deg)`;
  } else {
    const isAi = m.ai_probability >= state.threshold;
    elements.currentPredictionText.textContent = isAi ? "AI-like behavioral pattern" : "Predicted HUMAN";
    elements.currentPredictionText.className = `hero-prediction ${isAi ? "is-ai" : "is-human"}`;

    const aiPct = (m.ai_probability * 100).toFixed(1);
    const humanPct = (m.human_probability * 100).toFixed(1);
    elements.currentConfidenceText.textContent = `AI Prob: ${aiPct}% | Human Prob: ${humanPct}% (τ = ${state.threshold.toFixed(2)})`;

    // Circular gauge
    elements.meterVal.textContent = `${Math.round(aiPct)}%`;
    const deg = (aiPct / 100) * 360;
    const gaugeColor = isAi ? "var(--color-ai)" : "var(--color-human)";
    elements.circularMeter.style.background = `conic-gradient(${gaugeColor} ${deg}deg, #1e293b ${deg}deg)`;
  }

  // Signals list
  renderSignalList(m.behavioral_signals || []);

  // 10 Feature Meters
  renderFeatureMetersGrid(m.features_10 || null);
}

function formatClock(seconds) {
  if (seconds === undefined || seconds === null || isNaN(seconds)) return "--:--";
  const s = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
}

function renderSignalList(signals) {
  elements.signalList.innerHTML = "";
  if (!signals || signals.length === 0) {
    const li = document.createElement("li");
    li.className = "signal-item";
    li.textContent = "Observed timing and clock parameters align with standard deliberate human deliberation.";
    elements.signalList.appendChild(li);
    return;
  }

  signals.forEach(s => {
    const li = document.createElement("li");
    li.className = "signal-item";
    li.innerHTML = `<strong>${s.name} (${s.value}):</strong> ${s.observation}`;
    elements.signalList.appendChild(li);
  });
}

function renderFeatureMetersGrid(f) {
  if (!f) {
    elements.featuresMetersGrid.innerHTML = `
      <div class="meter-card" style="grid-column: 1 / -1; padding: 1.25rem; text-align: center; color: var(--text-muted);">
        <em>Behavioral timing features unavailable (no [%clk] comments detected in PGN).</em>
      </div>
    `;
    return;
  }

  const fastVal = f.consecutive_fast_moves ?? 0;
  const pmrVal = f.premove_rate_10 ?? 0;
  const cumPmrVal = f.cumulative_premove_rate ?? 0;
  const cvVal = f.move_time_cv_10 ?? 0;
  const pdrVal = f.phase_deliberation_ratio ?? 1.0;
  const relTimeVal = f.relative_move_time ?? 0;
  const timeSpentVal = f.time_spent_ratio ?? 0;
  const egClockVal = f.current_endgame_clock_ratio ?? 0;
  const tankVal = f.tank_move_count_10 ?? 0;
  const tpJitterVal = f.time_pressure_jitter ?? 0;

  const featureConfigs = [
    {
      key: "consecutive_fast_moves",
      name: "Fast Move Streak",
      valTxt: `${Math.round(fastVal)} moves`,
      pct: Math.min(100, (fastVal / 6) * 100),
      isAi: fastVal >= 3
    },
    {
      key: "premove_rate_10",
      name: "Premove Rate (W=10)",
      valTxt: `${(pmrVal * 100).toFixed(0)}%`,
      pct: pmrVal * 100,
      isAi: pmrVal >= 0.50
    },
    {
      key: "cumulative_premove_rate",
      name: "Cumulative Premove Rate",
      valTxt: `${(cumPmrVal * 100).toFixed(0)}%`,
      pct: cumPmrVal * 100,
      isAi: cumPmrVal >= 0.40
    },
    {
      key: "move_time_cv_10",
      name: "Timing Volatility (CV)",
      valTxt: cvVal.toFixed(2),
      pct: Math.min(100, (cvVal / 2.0) * 100),
      isAi: cvVal < 0.25
    },
    {
      key: "phase_deliberation_ratio",
      name: "Phase Deliberation Ratio",
      valTxt: `${pdrVal.toFixed(2)}x`,
      pct: Math.min(100, (pdrVal / 4.0) * 100),
      isAi: pdrVal < 1.1
    },
    {
      key: "relative_move_time",
      name: "Relative Move Time",
      valTxt: `${relTimeVal.toFixed(2)}x budget`,
      pct: Math.min(100, (relTimeVal / 3.0) * 100),
      isAi: false
    },
    {
      key: "time_spent_ratio",
      name: "Time Spent / Clock Bank",
      valTxt: `${(timeSpentVal * 100).toFixed(1)}%`,
      pct: timeSpentVal * 100,
      isAi: false
    },
    {
      key: "current_endgame_clock_ratio",
      name: "Endgame Clock Ratio",
      valTxt: egClockVal.toFixed(2),
      pct: Math.min(100, (egClockVal / 2.0) * 100),
      isAi: false
    },
    {
      key: "tank_move_count_10",
      name: "Tank Moves (>30s)",
      valTxt: `${Math.round(tankVal)} deep thinks`,
      pct: Math.min(100, (tankVal / 3) * 100),
      isAi: false
    },
    {
      key: "time_pressure_jitter",
      name: "Time Pressure Jitter",
      valTxt: `${tpJitterVal.toFixed(2)}s`,
      pct: Math.min(100, (tpJitterVal / 5.0) * 100),
      isAi: false
    }
  ];

  elements.featuresMetersGrid.innerHTML = "";
  featureConfigs.forEach(cfg => {
    const card = document.createElement("div");
    card.className = "meter-card";
    const barClass = cfg.isAi ? "meter-bar meter-ai" : "meter-bar";

    card.innerHTML = `
      <div class="meter-header">
        <span class="meter-name">${cfg.name}</span>
        <span class="meter-value-txt">${cfg.valTxt}</span>
      </div>
      <div class="meter-track">
        <div class="${barClass}" style="width: ${cfg.pct}%;"></div>
      </div>
    `;
    elements.featuresMetersGrid.appendChild(card);
  });
}

function recalculateSummariesUnderThreshold() {
  if (!state.gameData || !state.gameData.moves || !state.gameData.has_clock_data) return;

  const whiteProbs = [];
  const blackProbs = [];

  state.gameData.moves.forEach(m => {
    m.prediction = m.ai_probability >= state.threshold ? "AI-like behavioral pattern" : "Predicted HUMAN";
    m.predicted_class = m.ai_probability >= state.threshold ? "AI" : "HUMAN";
    if (m.player === "White") {
      whiteProbs.push(m.ai_probability);
    } else {
      blackProbs.push(m.ai_probability);
    }
  });

  function recalcPlayer(probs, name) {
    if (!probs.length) return { name, moves_count: 0 };
    const meanP = probs.reduce((a, b) => a + b, 0) / probs.length;
    const maxP = Math.max(...probs);
    const flagged = probs.filter(p => p >= state.threshold).length;
    const flaggedRatio = flagged / probs.length;
    const isAi = meanP >= state.threshold || flaggedRatio >= 0.50;

    return {
      name,
      moves_count: probs.length,
      mean_ai_probability_pct: Math.round(meanP * 1000) / 10,
      mean_human_probability_pct: Math.round((1 - meanP) * 1000) / 10,
      max_ai_probability_pct: Math.round(maxP * 1000) / 10,
      ai_flagged_move_pct: Math.round(flaggedRatio * 1000) / 10,
      overall_prediction: isAi ? "AI-like behavioral pattern" : "Predicted HUMAN",
      predicted_class: isAi ? "AI" : "HUMAN"
    };
  }

  state.gameData.summary.white = recalcPlayer(whiteProbs, state.gameData.metadata.white);
  state.gameData.summary.black = recalcPlayer(blackProbs, state.gameData.metadata.black);

  updatePlayerCard(elements.whiteCard, elements.whiteName, elements.whiteRating, elements.whitePill,
    elements.whiteAiPct, elements.whiteHumanPct, elements.whiteProbBar,
    elements.whiteMovesCount, elements.whiteFlaggedCount, elements.whitePeakAi,
    state.gameData.summary.white, state.gameData.metadata.white, state.gameData.metadata.white_elo);

  updatePlayerCard(elements.blackCard, elements.blackName, elements.blackRating, elements.blackPill,
    elements.blackAiPct, elements.blackHumanPct, elements.blackProbBar,
    elements.blackMovesCount, elements.blackFlaggedCount, elements.blackPeakAi,
    state.gameData.summary.black, state.gameData.metadata.black, state.gameData.metadata.black_elo);

  updateChartsData(state.gameData.moves);
}
