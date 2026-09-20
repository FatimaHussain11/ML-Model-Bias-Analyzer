const $ = (id) => document.getElementById(id);

const CATEGORICAL_FIELDS = [
  "workclass", "education", "marital_status", "occupation",
  "relationship", "race", "sex", "native_country",
];

function apiBase() {
  return $("apiBase").value.replace(/\/+$/, "");
}

function setStatus(state, text) {
  const el = $("connStatus");
  el.dataset.state = state;
  el.textContent = text;
}

async function checkHealth() {
  try {
    const res = await fetch(`${apiBase()}/api/health`);
    if (!res.ok) throw new Error();
    setStatus("online", "connected");
    return true;
  } catch {
    setStatus("offline", "not connected");
    return false;
  }
}

async function loadOptions() {
  try {
    const res = await fetch(`${apiBase()}/api/options`);
    if (!res.ok) throw new Error("Could not load options");
    const data = await res.json();
    for (const field of CATEGORICAL_FIELDS) {
      const select = $(field);
      select.innerHTML = "";
      for (const value of data.categorical_options[field]) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = value;
        select.appendChild(opt);
      }
    }
    // sensible defaults matching the sample profile
    trySelect("sex", "Male");
    trySelect("race", "White");
    setStatus("online", "connected");
  } catch (err) {
    setStatus("offline", "not connected");
  }
}

function trySelect(id, value) {
  const el = $(id);
  if (el && [...el.options].some((o) => o.value === value)) el.value = value;
}

function collectPayload() {
  const form = $("profileForm");
  const data = new FormData(form);
  const payload = {};
  for (const [key, value] of data.entries()) payload[key] = value;
  return payload;
}

function fmtPct(x) {
  return (x * 100).toFixed(1) + "%";
}

// Blends from the calm accent (low end of this profile's group spread)
// to the flag color (high end), so the bias-check bars read as a scale.
function mixColor(t) {
  const low = [15, 122, 99];    // --teal
  const high = [209, 85, 42];   // --coral
  const c = low.map((c0, i) => Math.round(c0 + (high[i] - c0) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

async function runPrediction(e) {
  e.preventDefault();
  const btn = $("predictBtn");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const res = await fetch(`${apiBase()}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");

    $("predictionEmpty").hidden = true;
    const resultEl = $("predictionResult");
    resultEl.hidden = false;
    resultEl.style.animation = "none";
    void resultEl.offsetWidth;
    resultEl.style.animation = "";
    $("verdictLabel").textContent = data.label;
    $("confidenceFill").style.width = fmtPct(data.probability_gt_50k);
    $("confidenceLow").textContent = `P(\u226450K) ${fmtPct(data.probability_le_50k)}`;
    $("confidenceHigh").textContent = `P(>50K) ${fmtPct(data.probability_gt_50k)}`;
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run prediction";
  }
}

async function runBiasCheck() {
  const btn = $("biasBtn");
  btn.disabled = true;
  btn.textContent = "Checking…";
  try {
    const res = await fetch(`${apiBase()}/api/bias-check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Bias check failed");

    $("biasEmpty").hidden = true;
    const biasResultEl = $("biasResult");
    biasResultEl.hidden = false;

    const spreadEl = $("spreadValue");
    spreadEl.textContent = fmtPct(data.max_probability_spread);
    spreadEl.className = "spread-value " + (data.max_probability_spread > 0.1 ? "flag" : "calm");

    const sorted = [...data.results].sort((a, b) => b.probability_gt_50k - a.probability_gt_50k);
    const values = sorted.map((r) => r.probability_gt_50k);
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const grid = $("biasGrid");
    grid.innerHTML = "";
    sorted.forEach((row, i) => {
      const t = hi > lo ? (row.probability_gt_50k - lo) / (hi - lo) : 0.5;
      const barColor = mixColor(t);
      const el = document.createElement("div");
      el.className = "bias-row";
      el.style.animationDelay = `${i * 35}ms`;
      el.innerHTML = `
        <span class="race-name">${row.race}</span>
        <span class="sex-name">${row.sex === "Male" ? "M" : "F"}</span>
        <span class="bias-bar-track"><span class="bias-bar-fill" style="width:${fmtPct(row.probability_gt_50k)};background:${barColor}"></span></span>
        <span class="pct">${fmtPct(row.probability_gt_50k)}</span>
      `;
      grid.appendChild(el);
    });
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run bias check";
  }
}

$("profileForm").addEventListener("submit", runPrediction);
$("biasBtn").addEventListener("click", runBiasCheck);
$("apiBase").addEventListener("change", () => {
  checkHealth();
  loadOptions();
});

loadOptions();
