const byId = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[char]));
const number = (value, digits = 0) => new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: digits,
  minimumFractionDigits: digits,
}).format(value);
const percent = (value) => `${number(value * 100, 1)}%`;

let dashboardData = null;
let toastTimer;

function showToast(message) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3200);
}

function setView(view) {
  document.querySelectorAll(".view").forEach((section) => section.classList.toggle("active", section.id === `view-${view}`));
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  const title = { overview: "Overview", insights: "Insights", prediction: "Prediction lab" }[view] || "Overview";
  byId("current-section").textContent = title;
  history.replaceState(null, "", `#${view}`);
  if (view === "insights" && dashboardData) renderInsights(dashboardData);
}

function chartRows(rows, valueKey, options = {}) {
  const max = Math.max(...rows.map((row) => Number(row[valueKey]) || 0), 0.00001);
  const sorted = [...rows].sort((a, b) => (Number(b[valueKey]) || 0) - (Number(a[valueKey]) || 0));
  return sorted.map((row) => {
    const raw = Number(row[valueKey]) || 0;
    const width = Math.max(raw > 0 ? 1.5 : 0, raw / max * 100);
    let display = options.percent ? percent(raw) : number(raw, options.digits ?? 0);
    if (options.currency) display = `₹${number(raw)}`;
    return `<div class="chart-row"><span class="chart-label" title="${escapeHtml(row.Condition)}">${escapeHtml(row.Condition)}</span><div class="chart-track"><div class="chart-fill ${options.color || ""}" style="width:${width}%"></div></div><span class="chart-value">${display}</span></div>`;
  }).join("");
}

function renderOverview(data) {
  byId("metric-records").textContent = number(data.record_count);
  byId("metric-cost").textContent = number(data.average_cost);
  byId("metric-readmission").textContent = percent(data.readmission_rate);
  byId("metric-satisfaction").textContent = `${number(data.average_satisfaction, 1)} / 5`;
  byId("readmission-chart").innerHTML = chartRows(data.condition_summary, "readmission_rate", { percent: true, color: "mint" });
  byId("volume-chart").innerHTML = chartRows(data.condition_summary, "patients", { color: "" });
  byId("summary-table").innerHTML = data.condition_summary.map((row) => `
    <tr>
      <td><span class="condition-dot"></span>${escapeHtml(row.Condition)}</td>
      <td>${number(row.patients)}</td>
      <td>${number(row.average_cost)}</td>
      <td><span class="rate-pill">${percent(row.readmission_rate)}</span></td>
      <td>${number(row.average_satisfaction, 1)} / 5</td>
      <td>${percent(row.recovery_rate)}</td>
    </tr>`).join("");
  const genderSelect = byId("gender");
  genderSelect.innerHTML = Object.keys(data.gender_counts).sort().map((value) => `<option>${escapeHtml(value)}</option>`).join("");
  const conditionSelect = byId("condition");
  conditionSelect.innerHTML = data.conditions.map((value) => `<option>${escapeHtml(value)}</option>`).join("");
  updateProcedureOptions();
}

function renderInsights(data) {
  byId("cost-chart").innerHTML = chartRows(data.condition_summary, "average_cost", { color: "" });
  byId("stay-chart").innerHTML = chartRows(data.condition_summary, "average_stay", { digits: 1, color: "amber" });
  byId("recovery-chart").innerHTML = chartRows(data.condition_summary, "recovery_rate", { percent: true, color: "mint" });
  byId("satisfaction-chart").innerHTML = chartRows(data.condition_summary, "average_satisfaction", { digits: 1, color: "" });
}

function updateProcedureOptions() {
  if (!dashboardData) return;
  const condition = byId("condition").value;
  const options = dashboardData.procedures_by_condition[condition] || [];
  byId("procedure").innerHTML = options.map((value) => `<option>${escapeHtml(value)}</option>`).join("");
}

function renderProbabilityList(targetId, probabilities) {
  const entries = Object.entries(probabilities || {}).sort((a, b) => b[1] - a[1]);
  byId(targetId).innerHTML = entries.map(([label, value]) => `
    <div class="prob-row"><span>${escapeHtml(label)}</span><div class="prob-bar"><i style="width:${Math.max(0, Math.min(100, value * 100))}%"></i></div><b>${percent(value)}</b></div>`).join("");
}

async function loadSummary() {
  try {
    const response = await fetch("/api/summary", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error("The dashboard summary could not be loaded.");
    dashboardData = await response.json();
    renderOverview(dashboardData);
    if (location.hash === "#insights") setView("insights");
    if (location.hash === "#prediction") setView("prediction");
  } catch (error) {
    showToast(error.message || "Could not connect to the dashboard API.");
  }
}

document.querySelectorAll("[data-view]").forEach((item) => item.addEventListener("click", () => setView(item.dataset.view)));
document.querySelectorAll("[data-open-view]").forEach((item) => item.addEventListener("click", () => setView(item.dataset.openView)));
document.querySelector(".notice-close")?.addEventListener("click", (event) => event.currentTarget.closest(".notice").remove());
byId("condition").addEventListener("change", updateProcedureOptions);
byId("age").addEventListener("input", (event) => { byId("age-value").textContent = event.currentTarget.value; });

byId("prediction-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = byId("predict-button");
  button.disabled = true;
  button.textContent = "Calculating estimates…";
  byId("result-status").textContent = "RUNNING";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        age: Number(byId("age").value),
        gender: byId("gender").value,
        condition: byId("condition").value,
        procedure: byId("procedure").value,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Prediction request failed.");
    const results = payload.predictions;
    byId("pred-readmission").textContent = results.Readmission.prediction;
    byId("pred-outcome").textContent = results.Outcome.prediction;
    byId("pred-cost").textContent = number(results.Cost.prediction);
    byId("pred-stay").textContent = number(results.Length_of_Stay.prediction, 1);
    byId("pred-satisfaction").textContent = `${number(results.Satisfaction.prediction)} / 5`;
    renderProbabilityList("prob-readmission", results.Readmission.probabilities);
    renderProbabilityList("prob-outcome", results.Outcome.probabilities);
    renderProbabilityList("prob-satisfaction", results.Satisfaction.probabilities);
    byId("results-placeholder").classList.add("hidden");
    byId("results-content").classList.remove("hidden");
    byId("result-status").textContent = "ESTIMATE READY";
  } catch (error) {
    byId("result-status").textContent = "UNAVAILABLE";
    showToast(error.message || "The model could not return an estimate.");
  } finally {
    button.disabled = false;
    button.innerHTML = 'Generate estimates <span>→</span>';
  }
});

byId("los-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = byId("los-button");
  const result = byId("los-result");
  button.disabled = true;
  button.textContent = "Calculating estimate…";
  try {
    const response = await fetch("/api/predict/length-of-stay", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        age: Number(byId("los-age").value),
        gender: byId("los-gender").value,
        primary_diagnosis: byId("los-diagnosis").value,
        comorbidities_count: Number(byId("los-comorbidities").value),
        surgery_flag: Number(byId("los-surgery").value),
        insurance_type: byId("los-insurance").value,
        social_support: Number(byId("los-support").value),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Length-of-stay request failed.");
    result.innerHTML = `<span class="placeholder-icon">⌁</span><span class="los-value">${number(payload.prediction_days, 1)} days</span><strong>Estimated length of stay</strong><span>Admission-time linear regression</span>`;
  } catch (error) {
    showToast(error.message || "The length-of-stay model could not return an estimate.");
  } finally {
    button.disabled = false;
    button.innerHTML = 'Estimate length of stay <span>→</span>';
  }
});

loadSummary();

fetch("/api/model-info").then((response) => response.ok ? response.json() : null).then((info) => {
  if (!info) return;
  const holdout = info.holdout;
  byId("model-metrics").textContent = `Holdout MAE ${number(holdout.mae_days, 2)} days · RMSE ${number(holdout.rmse_days, 2)} days · R² ${number(holdout.r2, 2)} · 5-fold CV MAE ${number(info.five_fold_cv_mae_days_mean, 2)} ± ${number(info.five_fold_cv_mae_days_std, 2)} days. Educational sample only.`;
}).catch(() => {});
