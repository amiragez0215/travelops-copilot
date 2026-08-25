/*
 * 页面只调用 FastAPI 已公开的三个契约：
 *   POST /api/v1/travel/invoke
 *   GET  /api/v1/travel/runs/{thread_id}
 *   POST /api/v1/travel/decision/{thread_id}
 *
 * 页面分为两个视图：用户页只展示旅行结果；管理员页才展示运行状态和 Trace。
 * 两个视图共用 state.currentRun，避免切换页面时重新执行 Agent 或读取私有 State。
 */

const state = { currentRun: null, pendingDecision: null, toastTimer: null };

const elements = {
  userPage: document.querySelector("#user-page"),
  managerPage: document.querySelector("#manager-page"),
  userPageButton: document.querySelector("#user-page-button"),
  managerPageButton: document.querySelector("#manager-page-button"),
  planForm: document.querySelector("#plan-form"),
  message: document.querySelector("#message"),
  userId: document.querySelector("#user-id"),
  referenceDate: document.querySelector("#reference-date"),
  submitButton: document.querySelector("#submit-button"),
  exampleButton: document.querySelector("#example-button"),
  refreshButton: document.querySelector("#refresh-button"),
  userResultCard: document.querySelector("#user-result-card"),
  userResultContent: document.querySelector("#user-result-content"),
  managerResultCard: document.querySelector("#manager-result-card"),
  managerResultContent: document.querySelector("#manager-result-content"),
  threadId: document.querySelector("#thread-id"),
  runId: document.querySelector("#run-id"),
  verifierStatus: document.querySelector("#verifier-status"),
  statusCard: document.querySelector("#status-card"),
  statusTitle: document.querySelector("#status-title"),
  statusDescription: document.querySelector("#status-description"),
  connectionDot: document.querySelector("#connection-dot"),
  connectionLabel: document.querySelector("#connection-label"),
  decisionDialog: document.querySelector("#decision-dialog"),
  decisionForm: document.querySelector("#decision-form"),
  decisionEyebrow: document.querySelector("#decision-eyebrow"),
  decisionTitle: document.querySelector("#decision-title"),
  decisionDescription: document.querySelector("#decision-description"),
  reasonField: document.querySelector("#reason-field"),
  decisionReason: document.querySelector("#decision-reason"),
  decisionSubmit: document.querySelector("#decision-submit"),
  dialogClose: document.querySelector("#dialog-close"),
  dialogCancel: document.querySelector("#decision-cancel"),
  toast: document.querySelector("#toast"),
};

const API_ROOT = "/api/v1/travel";
const PERIOD_LABELS = {
  arrival: "抵达后", morning: "上午", afternoon: "下午", evening: "晚上",
  full_day: "全天", departure: "返程前",
};

/** 初始化后端连接提示；提示只位于管理员页，不干扰用户阅读旅行方案。 */
async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("health endpoint unavailable");
    const health = await response.json();
    elements.connectionDot.className = "connection-dot is-ok";
    elements.connectionLabel.textContent = health.workflow_runtime_ready
      ? "后端服务已就绪"
      : "后端已连接，运行时将按需初始化";
  } catch (_) {
    elements.connectionDot.className = "connection-dot is-error";
    elements.connectionLabel.textContent = "暂时无法连接服务";
  }
}

/** 统一解析 FastAPI 的字符串、Pydantic 数组等不同错误格式。 */
async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map((item) => item.msg || "输入不符合要求").join("；")
      : body.detail;
    throw new Error(detail || `请求失败（HTTP ${response.status}）`);
  }
  return body;
}

/**
 * 只改变展示层可见性，不会请求 API。aria-selected 与 hidden 同步更新，
 * 既支持鼠标切换，也让读屏软件知道当前正在阅读哪一个工作台。
 */
function switchPage(page) {
  const isUserPage = page === "user";
  elements.userPage.hidden = !isUserPage;
  elements.managerPage.hidden = isUserPage;
  elements.userPageButton.classList.toggle("is-active", isUserPage);
  elements.managerPageButton.classList.toggle("is-active", !isUserPage);
  elements.userPageButton.setAttribute("aria-selected", String(isUserPage));
  elements.managerPageButton.setAttribute("aria-selected", String(!isUserPage));
}

function setPlanningLoading(loading) {
  elements.submitButton.disabled = loading;
  elements.submitButton.classList.toggle("is-loading", loading);
}

function setStatus(kind, title, description) {
  elements.statusCard.className = `status-card status-${kind}`;
  elements.statusTitle.textContent = title;
  elements.statusDescription.textContent = description;
}

function showToast(message, isError = false) {
  window.clearTimeout(state.toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.toggle("is-error", isError);
  elements.toast.classList.add("is-visible");
  state.toastTimer = window.setTimeout(() => elements.toast.classList.remove("is-visible"), 4200);
}

/** API 文本、Mock 数据与模型文本都必须转义，不能直接拼进 innerHTML。 */
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function text(value, fallback = "—") {
  const normalized = String(value ?? "").trim();
  return normalized || fallback;
}

function formatCurrency(value) {
  const number = Number(value);
  return Number.isFinite(number)
    ? `¥${number.toLocaleString("zh-CN", { maximumFractionDigits: 0 })}`
    : "—";
}

function formatDate(value) {
  const raw = text(value, "");
  if (!raw) return "—";
  const date = new Date(`${raw}T00:00:00`);
  if (Number.isNaN(date.getTime())) return raw;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "long", day: "numeric", weekday: "short",
  }).format(date);
}

function listItems(items, className = "plain-list") {
  if (!Array.isArray(items) || !items.length) return "";
  return `<ul class="${className}">${items.filter(Boolean).map((item) => {
    const content = typeof item === "string" ? item : item.text;
    return `<li>${escapeHtml(content)}</li>`;
  }).join("")}</ul>`;
}

function statusCopy(run) {
  if (run.run_status === "waiting_for_decision") {
    return ["waiting", "等待用户决定", "方案已完成校验，正在等待用户确认、修改或取消。"];
  }
  if (run.run_status === "completed") {
    return ["completed", "本次流程已完成", run.final_response?.message || "旅行规划流程已到达终点。"];
  }
  if (run.run_status === "failed") {
    return ["error", "流程未完成", run.final_response?.message || "系统未能安全完成本次规划。"];
  }
  return ["loading", "Agent 正在处理", "正在等待工作流返回公开状态。"];
}

/**
 * 用户页使用 TripProposal 的明确字段，而不再把任何嵌套对象压缩成一行。
 * daily_plan 是最终用户最需要的内容，因此按“日期 → 时段活动 → 实用提醒”展开。
 */
function renderUserProposal(proposal) {
  if (!proposal || typeof proposal !== "object") return "";
  const overview = proposal.trip_overview || {};
  const flights = proposal.selected_flights || {};
  const hotel = proposal.selected_hotel || {};
  const budget = proposal.budget_summary || {};

  return `
    <section class="trip-intro">
      <p class="eyebrow">YOUR TRIP</p>
      <h2>${escapeHtml(text(overview.origin))} → ${escapeHtml(text(overview.destination))}</h2>
      <p>${escapeHtml(text(proposal.summary, "已为你整理旅行方案。"))}</p>
      ${listItems(proposal.highlights, "highlight-list")}
    </section>
    <section class="trip-section">
      <div class="section-title"><div><p class="eyebrow">TRIP OVERVIEW</p><h3>行程概览</h3></div></div>
      <div class="overview-grid">
        <div><span>出行日期</span><strong>${escapeHtml(formatDate(overview.start_date))} 至 ${escapeHtml(formatDate(overview.end_date))}</strong></div>
        <div><span>旅行时长</span><strong>${escapeHtml(text(overview.days))} 天 ${escapeHtml(text(overview.nights, "0"))} 晚</strong></div>
        <div><span>同行人数</span><strong>${escapeHtml(text(overview.people_count))} 人</strong></div>
        <div><span>入住安排</span><strong>${escapeHtml(text(overview.room_count))} 间客房</strong></div>
      </div>
    </section>
    <section class="trip-section">
      <div class="section-title"><div><p class="eyebrow">TRANSPORT</p><h3>往返航班</h3></div></div>
      <div class="travel-card-grid">${renderFlightCard(flights.outbound, "去程")}${renderFlightCard(flights.return_flight, "返程")}</div>
    </section>
    <section class="trip-section">
      <div class="section-title"><div><p class="eyebrow">STAY</p><h3>住宿安排</h3></div></div>
      ${renderHotelCard(hotel, proposal.hotel_context)}
    </section>
    <section class="trip-section">
      <div class="section-title"><div><p class="eyebrow">DAY BY DAY</p><h3>每天怎么玩</h3></div><span class="section-count">${Array.isArray(proposal.daily_plan) ? proposal.daily_plan.length : 0} 天</span></div>
      <div class="itinerary-list">${renderDailyPlan(proposal.daily_plan)}</div>
    </section>
    <section class="trip-section">
      <div class="section-title"><div><p class="eyebrow">BUDGET</p><h3>预算安排</h3></div></div>
      ${renderBudgetCard(budget)}
    </section>
    ${renderPreparation(proposal)}
  `;
}

function renderFlightCard(flight, label) {
  if (!flight || typeof flight !== "object") {
    return `<div class="travel-fact-card"><span>${escapeHtml(label)}</span><p>暂未获得航班信息。</p></div>`;
  }
  const time = [flight.depart_time, flight.arrive_time].filter(Boolean).join(" → ") || "时间待确认";
  const route = [flight.departure_airport || flight.departure_city, flight.arrival_airport || flight.arrival_city].filter(Boolean).join(" → ") || "航线待确认";
  return `<article class="travel-fact-card">
    <div class="fact-card-heading"><span>${escapeHtml(label)}</span><strong>${escapeHtml(text(flight.flight_no, "航班待确认"))}</strong></div>
    <p class="flight-time">${escapeHtml(time)}</p>
    <p>${escapeHtml(route)} · ${escapeHtml(text(flight.airline, "承运航司待确认"))}</p>
    <div class="fact-card-footer"><span>${escapeHtml(text(flight.cabin, "经济舱"))}${flight.is_direct ? " · 直飞" : " · 含中转"}</span><strong>${escapeHtml(formatCurrency(flight.total_price || flight.price))}</strong></div>
  </article>`;
}

function renderHotelCard(hotel, hotelContext) {
  if (!hotel || typeof hotel !== "object") return "<div class='travel-fact-card'><p>暂未获得酒店信息。</p></div>";
  const context = hotelContext && typeof hotelContext === "object" ? hotelContext : {};
  const subway = hotel.near_subway ? `距地铁约 ${text(hotel.distance_to_subway_meters)} 米` : "地铁距离请出行前确认";
  return `<article class="hotel-card">
    <div><p class="hotel-location">${escapeHtml([hotel.city, hotel.district].filter(Boolean).join(" · ") || "酒店")}</p><h4>${escapeHtml(text(hotel.name, "推荐酒店"))}</h4><p>${escapeHtml(text(hotel.address, "地址待确认"))}</p></div>
    <div class="hotel-price"><strong>${escapeHtml(formatCurrency(hotel.estimated_total_price))}</strong><span>${escapeHtml(text(hotel.planned_nights, "0"))} 晚合计</span></div>
    <div class="hotel-meta"><span>评分 ${escapeHtml(text(hotel.rating))}</span><span>${escapeHtml(subway)}</span>${hotel.quiet_score ? `<span>安静度 ${escapeHtml(text(hotel.quiet_score))}</span>` : ""}</div>
    ${context.summary ? `<p class="hotel-context">${escapeHtml(context.summary)}</p>` : ""}
    ${listItems(context.cautions, "notice-list")}
  </article>`;
}

function renderDailyPlan(days) {
  if (!Array.isArray(days) || !days.length) return "<p class='empty-copy'>当前方案没有返回逐日行程。</p>";
  return days.map((day, index) => {
    const weather = day.weather || {};
    const temperature = [weather.temperature_low, weather.temperature_high]
      .filter((value) => value !== null && value !== undefined).join("–");
    const weatherText = [weather.condition, temperature ? `${temperature}℃` : ""].filter(Boolean).join(" · ");
    const activities = Array.isArray(day.activities) ? day.activities : [];
    return `<article class="day-card">
      <div class="day-rail"><span>DAY</span><strong>${escapeHtml(String(day.day_index || index + 1).padStart(2, "0"))}</strong></div>
      <div class="day-content">
        <div class="day-heading"><div><p>${escapeHtml(formatDate(day.date))}</p><h4>${escapeHtml(text(day.theme, "当天行程"))}</h4></div>${weatherText ? `<span class="weather-pill">${escapeHtml(weatherText)}</span>` : ""}</div>
        <div class="activity-list">${activities.map((activity) => renderActivity(activity)).join("")}</div>
        ${day.weather_adjustment ? `<p class="weather-adjustment"><strong>天气安排：</strong>${escapeHtml(day.weather_adjustment)}</p>` : ""}
        ${listItems(day.day_notes, "day-note-list")}
      </div>
    </article>`;
  }).join("");
}

function renderActivity(activity) {
  if (!activity || typeof activity !== "object") return "";
  const notes = Array.isArray(activity.practical_notes) && activity.practical_notes.length
    ? `<p class="activity-note">${escapeHtml(activity.practical_notes.join("；"))}</p>` : "";
  return `<article class="activity-item"><span class="period-label">${escapeHtml(PERIOD_LABELS[activity.period] || text(activity.period))}</span><div><h5>${escapeHtml(text(activity.title, "活动安排"))}</h5><p>${escapeHtml(text(activity.description, ""))}</p>${notes}</div></article>`;
}

function renderBudgetCard(budget) {
  const knownCosts = budget && typeof budget.known_costs === "object" ? budget.known_costs : {};
  return `<div class="budget-card">
    <div><span>总预算</span><strong>${escapeHtml(budget.total_budget === null || budget.total_budget === undefined ? "未设置" : formatCurrency(budget.total_budget))}</strong></div>
    <div><span>航班与酒店已知费用</span><strong>${escapeHtml(formatCurrency(knownCosts.total || knownCosts.known_subtotal || knownCosts.subtotal))}</strong></div>
    <div><span>可灵活安排的预算</span><strong>${escapeHtml(formatCurrency(budget.remaining_budget))}</strong></div>
    ${budget.note ? `<p>${escapeHtml(budget.note)}</p>` : ""}
  </div>`;
}

function renderPreparation(proposal) {
  const hasPacking = Array.isArray(proposal.packing_tips) && proposal.packing_tips.length;
  const hasSafety = Array.isArray(proposal.safety_notes) && proposal.safety_notes.length;
  const hasPreferences = Array.isArray(proposal.preference_alignment) && proposal.preference_alignment.length;
  const hasLimits = Array.isArray(proposal.limitations) && proposal.limitations.length;
  if (!hasPacking && !hasSafety && !hasPreferences && !hasLimits) return "";
  return `<section class="trip-section preparation-section"><div class="section-title"><div><p class="eyebrow">BEFORE YOU GO</p><h3>出行提示</h3></div></div><div class="preparation-grid">
    ${hasPacking ? `<article><h4>行前准备</h4>${listItems(proposal.packing_tips, "notice-list")}</article>` : ""}
    ${hasSafety ? `<article><h4>安全提醒</h4>${listItems(proposal.safety_notes, "notice-list")}</article>` : ""}
    ${hasPreferences ? `<article><h4>已照顾到的偏好</h4>${listItems(proposal.preference_alignment, "notice-list")}</article>` : ""}
    ${hasLimits ? `<article><h4>出行前请留意</h4>${listItems(proposal.limitations, "notice-list")}</article>` : ""}
  </div></section>`;
}

/** 用户在没有生成 Proposal 时，只看自然语言结果和可执行建议，不接触 issue/trace 等工程术语。 */
function renderUserFallback(run) {
  const finalResponse = run.final_response || {};
  const details = finalResponse.details || {};
  const suggestions = Array.isArray(details.suggestions) ? details.suggestions : [];
  return `<section class="user-message-state"><p class="eyebrow">TRIP UPDATE</p><h2>${escapeHtml(run.run_status === "failed" ? "暂时无法完成方案" : "本次旅行规划结果")}</h2><p>${escapeHtml(text(finalResponse.message || run.interrupt?.message, "暂未获得可展示的旅行方案。"))}</p>${suggestions.length ? `<div><h3>你可以尝试</h3>${listItems(suggestions, "notice-list")}</div>` : ""}</section>`;
}

function renderDecisionBar(interrupt) {
  const actions = Array.isArray(interrupt.available_actions) ? interrupt.available_actions : [];
  const approveAction = actions.find((action) => action.decision === "approve");
  const cancelAction = actions.find((action) => action.decision === "cancel");
  const changeAction = actions.find((action) => action.decision === "request_changes");

  /*
   * “同意”和“取消”是明确的、低频的状态决定，所以使用独立按钮；
   * “修改”本质上是一次自然语言补充，使用常驻小型 Composer 更符合用户
   * 连续对话的心智模型，也避免了先点“修改”再弹窗输入的两步操作。
   */
  return `<section class="decision-studio">
    <div class="decision-topline">
      <div><p class="eyebrow">NEXT STEP</p><h3>这份计划怎么样？</h3><p>${escapeHtml(interrupt.message || "可以确认方案，或者告诉我想调整什么。")}</p></div>
      <div class="approval-actions">
        ${approveAction ? `<button class="approval-button approval-button-primary" data-decision="approve" type="button"><span aria-hidden="true">✓</span>${escapeHtml(approveAction.label)}</button>` : ""}
        ${cancelAction ? `<button class="approval-button approval-button-cancel" data-decision="cancel" type="button"><span aria-hidden="true">×</span>${escapeHtml(cancelAction.label)}</button>` : ""}
      </div>
    </div>
    ${changeAction ? `<form id="inline-change-form" class="revision-composer">
      <textarea id="inline-change-request" maxlength="4000" rows="2" placeholder="想调整什么？例如：第二天下午留出休息时间，酒店换到地铁站旁。"></textarea>
      <button id="inline-change-submit" class="composer-send" type="submit" aria-label="发送修改要求" title="发送修改要求"><span aria-hidden="true">↑</span></button>
    </form>
    <p class="composer-hint">直接描述修改要求；按 Ctrl / ⌘ + Enter 也可发送。</p>` : ""}
  </section>`;
}

function renderUserRun(run) {
  elements.userResultCard.classList.remove("is-empty");
  elements.userResultContent.hidden = false;
  const proposalContent = run.proposal ? renderUserProposal(run.proposal) : renderUserFallback(run);
  const decisionBar = run.run_status === "waiting_for_decision" && run.interrupt ? renderDecisionBar(run.interrupt) : "";
  const completionMessage = run.run_status === "completed" && run.final_response?.message
    ? `<p class="completion-message">${escapeHtml(run.final_response.message)}</p>` : "";
  elements.userResultContent.innerHTML = `${completionMessage}${proposalContent}${decisionBar}`;
  document.querySelectorAll("[data-decision]").forEach((button) => button.addEventListener("click", () => openDecisionDialog(button.dataset.decision)));

  // Composer 在每次 renderRun 后重新创建，因此在这里绑定，避免保留旧 Proposal 的 action_id。
  const inlineChangeForm = document.querySelector("#inline-change-form");
  const inlineChangeRequest = document.querySelector("#inline-change-request");
  inlineChangeForm?.addEventListener("submit", submitInlineChange);
  inlineChangeRequest?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      inlineChangeForm.requestSubmit();
    }
  });
}

/** 管理员页只展示后端已经脱敏、白名单化的字段，便于演示状态与诊断链路。 */
function renderManagerRun(run) {
  const [kind, title, description] = statusCopy(run);
  setStatus(kind, title, description);
  elements.threadId.textContent = run.thread_id || "—";
  elements.runId.textContent = run.run_id || "—";
  elements.verifierStatus.textContent = run.verification?.passed === true ? "通过" : run.verification?.passed === false ? "未通过" : "—";
  elements.refreshButton.disabled = !run.thread_id;
  elements.managerResultCard.classList.remove("is-empty");
  elements.managerResultContent.hidden = false;
  const verification = run.verification || {};
  elements.managerResultContent.innerHTML = `
    <div class="result-topline"><div><p class="eyebrow">PUBLIC RUN RESPONSE</p><h2>运行详情</h2></div><span class="status-badge is-${escapeHtml(run.run_status || "unknown")}">${escapeHtml(run.run_status || "unknown")}</span></div>
    <div class="admin-summary-grid">
      ${adminMetric("Workflow", run.workflow)}${adminMetric("Itinerary", run.itinerary_status)}
      ${adminMetric("Next action", verification.next_action)}${adminMetric("Verifier", verification.passed === true ? "passed" : verification.passed === false ? "failed" : "—")}
      ${adminMetric("Warnings", verification.warning_count)}${adminMetric("Errors", verification.error_count)}
      ${adminMetric("Proposal", run.proposal?.proposal_id)}${adminMetric("Checkpoint", run.checkpoint_id)}
    </div>
    ${renderAdminDetails(run.final_response?.details)}
    ${renderTrace(run.trace)}
  `;
}

function adminMetric(label, value) {
  return `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(text(value))}</strong></div>`;
}

function renderAdminDetails(details) {
  if (!details || typeof details !== "object" || !Object.keys(details).length) return "";
  const rows = Object.entries(details).map(([key, value]) => {
    const rendered = Array.isArray(value)
      ? value.map((item) => typeof item === "object" ? JSON.stringify(item) : String(item)).join("；")
      : typeof value === "object" && value !== null ? JSON.stringify(value) : String(value ?? "—");
    return `<div class="admin-detail-row"><span>${escapeHtml(key.replaceAll("_", " "))}</span><p>${escapeHtml(rendered)}</p></div>`;
  }).join("");
  return `<section class="admin-details"><h3>后端公开返回详情</h3>${rows}</section>`;
}

function renderTrace(trace) {
  if (!Array.isArray(trace) || !trace.length) return "";
  return `<details class="trace-details"><summary>查看公开 Trace 摘要（${trace.length} 条）</summary><div class="trace-list">${trace.map((item) => `<article class="trace-row"><div><strong>${escapeHtml(item.node_name || item.tool_name || "node")}</strong><span>${escapeHtml(item.status || "—")} · ${escapeHtml(item.latency_ms ?? "—")} ms</span></div>${item.input_summary ? `<p>输入：${escapeHtml(item.input_summary)}</p>` : ""}${item.output_summary ? `<p>输出：${escapeHtml(item.output_summary)}</p>` : ""}</article>`).join("")}</div></details>`;
}

/** 一次 API 响应同时驱动两个页面；用户视图永远不插入 Trace、Thread 或 Verifier 字段。 */
function renderRun(run) {
  state.currentRun = run;
  renderUserRun(run);
  renderManagerRun(run);
}

function openDecisionDialog(decision) {
  const interrupt = state.currentRun?.interrupt;
  if (!interrupt) return;
  const action = (interrupt.available_actions || []).find((item) => item.decision === decision);
  if (!action) return;
  state.pendingDecision = { decision, action };
  elements.decisionEyebrow.textContent = decision.toUpperCase();
  elements.decisionTitle.textContent = action.label;
  elements.decisionDescription.textContent = action.description || interrupt.message || "";
  // 修改请求不会进入这个确认弹窗：它已经在用户页的 Composer 中直接提交。
  elements.reasonField.hidden = false;
  elements.decisionReason.value = "";
  elements.decisionSubmit.textContent = action.label;
  elements.decisionDialog.showModal();
  window.setTimeout(() => elements.decisionReason.focus(), 0);
}

function closeDecisionDialog() {
  state.pendingDecision = null;
  elements.decisionDialog.close();
}

async function submitDecision(event) {
  event.preventDefault();
  const run = state.currentRun;
  const pending = state.pendingDecision;
  if (!run?.thread_id || !run.interrupt || !pending) return;
  const payload = { decision: pending.decision, proposal_id: run.interrupt.proposal_id, action_id: run.interrupt.action_id };
  if (elements.decisionReason.value.trim()) payload.reason = elements.decisionReason.value.trim();
  elements.decisionSubmit.disabled = true;
  try {
    const nextRun = await requestJson(`${API_ROOT}/decision/${encodeURIComponent(run.thread_id)}`, { method: "POST", body: JSON.stringify(payload) });
    closeDecisionDialog();
    renderRun(nextRun);
    showToast("你的决定已提交。", false);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.decisionSubmit.disabled = false;
  }
}

/**
 * 内联 Composer 直接复用 Decision API 的 request_changes 合同。
 * 它只发送当前 interrupt 提供的 action_id / proposal_id，不能让用户自行
 * 构造或恢复其他会话的审批动作。
 */
async function submitInlineChange(event) {
  event.preventDefault();
  const run = state.currentRun;
  const input = document.querySelector("#inline-change-request");
  const submitButton = document.querySelector("#inline-change-submit");
  const action = (run?.interrupt?.available_actions || []).find(
    (item) => item.decision === "request_changes",
  );
  const changeRequest = input?.value.trim() || "";

  if (!run?.thread_id || !run.interrupt || !action || !input) return;
  if (!changeRequest) {
    showToast("请先告诉我你想怎样调整这份计划。", true);
    input.focus();
    return;
  }

  submitButton.disabled = true;
  try {
    const nextRun = await requestJson(
      `${API_ROOT}/decision/${encodeURIComponent(run.thread_id)}`,
      {
        method: "POST",
        body: JSON.stringify({
          decision: "request_changes",
          proposal_id: run.interrupt.proposal_id,
          action_id: run.interrupt.action_id,
          change_request: changeRequest,
        }),
      },
    );
    renderRun(nextRun);
    showToast("收到，我正在根据你的要求调整旅行方案。", false);
  } catch (error) {
    showToast(error.message, true);
    submitButton.disabled = false;
  }
}

async function submitPlan(event) {
  event.preventDefault();
  const message = elements.message.value.trim();
  if (!message) {
    elements.message.focus();
    return;
  }
  setPlanningLoading(true);
  try {
    const payload = { user_id: elements.userId.value.trim(), message };
    if (elements.referenceDate.value) payload.reference_date = elements.referenceDate.value;
    const run = await requestJson(`${API_ROOT}/invoke`, { method: "POST", body: JSON.stringify(payload) });
    renderRun(run);
    showToast(run.run_status === "waiting_for_decision" ? "方案已生成，请查看每天的行程安排。" : "旅行规划已返回结果。", false);
  } catch (error) {
    renderRun({ run_status: "failed", final_response: { message: error.message } });
    showToast(error.message, true);
  } finally {
    setPlanningLoading(false);
  }
}

async function refreshCurrentRun() {
  const threadId = state.currentRun?.thread_id;
  if (!threadId) return;
  elements.refreshButton.disabled = true;
  try {
    const run = await requestJson(`${API_ROOT}/runs/${encodeURIComponent(threadId)}?include_trace=true`);
    renderRun(run);
    showToast("已刷新当前工作流状态。", false);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.refreshButton.disabled = false;
  }
}

elements.userPageButton.addEventListener("click", () => switchPage("user"));
elements.managerPageButton.addEventListener("click", () => {
  switchPage("manager");

  // POST /invoke 的同步响应默认不携带 Trace。管理员主动打开面板时再读取
  // 同一 thread 的 include_trace=true 公开摘要，既能看到细节，也不会重跑 Agent。
  if (state.currentRun?.thread_id) refreshCurrentRun();
});
elements.planForm.addEventListener("submit", submitPlan);
elements.refreshButton.addEventListener("click", refreshCurrentRun);
elements.decisionForm.addEventListener("submit", submitDecision);
elements.dialogClose.addEventListener("click", closeDecisionDialog);
elements.dialogCancel.addEventListener("click", closeDecisionDialog);
elements.exampleButton.addEventListener("click", () => {
  elements.message.value = "2026 年 7 月 2 日从杭州去成都玩三天，预算 4000，想住安静一点，行程不要太赶。";
  elements.referenceDate.value = "2026-07-01";
  elements.message.focus();
});

checkHealth();
