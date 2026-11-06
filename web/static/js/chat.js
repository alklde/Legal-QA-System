/* ============================================================
   法律知识图谱智能问答系统 - 前端交互逻辑（SSE 流式 + SPA）
   ============================================================ */

let sessionId = localStorage.getItem("legal_kg_session_id") || null;
let currentController = null;  // AbortController for stop

marked.setOptions({ breaks: true, gfm: true });

document.addEventListener("DOMContentLoaded", function () {
    const input = document.getElementById("chat-input");
    input.addEventListener("input", autoResize);
    restoreHistory();
    updateSessionBadge();
});

/* ==================== 视图切换 ==================== */

function switchView(view) {
    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    document.getElementById(`view-${view}`).classList.add("active");
    document.querySelectorAll(".nav-item").forEach(n => {
        n.classList.toggle("active", n.dataset.view === view);
    });
    if (view === "stats") loadStats();
    if (view === "health") loadHealth();
}

/* ==================== 统计/健康 ==================== */

async function loadStats() {
    const el = document.getElementById("stats-content");
    el.innerHTML = '<div class="loading-placeholder">加载中...</div>';
    try {
        const res = await fetch("/api/kg/stats");
        const data = await res.json();
        const nc = data.node_counts || {};
        const rc = data.relation_counts || {};
        const top = data.top_accusations || [];
        const nodeCards = [
            { label: "案例", value: nc.Case || 0, icon: "📁" },
            { label: "被告人", value: nc.Criminal || 0, icon: "👤" },
            { label: "罪名", value: nc.Accusation || 0, icon: "🏷️" },
            { label: "法条", value: nc.Article || 0, icon: "📖" },
            { label: "判决", value: nc.Sentence || 0, icon: "⚖️" },
        ];
        const relCards = [
            { label: "INVOLVES", value: rc.INVOLVES || 0 },
            { label: "SENTENCED_TO", value: rc.SENTENCED_TO || 0 },
            { label: "CITES", value: rc.CITES || 0 },
            { label: "CHARGED_WITH", value: rc.CHARGED_WITH || 0 },
            { label: "BASED_ON", value: rc.BASED_ON || 0 },
        ];
        const topRows = top.map((t, i) =>
            `<tr><td>${i + 1}</td><td>${escHtml(t.name)}</td><td>${t.count.toLocaleString()}</td></tr>`
        ).join("");

        el.innerHTML = `
            <h3>节点统计</h3>
            <div class="stat-cards">${nodeCards.map(c => `
                <div class="stat-card">
                    <div class="stat-icon">${c.icon}</div>
                    <div class="stat-value">${c.value.toLocaleString()}</div>
                    <div class="stat-label">${c.label}</div>
                </div>`).join("")}
            </div>
            <h3>关系统计</h3>
            <div class="stat-cards">${relCards.map(c => `
                <div class="stat-card stat-card-rel">
                    <div class="stat-value">${c.value.toLocaleString()}</div>
                    <div class="stat-label">${c.label}</div>
                </div>`).join("")}
            </div>
            <h3>高频罪名 TOP 10</h3>
            <table class="stat-table">
                <thead><tr><th>#</th><th>罪名</th><th>案件数</th></tr></thead>
                <tbody>${topRows}</tbody>
            </table>`;
    } catch (e) {
        el.innerHTML = `<div class="error-banner">加载失败：${escHtml(e.message)} <button onclick="loadStats()">重试</button></div>`;
    }
}

async function loadHealth() {
    const el = document.getElementById("health-content");
    el.innerHTML = '<div class="loading-placeholder">加载中...</div>';
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        const statusColor = data.status === "healthy" ? "var(--accent)" : "#f59e0b";
        el.innerHTML = `
            <div class="health-card" style="border-left: 4px solid ${statusColor}">
                <h3>系统状态：<span style="color:${statusColor}">${escHtml(data.status)}</span></h3>
                <table class="info-table">
                    <tr><td>版本</td><td>${escHtml(data.version)}</td></tr>
                    <tr><td>LLM 模型</td><td>${escHtml(data.llm?.model || "-")}</td></tr>
                    <tr><td>LLM 端点</td><td>${escHtml(data.llm?.provider || "-")}</td></tr>
                    <tr><td>Neo4j 状态</td><td><span style="color:${data.neo4j?.status === "connected" ? "var(--accent)" : "#ef4444"}">${escHtml(data.neo4j?.status || "-")}</span></td></tr>
                    <tr><td>Neo4j 地址</td><td>${escHtml(data.neo4j?.uri || "-")}</td></tr>
                    <tr><td>内存占用</td><td>${data.memory_mb || 0} MB</td></tr>
                </table>
            </div>`;
    } catch (e) {
        el.innerHTML = `<div class="error-banner">加载失败：${escHtml(e.message)} <button onclick="loadHealth()">重试</button></div>`;
    }
}

/* ==================== 聊天 ==================== */

function autoResize() {
    const input = document.getElementById("chat-input");
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
}

function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function useSuggestion(text) {
    const input = document.getElementById("chat-input");
    input.value = text;
    autoResize();
    sendMessage();
}

/* ---------------- 消息构建 ---------------- */

function createMessageBubble(role) {
    const container = document.getElementById("chat-messages");
    const div = document.createElement("div");
    div.className = `message ${role}`;
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "👤" : "⚖️";
    const content = document.createElement("div");
    content.className = "message-content";
    div.appendChild(avatar);
    div.appendChild(content);
    container.appendChild(div);
    scrollToBottom();
    return content;
}

function renderMarkdown(text) {
    if (!text) return "";
    return DOMPurify.sanitize(marked.parse(text));
}

function escHtml(s) {
    if (!s) return "";
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/* ---------------- 思考块 ---------------- */

function createThinkingBlock() {
    const wrap = document.createElement("div");
    wrap.className = "thinking";
    const header = document.createElement("div");
    header.className = "thinking-header";
    const icon = document.createElement("span");
    icon.className = "thinking-icon";
    icon.innerHTML = "🧠";
    const label = document.createElement("span");
    label.textContent = "思考中…";
    const chevron = document.createElement("span");
    chevron.className = "thinking-chevron";
    chevron.innerHTML = "▾";
    header.appendChild(icon);
    header.appendChild(label);
    header.appendChild(chevron);
    const body = document.createElement("div");
    body.className = "thinking-body";
    header.addEventListener("click", () => {
        const isOpen = wrap.classList.toggle("thinking-open");
        body.style.display = isOpen ? "block" : "none";
        chevron.innerHTML = isOpen ? "▾" : "▸";
    });
    wrap.appendChild(header);
    wrap.appendChild(body);
    return wrap;
}

/* ---------------- 会话持久化 ---------------- */

function saveHistory(userText, answerText) {
    const key = "legal_kg_history";
    const history = JSON.parse(localStorage.getItem(key) || "[]");
    history.push({ role: "user", text: userText, ts: Date.now() });
    history.push({ role: "assistant", text: answerText, ts: Date.now() });
    // keep last 50 messages
    if (history.length > 50) history.splice(0, history.length - 50);
    localStorage.setItem(key, JSON.stringify(history));
}

function restoreHistory() {
    const key = "legal_kg_history";
    const history = JSON.parse(localStorage.getItem(key) || "[]");
    if (!history.length) return;
    // only restore last 10 pairs (20 msgs)
    const recent = history.slice(-20);
    recent.forEach(m => {
        if (m.role === "user") {
            const content = createMessageBubble("user");
            content.innerHTML = renderMarkdown(m.text);
        } else if (m.role === "assistant") {
            const content = createMessageBubble("assistant");
            content.innerHTML = renderMarkdown(m.text);
        }
    });
    scrollToBottom();
}

function updateSessionBadge() {
    const badge = document.getElementById("session-id");
    if (sessionId) badge.textContent = `会话: ${sessionId.replace("web-", "").slice(0, 8)}`;
}

/* ---------------- 停止生成 ---------------- */

function stopGeneration() {
    if (currentController) {
        currentController.abort();
        currentController = null;
    }
    document.getElementById("send-btn").classList.remove("hidden");
    document.getElementById("stop-btn").classList.add("hidden");
    document.getElementById("chat-input").disabled = false;
}

/* ---------------- 主发送流程（SSE） ---------------- */

async function sendMessage() {
    const input = document.getElementById("chat-input");
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    autoResize();

    const sendBtn = document.getElementById("send-btn");
    const stopBtn = document.getElementById("stop-btn");
    sendBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    input.disabled = true;

    // user bubble
    const userContent = createMessageBubble("user");
    userContent.innerHTML = renderMarkdown(message);

    // assistant bubble + thinking
    const assistantContent = createMessageBubble("assistant");
    const thinking = createThinkingBlock();
    const thinkingBody = thinking.querySelector(".thinking-body");
    assistantContent.appendChild(thinking);
    const answerDiv = document.createElement("div");
    answerDiv.className = "answer-text";
    assistantContent.appendChild(answerDiv);
    const statusDiv = document.createElement("div");
    statusDiv.className = "answer-status";
    assistantContent.appendChild(statusDiv);

    if (!sessionId) sessionId = `web-${Date.now().toString(16)}${Math.floor(Math.random() * 0xffff).toString(16)}`;
    localStorage.setItem("legal_kg_session_id", sessionId);
    updateSessionBadge();

    let answerBuf = "", thinkingBuf = "";
    let answerStarted = false;

    currentController = new AbortController();
    try {
        const resp = await fetch("/api/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, session_id: sessionId }),
            signal: currentController.signal,
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            let idx;
            while ((idx = buffer.indexOf("\n\n")) !== -1) {
                const block = buffer.slice(0, idx);
                buffer = buffer.slice(idx + 2);
                const lines = block.split("\n");
                let event = "message", data = "";
                for (const line of lines) {
                    if (line.startsWith("event:")) event = line.slice(6).trim();
                    else if (line.startsWith("data:")) data += line.slice(5).trim();
                }
                if (!data) continue;
                let payload;
                try { payload = JSON.parse(data); } catch (e) { continue; }

                if (event === "status") {
                    if (payload.step === "intent") statusDiv.textContent = `🧠 意图: ${payload.intent}`;
                    else if (payload.step === "plan") statusDiv.textContent += `  → ${payload.tools?.join(", ") || "无工具"}`;
                    else if (payload.step === "tool_result") {
                        const tools = payload.sources?.map(s => s.tool).filter(Boolean) || [];
                        if (tools.length) statusDiv.textContent += `  ✓ ${tools.join(", ")}`;
                    }
                } else if (event === "thinking_chunk") {
                    if (!answerStarted) {
                        thinkingBuf += payload.text;
                        thinkingBody.innerHTML = renderMarkdown(thinkingBuf);
                    }
                } else if (event === "answer_chunk") {
                    if (!answerStarted) {
                        answerStarted = true;
                        thinking.classList.remove("thinking-open");
                        thinkingBody.style.display = "none";
                        thinking.querySelector(".thinking-chevron").innerHTML = "▸";
                        thinking.querySelector(".thinking-header span").textContent = "查看思考过程";
                    }
                    answerBuf += payload.text;
                    answerDiv.innerHTML = renderMarkdown(answerBuf);
                    scrollToBottom();
                } else if (event === "error") {
                    answerDiv.innerHTML = renderMarkdown(`😅 错误：${payload.message}`);
                } else if (event === "done") {
                    if (!answerStarted) {
                        thinking.classList.remove("thinking-open");
                        thinkingBody.style.display = "none";
                        thinking.querySelector(".thinking-chevron").innerHTML = "▸";
                        thinking.querySelector(".thinking-header span").textContent = thinkingBuf ? "查看思考过程" : "";
                    }
                    if (payload.sources?.length) {
                        const srcDiv = document.createElement("div");
                        srcDiv.style.marginTop = "8px";
                        srcDiv.innerHTML = '<div style="margin-bottom:4px;font-size:12px;color:var(--text-muted)">📎 信息来源：</div>';
                        payload.sources.forEach(s => {
                            if (s?.tool) {
                                const tag = document.createElement("span");
                                tag.className = "source-tag";
                                tag.textContent = s.tool + (s.type ? ` · ${s.type}` : "");
                                srcDiv.appendChild(tag);
                            }
                        });
                        answerDiv.appendChild(srcDiv);
                    }
                    if (typeof payload.confidence === "number") {
                        const confDiv = document.createElement("div");
                        confDiv.className = "confidence";
                        confDiv.textContent = `可信度: ${Math.round(payload.confidence * 100)}%`;
                        answerDiv.appendChild(confDiv);
                    }
                    break;
                }
            }
        }
        // save to localStorage
        if (answerBuf) saveHistory(message, answerBuf);
    } catch (err) {
        if (err.name === "AbortError") {
            if (answerBuf) answerDiv.innerHTML += '<div style="color:var(--text-muted);font-size:12px;margin-top:4px">（已停止）</div>';
            else answerDiv.innerHTML = renderMarkdown("（已停止）");
        } else {
            answerDiv.innerHTML = renderMarkdown(`😅 请求失败：${err.message}`);
        }
    } finally {
        currentController = null;
        sendBtn.classList.remove("hidden");
        stopBtn.classList.add("hidden");
        input.disabled = false;
        input.focus();
        scrollToBottom();
    }
}

/* ---------------- 清空会话 ---------------- */

function clearSession() {
    sessionId = null;
    localStorage.removeItem("legal_kg_session_id");
    localStorage.removeItem("legal_kg_history");
    const container = document.getElementById("chat-messages");
    container.innerHTML = "";
    const content = createMessageBubble("assistant");
    content.innerHTML = renderMarkdown("对话已清空。有什么可以帮您？");
    updateSessionBadge();
}

function scrollToBottom() {
    const c = document.getElementById("chat-messages");
    c.scrollTop = c.scrollHeight;
}
