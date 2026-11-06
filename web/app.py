"""Web前端服务入口（含流式聊天）"""

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

_env_file = PROJECT_ROOT / ".env"
if not _env_file.exists():
    _env_file = PROJECT_ROOT / ".env.example"
load_dotenv(_env_file, override=False)

import urllib.request
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.utils.config import load_config
from src.utils.logger import setup_logging
from src.kg.schema import LegalKGSchema
from src.kg.store import Neo4jKGStore
from src.kg.query import KGQueryEngine
from src.agent.graph import LegalLangGraphAgent
from src.agent.tools import set_kg_engine

config = load_config()
setup_logging()

app = FastAPI(
    title="法律知识图谱 - Web前端",
    version="2.0.0",
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

_agent: LegalLangGraphAgent = None
_qe: KGQueryEngine = None


def _get_qe() -> KGQueryEngine:
    global _qe
    if _qe is None:
        _get_agent()  # ensure agent/qe initialized
    return _qe


def _get_agent() -> LegalLangGraphAgent:
    global _agent, _qe
    if _agent is None:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        password = os.getenv("NEO4J_PASSWORD", "legal_kg_2026")
        driver = GraphDatabase.driver(uri, auth=("neo4j", password))
        store = Neo4jKGStore(driver, LegalKGSchema())
        qe = KGQueryEngine(store=store, schema=LegalKGSchema())
        set_kg_engine(qe)
        _qe = qe

        router = None
        if os.getenv("TOOL_FT_ROUTER", "").strip() not in ("", "0", "false", "False"):
            try:  # 可选：用项目二微调模型做 LLM 工具路由（默认关闭，规则路由）
                from src.agent.llm_router import build_default_router
                router = build_default_router()
            except Exception as e:  # 加载失败不影响服务，回退规则
                print(f"[web] TOOL_FT_ROUTER 启用失败，回退规则路由: {e}")

        _agent = LegalLangGraphAgent(config={
            "model": OPENAI_MODEL,
            "base_url": OPENAI_BASE_URL,
            "api_key": OPENAI_API_KEY,
            "max_tokens": os.getenv("OPENAI_MAX_TOKENS", "").strip() or None,
            "temperature": 0.2,
        }, router=router)
    return _agent


@app.get("/", response_class=HTMLResponse)
async def index():
    template_path = Path(__file__).parent / "templates" / "index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat_api(request: Request):
    body = await request.json()
    message = body.get("message", "").strip()
    session_id = body.get("session_id") or f"web-{uuid.uuid4().hex[:8]}"
    if not message:
        return {"error": "消息不能为空"}

    agent = _get_agent()
    result = agent.run(message, session_id=session_id)
    return result


def _build_initial_state(agent, session_id, message):
    from src.agent.nodes import classify_intent, create_plan, execute_step, reflect_and_validate

    state = {
        "user_input": message,
        "session_id": session_id,
        "messages": [],
        "history": agent._get_history(session_id),
        "intent": "",
        "plan_steps": [],
        "current_step_index": 0,
        "tool_results": [],
        "sources": [],
        "validation_passed": False,
        "validation_reason": "",
        "final_answer": "",
        "confidence": 0.0,
        "error": None,
        "attempts": 0,
        "max_attempts": agent.config.get("max_attempts", 5),
    }
    state.update(classify_intent(state))
    state.update(create_plan(state))
    state.update(execute_step(state))
    state.update(reflect_and_validate(state))
    return state


def _llm_stream_raw(prompt: str, max_tokens: Optional[int] = None):
    if not OPENAI_BASE_URL or not OPENAI_API_KEY:
        yield ("thinking", "未配置 LLM API Key，将使用规则模板回答问题。")
        return
    body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": True,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens
    body = json.dumps(body).encode()

    req = urllib.request.Request(
        OPENAI_BASE_URL + "/chat/completions",
        data=body,
        headers={
            "Authorization": "Bearer " + OPENAI_API_KEY,
            "Content-Type": "application/json",
        },
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            rc = delta.get("reasoning_content") or delta.get("reasoning")
            content = delta.get("content")
            if rc:
                yield ("thinking", rc)
            if content:
                yield ("answer", content)


def _stream_chat(message: str, session_id: str):
    """Generator for SSE chat stream."""
    agent = _get_agent()
    state = _build_initial_state(agent, session_id, message)

    intent = state.get("intent", "")
    yield _sse("status", {"step": "intent", "intent": intent})
    plan_tools = [s.get("tool") for s in state.get("plan_steps", [])]
    yield _sse("status", {"step": "plan", "tools": plan_tools})
    yield _sse("status", {"step": "tool_result", "sources": state.get("sources", [])})

    # --- chitchat short-circuit ---
    if intent == "chitchat":
        answer = "您好！我是法律智能问答助手，可以帮助您查询法律知识、罪名量刑信息、相关法条和类似案例。请问您想了解什么？"
        agent._save_turn(session_id, message, answer)
        yield _sse("answer_chunk", {"text": answer})
        yield _sse("done", {"confidence": 1.0, "sources": [], "thinking": ""})
        return

    tool_results = state.get("tool_results", [])
    success_results = [r for r in tool_results if r.get("success", False)]
    if not success_results:
        answer = "抱歉，没有找到相关信息。请尝试更具体的问题。"
        agent._save_turn(session_id, message, answer)
        yield _sse("answer_chunk", {"text": answer})
        yield _sse("done", {"confidence": 0.1, "sources": state.get("sources", []), "thinking": ""})
        return

    prompt = (
        "你是一位法律知识图谱问答助手。请基于下面从中国刑事裁判文书知识图谱检索到的结构化数据，"
        "用简洁自然的中文回答用户的问题。可以适当概括统计结果，但不要编造数据中没有的信息。"
        "如果数据不足以回答，请如实说明。请先思考，再回答。\n\n"
        f"用户问题：{message}\n"
        f"检索结果：\n{json.dumps(success_results, ensure_ascii=False)[:3000]}\n\n"
        "请直接输出回答。"
    )

    full_thinking = []
    full_answer = []
    try:
        for kind, text in _llm_stream_raw(prompt, max_tokens=agent.config.get("max_tokens")):
            if kind == "thinking":
                full_thinking.append(text)
                yield _sse("thinking_chunk", {"text": text})
            elif kind == "answer":
                full_answer.append(text)
                yield _sse("answer_chunk", {"text": text})
    except Exception as e:
        yield _sse("error", {"message": str(e)})

    final_answer = "".join(full_answer).strip()
    if final_answer:
        agent._save_turn(session_id, message, final_answer)
    from src.agent.nodes import compute_confidence

    confidence = compute_confidence(success_results)
    yield _sse("done", {
        "confidence": round(confidence, 2),
        "sources": state.get("sources", []),
        "thinking": "".join(full_thinking),
    })


@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()
    message = body.get("message", "").strip()
    session_id = body.get("session_id") or f"web-{uuid.uuid4().hex[:8]}"
    if not message:
        async def err_gen():
            yield _sse("error", {"message": "消息不能为空"})
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    def sync_gen():
        yield from _stream_chat(message, session_id)
    return StreamingResponse(sync_gen(), media_type="text/event-stream")


@app.get("/api/kg/stats")
async def kg_stats():
    qe = _get_qe()
    return qe.get_stats()


@app.get("/api/health")
async def health():
    import time as _time
    import resource
    memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    model = OPENAI_MODEL
    base_url = OPENAI_BASE_URL.split("//")[-1].split("/")[0] if OPENAI_BASE_URL else ""
    neo4j_uri = os.getenv("NEO4J_URI", "")
    neo4j_ok = False
    try:
        qe = _get_qe()
        with qe.store._session() as s:
            s.run("RETURN 1").consume()
        neo4j_ok = True
    except Exception:
        pass
    return {
        "status": "healthy" if neo4j_ok else "degraded",
        "version": "2.0.0",
        "llm": {"model": model, "provider": base_url or "未配置"},
        "neo4j": {"status": "connected" if neo4j_ok else "disconnected", "uri": neo4j_uri},
        "memory_mb": round(memory_mb, 1),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=5000, reload=True)