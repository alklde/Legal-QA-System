"""API路由定义"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse

from ..agent.core import LegalAgent, AgentResponse
from ..kg.query import KGQueryEngine, KGQueryError
from .schemas import (
    ChatRequest,
    ChatResponse,
    KGQueryRequest,
    CaseSearchRequest,
    StatsResponse,
    HealthResponse,
)
from .middleware import rate_limiter, metrics_collector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


def get_agent(request: Request) -> LegalAgent:
    return request.app.state.agent


def get_kg_engine(request: Request) -> KGQueryEngine:
    return request.app.state.kg_engine


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent: LegalAgent = Depends(get_agent),
    raw_request: Request = None,
):
    start = time.time()
    client_ip = raw_request.client.host if raw_request else "unknown"

    allowed, retry_after = rate_limiter.check(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={"message": "请求过于频繁，请稍后再试", "retry_after": retry_after},
        )

    try:
        result = await agent.run(request.message, request.session_id)
        latency = time.time() - start
        metrics_collector.record_request("/api/chat", latency, 200)

        return ChatResponse(**{
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
            "session_id": result.session_id,
            "metadata": result.metadata,
            "error": result.error,
        })
    except Exception as e:
        logger.exception(f"聊天接口异常: {e}")
        metrics_collector.record_request("/api/chat", time.time() - start, 500)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.post("/kg/query")
async def query_kg(
    request: KGQueryRequest,
    kg_engine: KGQueryEngine = Depends(get_kg_engine),
    raw_request: Request = None,
):
    start = time.time()
    client_ip = raw_request.client.host if raw_request else "unknown"

    allowed, retry_after = rate_limiter.check(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail="请求过于频繁")

    try:
        params = request.params or {}

        if request.query_type == "accusation":
            result = kg_engine.query_by_accusation(
                params.get("name", ""),
                include_cases=params.get("include_cases", False),
                case_limit=params.get("case_limit", 10),
            )
        elif request.query_type == "article":
            result = kg_engine.query_article(
                params.get("article_id"),
                include_cases=params.get("include_cases", False),
                case_limit=params.get("case_limit", 10),
            )
        elif request.query_type == "stats":
            result = kg_engine.get_stats()
        else:
            raise HTTPException(status_code=400, detail=f"不支持的查询类型: {request.query_type}")

        metrics_collector.record_request("/api/kg/query", time.time() - start, 200)

        if result is None:
            return {"type": request.query_type, "result": None, "found": False}
        return {"type": request.query_type, "result": result, "found": True}
    except KGQueryError as e:
        logger.error(f"知识图谱查询异常: {e}")
        metrics_collector.record_request("/api/kg/query", time.time() - start, 500)
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"知识图谱查询异常: {e}")
        metrics_collector.record_request("/api/kg/query", time.time() - start, 500)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.post("/kg/search")
async def search_cases(
    request: CaseSearchRequest,
    kg_engine: KGQueryEngine = Depends(get_kg_engine),
    raw_request: Request = None,
):
    start = time.time()
    try:
        results = kg_engine.search_case_text(request.query, limit=request.limit)
        metrics_collector.record_request("/api/kg/search", time.time() - start, 200)
        return {"total": len(results), "results": results}
    except Exception as e:
        metrics_collector.record_request("/api/kg/search", time.time() - start, 500)
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@router.get("/kg/stats", response_model=StatsResponse)
async def get_stats(kg_engine: KGQueryEngine = Depends(get_kg_engine)):
    try:
        return kg_engine.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


@router.get("/articles/{article_id}")
async def get_article(
    article_id: int,
    kg_engine: KGQueryEngine = Depends(get_kg_engine),
):
    try:
        result = kg_engine.query_article(article_id, include_cases=True, case_limit=5)
        if result is None:
            raise HTTPException(status_code=404, detail=f"法条 {article_id} 不存在")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health():
    import os
    import resource

    memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "api": {"status": "healthy"},
            "neo4j": {"status": "healthy" if not os.getenv("NEO4J_URI") else "connected"},
            "llm": {"status": "healthy"},
        },
        "uptime_seconds": time.time() - metrics_collector.start_time,
        "memory_usage_mb": round(memory_mb, 2),
    }


@router.get("/metrics")
async def get_metrics():
    return metrics_collector.get_metrics()