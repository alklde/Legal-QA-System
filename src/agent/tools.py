"""LangChain工具定义（Agent调用的工具集）"""

import json
import logging
from typing import Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_kg_engine = None
_vector_store = None


def set_kg_engine(engine):
    global _kg_engine
    _kg_engine = engine


def set_vector_store(vs):
    global _vector_store
    _vector_store = vs


@tool
def kg_query(accusation_name: str = "", article_id: str = "") -> str:
    """查询知识图谱：按罪名或法条编号查询法律信息、量刑统计和相关法条。"""
    if not _kg_engine:
        return json.dumps({"error": "知识图谱引擎未初始化"}, ensure_ascii=False)
    try:
        if accusation_name:
            data = _kg_engine.query_by_accusation(accusation_name, include_cases=False)
        elif article_id:
            data = _kg_engine.query_article(article_id, include_cases=False)
        else:
            data = _kg_engine.get_stats()
        if data is None:
            return json.dumps({"error": f"未找到相关数据: {accusation_name or article_id}"}, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def article_query(article_id: str) -> str:
    """查询刑法法条原文，传入法条编号如234、264。返回法条内容、相关罪名和相关案例数。"""
    if not _kg_engine:
        return json.dumps({"error": "知识图谱引擎未初始化"}, ensure_ascii=False)
    try:
        data = _kg_engine.query_article(article_id, include_cases=True, case_limit=5)
        if data is None:
            return json.dumps({"error": f"法条 {article_id} 不存在"}, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def stat_analysis() -> str:
    """统计知识图谱整体数据：各类型节点数量、罪名分布、刑期统计等。"""
    if not _kg_engine:
        return json.dumps({"error": "知识图谱引擎未初始化"}, ensure_ascii=False)
    try:
        data = _kg_engine.get_stats()
        return json.dumps(data, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def case_search(query: str, limit: int = 5) -> str:
    """基于语义检索或关键词匹配检索相关法律案例。输入案情描述，返回相关案例。"""
    import re

    cleaned = re.sub(r"(找|寻|查看|看看|了解|介绍|给我|提供|帮忙|一些|哪些|类似|几个|相关|相近)", "", query)
    cleaned = re.sub(r"(一下|类似|几个|相关)", "", cleaned).strip()
    search_q = cleaned if cleaned and len(cleaned) >= 2 else query

    if not _vector_store:
        if not _kg_engine:
            return json.dumps({"error": "检索服务未初始化"}, ensure_ascii=False)
        results = _kg_engine.search_case_text(search_q, limit)
        return json.dumps({"total": len(results), "results": results, "source": "kg"}, ensure_ascii=False)
    try:
        results = _vector_store.search(search_q, n_results=limit)
        return json.dumps({"total": len(results), "results": results, "source": "vector"}, ensure_ascii=False)
    except Exception as e:
        results = _kg_engine.search_case_text(search_q, limit) if _kg_engine else []
        return json.dumps({"total": len(results), "results": results, "source": "fallback", "error": str(e)}, ensure_ascii=False)


ALL_TOOLS = [kg_query, article_query, stat_analysis, case_search]


def get_tools():
    return list(ALL_TOOLS)