"""LangGraph图节点（意图识别、规划、执行、反思、验证、回答）"""

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional

from .state import AgentState
from .tools import get_tools

logger = logging.getLogger(__name__)

_KG_KEYWORDS = [
    "判几年", "刑期", "法条", "罪名", "犯什么罪", "刑法第",
    "量刑", "法律", "处罚", "刑罚", "有期徒刑",
]
_CASE_KEYWORDS = ["类似案例", "相似", "类案", "案例", "相似案例"]
_STAT_KEYWORDS = ["统计", "平均刑期", "分布", "比例", "多少件", "多少案件", "整体"]
_ARTICLE_KEYWORDS = ["第.+条", "法条.*内容", "条文"]
_CHITCHAT_KEYWORDS = ["你好", "你是谁", "谢谢", "再见", "帮助", "请问你是"]


def compute_confidence(success_results):
    """基于知识图谱统计特征的置信度评分。

    数据驱动，而非依赖工具数量：
      - 样本量（案件数）：样本越多，统计越可信
      - 离散度（变异系数 cv）：刑期分布越集中，均值越有代表性

    方案：基础分 0.35 + 样本量因子(0~0.5) + 数据完整度因子(0~0.1)，
    并按离散度对统计类结果打折。无数据 0.1。
    """
    if not success_results:
        return 0.1

    scores = []
    for item in success_results:
        result = item.get("result", {}) if isinstance(item, dict) else {}
        stats = result.get("statistics") or {}
        if stats:
            n_cases = stats.get("total_cases") or stats.get("total_defendants") or 0
            cv = stats.get("cv", 0)
            scores.append(_stat_based_score(n_cases, cv))
        elif result.get("article_id") or "case_count" in result:
            scores.append(_stat_based_score(result.get("case_count") or 0, cv=0))
        elif isinstance(result.get("results"), list):
            scores.append(0.35 + 0.5 * min(1.0, len(result["results"]) / 10.0))
        elif result.get("node_counts"):
            total = result["node_counts"].get("Case", 0)
            scores.append(_stat_based_score(total, cv=0))
        else:
            scores.append(0.1)

    return min(0.98, max(0.1, sum(scores) / len(scores)))


def _stat_based_score(n_cases: int, cv: float) -> float:
    sample_factor = 0.5 * (1.0 - math.exp(-n_cases / 300.0)) if n_cases > 0 else 0.0
    spread_penalty = 0.1 * min(1.0, cv) if cv else 0.0
    return 0.35 + sample_factor + 0.1 - spread_penalty


def _extract_accusation(text: str) -> Optional[str]:
    common = [
        "故意伤害", "故意杀人", "盗窃", "抢劫", "诈骗", "抢夺",
        "交通肇事", "危险驾驶", "寻衅滋事", "聚众斗殴", "贩卖毒品",
        "贪污", "受贿", "信用卡诈骗",
    ]
    for name in common:
        if name in text:
            return name
    return None


def _extract_article_id(text: str) -> Optional[str]:
    m = re.search(r"第\s*(\d+)\s*条", text)
    if m:
        return m.group(1)
    return None


def classify_intent(state: AgentState) -> dict:
    user_input = state.get("user_input", "")
    if any(kw in user_input for kw in _CHITCHAT_KEYWORDS):
        return {"intent": "chitchat", "attempts": 0}
    if any(kw in user_input for kw in _CASE_KEYWORDS):
        return {"intent": "case_search", "attempts": 0}
    if any(re.search(pat, user_input) for pat in _ARTICLE_KEYWORDS) or _extract_article_id(user_input):
        return {"intent": "article_query", "attempts": 0}
    has_accusation = bool(_extract_accusation(user_input))
    if any(kw in user_input for kw in _STAT_KEYWORDS) and not has_accusation:
        return {"intent": "stat_analysis", "attempts": 0}
    if any(kw in user_input for kw in _KG_KEYWORDS) or has_accusation:
        return {"intent": "kg_query", "attempts": 0}
    return {"intent": "kg_query", "attempts": 0}


def create_plan(state: AgentState) -> dict:
    intent = state.get("intent", "kg_query")
    user_input = state.get("user_input", "")
    steps: List[Dict[str, Any]] = []

    if intent == "kg_query":
        acc = _extract_accusation(user_input)
        art = _extract_article_id(user_input)
        if acc:
            steps.append({"tool": "kg_query", "kwargs": {"accusation_name": acc}})
        elif art:
            steps.append({"tool": "article_query", "kwargs": {"article_id": art}})
        else:
            steps.append({"tool": "stat_analysis", "kwargs": {}})

    elif intent == "article_query":
        art = _extract_article_id(user_input)
        if art:
            steps.append({"tool": "article_query", "kwargs": {"article_id": art}})
        else:
            steps.append({"tool": "kg_query", "kwargs": {}})

    elif intent == "case_search":
        steps.append({"tool": "case_search", "kwargs": {"query": user_input, "limit": 5}})

    elif intent == "stat_analysis":
        steps.append({"tool": "stat_analysis", "kwargs": {}})

    return {"plan_steps": steps, "current_step_index": 0}


def route_with_llm(state: AgentState, router) -> dict:
    """LLM 工具路由节点：微调模型产出 intent + plan_steps。

    替换 classify_intent + create_plan。模型不可用/异常时回退规则，保证可用性。
    """
    user_input = state.get("user_input", "")
    history = state.get("history") or []
    session_id = state.get("session_id") or None
    try:
        out = router.route(user_input, history=history, session_id=session_id)
    except Exception as e:
        logger.warning(f"LLM Router 失败，回退规则路由: {e}")
        intent_res = classify_intent(state)
        plan_res = create_plan({**state, **intent_res})
        return {**intent_res, **plan_res, "attempts": 0}

    logger.info("[llm_router] kind=%s intent=%s attempts=%s errors=%s",
                out.get("kind"), out.get("intent"), out.get("attempts"), out.get("errors"))
    return {
        "intent": out.get("intent") or "kg_query",
        "plan_steps": out.get("plan_steps") or [],
        "current_step_index": 0,
        "attempts": 0,
        "router_kind": out.get("kind", ""),
        "router_note": out.get("note") or "",
    }


def execute_step(state: AgentState) -> dict:
    steps = state.get("plan_steps", [])
    idx = state.get("current_step_index", 0)
    attempts = state.get("attempts", 0)

    if idx >= len(steps):
        return {"current_step_index": idx, "tool_results": [], "sources": []}

    step = steps[idx]
    tool_name = step.get("tool", "")
    kwargs = step.get("kwargs", {})

    all_tools = {t.name: t for t in get_tools()}
    tool_fn = all_tools.get(tool_name)

    if not tool_fn:
        return {
            "tool_results": [{"tool": tool_name, "error": f"工具 {tool_name} 不存在"}],
            "sources": [{"type": "error", "tool": tool_name}],
            "current_step_index": idx + 1,
            "attempts": attempts + 1,
        }

    try:
        result_str = tool_fn.invoke(kwargs)
        try:
            result = json.loads(result_str)
        except Exception:
            result = {"raw": result_str}

        has_data = not (isinstance(result, dict) and result.get("error"))
        source = {"tool": tool_name, "has_data": has_data}
        if "statistics" in result or "avg_months" in result:
            source["type"] = "kg_stats"
        elif "content" in result:
            source["type"] = "article"
        elif "results" in result:
            source["type"] = "case_search"
        else:
            source["type"] = "kg_raw"

        return {
            "tool_results": [{"tool": tool_name, "result": result, "success": has_data}],
            "sources": [source],
            "current_step_index": idx + 1,
            "attempts": attempts + 1,
        }
    except Exception as e:
        logger.error(f"工具执行异常: {e}")
        return {
            "tool_results": [{"tool": tool_name, "error": str(e), "success": False}],
            "sources": [{"tool": tool_name, "type": "error", "error": str(e)}],
            "current_step_index": idx + 1,
            "attempts": attempts + 1,
        }


def check_more_steps(state: AgentState) -> str:
    idx = state.get("current_step_index", 0)
    steps = state.get("plan_steps", [])
    attempts = state.get("attempts", 0)
    max_attempts = state.get("max_attempts", 5)
    if attempts >= max_attempts:
        return "validate"
    if idx < len(steps):
        return "execute"
    return "validate"


def reflect_and_validate(state: AgentState) -> dict:
    tool_results = state.get("tool_results", [])
    successful = [r for r in tool_results if r.get("success", False)]
    if successful:
        return {"validation_passed": True, "validation_reason": "查询成功"}
    if state.get("attempts", 0) < state.get("max_attempts", 5):
        return {"validation_passed": False, "validation_reason": "查询失败，建议重试"}
    return {"validation_passed": False, "validation_reason": "超过最大尝试次数，使用已有结果"}


def generate_answer(state: AgentState, llm=None) -> dict:
    user_input = state.get("user_input", "")
    intent = state.get("intent", "kg_query")
    tool_results = state.get("tool_results", [])
    sources = state.get("sources", [])

    if intent == "chitchat":
        note = state.get("router_note")
        if note:
            confidence = 0.1 if state.get("router_kind") == "clarify" else 0.5
            return {"final_answer": note, "confidence": confidence}
        return {"final_answer": "您好！我是法律智能问答助手，可以帮助您查询法律知识、罪名量刑信息、相关法条和类似案例。请问您想了解什么？", "confidence": 1.0}

    if not tool_results:
        return {"final_answer": "抱歉，没有找到相关信息。请尝试更具体的问题。", "confidence": 0.1}

    parts = []
    success_results = [r for r in tool_results if r.get("success", False)]
    context_lines = []

    for item in success_results:
        result = item.get("result", {})
        context_lines.append(f"[{item.get('tool', 'tool')}]\n{json.dumps(result, ensure_ascii=False)[:2000]}")
        if "statistics" in result:
            stats = result["statistics"]
            name = result.get("name", "")
            lines = [f"【{name}】量刑统计："]
            if "total_cases" in stats:
                lines.append(f"  - 涉及案件：{stats['total_cases']} 件")
            if "avg_months" in stats and stats["avg_months"] > 0:
                lines.append(f"  - 平均刑期：约 {stats['avg_months']:.1f} 个月")
            if "min_months" in stats:
                lines.append(f"  - 最低刑期：{stats['min_months']} 个月")
            if "max_months" in stats:
                lines.append(f"  - 最高刑期：{stats['max_months']} 个月")
            related = result.get("related_accusations", [])
            if related:
                lines.append(f"  - 关联罪名：{'、'.join(related[:5])}")
            parts.append("\n".join(lines))

        if "content" in result and "article_id" in result:
            accs = result.get("related_accusations", [])
            lines = [f"刑法第{result['article_id']}条：", result["content"]]
            if accs:
                lines.append(f"相关罪名：{'、'.join(accs)}")
            parts.append("\n".join(lines))

        if "results" in result:
            results = result["results"]
            if results:
                lines = [f"找到 {len(results)} 个相关结果："]
                for r in results[:3]:
                    if "fact" in r:
                        fact_preview = r["fact"][:80] + ("..." if len(r["fact"]) > 80 else "")
                        lines.append(f"  - {fact_preview}")
                parts.append("\n".join(lines))

    if not parts:
        raw = success_results[0].get("result", {}) if success_results else {}
        parts.append("查询结果：" + json.dumps(raw, ensure_ascii=False)[:500] if raw else "查询结果为空。")

    template_answer = "\n\n".join(parts)
    template_answer += "\n\n⚠️ 以上数据基于历史裁判文书统计，仅供参考。具体判决请咨询专业法律人士。"
    confidence = compute_confidence(success_results)

    if llm is None or not success_results:
        return {"final_answer": template_answer, "confidence": confidence}

    prompt = (
        "你是一位法律知识图谱问答助手。请基于下面从中国刑事裁判文书知识图谱检索到的结构化数据，"
        "用简洁自然的中文回答用户的问题。可以适当概括统计结果，但不要编造数据中没有的信息。"
        "如果数据不足以回答，请如实说明。\n\n"
        f"用户问题：{user_input}\n"
        f"检索结果：\n{json.dumps(success_results, ensure_ascii=False)[:3000]}\n\n"
        "请直接输出回答。"
    )
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        if content and content.strip():
            answer = content.strip()
            answer += "\n\n⚠️ 以上回答基于知识图谱检索的历史裁判文书数据，仅供参考。具体判决请咨询专业法律人士。"
            return {"final_answer": answer, "confidence": confidence}
    except Exception as e:
        logger.warning(f"LLM生成回答失败，使用模板: {e}")

    return {"final_answer": template_answer, "confidence": confidence}