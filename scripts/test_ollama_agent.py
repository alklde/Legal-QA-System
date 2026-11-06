#!/usr/bin/env python
"""使用本地 Ollama 模型（ndavat/Nanbeige4.2-3B）测试 Agent，展示完整对话链路"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["LOG_LEVEL"] = "WARNING"

from src.kg.schema import LegalKGSchema
from src.kg.store import Neo4jKGStore
from src.kg.query import KGQueryEngine
from src.agent.graph import LegalLangGraphAgent
from src.agent.tools import set_kg_engine, kg_query, article_query, stat_analysis, case_search

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434/v1")
MODEL = os.getenv("OLLAMA_MODEL", "ndavat/Nanbeige4.2-3B")


def ensure_driver():
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    password = os.getenv("NEO4J_PASSWORD", "legal_kg_2026")
    driver = GraphDatabase.driver(uri, auth=("neo4j", password))
    return driver


def render_tool(tool_name, kwargs, result_str):
    """渲染一次工具调用的输出"""
    print(f"      ┌── 工具调用: {tool_name}({json.dumps(kwargs, ensure_ascii=False)[:80]})")
    try:
        data = json.loads(result_str)
    except Exception:
        data = {"raw": result_str}
    if tool_name == "kg_query" and "statistics" in data:
        s = data["statistics"]
        print(f"      │   结果: 罪名={data.get('name')}, 类别={data.get('category')}")
        print(f"      │          统计={{total_cases={s.get('total_cases')}, "
              f"avg_months={s.get('avg_months')}, median_months={s.get('median_months')}, "
              f"min={s.get('min_months')}, max={s.get('max_months')}}}")
        if data.get("related_accusations"):
            print(f"      │          关联罪名={data['related_accusations'][:5]}")
        if data.get("articles"):
            print(f"      │          关联法条={data['articles'][:5]}")
    elif tool_name == "article_query" and "article_id" in data:
        print(f"      │   结果: 刑法第{data['article_id']}条")
        print(f"      │          content={str(data.get('content',''))[:60]}…")
        print(f"      │          相关罪名={data.get('related_accusations')}, "
              f"case_count={data.get('case_count')}")
    elif "results" in data:
        print(f"      │   结果: total={data.get('total')}, source={data.get('source')}")
        for i, r in enumerate(data["results"][:2], 1):
            print(f"      │          [{i}] {str(r.get('fact',''))[:70]}…")
    elif "error" in data:
        print(f"      │   结果: ERROR {data['error']}")
    else:
        print(f"      │   结果: {json.dumps(data, ensure_ascii=False)[:200]}")
    print("      └──")


def run_and_display(agent, question):
    print("\n" + "#" * 76)
    print(f"🧑 用户: {question}")
    result = agent.run(question, session_id=f"sv-{question[:6]}")

    meta = result["metadata"]
    print(f"🧠 意图识别: {meta.get('intent')}")
    plan_desc = {
        "kg_query": "按罪名查询量刑统计",
        "article_query": "查询法条原文",
        "stat_analysis": "整体统计",
        "case_search": "检索类似案例",
        "chitchat": "闲聊",
    }.get(meta.get("intent"))
    if plan_desc:
        print(f"📋 执行计划: 步骤1 → {plan_desc}")

    for s in result.get("sources", []):
        tool_name = s.get("tool")
        print(f"   [工具结果条目: {tool_name}, has_data={s.get('has_data')}, type={s.get('type')}]")

    print(f"🤖 回答(LLM: {MODEL}):")
    print(result["answer"])
    print(f"   [confidence={result['confidence']} | intent={meta.get('intent')} "
          f"| attempts={meta.get('attempts')} | validation={meta.get('validation_passed')}]")
    print("#" * 76)


def main():
    driver = ensure_driver()
    qe = KGQueryEngine(store=Neo4jKGStore(driver, LegalKGSchema()), schema=LegalKGSchema())
    set_kg_engine(qe)

    agent = LegalLangGraphAgent(config={
        "model": MODEL,
        "base_url": OLLAMA_BASE,
        "api_key": "ollama",
        "max_tokens": 220,
        "temperature": 0.2,
    })

    questions = [
        "盗窃罪一般判几年？",
        "刑法第264条是什么内容？",
        "诈骗罪的统计情况如何？",
        "帮我找几个类似的寻衅滋事案例",
        "你好",
    ]
    for q in questions:
        run_and_display(agent, q)

    driver.close()


if __name__ == "__main__":
    main()