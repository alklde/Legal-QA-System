#!/usr/bin/env python
"""使用 OpenAI 兼容端点（Azure 阿里云百炼 MaaS，配置见 .env.example）测试 Agent，展示完整对话链路"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["LOG_LEVEL"] = "WARNING"

from dotenv import load_dotenv

_env_file = PROJECT_ROOT / ".env"
if not _env_file.exists():
    _env_file = PROJECT_ROOT / ".env.example"
load_dotenv(_env_file, override=False)

from src.kg.schema import LegalKGSchema
from src.kg.store import Neo4jKGStore
from src.kg.query import KGQueryEngine
from src.agent.graph import LegalLangGraphAgent
from src.agent.tools import set_kg_engine

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("OPENAI_API_KEY", "")


def ensure_driver():
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    password = os.getenv("NEO4J_PASSWORD", "legal_kg_2026")
    driver = GraphDatabase.driver(uri, auth=("neo4j", password))
    return driver


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
        print(f"   [工具结果条目: {s.get('tool')}, has_data={s.get('has_data')}, type={s.get('type')}]")

    print(f"🤖 回答(LLM: {MODEL}):")
    print(result["answer"])
    print(f"   [confidence={result['confidence']} | intent={meta.get('intent')} "
          f"| attempts={meta.get('attempts')} | validation={meta.get('validation_passed')}]")
    print("#" * 76)


def main():
    if not API_KEY or not BASE_URL:
        print("错误：请在 .env / .env.example 中配置 OPENAI_API_KEY 与 OPENAI_BASE_URL")
        sys.exit(1)

    driver = ensure_driver()
    qe = KGQueryEngine(store=Neo4jKGStore(driver, LegalKGSchema()), schema=LegalKGSchema())
    set_kg_engine(qe)

    agent = LegalLangGraphAgent(config={
        "model": MODEL,
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "max_tokens": 512,
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