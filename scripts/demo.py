#!/usr/bin/env python
"""快速演示脚本：构建CAIL2024知识图谱 + LangGraph Agent问答"""

import os
import sys
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["LOG_LEVEL"] = "WARNING"

from src.kg.schema import LegalKGSchema
from src.kg.store import InMemoryKGStore
from src.kg.builder import KnowledgeGraphBuilder
from src.kg.query import KGQueryEngine
from src.data.sample_data import get_sample_cases
from src.agent.core import LegalAgent


def demo_kg_build():
    print("=" * 60)
    print("步骤1：构建知识图谱（CAIL2024多人多罪 内置示例，6个案例）")
    print("=" * 60)

    schema = LegalKGSchema()
    store = InMemoryKGStore(schema)
    cases = get_sample_cases()
    builder = KnowledgeGraphBuilder(store=store)
    stats = builder.build_from_cases(list(cases.cases))

    full = store.get_stats()
    print(f"📦 构建完成：{stats['total_cases']} 案例 / {stats['total_outcomes']} 被告判决")
    print(f"  节点：{full['node_counts']}")
    print(f"  关系：{full['relation_counts']}")
    print(f"  热门罪名：{full['top_accusations'][:3]}")

    print("\n" + "=" * 60)
    print("步骤2：KG查询演示")
    print("=" * 60)
    kg_engine = KGQueryEngine(store=store)
    r = kg_engine.query_by_accusation("盗窃", include_cases=True, case_limit=2)
    print(f"\n【盗窃罪】统计: {r['statistics']}")
    print(f"  关联法条: {r['articles']}")

    print("\n" + "=" * 60)
    print("步骤3：LangGraph Agent问答演示（无API Key → Fake LLM）")
    print("=" * 60)

    agent = LegalAgent(kg_engine=kg_engine)

    questions = [
        "盗窃罪一般判几年？",
        "刑法第234条是什么？",
        "找几个类似案例",
    ]

    for q in questions:
        print(f"\n👤 问题: {q}")
        response = asyncio.run(agent.run(q))
        print(f"🤖 回答:\n{response.answer}")
        print(
            f"   [置信度 {response.confidence:.0%} | intent={response.metadata.get('intent')} | "
            f"attempts={response.metadata.get('attempts', '')}]"
        )


if __name__ == "__main__":
    demo_kg_build()