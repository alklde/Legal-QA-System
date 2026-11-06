"""端到端联调：用项目二微调模型（Qwen2.5-0.5B-tool-ft）做 LangGraph 工具路由。

跑通「自然语言 → LLM 路由 → legal @tool → 回答」全链路。默认用样例数据
（InMemoryKGStore，无需 Neo4j）；LLM 后端默认本地 HF，也可用 --server 指向 llama-server。

用法：
  # 本地 HF（transformers + peft，需显存 ~1GB）
  python scripts/test_tool_router_agent.py

  # 走 llama-server（不加载本地权重）
  TOOL_FT_SERVER_URL=http://127.0.0.1:8080 python scripts/test_tool_router_agent.py

  # 规则路由对照（不加载模型）
  python scripts/test_tool_router_agent.py --rules
"""
import argparse
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.agent.graph import LegalLangGraphAgent
from src.agent.llm import build_llm
from src.agent.tools import set_kg_engine, set_vector_store
from src.kg.query import KGQueryEngine
from src.kg.store import InMemoryKGStore
from src.data.sample_data import get_sample_cases

DEMO_QUERIES = [
    "盗窃罪一般判几年？",
    "那故意伤害罪呢？",  # 多轮指代：应复用上文「量刑统计」意图，槽位换成故意伤害罪
    "刑法第264条讲的是什么？",
    "帮我找几个入室盗窃的类似案例",
    "你好",
]


def build_engine(use_neo4j=False):
    if use_neo4j:
        from neo4j import GraphDatabase
        from src.kg.store import Neo4jKGStore
        from src.kg.schema import LegalKGSchema
        uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        pwd = os.getenv("NEO4J_PASSWORD", "legal_kg_2026")
        driver = GraphDatabase.driver(uri, auth=("neo4j", pwd))
        return KGQueryEngine(store=Neo4jKGStore(driver, LegalKGSchema()))
    store = InMemoryKGStore()
    for c in get_sample_cases().cases:
        store.upsert_case(c)
    return KGQueryEngine(store=store)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", action="store_true", help="用关键词规则路由（对照，不加载模型）")
    ap.add_argument("--server", default=None, help="llama-server 地址（覆盖 TOOL_FT_SERVER_URL）")
    ap.add_argument("--neo4j", action="store_true", help="用真实 Neo4j big 库（默认样例内存库）")
    ap.add_argument("--session", default="router_demo")
    args = ap.parse_args()

    engine = build_engine(use_neo4j=args.neo4j)
    set_kg_engine(engine)
    set_vector_store(None)
    print(f"KG 后端：{'Neo4j big 库' if args.neo4j else 'InMemory 样例库'}")

    if args.rules:
        router = None
        print("=== 规则路由（baseline）===")
    else:
        from src.agent.llm_router import LegalToolRouter, HFBackend, OpenAICompatBackend
        url = args.server or os.getenv("TOOL_FT_SERVER_URL")
        backend = OpenAICompatBackend(url) if url else HFBackend()
        router = LegalToolRouter(backend, max_regen=2)
        print(f"=== LLM 工具路由（{'llama-server ' + url if url else '本地 HF'}）===")

    agent = LegalLangGraphAgent(llm=build_llm(api_key=None),
                                config={"max_attempts": 3}, router=router)

    for q in DEMO_QUERIES:
        result = agent.run(q, args.session)
        meta = result["metadata"]
        print(f"\n> {q}")
        print(f"  intent={meta.get('intent')} router_kind={meta.get('router_kind')} "
              f"confidence={result['confidence']:.2f}")
        answer = result["answer"].strip().replace("\n", " ")
        print(f"  answer: {answer[:200]}")


if __name__ == "__main__":
    main()
