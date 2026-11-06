#!/usr/bin/env python
"""构建知识图谱脚本（支持CAIL2024/2025多人多罪数据与CAIL2018）"""

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logging
from src.utils.config import load_config
from src.kg.builder import KnowledgeGraphBuilder
from src.kg.schema import LegalKGSchema
from src.kg.store import InMemoryKGStore, Neo4jKGStore
from src.data.loader import CAILParser
from src.data.sample_data import get_sample_cases


def main():
    parser = argparse.ArgumentParser(description="构建法律知识图谱")
    parser.add_argument("--input", default=None, help="输入数据文件或目录（CAIL2024/2025 JSON、CAIL2018 JSONL）")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/settings.yaml"), help="配置文件路径")
    parser.add_argument("--limit", type=int, default=None, help="最多处理的记录数")
    parser.add_argument("--use-sample", action="store_true", help="使用内置示例数据")
    parser.add_argument("--neo4j", action="store_true", help="写入Neo4j（需NEO4J_URI/NEO4J_PASSWORD）")
    args = parser.parse_args()

    setup_logging(log_dir=str(PROJECT_ROOT / "logs"))
    logger = logging.getLogger("build_kg")

    config = load_config(args.config)
    kg_config = config.get("kg", {})

    schema = LegalKGSchema()

    if args.neo4j:
        from neo4j import GraphDatabase

        uri = os.getenv("NEO4J_URI", config.get_resolved("neo4j.uri", "bolt://localhost:7687"))
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", config.get_resolved("neo4j.password", ""))
        if not password:
            logger.error("Neo4j密码未配置（NEO4J_PASSWORD）")
            sys.exit(1)
        driver = GraphDatabase.driver(uri, auth=(user, password))
        store = Neo4jKGStore(driver, schema)
        with driver.session() as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Case) REQUIRE c.case_id IS UNIQUE"
            )
        store_backend = "Neo4j"
    else:
        store = InMemoryKGStore(schema)
        store_backend = "InMemory"

    builder = KnowledgeGraphBuilder(store=store, config={**kg_config})

    try:
        if args.input:
            logger.info(f"加载数据: {args.input}")
            if os.path.isdir(args.input):
                stats = builder.build_from_directory_streaming(args.input, limit=args.limit)
            elif os.path.isfile(args.input):
                parser = CAILParser()
                ds = parser.parse_file(args.input, limit=args.limit)
                stats = builder.build_from_cases(list(ds.cases), limit=None)
            else:
                logger.error(f"输入路径不存在: {args.input}")
                sys.exit(1)
        elif args.use_sample:
            logger.info("使用内置示例数据（CAIL2024格式）")
            cases = get_sample_cases()
            stats = builder.build_from_cases(list(cases.cases), limit=args.limit)
        else:
            logger.error("请指定 --input 或 --use-sample")
            sys.exit(1)
    except Exception as e:
        logger.error(f"构建失败: {e}")
        sys.exit(1)

    # 构建期预计算各罪名统计特征（样本量/刑期离散度），供置信度计算使用
    if hasattr(store, "calculate_statistics"):
        acc_stats = store.calculate_statistics()
        logger.info(f"已预计算 {len(acc_stats)} 个罪名的统计特征")

    full = builder.get_stats()
    print("=" * 50)
    print(f"知识图谱构建完成（后端: {store_backend}）")
    print(f"  案例数: {stats['total_cases']}，被告判决: {stats['total_outcomes']}")
    print(f"  耗时: {stats['duration']}s")
    print(f"  节点数: {full['node_counts']}")
    print(f"  关系数: {full['relation_counts']}")
    print("=" * 50)


if __name__ == "__main__":
    main()