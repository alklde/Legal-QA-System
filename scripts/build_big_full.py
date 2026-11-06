#!/usr/bin/env python
"""续跑 train_big 全量写入 Neo4j（batch 100 避免内存事务超限）"""
import sys, time, logging, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
)

from src.kg.store import Neo4jKGStore
from src.kg.schema import LegalKGSchema
from src.data.loader import CAILParser
from neo4j import GraphDatabase

uri = os.getenv('NEO4J_URI', 'bolt://127.0.0.1:7687')
pwd = os.getenv('NEO4J_PASSWORD', 'legal_kg_2026')
driver = GraphDatabase.driver(uri, auth=('neo4j', pwd), max_connection_pool_size=5)
store = Neo4jKGStore(driver, LegalKGSchema())
p = CAILParser()

data_dir = os.getenv('CAIL_BIG_DATA_DIR', os.path.join(PROJECT_ROOT, 'data', 'big'))
files = [
    os.path.join(data_dir, name)
    for name in ('test_big.jsonl', 'train_big.jsonl', 'valid_big.jsonl')
]

start = time.time()
total = 0
batch_size = 100
pending = []

for fp in files:
    logging.info(f'开始写入文件: {fp}')
    for obj in p._iter_json_lines(fp):
        try:
            case = p.parse_record(obj)
        except Exception:
            continue
        pending.append(case)
        if len(pending) >= batch_size:
            store.upsert_cases_batch(pending, batch_size=batch_size)
            total += len(pending)
            pending = []
            if total % 10000 == 0:
                logging.info(f'已写入 {total} 条 ({time.time()-start:.0f}s)')
    logging.info(f'文件处理完成 {fp.split("/")[-1]}，累计 {total} 条')

if pending:
    store.upsert_cases_batch(pending, batch_size=batch_size)
    total += len(pending)
    pending = []

logging.info(f'DONE! 全量写入 {total} 条，耗时 {time.time()-start:.0f}s')
stats = store.calculate_statistics()
logging.info(f'预计算统计特征: {len(stats)} 个罪名')
driver.close()
