"""知识图谱构建器（支持CAIL2024多人多罪数据 → Neo4j/InMemory）"""

import glob
import logging
import os
import time
from typing import List, Optional

from ..data.loader import CAILParser, CAILDataset
from ..data.models import LegalCase
from .schema import LegalKGSchema, legal_kg_schema
from .store import InMemoryKGStore, Neo4jKGStore, build_store, KGStoreError

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """知识图谱构建器"""

    def __init__(self, store=None, schema: LegalKGSchema = None, config: dict = None):
        self.schema = schema or legal_kg_schema
        self.config = config or {}
        self.batch_size = self.config.get("batch_size", 500)
        if store is not None:
            self.store = store
        else:
            self.store = InMemoryKGStore(self.schema)
        self.stats = {"total_cases": 0, "total_outcomes": 0, "duration": 0.0}

    def build_from_cases(self, cases: List[LegalCase], limit: Optional[int] = None) -> dict:
        start = time.time()
        count = 0
        pending: List[LegalCase] = []
        batch_add = getattr(self.store, "upsert_cases_batch", None)

        def _flush():
            nonlocal pending
            if not pending:
                return 0
            n = 0
            if batch_add:
                n = batch_add(pending, batch_size=self.batch_size)
            else:
                for c in pending:
                    try:
                        self.store.upsert_case(c)
                        n += 1
                    except (KGStoreError, Exception):
                        continue
            self.stats["total_outcomes"] += sum(
                len(c.outcomes) for c in pending
            )
            pending = []
            return n

        for case in cases:
            if limit is not None and len(pending) + count >= limit:
                break
            pending.append(case)
            if len(pending) >= self.batch_size:
                n = _flush()
                count += n
                logger.info(f"已构建 {count} 条")
        n = _flush()
        count += n
        self.stats["total_cases"] = count
        self.stats["duration"] = round(time.time() - start, 2)
        logger.info(f"构建完成: {count} 案例, 耗时 {self.stats['duration']}s")
        return self.stats

    def build_from_file(self, file_path: str, limit: Optional[int] = None) -> dict:
        parser = CAILParser()
        try:
            ds = parser.parse_directory(file_path, limit=limit)
        except Exception:
            ds = parser.parse_file(file_path, limit=limit)
        return self.build_from_cases(list(ds.cases), limit=None)

    def build_from_directory(
        self, directory: str, pattern: str = "*.json*", limit: Optional[int] = None
    ) -> dict:
        parser = CAILParser()
        ds = parser.parse_directory(directory, pattern=pattern, limit=limit)
        return self.build_from_cases(list(ds.cases), limit=None)

    def build_from_directory_streaming(
        self, directory: str, pattern: str = "*.json*", limit: Optional[int] = None
    ) -> dict:
        """流式构建目录数据：逐行解析、按 batch 写入，内存占用恒定。

        适用于大文件（如 CMDL big 全量 39.4 万条），避免全量解析入内存导致 OOM。
        """
        import glob

        parser = CAILParser()
        files = sorted(glob.glob(os.path.join(directory, pattern)))
        files = [
            f
            for f in files
            if not os.path.basename(f).lower().startswith(("article2idx", "charge2idx"))
        ]
        if not files:
            raise FileNotFoundError(f"目录中没有匹配文件: {directory} {pattern}")

        start = time.time()
        count = 0
        pending: List[LegalCase] = []

        def _flush():
            nonlocal pending
            if not pending:
                return 0
            n = 0
            if callable(getattr(self.store, "upsert_cases_batch", None)):
                n = self.store.upsert_cases_batch(pending, batch_size=self.batch_size)
            else:
                for c in pending:
                    try:
                        self.store.upsert_case(c)
                        n += 1
                    except Exception:
                        continue
            self.stats["total_outcomes"] += sum(len(c.outcomes) for c in pending)
            pending = []
            return n

        for fp in files:
            logger.info(f"流式构建文件: {fp}")
            for obj in parser._iter_json_lines(fp):
                if limit is not None and count + len(pending) >= limit:
                    break
                try:
                    case = parser.parse_record(obj)
                except Exception as e:
                    logger.debug(f"跳过无效记录: {e}")
                    continue
                pending.append(case)
                if len(pending) >= self.batch_size:
                    count += _flush()
            if limit is not None and count + len(pending) >= limit:
                break
        count += _flush()

        self.stats["total_cases"] = count
        self.stats["duration"] = round(time.time() - start, 2)
        logger.info(f"流式构建完成: {count} 案例, 耗时 {self.stats['duration']}s")
        return dict(self.stats)

    def get_stats(self) -> dict:
        store_stats = self.store.get_stats()
        store_stats["builder"] = self.stats
        return store_stats


def build_kg(
    store,
    cases: Optional[List[LegalCase]] = None,
    directory: Optional[str] = None,
    pattern: str = "*.json*",
    limit: Optional[int] = None,
) -> dict:
    builder = KnowledgeGraphBuilder(store=store)
    if directory:
        return builder.build_from_directory(directory, pattern=pattern, limit=limit)
    if cases:
        return builder.build_from_cases(cases, limit=limit)
    from ..data.sample_data import get_sample_cases

    ds = get_sample_cases()
    return builder.build_from_cases(list(ds.cases), limit=None)