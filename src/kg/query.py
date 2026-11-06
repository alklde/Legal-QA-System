"""知识图谱查询引擎（包装Store接口，供Agent工具与API调用）"""

import logging
from typing import Dict, List, Optional

from .schema import LegalKGSchema, legal_kg_schema
from .store import InMemoryKGStore, Neo4jKGStore, build_store

logger = logging.getLogger(__name__)


class KGQueryError(Exception):
    pass


class KGQueryEngine:
    def __init__(self, store=None, driver=None, schema: LegalKGSchema = None, config: dict = None):
        self.schema = schema or legal_kg_schema
        self.config = config or {}
        if store is not None:
            self.store = store
        elif driver is not None:
            self.store = Neo4jKGStore(driver, self.schema)
        else:
            self.store = InMemoryKGStore(self.schema)
        self._use_mock = isinstance(self.store, InMemoryKGStore)

    def query_by_accusation(
        self, accusation_name: str, include_cases: bool = False, case_limit: int = 10
    ) -> Optional[dict]:
        try:
            return self.store.query_by_accusation(accusation_name, include_cases, case_limit)
        except Exception as e:
            logger.error(f"query_by_accusation failed: {e}")
            raise KGQueryError(str(e))

    def query_article(self, article_id, include_cases: bool = False, case_limit: int = 10) -> Optional[dict]:
        try:
            return self.store.query_article(article_id, include_cases, case_limit)
        except Exception as e:
            logger.error(f"query_article failed: {e}")
            raise KGQueryError(str(e))

    def query_sentence_distribution(self, accusation_name: str = None) -> dict:
        try:
            return self.store.query_sentence_distribution(accusation_name)
        except Exception as e:
            logger.error(f"query_sentence_distribution failed: {e}")
            raise KGQueryError(str(e))

    def query_related_accusations(self, accusation_name: str, limit: int = 10) -> list:
        try:
            return self.store.query_related_accusations(accusation_name, limit)
        except Exception as e:
            logger.error(f"query_related_accusations failed: {e}")
            raise KGQueryError(str(e))

    def get_stats(self) -> dict:
        try:
            return self.store.get_stats()
        except Exception as e:
            logger.error(f"get_stats failed: {e}")
            raise KGQueryError(str(e))

    def search_case_text(self, query: str, limit: int = 10) -> List[dict]:
        try:
            return self.store.search_case_text(query, limit)
        except Exception as e:
            logger.error(f"search_case_text failed: {e}")
            return []