"""测试知识图谱模块"""

import os
import sys
import json
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.kg.schema import LegalKGSchema
from src.kg.store import InMemoryKGStore, build_store
from src.kg.builder import KnowledgeGraphBuilder
from src.kg.query import KGQueryEngine
from src.data.loader import CAILParser
from src.data.models import LegalCase, Outcome, Judgment, Penalty


class TestLegalKGSchema:
    def test_node_types_valid(self):
        schema = LegalKGSchema()
        assert "Case" in schema.NODE_TYPES
        assert "Criminal" in schema.NODE_TYPES
        assert "Accusation" in schema.NODE_TYPES
        assert "Article" in schema.NODE_TYPES
        assert "Sentence" in schema.NODE_TYPES

    def test_relation_types_valid(self):
        schema = LegalKGSchema()
        assert "INVOLVES" in schema.RELATION_TYPES
        assert "CHARGED_WITH" in schema.RELATION_TYPES
        assert "CITES" in schema.RELATION_TYPES
        assert "SENTENCED_TO" in schema.RELATION_TYPES
        assert "BASED_ON" in schema.RELATION_TYPES

    def test_relation_endpoints(self):
        schema = LegalKGSchema()
        assert schema.get_relation_endpoints("INVOLVES") == ("Case", "Criminal")
        assert schema.get_relation_endpoints("CHARGED_WITH") == ("Criminal", "Accusation")

    def test_categorize_accusation(self):
        schema = LegalKGSchema()
        assert schema.categorize_accusation("盗窃") == "侵犯财产罪"
        assert schema.categorize_accusation("故意伤害") == "侵犯公民人身权利、民主权利罪"
        assert schema.categorize_accusation("未知罪名") == "其他犯罪"

    def test_validate_relation(self):
        schema = LegalKGSchema()
        assert schema.is_valid_relation("INVOLVES", "Case", "Criminal")
        assert not schema.is_valid_relation("INVOLVES", "Case", "Accusation")


class TestCAILParserAndModels:
    def test_penalty_from_dict(self):
        p = Penalty.from_dict({
            "Imprisonment": 12, "Fine": 3000, "Death Penalty": False,
            "Life Imprisonment": False, "Detention": 0, "Surveillance": 0,
        })
        assert p.imprisonment == 12
        assert p.fine == 3000
        assert p.months == 12

    def test_judgment_from_dict(self):
        j = Judgment.from_dict({
            "Accusation": "盗窃罪", "Standard Accusation": "盗窃罪",
            "Related Law": ["264"],
            "Penalty": {"Imprisonment": 14, "Fine": 3000},
        })
        assert j.accusation == "盗窃罪"
        assert j.related_law == ["264"]
        assert j.penalty.imprisonment == 14

    def test_parser_cail2024(self):
        parser = CAILParser()
        case = parser.parse_record({
            "Facts": "2024年1月，被告人林某A、林某B盗窃商场商品价值4800元。",
            "Court View": "公诉机关指控事实清楚。",
            "Defendants": ["林某A", "林某B"],
            "Outcomes": [{
                "Name": "林某A",
                "Judgment": [{"Accusation": "盗窃罪", "Standard Accusation": "盗窃罪",
                              "Related Law": ["264"],
                              "Penalty": {"Imprisonment": 14, "Fine": 3000}}]
            }]
        })
        assert len(case.defendants) == 2
        assert len(case.outcomes) == 2
        assert case.outcomes[0].name == "林某A"
        assert case.accusations == ["盗窃罪"]

    def test_parser_cail2018(self):
        parser = CAILParser()
        case = parser.parse_record({
            "fact": "2015年，被告人胡某用木制坐垫打伤被害人孙某左腹部。",
            "meta": {
                "relevant_articles": [234],
                "accusation": ["故意伤害"],
                "criminals": ["胡某"],
                "punish_of_money": 0,
                "term_of_imprisonment": {"death_penalty": False, "imprisonment": 12, "life_imprisonment": False}
            }
        })
        assert case.defendants == ["胡某"]
        assert case.accusations == ["故意伤害"]
        assert case.outcomes[0].judgments[0].penalty.imprisonment == 12


class TestKnowledgeGraphBuilder:
    @pytest.fixture
    def store(self):
        return InMemoryKGStore()

    @pytest.fixture
    def sample_cases(self):
        return [
            LegalCase(
                case_id="case_001",
                fact="被告人胡某在公司车间用木制坐垫打伤被害人孙某左腹部，构成故意伤害罪。",
                defendants=["胡某"],
                outcomes=[Outcome(name="胡某", judgments=[
                    Judgment(accusation="故意伤害罪", standard_accusation="故意伤害罪",
                             related_law=["234"],
                             penalty=Penalty(imprisonment=12))
                ])]
            ),
            LegalCase(
                case_id="case_002",
                fact="被告人张某在超市盗窃现金5000元及手机一部，构成盗窃罪。",
                defendants=["张某"],
                outcomes=[Outcome(name="张某", judgments=[
                    Judgment(accusation="盗窃罪", standard_accusation="盗窃罪",
                             related_law=["264"],
                             penalty=Penalty(imprisonment=8))
                ])]
            ),
        ]

    def test_build_from_cases(self, store, sample_cases):
        builder = KnowledgeGraphBuilder(store=store)
        stats = builder.build_from_cases(sample_cases)
        assert stats["total_cases"] == 2
        assert stats["total_outcomes"] == 2
        assert isinstance(stats["duration"], float)

    def test_build_with_limit(self, store, sample_cases):
        builder = KnowledgeGraphBuilder(store=store)
        stats = builder.build_from_cases(sample_cases, limit=1)
        assert stats["total_cases"] == 1

    def test_build_sample_data(self, store):
        from src.data.sample_data import get_sample_cases
        ds = get_sample_cases()
        builder = KnowledgeGraphBuilder(store=store)
        stats = builder.build_from_cases(list(ds.cases))
        assert stats["total_cases"] == 6

    def test_get_stats(self, store, sample_cases):
        builder = KnowledgeGraphBuilder(store=store)
        builder.build_from_cases(sample_cases)
        stats = builder.get_stats()
        assert stats["node_counts"]["Case"] == 2
        assert stats["node_counts"]["Criminal"] == 2
        assert stats["relation_counts"]["INVOLVES"] == 2


class TestInMemoryKGStore:
    @pytest.fixture
    def populated_store(self):
        store = InMemoryKGStore()
        from src.data.sample_data import get_sample_cases
        ds = get_sample_cases()
        for c in ds.cases:
            store.upsert_case(c)
        return store

    def test_get_stats(self, populated_store):
        stats = populated_store.get_stats()
        assert stats["node_counts"]["Case"] == 6
        assert len(stats["top_accusations"]) > 0

    def test_query_by_accusation(self, populated_store):
        result = populated_store.query_by_accusation("盗窃")
        assert result is not None
        assert "盗窃" in result["name"]
        assert "statistics" in result

    def test_query_by_accusation_not_found(self, populated_store):
        result = populated_store.query_by_accusation("不存在的罪名")
        assert result is None

    def test_query_article(self, populated_store):
        result = populated_store.query_article("264")
        assert result is not None
        assert "264" in str(result["article_id"])
        assert result["content"]
        assert "case_count" in result

    def test_query_article_not_found(self, populated_store):
        result = populated_store.query_article("9999")
        assert result is None

    def test_search_case_text(self, populated_store):
        results = populated_store.search_case_text("盗窃商场", limit=3)
        assert len(results) > 0
        assert results[0]["case_id"]

    def test_related_accusations(self, populated_store):
        acc = populated_store.query_by_accusation("故意伤害")
        assert acc is not None
        assert isinstance(acc["related_accusations"], list)


class TestKGQueryEngine:
    @pytest.fixture
    def engine(self):
        store = InMemoryKGStore()
        from src.data.sample_data import get_sample_cases
        ds = get_sample_cases()
        for c in ds.cases:
            store.upsert_case(c)
        return KGQueryEngine(store=store)

    def test_query_by_accusation(self, engine):
        result = engine.query_by_accusation("盗窃")
        assert result is not None
        assert "盗窃" in result["name"]
        assert result["statistics"]["avg_months"] > 0

    def test_query_article(self, engine):
        result = engine.query_article("264")
        assert result is not None
        assert "content" in result

    def test_get_stats(self, engine):
        stats = engine.get_stats()
        assert stats["node_counts"]["Case"] > 0
        assert len(stats["top_accusations"]) > 0

    def test_search_case_text(self, engine):
        results = engine.search_case_text("故意伤害", limit=3)
        assert len(results) > 0


class TestNeo4jKGStore:
    """Neo4j 集成测试（有 NEO4J_URI + Neo4j 运行时执行，否则跳过）"""

    @pytest.fixture
    def neo4j_store(self):
        import os

        uri = os.getenv("NEO4J_URI")
        password = os.getenv("NEO4J_PASSWORD")
        if not uri or not password:
            pytest.skip("未设置 NEO4J_URI / NEO4J_PASSWORD")
        try:
            from neo4j import GraphDatabase
            from src.kg.store import Neo4jKGStore

            driver = GraphDatabase.driver(uri, auth=("neo4j", password))
            store = Neo4jKGStore(driver)
            driver.session().run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Case) REQUIRE c.case_id IS UNIQUE"
            )
        except Exception as e:  # pragma: no cover
            pytest.skip(f"无法连接 Neo4j: {e}")
        yield store
        try:  # 清理所有测试残留节点，避免污染真实统计
            with driver.session() as s:
                s.run(
                    "MATCH (c:Case) WHERE c.case_id STARTS WITH 'neo4j_test_' "
                    "DETACH DELETE c"
                )
        except Exception:
            pass
        driver.close()

    def test_upsert_and_stats(self, neo4j_store):
        import uuid

        from src.data.models import Judgment, LegalCase, Outcome

        before = neo4j_store.get_stats()["node_counts"]["Case"]
        case = LegalCase(
            case_id=f"neo4j_test_{uuid.uuid4().hex[:12]}",
            fact="测试用案例事实……盗窃数额较大",
            court_view="本院认为……构成盗窃罪",
            defendants=["测试被告"],
            outcomes=[
                Outcome(
                    name="测试被告",
                    judgments=[
                        Judgment(
                            accusation="盗窃罪",
                            standard_accusation="盗窃罪",
                            related_law=["264"],
                            penalty=None,
                        )
                    ],
                )
            ],
        )
        neo4j_store.upsert_case(case)
        after = neo4j_store.get_stats()["node_counts"]["Case"]
        assert after == before + 1
        assert after >= 393000

    def test_query_by_accusation(self, neo4j_store):
        result = neo4j_store.query_by_accusation("盗窃罪")
        assert result is not None
        st = result["statistics"]
        assert result["category"]
        assert st["total_defendants"] >= 5000
        assert st["avg_months"] > 0
        assert st["median_months"] > 0

    def test_query_article_case_count(self, neo4j_store):
        result = neo4j_store.query_article("264", include_cases=False)
        assert result is not None
        assert result["case_count"] >= 90000
        assert "盗窃罪" in result["related_accusations"]

    def test_get_stats_top_accusations(self, neo4j_store):
        stats = neo4j_store.get_stats()
        assert stats["node_counts"]["Case"] >= 393000
        assert stats["top_accusations"]
        assert stats["top_accusations"][0]["name"] in ("盗窃罪", "诈骗罪", "寻衅滋事罪")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])