"""ChromaDB 向量检索层测试"""

import shutil

import pytest

from src.data.sample_data import get_sample_cases
from src.retrieval.vector_store import ChromaCaseStore


@pytest.fixture
def chroma(tmp_path):
    store = ChromaCaseStore(persist_directory=str(tmp_path / "vector_db"))
    yield store
    shutil.rmtree(str(tmp_path), ignore_errors=True)


def _to_records():
    recs = []
    for c in get_sample_cases().cases:
        accs = [j.accusation for o in c.outcomes for j in o.judgments]
        recs.append({"case_id": c.case_id, "fact": c.fact, "metadata": {"accusations": accs}})
    return recs


def test_collection_name():
    store = ChromaCaseStore(collection_name="custom_cases")
    assert store.collection_name == "custom_cases"


def test_add_and_count(chroma):
    assert chroma.count() == 0
    chroma.add_cases_batch(_to_records())
    assert chroma.count() == 6


def test_add_case_single(chroma):
    rec = _to_records()[0]
    chroma.add_case(rec["case_id"], rec["fact"], rec["metadata"])
    assert chroma.count() == 1


def test_search_returns_results(chroma):
    chroma.add_cases_batch(_to_records())
    results = chroma.search("入室盗窃现金", n_results=3)
    assert len(results) == 3
    for r in results:
        assert r["case_id"]
        assert "fact" in r
        assert 0.0 <= r["relevance"] <= 1.0


def test_search_empty_query(chroma):
    chroma.add_cases_batch(_to_records())
    assert chroma.search("", n_results=3) == []


def test_search_without_data(chroma):
    assert chroma.search("随便查查", n_results=3) == []


def test_delete_case(chroma):
    rec = _to_records()[0]
    chroma.add_cases_batch(_to_records())
    chroma.delete(rec["case_id"])
    assert chroma.count() == 5


def test_add_empty_fact_ignored(chroma):
    chroma.add_case("case_x", "", {})
    assert chroma.count() == 0