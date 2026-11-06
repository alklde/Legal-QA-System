"""测试API模块"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.kg.query import KGQueryEngine
from src.kg.store import InMemoryKGStore
from src.data.sample_data import get_sample_cases
from src.agent.core import LegalAgent
from src.agent.llm import build_llm
from src.agent import tools as agent_tools
from src.api.routes import router


@pytest.fixture(scope="module")
def app():
    store = InMemoryKGStore()
    ds = get_sample_cases()
    for c in ds.cases:
        store.upsert_case(c)
    kg_engine = KGQueryEngine(store=store)

    agent_tools.set_kg_engine(kg_engine)
    agent_tools.set_vector_store(None)
    fake_llm = build_llm(api_key=None)
    agent = LegalAgent(llm=fake_llm, kg_engine=kg_engine)

    test_app = FastAPI(title="test")
    test_app.state.kg_engine = kg_engine
    test_app.state.agent = agent
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"


def test_kg_stats(client):
    resp = client.get("/api/kg/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_counts"]["Case"] > 0


def test_kg_query_accusation(client):
    resp = client.post("/api/kg/query", json={
        "query_type": "accusation",
        "params": {"name": "盗窃"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert "盗窃" in data["result"]["name"]


def test_kg_query_article(client):
    resp = client.post("/api/kg/query", json={
        "query_type": "article",
        "params": {"article_id": "264", "include_cases": False},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert "content" in data["result"]


def test_kg_query_stats(client):
    resp = client.post("/api/kg/query", json={"query_type": "stats", "params": {}})
    assert resp.status_code == 200
    assert resp.json()["found"] is True


def test_kg_query_unknown_type(client):
    resp = client.post(
        "/api/kg/query", json={"query_type": "unknown", "params": {}}
    )
    assert resp.status_code == 400


def test_kg_search(client):
    resp = client.post("/api/kg/search", json={"query": "盗窃", "limit": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 0


def test_chat(client):
    resp = client.post("/api/chat", json={"message": "盗窃罪一般判几年？"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"]
    assert data["session_id"]


def test_chat_empty_message(client):
    resp = client.post("/api/chat", json={"message": " "})
    assert resp.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])