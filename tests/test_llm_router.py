"""测试 LLM 工具路由（llm_router）与 LangGraph 接入。"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.agent.graph import LegalLangGraphAgent
from src.agent.llm import build_llm
from src.agent.llm_router import (
    LegalToolRouter,
    parse_tool_call,
    validate_tool_call,
    _to_legal_plan,
)
from src.agent.nodes import route_with_llm
from src.agent.tools import set_kg_engine, set_vector_store
from src.kg.query import KGQueryEngine
from src.kg.store import InMemoryKGStore
from src.data.sample_data import get_sample_cases


class FakeBackend:
    """按脚本返回固定文本，记录收到的 messages。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages):
        self.calls.append(list(messages))
        if not self.responses:
            return ""
        r = self.responses.pop(0)
        return r


def _tc(name, args):
    import json
    return f'<tool_call>{{"name": "{name}", "arguments": {json.dumps(json.dumps(args, ensure_ascii=False))}}}</tool_call>'


def test_parse_and_validate():
    tc = parse_tool_call(_tc("query_article", {"article_id": 264}))
    assert tc == {"name": "query_article", "arguments": {"article_id": 264}}
    assert validate_tool_call(tc) is None

    bad = parse_tool_call(_tc("query_article", {"article_id": 3000}))
    assert "1~453" in validate_tool_call(bad)

    assert validate_tool_call({"name": "unknown", "arguments": {}}) is not None
    assert validate_tool_call({"name": "search_case", "arguments": {}}) is not None
    assert parse_tool_call("没有工具调用") is None


def test_tool_mapping():
    assert _to_legal_plan({"name": "search_case", "arguments": {"query": "入室盗窃"}}) == \
        {"tool": "case_search", "kwargs": {"query": "入室盗窃", "limit": 5}}
    assert _to_legal_plan({"name": "query_article", "arguments": {"article_id": 264}}) == \
        {"tool": "article_query", "kwargs": {"article_id": "264"}}
    assert _to_legal_plan({"name": "stat_accusation", "arguments": {"name": "盗窃罪"}}) == \
        {"tool": "kg_query", "kwargs": {"accusation_name": "盗窃罪"}}


def test_route_tool_ok():
    backend = FakeBackend([_tc("stat_accusation", {"name": "盗窃罪"})])
    out = LegalToolRouter(backend).route("统计盗窃罪")
    assert out["kind"] == "tool"
    assert out["intent"] == "kg_query"
    assert out["plan_steps"][0]["kwargs"] == {"accusation_name": "盗窃罪"}
    assert out["errors"] == []


def test_route_regenerate_then_ok():
    backend = FakeBackend([
        _tc("query_article", {"article_id": 3000}),
        _tc("query_article", {"article_id": 264}),
    ])
    out = LegalToolRouter(backend, max_regen=2).route("刑法第3000条")
    assert out["kind"] == "tool"
    assert out["attempts"] == 2
    assert len(out["errors"]) == 1
    assert out["plan_steps"][0]["kwargs"] == {"article_id": "264"}
    assert len(backend.calls) == 2


def test_route_clarify_after_max_regen():
    bad = _tc("query_article", {"article_id": 9999})
    backend = FakeBackend([bad, bad, bad])
    out = LegalToolRouter(backend, max_regen=2).route("乱问")
    assert out["kind"] == "clarify"
    assert out["intent"] == "chitchat"
    assert out["plan_steps"] == []
    assert out["note"]


def test_route_chitchat_no_tool():
    backend = FakeBackend(["你好，我是法律助手。"])
    out = LegalToolRouter(backend).route("你好")
    assert out["kind"] == "no_tool"
    assert out["intent"] == "chitchat"
    assert out["plan_steps"] == []


def test_route_with_llm_fallback_on_error():
    class Boom:
        def route(self, *a, **k):
            raise RuntimeError("backend down")

    state = {"user_input": "找几个类似案例", "session_id": "t"}
    result = route_with_llm(state, Boom())
    assert result["intent"] == "case_search"
    assert result["plan_steps"][0]["tool"] == "case_search"


@pytest.fixture
def store():
    s = InMemoryKGStore()
    for c in get_sample_cases().cases:
        s.upsert_case(c)
    return s


def test_graph_with_router_end_to_end(store):
    set_kg_engine(KGQueryEngine(store=store))
    set_vector_store(None)
    backend = FakeBackend([_tc("stat_accusation", {"name": "盗窃罪"})])
    router = LegalToolRouter(backend)
    agent = LegalLangGraphAgent(llm=build_llm(api_key=None),
                                config={"max_attempts": 3}, router=router)
    result = agent.run("盗窃罪一般判几年？", "router_e2e")
    assert result["answer"]
    assert result["metadata"]["intent"] == "kg_query"
    assert result["confidence"] > 0


def test_graph_with_router_chitchat(store):
    set_kg_engine(KGQueryEngine(store=store))
    set_vector_store(None)
    backend = FakeBackend(["你好，我是法律助手。"])
    agent = LegalLangGraphAgent(llm=build_llm(api_key=None),
                                config={"max_attempts": 3}, router=LegalToolRouter(backend))
    result = agent.run("你好", "router_chat")
    assert result["answer"]
    assert result["metadata"]["intent"] == "chitchat"


def test_graph_with_router_clarify(store):
    set_kg_engine(KGQueryEngine(store=store))
    set_vector_store(None)
    bad = _tc("query_article", {"article_id": 9999})
    agent = LegalLangGraphAgent(llm=build_llm(api_key=None),
                                config={"max_attempts": 3},
                                router=LegalToolRouter(FakeBackend([bad, bad, bad]), max_regen=2))
    result = agent.run("查个不存在的法条", "router_clarify")
    assert result["answer"]
    assert result["metadata"]["intent"] == "chitchat"
    assert result["confidence"] <= 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
