"""测试Agent模块"""

import os
import sys
import json
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.agent.core import LegalAgent, AgentResponse, ConversationMemory
from src.agent.llm import build_llm
from src.agent.tools import set_kg_engine, set_vector_store, get_tools
from src.agent.nodes import classify_intent, create_plan, execute_step, generate_answer, compute_confidence, _extract_accusation, _extract_article_id
from src.agent.state import AgentState
from src.agent.graph import LegalLangGraphAgent
from src.kg.query import KGQueryEngine
from src.kg.store import InMemoryKGStore
from src.data.sample_data import get_sample_cases


@pytest.fixture
def store():
    s = InMemoryKGStore()
    ds = get_sample_cases()
    for c in ds.cases:
        s.upsert_case(c)
    return s


@pytest.fixture
def kg_engine(store):
    return KGQueryEngine(store=store)


@pytest.fixture
def fake_llm():
    return build_llm(api_key=None)


@pytest.fixture
def agent(kg_engine, fake_llm):
    set_kg_engine(kg_engine)
    set_vector_store(None)
    return LegalAgent(llm=fake_llm, kg_engine=kg_engine, config={"max_history": 20, "max_attempts": 3})


@pytest.mark.asyncio
async def test_agent_kg_query(agent):
    response = await agent.run("盗窃罪一般判几年？")
    assert isinstance(response, AgentResponse)
    assert response.answer
    assert response.confidence > 0
    assert response.session_id


@pytest.mark.asyncio
async def test_agent_article_query(agent):
    response = await agent.run("刑法第234条是什么？")
    assert response.answer
    assert response.confidence > 0


@pytest.mark.asyncio
async def test_agent_stat_analysis(agent):
    response = await agent.run("统计一下数据库里有多少案件？")
    assert response.answer
    assert response.session_id


@pytest.mark.asyncio
async def test_agent_case_search(agent):
    response = await agent.run("找几个盗窃案的类似案例")
    assert response.answer


@pytest.mark.asyncio
async def test_agent_chitchat(agent):
    response = await agent.run("你好")
    assert response.answer
    assert response.error is None


@pytest.mark.asyncio
async def test_multi_turn_conversation(agent):
    session_id = "test_session"
    resp1 = await agent.run("盗窃罪一般判几年？", session_id)
    assert resp1.session_id == session_id
    resp2 = await agent.run("那故意伤害罪呢？", session_id)
    assert resp2.session_id == session_id
    history = agent.memory.get_history(session_id)
    assert len(history) == 4


def test_session_management(agent):
    session_id = agent.memory.create_session()
    assert session_id
    agent.memory.update(session_id, "用户问题", "Agent回答")
    history = agent.memory.get_history(session_id)
    assert len(history) == 2
    agent.memory.delete_session(session_id)
    assert agent.memory.get_session(session_id) is None


def test_extract_accusation():
    assert _extract_accusation("盗窃罪一般判几年？") == "盗窃"
    assert _extract_accusation("故意伤害怎么判？") == "故意伤害"
    assert _extract_accusation("今天天气怎么样？") is None


def test_extract_article_id():
    assert _extract_article_id("刑法第234条是什么？") == "234"
    assert _extract_article_id("查看第133条") == "133"
    assert _extract_article_id("今天天气怎么样？") is None


def test_classify_intent():
    state: AgentState = {"user_input": "盗窃罪一般判几年？", "session_id": "test"}
    result = classify_intent(state)
    assert result["intent"] == "kg_query"

    state2: AgentState = {"user_input": "找几个类似案例", "session_id": "test"}
    result2 = classify_intent(state2)
    assert result2["intent"] == "case_search"

    state3: AgentState = {"user_input": "你好", "session_id": "test"}
    result3 = classify_intent(state3)
    assert result3["intent"] == "chitchat"

    state4: AgentState = {"user_input": "刑法第264条", "session_id": "test"}
    result4 = classify_intent(state4)
    assert result4["intent"] == "article_query"


def test_create_plan():
    state: AgentState = {"intent": "kg_query", "user_input": "盗窃罪判几年？", "session_id": "test", "attempts": 0, "max_attempts": 5}
    result = create_plan(state)
    assert len(result["plan_steps"]) == 1
    assert result["plan_steps"][0]["tool"] == "kg_query"

    state2: AgentState = {"intent": "article_query", "user_input": "刑法第234条", "session_id": "test", "attempts": 0, "max_attempts": 5}
    result2 = create_plan(state2)
    assert result2["plan_steps"][0]["tool"] == "article_query"


def test_execute_step():
    set_kg_engine(KGQueryEngine(store=InMemoryKGStore()))
    from src.data.sample_data import get_sample_cases
    ds = get_sample_cases()
    store = InMemoryKGStore()
    for c in ds.cases:
        store.upsert_case(c)
    set_kg_engine(KGQueryEngine(store=store))
    state: AgentState = {
        "user_input": "盗窃", "session_id": "test",
        "plan_steps": [{"tool": "kg_query", "kwargs": {"accusation_name": "盗窃"}}],
        "current_step_index": 0, "attempts": 0, "max_attempts": 5,
        "tool_results": [], "sources": [],
    }
    result = execute_step(state)
    assert len(result["tool_results"]) == 1
    assert result["current_step_index"] == 1


def test_tools_registered():
    tools = get_tools()
    names = [t.name for t in tools]
    assert "kg_query" in names
    assert "article_query" in names
    assert "stat_analysis" in names
    assert "case_search" in names


def test_compute_confidence_data_driven():
    """置信度应随样本量与离散度变化，而非工具数量恒定的 0.85。"""
    big_sample = [{"tool": "kg_query", "success": True, "result": {"statistics": {"total_cases": 20000, "cv": 0.2}}}]
    small_sample = [{"tool": "kg_query", "success": True, "result": {"statistics": {"total_cases": 10, "cv": 0.2}}}]
    dispersed = [{"tool": "kg_query", "success": True, "result": {"statistics": {"total_cases": 20000, "cv": 2.0}}}]

    c_big = compute_confidence(big_sample)
    c_small = compute_confidence(small_sample)
    c_dispersed = compute_confidence(dispersed)

    assert c_big > c_small
    assert c_big > c_dispersed
    assert c_big > 0.7
    assert c_small < c_big
    assert c_dispersed < c_big


def test_compute_confidence_boundaries():
    assert compute_confidence([]) == 0.1
    no_data = [{"tool": "kg_query", "success": True, "result": {"error": "未找到相关数据"}}]
    assert compute_confidence(no_data) <= 0.1
    big = [{"tool": "kg_query", "success": True, "result": {"statistics": {"total_cases": 10**6, "cv": 0}}}]
    assert 0 <= compute_confidence(big) <= 0.98


def test_compute_confidence_article_vs_search():
    article = [{"tool": "article_query", "success": True, "result": {"article_id": "264", "case_count": 5000}}]
    few_results = [{"tool": "case_search", "success": True, "result": {"results": [{"case_id": "a"}, {"case_id": "b"}]}}]
    many_results = [{"tool": "case_search", "success": True, "result": {"results": [{"case_id": f"c{i}"} for i in range(10)]}}]
    assert compute_confidence(article) > compute_confidence(few_results)
    assert compute_confidence(many_results) > compute_confidence(few_results)


class TestGraphAgent:
    @pytest.fixture
    def graph_agent(self, store, fake_llm):
        set_kg_engine(KGQueryEngine(store=store))
        set_vector_store(None)
        return LegalLangGraphAgent(llm=fake_llm, config={"max_attempts": 3})

    def test_run_chitchat(self, graph_agent):
        result = graph_agent.run("你好", "test1")
        assert result["answer"]
        assert result["confidence"] > 0

    def test_run_kg_query(self, graph_agent):
        result = graph_agent.run("盗窃罪判几年？", "test2")
        assert result["answer"]
        assert result["metadata"]["intent"] == "kg_query"

    def test_run_article_query(self, graph_agent):
        result = graph_agent.run("刑法第264条", "test3")
        assert result["answer"]

    def test_run_stat_analysis(self, graph_agent):
        result = graph_agent.run("统计所有案件", "test4")
        assert result["answer"]

    def test_run_case_search(self, graph_agent):
        result = graph_agent.run("类似案例", "test5")
        assert result["answer"]


class TestMemory:
    def test_buffer_memory(self):
        memory = ConversationMemory(max_history=6)
        session_id = memory.create_session()
        for i in range(5):
            memory.update(session_id, f"问题{i}", f"回答{i}")
        history = memory.get_history(session_id)
        assert len(history) == 6

    def test_memory_limits(self):
        memory = ConversationMemory(max_history=2)
        session_id = memory.create_session()
        for i in range(5):
            memory.update(session_id, f"问题{i}", f"回答{i}")
        history = memory.get_history(session_id)
        assert len(history) <= 4

    def test_extract_entities(self):
        memory = ConversationMemory()
        session_id = memory.create_session()
        entities = memory.extract_entities(session_id, ["刑法第234条的内容是什么？"])
        assert 234 in entities["articles"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])