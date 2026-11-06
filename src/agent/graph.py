"""LangGraph StateGraph定义（法律Agent核心图）"""

import logging
from typing import Any, Dict, List, Optional, Sequence

from langgraph.graph import END, StateGraph

from .llm import build_llm
from .nodes import (
    check_more_steps,
    classify_intent,
    create_plan,
    execute_step,
    generate_answer,
    reflect_and_validate,
    route_with_llm,
)
from .state import AgentState

logger = logging.getLogger(__name__)


def _build_legal_graph(llm=None, router=None):
    graph = StateGraph(AgentState)

    if router is not None:
        graph.add_node("route_with_llm", lambda state: route_with_llm(state, router))
        graph.set_entry_point("route_with_llm")
        graph.add_conditional_edges(
            "route_with_llm",
            lambda state: "generate_answer" if state.get("intent") == "chitchat" else "execute_step",
            {"execute_step": "execute_step", "generate_answer": "generate_answer"},
        )
    else:
        graph.add_node("classify_intent", classify_intent)
        graph.add_node("create_plan", create_plan)
        graph.set_entry_point("classify_intent")
        graph.add_edge("classify_intent", "create_plan")
        graph.add_conditional_edges(
            "create_plan",
            lambda state: "generate_answer" if state.get("intent") == "chitchat" else "execute_step",
            {"execute_step": "execute_step", "generate_answer": "generate_answer"},
        )

    graph.add_node("execute_step", execute_step)
    graph.add_node("reflect_and_validate", reflect_and_validate)
    graph.add_node(
        "generate_answer",
        lambda state: generate_answer(state, llm=llm),
    )

    graph.add_edge("execute_step", "reflect_and_validate")

    graph.add_conditional_edges(
        "reflect_and_validate",
        lambda state: (
            "generate_answer"
            if state.get("validation_passed")
            or state.get("attempts", 0) >= state.get("max_attempts", 5)
            else "generate_answer"
        ),
        {"generate_answer": "generate_answer"},
    )

    graph.add_edge("generate_answer", END)

    return graph.compile()


class LegalLangGraphAgent:
    """基于LangGraph的法律智能问答Agent"""

    def __init__(
        self,
        llm=None,
        config: dict = None,
        router=None,
    ):
        self.config = config or {}
        self.router = router
        if llm is not None:
            self.llm = llm
        else:
            self.llm = build_llm(
                model=self.config.get("model", "gpt-4o-mini"),
                api_key=self.config.get("api_key"),
                base_url=self.config.get("base_url"),
                temperature=self.config.get("temperature", 0.1),
                max_tokens=self.config.get("max_tokens", 2000),
            )
        self._graph = _build_legal_graph(llm=self.llm, router=router)
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def _get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self._sessions.get(session_id, [])

    def _save_turn(self, session_id: str, user_msg: str, assistant_msg: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ])

    def run(self, user_input: str, session_id: str = "default") -> dict:
        history = self._get_history(session_id)
        init_state: AgentState = {
            "user_input": user_input,
            "session_id": session_id,
            "messages": [],
            "history": history,
            "intent": "",
            "plan_steps": [],
            "current_step_index": 0,
            "tool_results": [],
            "sources": [],
            "validation_passed": False,
            "validation_reason": "",
            "router_note": "",
            "router_kind": "",
            "final_answer": "",
            "confidence": 0.0,
            "error": None,
            "attempts": 0,
            "max_attempts": self.config.get("max_attempts", 5),
        }
        try:
            result = self._graph.invoke(init_state, {"recursion_limit": 30})
            self._save_turn(session_id, user_input, result.get("final_answer", ""))
            return {
                "answer": result.get("final_answer", "抱歉，无法生成回答。"),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "session_id": session_id,
                "metadata": {
                    "intent": result.get("intent"),
                    "attempts": result.get("attempts", 0),
                    "validation_passed": result.get("validation_passed", False),
                    "validation_reason": result.get("validation_reason", ""),
                    "router_kind": result.get("router_kind", ""),
                },
            }
        except Exception as e:
            logger.error(f"Agent执行失败: {e}")
            return {
                "answer": "抱歉，处理您的问题时出现了错误，请稍后重试。",
                "sources": [],
                "confidence": 0.1,
                "session_id": session_id,
                "metadata": {"error": str(e)},
            }