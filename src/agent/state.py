"""Agent状态定义（LangGraph StateGraph共享状态）"""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    session_id: str
    messages: Annotated[List[Dict[str, str]], operator.add]
    history: List[Dict[str, str]]
    intent: str
    plan_steps: List[Dict[str, Any]]
    current_step_index: int
    tool_results: Annotated[List[Dict[str, Any]], operator.add]
    sources: Annotated[List[Dict[str, Any]], operator.add]
    validation_passed: bool
    validation_reason: str
    router_note: str
    router_kind: str
    final_answer: str
    confidence: float
    error: Optional[str]
    attempts: int
    max_attempts: int