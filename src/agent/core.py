"""Agent核心实现（LangGraph门面 + 会话记忆）"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .graph import LegalLangGraphAgent
from .llm import build_llm
from .tools import set_kg_engine, set_vector_store

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    answer: str
    sources: List[dict] = field(default_factory=list)
    confidence: float = 0.0
    session_id: str = ""
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None


class ConversationMemory:
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def create_session(self, session_id: str = None) -> str:
        sid = session_id or f"session_{uuid.uuid4().hex[:12]}"
        self._sessions[sid] = []
        return sid

    def update(self, session_id: str, user_msg: str, assistant_msg: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ])
        if len(self._sessions[session_id]) > self.max_history:
            self._sessions[session_id] = self._sessions[session_id][-self.max_history:]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return list(self._sessions.get(session_id, []))

    def get_session(self, session_id: str):
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    def extract_entities(self, session_id: str, messages: List[str] = None) -> dict:
        import re

        entities = {"articles": [], "accusations": []}
        text = " ".join(messages or [])
        article_matches = re.findall(r"第(\d+)条", text)
        entities["articles"] = [int(a) for a in article_matches]
        common = ["故意伤害", "故意杀人", "盗窃", "抢劫", "诈骗", "交通肇事", "寻衅滋事"]
        for name in common:
            if name in text:
                entities["accusations"].append(name)
        return entities


class LegalAgent:
    def __init__(
        self,
        llm=None,
        kg_engine=None,
        vector_store=None,
        config: dict = None,
        router=None,
    ):
        self.config = config or {}
        self.kg_engine = kg_engine
        self.router = router

        if kg_engine:
            set_kg_engine(kg_engine)
        if vector_store:
            set_vector_store(vector_store)

        llm_config = self.config.get("llm", self.config)
        if llm is not None:
            self._llm = llm
        else:
            self._llm = build_llm(
                model=llm_config.get("model", "gpt-4o-mini"),
                api_key=llm_config.get("api_key"),
                temperature=llm_config.get("temperature", 0.1),
                max_tokens=llm_config.get("max_tokens", 2000),
            )

        self._inner = LegalLangGraphAgent(llm=self._llm, config=self.config, router=router)
        self.memory = ConversationMemory(max_history=self.config.get("max_history", 20))

    async def run(self, user_input: str, session_id: str = None) -> AgentResponse:
        start_time = time.time()
        sid = session_id or f"session_{uuid.uuid4().hex[:12]}"

        if sid not in [s for s in self.memory._sessions]:
            self.memory.create_session(sid)

        result = self._inner.run(user_input, sid)
        self.memory.update(sid, user_input, result["answer"])
        result["metadata"]["execution_time"] = round(time.time() - start_time, 3)

        return AgentResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            session_id=sid,
            metadata=result.get("metadata", {}),
            error=result["metadata"].get("error"),
        )

    @property
    def tools(self):
        return self._inner._graph.get_graph().nodes