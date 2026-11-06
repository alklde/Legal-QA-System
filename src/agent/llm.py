"""LLM工厂（ChatOpenAI + Fake支持测试）"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_FAKE_ANSWERS = {
    "intent": '{"intent": "kg_query", "confidence": 0.9, "reasoning": "用户查询法律知识"}',
    "chitchat": "您好！我是法律智能问答助手，可以帮助您查询法律知识、罪名量刑信息、相关法条和类似案例。请问您想了解什么？",
    "plan": '{"steps": [{"tool": "kg_query", "description": "查询罪名信息"}], "analysis": "查询相关法律信息"}',
    "validate": '{"passed": true, "reason": "结果相关"}',
    "answer": "根据知识图谱查询，相关法律信息如下。请以上述查询结果为准，具体情况建议咨询专业法律人士。",
}


def build_llm(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    streaming: bool = False,
):
    if not api_key and not base_url:
        logger.warning("未配置OpenAI API Key，使用FakeChatModel")
        return _build_fake(model)
    try:
        from langchain_openai import ChatOpenAI

        kwargs = dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    except Exception as e:
        logger.error(f"ChatOpenAI创建失败: {e}，降级Fake")
        return _build_fake(model)


def _build_fake(model: str = "fake"):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    messages = iter([
        ("assistant", "您好！我是法律智能问答助手。请问您想了解什么？"),
        ("assistant", "根据知识图谱查询，盗窃罪相关法条为刑法第264条，平均刑期约18个月。"),
        ("assistant", "刑法第234条规定：故意伤害他人身体的，处三年以下有期徒刑、拘役或者管制。"),
        ("assistant", "根据知识图谱统计，相关数据如下，请以权威法律文书为准。"),
    ])

    class FakeChatModel(GenericFakeChatModel):
        def bind_tools(self, tools=None, **kwargs):
            return self

        def with_structured_output(self, schema=None, **kwargs):
            return self

    return FakeChatModel(messages=messages)