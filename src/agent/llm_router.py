"""LLM 工具路由（对接项目二：Qwen2.5-0.5B-tool-ft）。

用微调小模型把「自然语言 → 工具调用（工具名 + 参数）」这一步 LLM 化，
替换 nodes.py 里 `classify_intent` / `create_plan` 的确定性关键词规则。

设计要点（Agent 缺陷应对）：
  - 多轮 context：携带 history，支持指代追问；
  - 参数校验 + 重生成：validate 拦截幻觉参数（如条号越界），把错误回灌让模型重写；
  - 连续失败转澄清：不把脏参数传给下游工具；
  - 后端可插拔：本地 HF（transformers+peft）或 OpenAI 兼容 llama-server。

工具名映射（训练 schema → 主项目 @tool）见 `_to_legal_plan`。
"""
import json
import logging
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TC_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
ARTICLE_MIN, ARTICLE_MAX = 1, 453

# 与 tool_ft/scripts/01_build_dataset.py 中训练用的 schema 完全一致
TOOLS = [
    {"type": "function", "function": {
        "name": "search_case",
        "description": "在刑事案例知识库中检索与给定查询相关的相似案件，返回案号与基本案情摘要。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "案情关键词组合，例如'入室盗窃 数额较大'"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "query_article",
        "description": "查询《中华人民共和国刑法》中某一条文的准确法律文本内容。",
        "parameters": {"type": "object", "properties": {
            "article_id": {"type": "integer", "description": "刑法条文编号，如 264 表示第二百六十四条"}},
            "required": ["article_id"]}}},
    {"type": "function", "function": {
        "name": "stat_accusation",
        "description": "统计指定罪名的案例数量、常见刑罚区间等分布信息。",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "罪名全称，如'盗窃罪'"}},
            "required": ["name"]}}},
]


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """解析 <tool_call>{"name","arguments"}</tool_call>；arguments 支持对象或 JSON 字符串。"""
    m = TC_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    args = obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return None
    if args is None:
        args = {}
    return {"name": obj.get("name"), "arguments": args}


def validate_tool_call(tc: Optional[dict]) -> Optional[str]:
    """校验 tool-call。合法返回 None，否则返回中文错误串。"""
    if tc is None:
        return "空工具调用"
    name, args = tc.get("name"), tc.get("arguments") or {}
    if not isinstance(args, dict):
        return f"arguments 应为对象，收到 {type(args).__name__}"
    if name not in {t["function"]["name"] for t in TOOLS}:
        return f"未定义的工具名：{name!r}"
    if name == "query_article":
        aid = args.get("article_id")
        if isinstance(aid, bool) or not isinstance(aid, int):
            return f"article_id 应为整数，收到 {aid!r}"
        if not (ARTICLE_MIN <= aid <= ARTICLE_MAX):
            return f"刑法条文编号应在 {ARTICLE_MIN}~{ARTICLE_MAX}，收到 {aid!r}"
    elif name == "search_case":
        if not str(args.get("query") or "").strip():
            return "参数缺失：query"
    elif name == "stat_accusation":
        if not str(args.get("name") or "").strip():
            return "参数缺失：name"
    return None


def _to_legal_plan(tc: dict) -> Optional[Dict[str, Any]]:
    """微调工具 → legal_kg_agent 执行计划（工具名 + kwargs）。"""
    name, args = tc.get("name"), tc.get("arguments") or {}
    if name == "search_case":
        return {"tool": "case_search", "kwargs": {"query": args["query"], "limit": 5}}
    if name == "query_article":
        return {"tool": "article_query", "kwargs": {"article_id": str(args["article_id"])}}
    if name == "stat_accusation":
        # 训练里 stat_accusation 带罪名参数；legal 的 kg_query(accusation_name) 返回该罪名统计
        return {"tool": "kg_query", "kwargs": {"accusation_name": args["name"]}}
    return None


def _assistant_tool_msg(tc: dict) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [
        {"type": "function", "function": {
            "name": tc["name"],
            "arguments": json.dumps(tc["arguments"], ensure_ascii=False)}}]}


def _regen_hint(err: str) -> str:
    return (f"你上一步的工具调用不合法：{err}。"
            f"请只重新输出一个合法的 <tool_call>，不要包含解释或其他内容。")


# ------------------------- 后端 -------------------------
class HFBackend:
    """本地 transformers + peft 加载 Qwen2.5-0.5B-tool-ft。"""

    def __init__(self, model_dir: Optional[str] = None, lora_dir: Optional[str] = None):
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(here))  # 仓库根目录
        tf = os.path.join(repo_root, "tool_ft", "output")
        self.model_dir = model_dir or os.getenv("TOOL_FT_MODEL_DIR", os.path.join(tf, "base_model"))
        self.lora_dir = lora_dir or os.getenv("TOOL_FT_LORA_DIR", os.path.join(tf, "lora_ckpt", "best"))
        self._model = None
        self._tok = None
        self._torch = None
        self._dev = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self._torch = torch
        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[llm_router] 加载 {self.model_dir} + LoRA {self.lora_dir} ({self._dev})")
        self._tok = AutoTokenizer.from_pretrained(self.model_dir)
        base = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=torch.bfloat16)
        self._model = PeftModel.from_pretrained(base, self.lora_dir).to(self._dev).eval()

    def generate(self, messages: List[dict]) -> str:
        self._lazy_load()
        prompt = self._tok.apply_chat_template(
            messages, tools=TOOLS, tokenize=False, add_generation_prompt=True)
        inp = self._tok(prompt, return_tensors="pt").to(self._dev)
        with self._torch.no_grad():
            out = self._model.generate(**inp, max_new_tokens=256, do_sample=False,
                                       pad_token_id=self._tok.pad_token_id)
        return self._tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class OpenAICompatBackend:
    """OpenAI 兼容端点（llama-server /v1），不加载本地权重。"""

    def __init__(self, base_url: str, api_key: str = "llama.cpp", model: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.api_key = api_key
        self.model = model

    def generate(self, messages: List[dict]) -> str:
        payload = {"messages": messages, "tools": TOOLS, "temperature": 0, "max_tokens": 256}
        if self.model:
            payload["model"] = self.model
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return (data["choices"][0]["message"].get("content") or "").strip()


# ------------------------- 路由 -------------------------
class LegalToolRouter:
    def __init__(self, backend, max_regen: int = 2, history_window: int = 8):
        self.backend = backend
        self.max_regen = max_regen
        self.history_window = history_window
        self._sessions: Dict[str, List[dict]] = {}

    @staticmethod
    def _normalize_history(history: Optional[List[dict]]) -> List[dict]:
        out = []
        for h in history or []:
            role, content = h.get("role"), h.get("content")
            if role in ("user", "assistant") and content:
                out.append({"role": role, "content": content})
        return out

    def _remember(self, session_id, messages):
        if not session_id:
            return
        window = self.history_window or len(messages)
        self._sessions[session_id] = messages[-window:]

    def route(self, query: str, history: Optional[List[dict]] = None,
              session_id: Optional[str] = None) -> Dict[str, Any]:
        """返回 {kind, intent, plan_steps, tool_call?, note?, attempts, errors, messages}。

        kind ∈ {tool, no_tool, clarify}。传 session_id 时内部维护多轮消息（含 tool_call
        上下文），比外部只传「用户+最终答案」的历史更利于指代追问。
        """
        if session_id and session_id in self._sessions:
            messages = list(self._sessions[session_id])
        else:
            messages = self._normalize_history(history)
            if self.history_window:
                messages = messages[-self.history_window:]
        messages = messages + [{"role": "user", "content": query}]
        errors: List[str] = []

        for attempt in range(self.max_regen + 1):
            text = self.backend.generate(messages)
            tc = parse_tool_call(text)

            if tc is None:
                if "<tool_call>" in (text or ""):
                    err = "tool_call JSON 解析失败"
                    errors.append(err)
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": _regen_hint(err)})
                    continue
                messages.append({"role": "assistant", "content": text})
                if errors:  # 纠正失败 → 澄清
                    return self._finalize(self._clarify(errors[-1], messages, attempt + 1, errors, text),
                                          session_id)
                return self._finalize(
                    {"kind": "no_tool", "intent": "chitchat", "plan_steps": [],
                     "note": text, "tool_call": None, "attempts": attempt + 1,
                     "errors": errors}, session_id, messages)

            err = validate_tool_call(tc)
            if err is None:
                plan = _to_legal_plan(tc)
                messages.append(_assistant_tool_msg(tc))
                return self._finalize(
                    {"kind": "tool", "intent": plan["tool"], "plan_steps": [plan],
                     "tool_call": tc, "note": "", "attempts": attempt + 1,
                     "errors": errors}, session_id, messages)

            errors.append(err)
            messages.append(_assistant_tool_msg(tc))
            messages.append({"role": "tool",
                             "content": json.dumps({"error": err}, ensure_ascii=False)})
            messages.append({"role": "user", "content": _regen_hint(err)})

        return self._finalize(
            self._clarify(errors[-1], messages, self.max_regen + 1, errors, ""), session_id)

    def _finalize(self, result: Dict[str, Any], session_id, messages=None) -> Dict[str, Any]:
        result["messages"] = messages if messages is not None else result.get("messages", [])
        self._remember(session_id, result["messages"])
        return result

    def _clarify(self, err, messages, attempts, errors, text) -> Dict[str, Any]:
        note = text or f"抱歉，我没能确定正确的工具参数（{err}）。请补充更明确的罪名或法条编号。"
        return {"kind": "clarify", "intent": "chitchat", "plan_steps": [], "tool_call": None,
                "note": note, "attempts": attempts, "errors": errors, "messages": messages}


def build_default_router() -> LegalToolRouter:
    """按环境变量构建：优先 llama-server，其次本地 HF。"""
    url = os.getenv("TOOL_FT_SERVER_URL")
    if url:
        logger.info(f"[llm_router] 使用 llama-server: {url}")
        return LegalToolRouter(OpenAICompatBackend(url))
    return LegalToolRouter(HFBackend())
