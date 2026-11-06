"""API请求/响应数据模型"""

from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """聊天请求"""

    message: str = Field(..., description="用户输入的问题", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="会话ID")
    context: Optional[Dict[str, Any]] = Field(default={}, description="上下文信息")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("消息不能为空")
        return v


class ChatResponse(BaseModel):
    """聊天响应"""

    answer: str = Field(..., description="Agent生成的回答")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="信息来源")
    confidence: float = Field(0.0, description="回答置信度")
    session_id: Optional[str] = Field(None, description="会话ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    error: Optional[str] = Field(None, description="错误信息")


class KGQueryRequest(BaseModel):
    """知识图谱查询请求"""

    query_type: str = Field(..., description="查询类型：accusation/article/case/sentence")
    params: Dict[str, Any] = Field(default={}, description="查询参数")


class KGQueryResponse(BaseModel):
    """知识图谱查询响应"""

    type: str = Field(..., description="查询类型")
    result: Any = Field(..., description="查询结果")


class CaseSearchRequest(BaseModel):
    """案例检索请求"""

    query: str = Field(..., description="查询文本", min_length=1, max_length=5000)
    limit: int = Field(10, description="返回结果数量", ge=1, le=50)
    filters: Optional[Dict[str, Any]] = Field(default={}, description="过滤条件")


class SessionRequest(BaseModel):
    """会话请求"""

    pass


class SessionResponse(BaseModel):
    """会话响应"""

    session_id: str
    created_at: str = Field(default="", description="创建时间")
    expires_at: str = Field(default="", description="过期时间")
    message_count: int = Field(0, description="消息数量")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field("healthy", description="状态")
    version: str = Field(..., description="版本")
    components: Dict[str, Any] = Field(default_factory=dict, description="各组件状态")
    uptime_seconds: float = Field(0.0, description="运行时长")
    memory_usage_mb: float = Field(0.0, description="内存占用")


class StatsResponse(BaseModel):
    """统计响应"""

    node_counts: Dict[str, int] = Field(default_factory=dict, description="节点数")
    relation_counts: Dict[str, int] = Field(default_factory=dict, description="关系数")
    top_accusations: List[Dict[str, Any]] = Field(default_factory=list, description="主要罪名")
    sentence_distribution: Dict[str, int] = Field(default_factory=dict, description="刑期分布")