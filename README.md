# 法律知识图谱智能问答系统

基于知识图谱与大模型的法律问答系统。解析 CAIL2024/2025 多人多罪判决数据，构建刑事法律知识图谱，结合 ChromaDB 检索与 LangGraph Agent，支持量刑咨询、法条查询、案例检索和统计分析。

## 特性

- **知识图谱**：Neo4j / 内存双后端，统一查询接口
- **Agent 编排**：意图识别 → 任务规划 → 工具调用 → 反思校验 → 答案生成
- **语义检索**：ChromaDB 向量检索，异常时回退图谱关键词检索
- **鲁棒降级**：无 OpenAI Key / Neo4j 时自动使用 Fake LLM / 内存后端
- **接口与界面**：FastAPI REST API + Web 前端

## 技术栈

Python 3.10+ · FastAPI · LangGraph · Neo4j · ChromaDB · OpenAI API · pytest

## 架构

```
Web / REST Client
        │
        ▼
FastAPI ──► LangGraph Agent ──► Tools
                              ├─ 知识图谱（Neo4j / InMemory）
                              ├─ 向量检索（ChromaDB）
                              └─ 大模型（OpenAI / Fake）
```

## 目录

```
src/agent/        Agent 编排、工具、LLM 后端
src/api/          FastAPI 接口
src/data/         CAIL 解析、统一模型、示例数据
src/kg/           知识图谱构建、存储与查询
src/retrieval/    ChromaDB 向量检索
scripts/          图谱构建、演示、数据导入、测试脚本
tests/            单元与集成测试
web/              Web 前端
```

## 快速开始

```bash
pip install -r requirements.txt

# 运行时需本地提供 config/settings.yaml 与 .env（未随仓库提供）
python scripts/build_kg.py --use-sample
python scripts/demo.py

# API 服务
uvicorn src.api.main:app --port 8000

# Web 界面（可选）
python web/app.py
```

- API 文档：`http://localhost:8000/docs`
- Web 界面：`http://localhost:5000`

## API

`POST /api/chat`、`POST /api/kg/query`、`POST /api/kg/search`、`GET /api/kg/stats`、`GET /api/health`

## 测试

```bash
python -m pytest tests/ -q
# 75 passed, 4 skipped
```

## 数据

CAIL2024/2025 官方数据或 CMDL 开源数据集；内置 6 条示例数据用于开发与演示。