"""FastAPI应用入口"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from ..utils.config import Config, load_config
from ..utils.logger import setup_logging
from ..data.sample_data import get_sample_cases
from ..kg.schema import LegalKGSchema
from ..kg.store import InMemoryKGStore
from ..kg.builder import KnowledgeGraphBuilder
from ..kg.query import KGQueryEngine
from ..agent.core import LegalAgent, AgentResponse
from .routes import router

logger = logging.getLogger("legal_kg")


def create_app(config_path: str = None) -> FastAPI:
    config = load_config(config_path)
    log_level = config.get("logging.level", "INFO")
    log_file = config.get("logging.file", "logs/app.log")
    log_dir = str(Path(log_file).parent)
    setup_logging(log_dir=log_dir, log_level=log_level)

    schema = LegalKGSchema()
    store = InMemoryKGStore(schema)

    llm_config = config.llm_config
    api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
    kg_dir = os.getenv("CAIL_DATA_DIR")
    use_neo4j = os.getenv("NEO4J_URI")

    if use_neo4j:
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                use_neo4j,
                auth=(
                    llm_config.get("neo4j_user", "neo4j"),
                    os.getenv("NEO4J_PASSWORD", ""),
                ),
            )
            from ..kg.store import Neo4jKGStore

            store = Neo4jKGStore(driver, schema)
            logger.info(f"使用Neo4j存储: {use_neo4j}")
        except Exception as e:
            logger.warning(f"Neo4j连接失败，降级InMemory: {e}")

    builder = KnowledgeGraphBuilder(store=store)
    if kg_dir and os.path.isdir(kg_dir):
        logger.info(f"加载CAIL数据目录: {kg_dir}")
        builder.build_from_directory(kg_dir)
    else:
        logger.info("加载内置示例数据")
        cases = get_sample_cases()
        builder.build_from_cases(list(cases.cases))

    kg_engine = KGQueryEngine(store=store)

    llm_provider = llm_config.get("provider", "openai")
    if not api_key and llm_provider != "mock":
        logger.warning("未配置OpenAI API Key，Agent将使用Fake LLM（纯规则/模板回复）")

    agent = LegalAgent(
        kg_engine=kg_engine,
        config={
            "llm": llm_config,
            "max_history": config.get("agent.max_history", 20),
            "max_attempts": config.get("agent.max_iterations", 5),
        },
    )

    cors_config = config.get("security.cors", {})
    app = FastAPI(
        title=config.get("app.name", "法律知识图谱智能问答系统"),
        description=config.get("app.description", ""),
        version=config.get("app.version", "1.0.0"),
        debug=config.get("app.debug", False),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get("origins", ["*"]),
        allow_credentials=cors_config.get("allow_credentials", True),
        allow_methods=cors_config.get("methods", ["*"]),
        allow_headers=cors_config.get("headers", ["*"]),
    )

    app.state.config = config
    app.state.kg_engine = kg_engine
    app.state.agent = agent

    app.include_router(router)

    web_dir = Path(__file__).parent.parent.parent / "web"
    if web_dir.exists():
        app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")

    @app.get("/")
    async def root():
        return {
            "service": config.get("app.name", "法律知识图谱智能问答系统"),
            "version": config.get("app.version", "1.0.0"),
            "docs": "/docs",
            "health": "/api/health",
        }

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"未处理的异常: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": "服务器内部错误", "error_type": type(exc).__name__},
        )

    logger.info(f"应用启动完成: {config.get('app.name', '')} v{config.get('app.version', '1.0.0')}")
    return app


app = create_app()


def main():
    import uvicorn

    host = os.getenv("APP_HOST", app.state.config.get("app.host", "0.0.0.0"))
    port = int(os.getenv("APP_PORT", app.state.config.get("app.port", 8000)))
    debug = app.state.config.get("app.debug", False)

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=debug,
    )


if __name__ == "__main__":
    main()