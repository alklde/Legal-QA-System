"""法律知识图谱智能问答系统 - 配置管理模块"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class ConfigError(Exception):
    """配置错误"""

    pass


class Config:
    """配置管理器"""

    _instance: "Config" = None

    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._default_config_path()
        self._config_data = self._load_config()
        self._apply_env_overrides()

    @classmethod
    def get_instance(cls) -> "Config":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_instance(cls, config: "Config"):
        """设置单例实例（用于测试）"""
        cls._instance = config

    def _default_config_path(self) -> str:
        """获取默认配置文件路径"""
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "config" / "settings.yaml")

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        config_path = Path(self.config_path)
        if not config_path.exists():
            raise ConfigError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config or {}

    def _apply_env_overrides(self):
        """应用环境变量覆盖"""
        # Neo4j配置
        if os.getenv("NEO4J_URI"):
            self.set("neo4j.uri", os.getenv("NEO4J_URI"))
        if os.getenv("NEO4J_USER"):
            self.set("neo4j.user", os.getenv("NEO4J_USER"))
        if os.getenv("NEO4J_PASSWORD"):
            self.set("neo4j.password", os.getenv("NEO4J_PASSWORD"))

        # LLM配置
        if os.getenv("LLM_PROVIDER"):
            self.set("llm.provider", os.getenv("LLM_PROVIDER"))
        if os.getenv("OPENAI_API_KEY"):
            self.set("llm.openai.api_key", os.getenv("OPENAI_API_KEY"))
        if os.getenv("LLM_MODEL"):
            self.set("llm.model", os.getenv("LLM_MODEL"))

        # Redis配置
        if os.getenv("REDIS_URL"):
            self.set("cache.redis.url", os.getenv("REDIS_URL"))

        # 日志配置
        if os.getenv("LOG_LEVEL"):
            self.set("logging.level", os.getenv("LOG_LEVEL"))
        if os.getenv("LOG_DIR"):
            self.set("logging.file", f"{os.getenv('LOG_DIR')}/app.log")

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号路径"""
        value = self._config_data
        for part in key.split("."):
            if isinstance(value, dict):
                value = value.get(part)
                if value is None:
                    return default
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """设置配置值"""
        parts = key.split(".")
        current = self._config_data
        for i, part in enumerate(parts[:-1]):
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def resolve_env(self, value: Any) -> Any:
        """解析环境变量引用 ${VAR_NAME}"""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, "")
        if isinstance(value, dict):
            return {k: self.resolve_env(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve_env(v) for v in value]
        return value

    def get_resolved(self, key: str, default: Any = None) -> Any:
        """获取已解析环境变量的配置值"""
        value = self.get(key, default)
        return self.resolve_env(value)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return dict(self._config_data)

    @property
    def app_name(self) -> str:
        return self.get("app.name", "legal-kg-agent")

    @property
    def app_version(self) -> str:
        return self.get("app.version", "1.0.0")

    @property
    def debug(self) -> bool:
        return self.get("app.debug", False)

    @property
    def neo4j_config(self) -> Dict[str, str]:
        return {
            "uri": self.get_resolved("neo4j.uri", "bolt://localhost:7687"),
            "user": self.get_resolved("neo4j.user", "neo4j"),
            "password": self.get_resolved("neo4j.password", ""),
        }

    @property
    def llm_config(self) -> Dict[str, Any]:
        return {
            "provider": self.get("llm.provider", "openai"),
            "model": self.get("llm.model", "gpt-4"),
            "temperature": self.get("llm.temperature", 0.1),
            "max_tokens": self.get("llm.max_tokens", 2000),
            "timeout": self.get("llm.timeout", 30),
            "api_key": self.get_resolved("llm.openai.api_key", ""),
        }


def load_config(config_path: str = None) -> Config:
    """加载配置"""
    return Config(config_path)


def get_config() -> Config:
    """获取全局配置"""
    return Config.get_instance()