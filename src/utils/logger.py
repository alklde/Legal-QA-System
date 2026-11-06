"""法律知识图谱智能问答系统 - 日志管理模块"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path


class LogManager:
    """日志管理器"""

    _initialized = False

    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)

    def setup(self):
        """初始化日志配置"""
        if LogManager._initialized:
            return

        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)

        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # 日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 文件处理器（轮转）
        log_file = os.path.join(self.log_dir, "app.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self.log_level)

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self.log_level)

        # 添加处理器
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        LogManager._initialized = True

        # 记录启动日志
        root_logger.info("日志系统初始化完成")

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """获取日志器"""
        return logging.getLogger(name)


# 便捷函数
def get_logger(name: str = "legal_kg") -> logging.Logger:
    """获取日志器"""
    return LogManager.get_logger(name)


def setup_logging(log_dir: str = None, log_level: str = None):
    """设置日志配置"""
    log_dir = log_dir or os.getenv("LOG_DIR", "logs")
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    LogManager(log_dir=log_dir, log_level=log_level).setup()