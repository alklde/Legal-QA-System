"""API中间件"""

import time
import logging
from typing import Dict, List, Tuple


class RateLimiter:
    """限流器"""

    def __init__(self, max_requests: int = 100, window: int = 60):
        self.max_requests = max_requests
        self.window = window  # 秒
        self.requests: Dict[str, List[float]] = {}

    def check(self, client_key: str) -> Tuple[bool, int]:
        """
        检查是否允许请求

        Returns:
            (是否允许, 剩余秒数)
        """
        now = time.time()

        # 清理过期记录
        self.requests = {
            key: timestamps
            for key, timestamps in self.requests.items()
            if any(t > now - self.window for t in timestamps)
        }

        timestamps = self.requests.get(client_key, [])

        if len(timestamps) >= self.max_requests:
            oldest = min(timestamps)
            retry_after = int(self.window - (now - oldest))
            return False, max(retry_after, 1)

        timestamps.append(now)
        self.requests[client_key] = timestamps

        return True, 0


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.start_time = time.time()
        self.endpoint_stats: Dict[str, Dict] = {}

    def record_request(self, endpoint: str, latency: float, status_code: int):
        """记录请求"""
        self.request_count += 1
        self.total_latency += latency

        if status_code >= 400:
            self.error_count += 1

        stats = self.endpoint_stats.setdefault(
            endpoint, {"count": 0, "errors": 0, "total_latency": 0.0}
        )
        stats["count"] += 1
        stats["total_latency"] += latency
        if status_code >= 400:
            stats["errors"] += 1

    def get_metrics(self) -> dict:
        """获取指标"""
        uptime = time.time() - self.start_time
        avg_latency = self.total_latency / self.request_count if self.request_count else 0

        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count else 0,
            "avg_latency": avg_latency,
            "uptime_seconds": uptime,
            "endpoint_stats": self.endpoint_stats,
        }


# 全局实例
rate_limiter = RateLimiter()
metrics_collector = MetricsCollector()