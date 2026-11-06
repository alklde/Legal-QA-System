"""法律知识图谱智能问答系统 - 文本处理工具"""

import re
import hashlib
from typing import List, Optional


class TextProcessor:
    """文本处理工具"""

    # 中文标点正则
    CHINESE_PUNCTUATION = r"[，。；：""''（）【】《》、！？…—·％]"

    # 数字正则
    NUMBER_PATTERN = r"\d+"

    # 日期正则
    DATE_PATTERNS = [
        r"\d{4}年\d{1,2}月\d{1,2}日",
        r"\d{4}年\d{1,2}月",
        r"\d{1,2}月\d{1,2}日",
    ]

    # 法条引用正则
    ARTICLE_PATTERNS = [
        r"第(\d+)条",
        r"《[^》]*?》第(\d+)条",
    ]

    @classmethod
    def clean_text(cls, text: str) -> str:
        """清理文本"""
        if not text:
            return ""

        # 去除多余空白
        text = re.sub(r"\s+", " ", text)

        # 统一空格
        text = text.strip()

        return text

    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """规范化空白字符"""
        return re.sub(r"\s+", "", text)

    @classmethod
    def extract_numbers(cls, text: str) -> List[int]:
        """提取文本中的数字"""
        matches = re.findall(cls.NUMBER_PATTERN, text)
        return [int(m) for m in matches]

    @classmethod
    def extract_article_ids(cls, text: str) -> List[int]:
        """提取法条编号"""
        article_ids = []
        for pattern in cls.ARTICLE_PATTERNS:
            matches = re.findall(pattern, text)
            article_ids.extend(int(m) for m in matches)

        return list(set(article_ids))

    @classmethod
    def extract_dates(cls, text: str) -> List[str]:
        """提取日期"""
        dates = []
        for pattern in cls.DATE_PATTERNS:
            matches = re.findall(pattern, text)
            dates.extend(matches)

        return list(set(dates))

    @classmethod
    def extract_court_name(cls, text: str) -> Optional[str]:
        """提取法院名称"""
        pattern = r"([\u4e00-\u9fa5]{2,}(?:市|区|县|省|自治[区县])人民法院)"
        matches = re.findall(pattern, text)
        return matches[0] if matches else None

    @classmethod
    def extract_location(cls, text: str) -> List[str]:
        """提取地点"""
        pattern = r"([\u4e00-\u9fa5]{2,}(?:市|区|县|镇|乡|街道|路|村|省))"
        matches = re.findall(pattern, text)

        # 过滤掉法院等机构名
        locations = []
        for m in matches:
            if "人民" not in m and "法院" not in m:
                locations.append(m)

        return list(set(locations))

    @classmethod
    def truncate_text(cls, text: str, max_length: int = 200) -> str:
        """截断文本"""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    @classmethod
    def compute_hash(cls, text: str) -> str:
        """计算文本哈希"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @classmethod
    def is_valid_fact(cls, text: str, min_length: int = 10) -> bool:
        """验证案情文本是否有效"""
        return isinstance(text, str) and len(text) >= min_length

    @classmethod
    def clean_entity_name(cls, name: str) -> str:
        """清理实体名称"""
        if not name:
            return ""

        # 去除特殊字符
        name = re.sub(cls.CHINESE_PUNCTUATION, "", name)

        # 去除空白
        name = re.sub(r"\s+", "", name)

        return name.strip()

    @classmethod
    def format_months(cls, months: int) -> str:
        """格式化月份为可读文本"""
        if months <= 0:
            return "无期" if months == -2 else ("死刑" if months == -1 else "未知")

        years, rem_months = divmod(months, 12)

        if years == 0:
            return f"{rem_months}个月"
        elif rem_months == 0:
            return f"{years}年"
        else:
            return f"{years}年{rem_months}个月"


text_processor = TextProcessor()