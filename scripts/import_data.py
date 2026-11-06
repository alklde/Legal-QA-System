#!/usr/bin/env python
"""从CAIL目录导入数据到知识图谱（对齐scripts/build_kg.py）"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logging
from src.data.loader import CAILParser
from src.data.sample_data import get_sample_cases


def main():
    parser = argparse.ArgumentParser(description="导入CAIL数据（CAIL2024/2025 JSON 或 CAIL2018 JSONL）")
    parser.add_argument("--input", required=True, help="数据文件或目录路径")
    parser.add_argument("--limit", type=int, default=None, help="最多记录数")
    parser.add_argument("--format", default=None, help="指定格式（cail2024/cail2018）")
    print_only = parser.add_argument("--print-only", action="store_true", help="仅打印信息")
    args = parser.parse_args()

    setup_logging(log_dir=str(PROJECT_ROOT / "logs"))
    logger = logging.getLogger("import_data")

    parser_p = CAILParser()
    ds = (
        parser_p.parse_directory(args.input, limit=args.limit)
        if Path(args.input).is_dir()
        else parser_p.parse_file(args.input, limit=args.limit)
    )

    print(f"成功解析 {len(ds)} 条案例")
    for case in ds.cases[:5]:
        print(f"  - {case.case_id[:22]}… 被告={case.defendants} 罪名={case.accusations}")


if __name__ == "__main__":
    main()