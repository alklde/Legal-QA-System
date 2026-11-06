"""CAIL数据加载与解析（支持CAIL2024/2025多人多罪格式与CAIL2018兼容格式）"""

import glob
import hashlib
import json
import logging
import os
import re
from typing import Dict, Generator, List, Optional

from .models import Judgment, LegalCase, Outcome, Penalty

logger = logging.getLogger(__name__)


class CAILDataError(Exception):
    pass


class CAILDataset:
    """统一数据集访问接口"""

    def __init__(self, cases: List[LegalCase]):
        self.cases = cases
        self._by_id = {c.case_id: c for c in cases}

    def __len__(self):
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def get(self, case_id: str) -> Optional[LegalCase]:
        return self._by_id.get(case_id)

    def filter_by_charge(self, keyword: str) -> List[LegalCase]:
        return [c for c in self.cases if keyword in c.accusations]


class CAILParser:
    """CAIL多版本解析器"""

    def __init__(self):
        self._article_db = LaneArticleDB()

    def detect_format(self, obj: dict) -> str:
        if "Facts" in obj:
            return "cail2024"
        if "fact" in obj and "outcomes" in obj and "court_view" in obj:
            return "cmdl"
        if "fact" in obj and "meta" in obj:
            return "cail2018"
        raise CAILDataError(f"无法识别的数据格式: {list(obj.keys())[:10]}")

    def parse_record(self, obj: dict) -> LegalCase:
        fmt = self.detect_format(obj)
        try:
            if fmt in ("cail2024", "cmdl"):
                return self._parse_cail2024(obj)
            return self._parse_cail2018(obj)
        except Exception as e:
            logger.warning(f"解析失败({fmt}): {e}")
            raise

    def parse_file(self, path: str, limit: Optional[int] = None) -> CAILDataset:
        cases: List[LegalCase] = []
        for obj in self._iter_json_lines(path):
            if limit is not None and len(cases) >= limit:
                break
            try:
                parsed = self.parse_record(obj)
                if parsed:
                    cases.append(parsed)
            except (CAILDataError, Exception) as e:
                logger.debug(f"跳过无效记录: {e}")
                continue
        logger.info(f"从 {path} 解析 {len(cases)} 条案例")
        return CAILDataset(cases)

    def parse_directory(
        self, directory: str, pattern: str = "*.json*", limit: Optional[int] = None
    ) -> CAILDataset:
        files = sorted(glob.glob(os.path.join(directory, pattern)))
        # 过滤掉辅助索引文件（article2idx/charge2idx 等 map 文件，非案例数据）
        files = [
            f
            for f in files
            if not os.path.basename(f).lower().startswith(("article2idx", "charge2idx"))
        ]
        if not files:
            raise CAILDataError(f"目录中没有匹配文件: {directory} {pattern}")
        all_cases: List[LegalCase] = []
        for f in files:
            try:
                ds = self.parse_file(f, limit=None if limit is None else max(0, limit - len(all_cases)))
                all_cases.extend(ds.cases)
            except Exception as e:
                logger.warning(f"跳过文件 {f}: {e}")
        return CAILDataset(all_cases)

    def _iter_json_lines(self, path: str) -> Generator[dict, None, None]:
        """流式逐行解析 JSONL（支持数组 JSON）。逐行读取避免大文件整读占内存。"""
        enc = self._pick_encoding(path)
        with open(path, "r", encoding=enc) as f:
            head = f.read(4)
            f.seek(0)
            if head.lstrip().startswith("["):
                arr = json.load(f)
                for obj in arr if isinstance(arr, list) else []:
                    if isinstance(obj, dict):
                        yield obj
                return
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except json.JSONDecodeError:
                    continue

    def _pick_encoding(self, path: str) -> str:
        for enc in ["utf-8", "gb18030", "utf-8-sig"]:
            try:
                with open(path, "r", encoding=enc) as f:
                    f.readline()
                return enc
            except (UnicodeDecodeError, OSError):
                continue
        raise CAILDataError(f"无法读取文件: {path}")

    def _parse_cail2024(self, data: dict) -> LegalCase:
        fact = data.get("Facts", data.get("fact", "")).strip()
        if len(fact) < 10:
            raise CAILDataError("fact 过短")

        defendants_raw = data.get("Defendants", data.get("defendants", []))
        if isinstance(defendants_raw, str):
            defendants_raw = [defendants_raw]

        outcomes_raw = data.get("Outcomes", data.get("outcomes", []))
        if isinstance(outcomes_raw, dict):
            outcomes_raw = [outcomes_raw]

        outcomes: List[Outcome] = []
        for out in outcomes_raw:
            try:
                outcomes.append(Outcome.from_dict(out))
            except Exception as e:
                logger.debug(f"跳过无效判决 {e}")

        if defendants_raw:
            existing = {o.name for o in outcomes}
            for name in defendants_raw:
                if name not in existing:
                    outcomes.append(Outcome(name=name))

        case_id = self._case_id(fact)
        return LegalCase(
            case_id=case_id,
            fact=fact,
            court_view=data.get("Court View", data.get("court_view", "")),
            defendants=defendants_raw,
            outcomes=outcomes,
            metadata={"source": "cail2024"},
        )

    def _parse_cail2018(self, data: dict) -> LegalCase:
        fact = data.get("fact", "").strip()
        meta = data.get("meta", {})
        if len(fact) < 10 or not isinstance(meta, dict):
            raise CAILDataError("cail2018 记录无效")

        criminals = meta.get("criminals", [])
        accusation = meta.get("accusation", [])
        articles = meta.get("relevant_articles", [])

        if not criminals:
            criminals = ["被告人"]

        outcomes = []
        for idx, name in enumerate(criminals):
            charge = accusation[0] if accusation else "其他犯罪"
            art = articles[0] if articles else ""
            term = meta.get("term_of_imprisonment", {})
            penalty = None
            if isinstance(term, dict):
                if term.get("death_penalty"):
                    penalty = Penalty(death_penalty=True)
                elif term.get("life_imprisonment"):
                    penalty = Penalty(life_imprisonment=True)
                elif term.get("imprisonment", 0) > 0:
                    penalty = Penalty(imprisonment=int(term["imprisonment"]))
                penalty = penalty or Penalty(imprisonment=int(term.get("imprisonment", 0)))
            law = [str(a) for a in articles] if articles else ([str(art)] if art else [])
            outcomes.append(
                Outcome(
                    name=name,
                    judgments=[
                        Judgment(
                            accusation=charge,
                            standard_accusation=charge,
                            related_law=law,
                            penalty=penalty,
                        )
                    ],
                )
            )

        return LegalCase(
            case_id=self._case_id(fact),
            fact=fact,
            defendants=criminals,
            outcomes=outcomes,
            metadata={
                "source": "cail2018",
                "punish_of_money": meta.get("punish_of_money", 0),
            },
        )

    def _case_id(self, fact: str) -> str:
        return f"case_{hashlib.md5(fact.encode('utf-8')).hexdigest()[:16]}"


class LaneArticleDB:
    """刑法条文数据库（少量内置，用于图谱Article节点）"""

    _FALLBACK_CONTENT = "（具体条文内容缺失，请以权威法条文本为准。）"

    _ARTICLES: Dict[int, str] = {
        133: "在道路上驾驶机动车，有下列情形之一的，处拘役，并处罚金。……违反交通运输管理法规，因而发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失的，处三年以下有期徒刑或者拘役；交通运输肇事后逃逸或者有其他特别恶劣情节的，处三年以上七年以下有期徒刑；因逃逸致人死亡的，处七年以上有期徒刑。",
        196: "有下列情形之一，进行信用卡诈骗活动，数额较大的，处五年以下有期徒刑或者拘役，并处二万元以上二十万元以下罚金；数额巨大或者有其他严重情节的，处五年以上十年以下有期徒刑，并处五万元以上五十万元以下罚金；数额特别巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处五万元以上五十万元以下罚金或者没收财产：……",
        232: "故意杀人的，处死刑、无期徒刑或者十年以上有期徒刑；情节较轻的，处三年以上十年以下有期徒刑。",
        234: "故意伤害他人身体的，处三年以下有期徒刑、拘役或者管制。犯前款罪，致人重伤的，处三年以上十年以下有期徒刑；致人死亡或者以特别残忍手段致人重伤造成严重残疾的，处十年以上有期徒刑、无期徒刑或者死刑。本法另有规定的，依照规定。",
        263: "以暴力、胁迫或者其他方法抢劫公私财物的，处三年以上十年以下有期徒刑，并处罚金；有下列情形之一的，处十年以上有期徒刑、无期徒刑或者死刑，并处罚金或者没收财产：……",
        264: "盗窃公私财物，数额较大的，或者多次盗窃、入户盗窃、携带凶器盗窃、扒窃的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；数额巨大或者有其他严重情节的，处三年以上十年以下有期徒刑，并处罚金；数额特别巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处罚金或者没收财产。",
        266: "诈骗公私财物，数额较大的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；数额巨大或者有其他严重情节的，处三年以上十年以下有期徒刑，并处罚金；数额特别巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处罚金或者没收财产。本法另有规定的，依照规定。",
        293: "有下列寻衅滋事行为之一，破坏社会秩序的，处五年以下有期徒刑、拘役或者管制：……",
    }

    def content(self, article_id) -> str:
        return self._ARTICLES.get(int(article_id), self._FALLBACK_CONTENT)

    def known(self, article_id) -> bool:
        return int(article_id) in self._ARTICLES


def load_cail_directory(directory: str, limit: Optional[int] = None) -> CAILDataset:
    parser = CAILParser()
    return parser.parse_directory(directory, limit=limit)