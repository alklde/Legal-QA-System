"""知识图谱存储层（Neo4j 与 内存实现共用同一查询接口）"""

import logging
import re
from typing import Dict, List, Optional

from ..data.loader import LaneArticleDB
from ..data.models import LegalCase
from .schema import LegalKGSchema, legal_kg_schema

logger = logging.getLogger(__name__)


class KGStoreError(Exception):
    pass


def _months_of(penalty) -> int:
    if penalty is None:
        return 0
    return penalty.months


def _sentence_type(penalty) -> str:
    if penalty is None:
        return "none"
    if penalty.death_penalty:
        return "death_penalty"
    if penalty.life_imprisonment:
        return "life_imprisonment"
    if penalty.imprisonment > 0:
        return "imprisonment"
    if penalty.detention > 0:
        return "detention"
    if penalty.surveillance > 0:
        return "surveillance"
    return "none"


class InMemoryKGStore:
    """内存知识图谱存储（无需Neo4j，用于开发/测试/教学）"""

    def __init__(self, schema: LegalKGSchema = None):
        self.schema = schema or legal_kg_schema
        self.nodes: Dict[str, dict] = {}
        self.edges: List[dict] = []

    # ---------- 构建 ----------
    def upsert_case(self, case: LegalCase):
        case_key = self._key("Case", case.case_id)
        self.nodes.setdefault(
            case_key,
            {
                "type": "Case",
                "id": case.case_id,
                "props": {"case_id": case.case_id, "fact": case.fact, "court_view": case.court_view},
            },
        )

        for outcome in case.outcomes:
            if not outcome.name:
                continue
            person_key = self._key("Criminal", outcome.name)
            self.nodes.setdefault(
                person_key,
                {"type": "Criminal", "id": outcome.name, "props": {"name": outcome.name}},
            )
            self._add_edge(case_key, "INVOLVES", person_key)

            for judgment in outcome.judgments:
                acc_name = judgment.standard_accusation or judgment.accusation
                if not acc_name:
                    continue
                acc_key = self._key("Accusation", acc_name)
                self.nodes.setdefault(
                    acc_key,
                    {
                        "type": "Accusation",
                        "id": acc_name,
                        "props": {
                            "name": acc_name,
                            "category": self.schema.categorize_accusation(acc_name),
                        },
                    },
                )
                self._add_edge(person_key, "CHARGED_WITH", acc_key)

                p = judgment.penalty
                sent_type = _sentence_type(p)
                sent_id = f"{outcome.name}|{acc_name}|{sent_type}|{_months_of(p)}|{p.fine if p else 0}"
                sent_key = self._key("Sentence", sent_id)
                self.nodes.setdefault(
                    sent_key,
                    {
                        "type": "Sentence",
                        "id": sent_id,
                        "props": {
                            "id": sent_id,
                            "type": sent_type,
                            "months": _months_of(p),
                            "fine": p.fine if p else 0,
                        },
                    },
                )
                self._add_edge(person_key, "SENTENCED_TO", sent_key)

                for article_id in judgment.related_law:
                    art_key = self._key("Article", article_id)
                    self.nodes.setdefault(
                        art_key,
                        {
                            "type": "Article",
                            "id": article_id,
                            "props": {
                                "article_id": article_id,
                                "content": self._article_content(article_id),
                                "chapter": "",
                            },
                        },
                    )
                    self._add_edge(case_key, "CITES", art_key)
                    self._add_edge(acc_key, "BASED_ON", art_key)

    def _add_edge(self, head: str, rel: str, tail: str):
        triple = (head, rel, tail)
        if any(e for e in self.edges if e["head"] == head and e["rel"] == rel and e["tail"] == tail):
            return
        self.edges.append({"head": head, "rel": rel, "tail": tail})

    def _key(self, node_type: str, node_id) -> str:
        return f"{node_type}:{node_id}"

    def _article_content(self, article_id) -> str:
        from ..data.loader import LaneArticleDB

        return LaneArticleDB().content(article_id)

    # ---------- 构建期统计预计算 ----------
    def calculate_statistics(self) -> dict:
        """构建结束后预计算各罪名统计特征（样本量、刑期均值/标准差），供置信度计算使用。"""
        acc_stats = {}
        for node in self.nodes.values():
            if node["type"] != "Accusation":
                continue
            acc_key = self._key("Accusation", node["id"])
            stats = self._stats_for_accusation(acc_key)
            acc_stats[node["id"]] = {
                "n_cases": stats.get("total_cases", 0),
                "n_defendants": stats.get("total_defendants", 0),
                "avg_months": stats.get("avg_months", 0),
                "std_months": stats.get("std_months", 0),
                "cv": stats.get("cv", 0),
            }
        return acc_stats

    # ---------- 查询 ----------
    def get_stats(self) -> dict:
        node_counts = {}
        for node in self.nodes.values():
            node_counts[node["type"]] = node_counts.get(node["type"], 0) + 1
        relation_counts = {}
        for e in self.edges:
            relation_counts[e["rel"]] = relation_counts.get(e["rel"], 0) + 1

        top_accusations = []
        acc_counter = {}
        for e in self.edges:
            if e["rel"] == "CHARGED_WITH":
                acc_id = e["tail"]
                acc_counter[acc_id] = acc_counter.get(acc_id, 0) + 1
        for acc_key, count in sorted(acc_counter.items(), key=lambda kv: -kv[1])[:10]:
            node = self.nodes.get(acc_key)
            if node:
                top_accusations.append({"name": node["props"]["name"], "count": count})

        return {
            "node_counts": node_counts,
            "relation_counts": relation_counts,
            "top_accusations": top_accusations,
        }

    def query_by_accusation(
        self, name: str, include_cases: bool = False, case_limit: int = 10
    ) -> Optional[dict]:
        matching = [
            n
            for n in self.nodes.values()
            if n["type"] == "Accusation"
            and (name in n["props"]["name"] or n["props"]["name"] in name)
        ]
        if not matching:
            return None

        acc_node = min(matching, key=lambda n: abs(len(n["props"]["name"]) - len(name)))
        acc_key = self._key("Accusation", acc_node["id"])

        related = self._related_accusations_from(acc_key)
        stats = self._stats_for_accusation(acc_key)

        result = {
            "name": acc_node["props"]["name"],
            "category": acc_node["props"]["category"],
            "articles": self._articles_for_accusation(acc_key),
            "related_accusations": [r[0] for r in related],
            "statistics": stats,
        }
        if include_cases:
            result["cases"] = self._cases_for_accusation(acc_key, case_limit)
        return result

    def _stats_for_accusation(self, acc_key: str) -> dict:
        months = []
        total = 0
        case_ids = set()
        for e in self.edges:
            if e["rel"] == "CHARGED_WITH" and e["tail"] == acc_key:
                criminal = e["head"]
                for se in self.edges:
                    if se["rel"] == "SENTENCED_TO" and se["head"] == criminal:
                        total += 1
                        sent = self.nodes[se["tail"]]["props"]
                        if sent["type"] == "imprisonment" and sent["months"] > 0:
                            months.append(sent["months"])
                        for ie in self.edges:
                            if ie["rel"] == "INVOLVES" and ie["tail"] == criminal:
                                cnode = self.nodes.get(ie["head"])
                                if cnode and cnode["type"] == "Case":
                                    case_ids.add(cnode["props"]["case_id"])
        if not months:
            return {
                "total_defendants": total,
                "total_cases": len(case_ids),
                "avg_months": 0,
                "std_months": 0,
                "cv": 0,
            }
        import statistics as _stats_mod

        std_months = round(_stats_mod.stdev(months), 1) if len(months) > 1 else 0.0
        avg_months = sum(months) / len(months)
        return {
            "total_defendants": total,
            "total_cases": len(case_ids),
            "avg_months": round(avg_months, 1),
            "min_months": min(months),
            "max_months": max(months),
            "median_months": sorted(months)[len(months) // 2],
            "std_months": std_months,
            "cv": round(std_months / avg_months, 3) if avg_months > 0 else 0,
        }

    def _articles_for_accusation(self, acc_key: str) -> List[int]:
        arts = []
        for e in self.edges:
            if e["rel"] == "BASED_ON" and e["head"] == acc_key:
                arts.append(e["tail"])
        return list(dict.fromkeys(arts))

    def _related_accusations_from(self, acc_key: str, limit: int = 5) -> List[tuple]:
        counter = {}
        for e in self.edges:
            if e["rel"] == "CHARGED_WITH" and e["tail"] == acc_key:
                criminal = e["head"]
                for e2 in self.edges:
                    if (
                        e2["rel"] == "CHARGED_WITH"
                        and e2["head"] == criminal
                        and e2["tail"] != acc_key
                    ):
                        counter[e2["tail"]] = counter.get(e2["tail"], 0) + 1
        names = []
        for acc2_key, count in sorted(counter.items(), key=lambda kv: -kv[1])[:limit]:
            node = self.nodes.get(acc2_key)
            if node:
                names.append((node["props"]["name"], count))
        return names

    def _cases_for_accusation(self, acc_key: str, limit: int) -> List[dict]:
        case_ids = set()
        for e in self.edges:
            if e["rel"] == "CHARGED_WITH" and e["tail"] == acc_key:
                criminal = e["head"]
                for ie in self.edges:
                    if ie["rel"] == "INVOLVES" and ie["tail"] == criminal:
                        cnode = self.nodes.get(ie["head"])
                        if cnode and cnode["type"] == "Case":
                            case_ids.add(cnode["props"]["case_id"])
        cases = []
        for cid in sorted(case_ids)[:limit]:
            node = self.nodes.get(self._key("Case", cid))
            if not node:
                continue
            cases.append(
                {
                    "case_id": cid,
                    "fact": node["props"]["fact"][:200],
                    "accusation": self.nodes[acc_key]["props"]["name"],
                }
            )
        return cases

    def query_article(
        self, article_id, include_cases: bool = False, case_limit: int = 10
    ) -> Optional[dict]:
        art_key = self._key("Article", str(article_id))
        node = self.nodes.get(art_key)
        if not node:
            return None
        case_ids = []
        for e in self.edges:
            if e["rel"] == "CITES" and e["tail"] == art_key:
                cnode = self.nodes[e["head"]]
                if cnode["type"] == "Case":
                    case_ids.append(cnode["props"]["case_id"])
        acc_names = []
        for e in self.edges:
            if e["rel"] == "BASED_ON" and e["tail"] == art_key:
                acc_node = self.nodes[e["head"]]
                if acc_node:
                    acc_names.append(acc_node["props"]["name"])
        result = {
            "article_id": node["props"]["article_id"],
            "content": node["props"]["content"],
            "related_accusations": list(dict.fromkeys(acc_names)),
            "case_count": len(set(case_ids)),
        }
        if include_cases:
            result["cases"] = [
                {
                    "case_id": cid,
                    "fact": self.nodes[self._key("Case", cid)]["props"]["fact"][:200],
                }
                for cid in list(dict.fromkeys(case_ids))[:case_limit]
            ]
        return result

    def query_sentence_distribution(self, accusation_name: str = None) -> dict:
        if not accusation_name:
            months = []
            types = {}
            for node in self.nodes.values():
                if node["type"] == "Sentence":
                    p = node["props"]
                    types[p["type"]] = types.get(p["type"], 0) + 1
                    if p["type"] == "imprisonment" and p["months"] > 0:
                        months.append(p["months"])
            return {
                "total_defendants": types.get("imprisonment", 0),
                "avg_months": round(sum(months) / len(months), 1) if months else 0,
                "distribution": types,
            }
        acc = self.query_by_accusation(accusation_name)
        if acc and acc.get("statistics"):
            return acc["statistics"]
        return {"total_defendants": 0, "avg_months": 0}

    def query_related_accusations(self, accusation_name: str, limit: int = 10) -> list:
        matching = [
            n
            for n in self.nodes.values()
            if n["type"] == "Accusation"
            and (accusation_name in n["props"]["name"] or n["props"]["name"] in accusation_name)
        ]
        if not matching:
            return []
        acc = min(matching, key=lambda n: abs(len(n["props"]["name"]) - len(accusation_name)))
        return [name for name, _ in self._related_accusations_from(acc["id"], limit)]

    def _to_keywords(self, text: str) -> list:
        tokens = re.findall(r"[\u4e00-\u9fa5]{2,}|[a-zA-Z0-9]+", text.lower())
        keywords = set()
        for token in tokens:
            if re.match(r"^[a-zA-Z0-9]+$", token):
                keywords.add(token)
            for i in range(len(token)):
                unit = token[i : i + 2]
                if len(unit) == 2:
                    keywords.add(unit)
            keywords.add(token)
        return list(keywords)

    def search_case_text(self, query: str, limit: int = 10) -> List[dict]:
        keywords = self._to_keywords(query)
        results = []
        for node in self.nodes.values():
            if node["type"] != "Case":
                continue
            props = node["props"]
            fact = props.get("fact", "")
            court_view = props.get("court_view", "")
            accusation_names = self._accusations_for_case(node["id"])
            searchable = f"{fact} {court_view} {' '.join(accusation_names)}".lower()
            score = sum(1 for kw in keywords if kw and kw in searchable)
            if score > 0:
                results.append(
                    {
                        "case_id": props.get("case_id", node["id"]),
                        "fact": fact[:300],
                        "relevance": score / max(len(keywords), 1),
                    }
                )
        results.sort(key=lambda r: -r["relevance"])
        if results:
            return results[:limit]
        if keywords and all(k in {"类似", "案例", "相似", "类案"} for k in keywords):
            return [
                {
                    "case_id": node["props"]["case_id"],
                    "fact": node["props"]["fact"][:300],
                    "relevance": 0.5,
                }
                for node in self.nodes.values()
                if node["type"] == "Case"
            ][:limit]
        return results[:limit]

    def _accusations_for_case(self, case_key: str) -> List[str]:
        names = []
        for e in self.edges:
            if e["rel"] == "INVOLVES" and e["head"] == case_key:
                criminal_key = e["tail"]
                for e2 in self.edges:
                    if e2["rel"] == "CHARGED_WITH" and e2["head"] == criminal_key:
                        acc_node = self.nodes.get(e2["tail"])
                        if acc_node:
                            names.append(acc_node["props"]["name"])
        return names


class Neo4jKGStore:
    """Neo4j知识图谱存储"""

    def __init__(self, driver, schema: LegalKGSchema = None):
        self.driver = driver
        self.schema = schema or legal_kg_schema

    def _session(self):
        return self.driver.session()

    def upsert_case(self, case: LegalCase):
        cypher = self._upsert_cypher()
        outcomes = []
        for o in case.outcomes:
            js = []
            for j in o.judgments:
                js.append(
                    {
                        "name": o.name,
                        "accusation": j.standard_accusation or j.accusation,
                        "law": [str(a) for a in j.related_law],
                        "penalty": j.penalty.to_dict() if j.penalty else {},
                    }
                )
            outcomes.extend(js)
        try:
            with self._session() as session:
                for item in outcomes:
                    session.run(
                        cypher,
                        case_id=case.case_id,
                        fact=case.fact,
                        court_view=case.court_view,
                        name=item["name"],
                        accusation=item["accusation"],
                        category=self.schema.categorize_accusation(item["accusation"]),
                        law=item["law"],
                        penalty=item["penalty"],
                        sentence_type=_sentence_type(self._penalty_obj(item["penalty"])),
                        months=_months_of(self._penalty_obj(item["penalty"])),
                        fine=item["penalty"].get("fine", 0),
                    )
        except Exception as e:
            logger.error(f"Neo4j写入失败: {e}")
            raise KGStoreError(f"Neo4j写入失败: {e}")

    def _penalty_obj(self, d: dict):
        from ..data.models import Penalty

        return Penalty.from_dict(d)

    def upsert_cases_batch(self, cases: List[LegalCase], batch_size: int = 500) -> int:
        """批量写入多个案例（UNWIND 一次提交，返回成功写入的案例数）"""
        if not cases:
            return 0
        rows: List[dict] = []
        for case in cases:
            for o in case.outcomes:
                for j in o.judgments:
                    accusation = j.standard_accusation or j.accusation
                    if not accusation:
                        continue
                    p = j.penalty
                    penalty_dict = p.to_dict() if p else {}
                    sentence_id = f"{o.name}|{accusation}|{_sentence_type(p)}|{_months_of(p)}|{penalty_dict.get('fine', 0)}"
                    rows.append(
                        {
                            "case_id": case.case_id,
                            "fact": case.fact,
                            "court_view": case.court_view,
                            "name": o.name,
                            "accusation": accusation,
                            "category": self.schema.categorize_accusation(accusation),
                            "law": [str(a) for a in j.related_law],
                            "penalty": penalty_dict,
                            "sentence_type": _sentence_type(p),
                            "months": _months_of(p),
                            "fine": penalty_dict.get("fine", 0),
                            "sentence_id": sentence_id,
                        }
                    )
        if not rows:
            return len(cases)
        succeeded = 0
        try:
            with self._session() as session:
                for i in range(0, len(rows), batch_size):
                    chunk = rows[i : i + batch_size]
                    session.run(self._upsert_batch_cypher(), rows=chunk)
            succeeded = len(cases)
        except Exception as e:
            logger.error(f"Neo4j批量写入失败: {e}")
            for case in cases:
                try:
                    self.upsert_case(case)
                    succeeded += 1
                except Exception as e2:
                    logger.warning(f"批量回退失败 {case.case_id}: {e2}")
            raise KGStoreError(f"Neo4j批量写入失败(含回退): {e}")
        return succeeded

    def _upsert_batch_cypher(self) -> str:
        return """
        UNWIND $rows AS row
        MERGE (c:Case {case_id: row.case_id})
        SET c.fact = row.fact, c.court_view = row.court_view
        MERGE (person:Criminal {name: row.name})
        MERGE (c)-[:INVOLVES]->(person)
        MERGE (acc:Accusation {name: row.accusation})
        SET acc.category = row.category
        MERGE (person)-[:CHARGED_WITH]->(acc)
        MERGE (s:Sentence {id: row.sentence_id})
        SET s.type = row.sentence_type, s.months = row.months, s.fine = row.fine
        MERGE (person)-[:SENTENCED_TO]->(s)
        FOREACH (aid IN row.law |
            MERGE (art:Article {article_id: aid})
            MERGE (c)-[:CITES]->(art)
            MERGE (acc)-[:BASED_ON]->(art)
        )
        """

    def _upsert_cypher(self) -> str:
        return """
        MERGE (c:Case {case_id: $case_id})
        SET c.fact = $fact, c.court_view = $court_view
        MERGE (person:Criminal {name: $name})
        MERGE (c)-[:INVOLVES]->(person)
        MERGE (acc:Accusation {name: $accusation})
        SET acc.category = $category
        MERGE (person)-[:CHARGED_WITH]->(acc)
        MERGE (s:Sentence {id: $name + '|' + $accusation + '|' + $sentence_type + '|' + toString($months) + '|' + toString($fine)})
        SET s.type = $sentence_type, s.months = $months, s.fine = $fine
        MERGE (person)-[:SENTENCED_TO]->(s)
        FOREACH (aid IN $law |
            MERGE (art:Article {article_id: aid})
            MERGE (c)-[:CITES]->(art)
            MERGE (acc)-[:BASED_ON]->(art)
        )
        """

    def calculate_statistics(self) -> dict:
        """构建结束后预计算各罪名统计特征（样本量、刑期均值/标准差/中位数）并写回节点属性。

        供置信度计算使用；无需权重/向量等复杂设计，仅两次轻量聚合查询。
        返回 {罪名: 统计dict}，并 SET 到 Accusation 节点（stats_* 属性）供运行时直接读取。
        """
        try:
            with self._session() as session:
                count_rows = session.run(
                    """
                    MATCH (a:Accusation)<-[:CHARGED_WITH]-(cr:Criminal)
                    OPTIONAL MATCH (c:Case)-[:INVOLVES]->(cr)
                    RETURN a.name AS name,
                           count(DISTINCT cr) AS n_defendants,
                           count(DISTINCT c) AS n_cases
                    """
                ).data()
                months_rows = session.run(
                    """
                    MATCH (a:Accusation)<-[:CHARGED_WITH]-(cr:Criminal)
                    MATCH (cr)-[:SENTENCED_TO]->(s:Sentence)
                    WHERE s.type = 'imprisonment' AND s.months > 0
                    RETURN a.name AS name,
                           avg(s.months) AS avg_months,
                           min(s.months) AS min_months,
                           max(s.months) AS max_months,
                           stDev(s.months) AS std_months,
                           percentileCont(s.months, 0.5) AS median_months
                    """
                ).data()
            acc_stats = {}
            for row in count_rows:
                acc_stats[row["name"]] = {
                    "n_cases": row["n_cases"],
                    "n_defendants": row["n_defendants"],
                    "avg_months": 0,
                    "min_months": 0,
                    "max_months": 0,
                    "std_months": 0,
                    "median_months": 0,
                    "cv": 0,
                }
            for row in months_rows:
                name = row["name"]
                if name not in acc_stats:
                    continue
                avg = row["avg_months"] or 0
                std = row["std_months"] or 0
                med = row["median_months"] or 0
                acc_stats[name]["avg_months"] = round(float(avg), 2)
                acc_stats[name]["min_months"] = row["min_months"] or 0
                acc_stats[name]["max_months"] = row["max_months"] or 0
                acc_stats[name]["std_months"] = round(float(std), 2)
                acc_stats[name]["median_months"] = round(float(med), 2)
                acc_stats[name]["cv"] = round(float(std) / avg, 3) if avg > 0 else 0
            self._persist_statistics(acc_stats)
            return acc_stats
        except Exception as e:
            logger.error(f"Neo4j统计特征计算失败: {e}")
            raise KGStoreError(f"Neo4j统计特征计算失败: {e}")

    def _persist_statistics(self, acc_stats: dict) -> None:
        """把预计算统计写回 Accusation 节点属性（stats_*），供运行时直接读取避免重复聚合。"""
        try:
            with self._session() as session:
                for name, st in acc_stats.items():
                    session.run(
                        """
                        MATCH (a:Accusation {name: $name})
                        SET a.stats_n_cases = $n_cases,
                            a.stats_n_defendants = $n_defendants,
                            a.stats_avg_months = $avg_months,
                            a.stats_min_months = $min_months,
                            a.stats_max_months = $max_months,
                            a.stats_std_months = $std_months,
                            a.stats_median_months = $median_months,
                            a.stats_cv = $cv
                        """,
                        name=name,
                        n_cases=int(st["n_cases"]),
                        n_defendants=int(st["n_defendants"]),
                        avg_months=st["avg_months"],
                        min_months=int(st["min_months"]),
                        max_months=int(st["max_months"]),
                        std_months=st["std_months"],
                        median_months=st["median_months"],
                        cv=st["cv"],
                    )
        except Exception as e:
            logger.warning(f"写回罪名统计属性失败: {e}")

    def get_stats(self) -> dict:
        node_counts, relation_counts = {}, {}
        try:
            with self._session() as session:
                for ntype in self.schema.NODE_TYPES:
                    r = session.run(f"MATCH (n:{ntype}) RETURN count(n) AS c").single()
                    node_counts[ntype] = r["c"]
                for rel in self.schema.RELATION_TYPES:
                    r = session.run(
                        f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c"
                    ).single()
                    relation_counts[rel] = r["c"]
                top = session.run(
                    "MATCH (cr:Criminal)-[:CHARGED_WITH]->(a:Accusation) "
                    "RETURN a.name AS name, count(*) AS count ORDER BY count DESC LIMIT 10"
                ).data()
            return {
                "node_counts": node_counts,
                "relation_counts": relation_counts,
                "top_accusations": top,
            }
        except Exception as e:
            logger.error(f"Neo4j统计失败: {e}")
            raise KGStoreError(f"Neo4j统计失败: {e}")

    def query_by_accusation(
        self, name: str, include_cases: bool = False, case_limit: int = 10
    ) -> Optional[dict]:
        try:
            with self._session() as session:
                r = session.run(
                    """
                    MATCH (a:Accusation)
                    WHERE a.name CONTAINS $name OR $name CONTAINS a.name
                    RETURN a
                    """,
                    name=name,
                ).data()
                if not r:
                    return None
                acc = r[0]["a"]
                acc_name = acc["name"]

                node_st = {
                    k: acc.get(k)
                    for k in (
                        "stats_n_cases",
                        "stats_n_defendants",
                        "stats_avg_months",
                        "stats_min_months",
                        "stats_max_months",
                        "stats_std_months",
                        "stats_median_months",
                        "stats_cv",
                    )
                }
                if node_st["stats_n_defendants"] is not None:
                    stat_dict = {
                        "total_defendants": int(node_st["stats_n_defendants"]),
                        "total_cases": int(node_st["stats_n_cases"] or 0),
                        "avg_months": node_st["stats_avg_months"] or 0,
                        "min_months": node_st["stats_min_months"] or 0,
                        "max_months": node_st["stats_max_months"] or 0,
                        "median_months": node_st["stats_median_months"] or 0,
                        "std_months": node_st["stats_std_months"] or 0,
                        "cv": node_st["stats_cv"] or 0,
                    }
                else:
                    stats = session.run(
                        """
                        MATCH (cr:Criminal)-[:CHARGED_WITH]->(a:Accusation {name: $name})
                        MATCH (cr)-[:SENTENCED_TO]->(s:Sentence)
                        WHERE s.type = 'imprisonment' AND s.months > 0
                        OPTIONAL MATCH (c:Case)-[:INVOLVES]->(cr)
                        RETURN count(DISTINCT cr) AS total_defendants,
                               count(DISTINCT c) AS total_cases,
                               avg(s.months) AS avg_months,
                               min(s.months) AS min_months,
                               max(s.months) AS max_months,
                               stDev(s.months) AS std_months,
                               percentileCont(s.months, 0.5) AS median_months
                        """,
                        name=acc_name,
                    ).single()
                    avg_m = float(stats["avg_months"] or 0)
                    std_months = round(float(stats["std_months"] or 0), 1)
                    cv = round(std_months / avg_m, 3) if avg_m > 0 else 0
                    stat_dict = {
                        "total_defendants": stats["total_defendants"],
                        "total_cases": stats["total_cases"],
                        "avg_months": round(avg_m, 1),
                        "min_months": stats["min_months"] or 0,
                        "max_months": stats["max_months"] or 0,
                        "median_months": stats["median_months"] or 0,
                        "std_months": std_months,
                        "cv": cv,
                    }

                related = session.run(
                    """
                    MATCH (a1:Accusation {name: $name})
                    MATCH (cr:Criminal)-[:CHARGED_WITH]->(a1)
                    MATCH (cr)-[:CHARGED_WITH]->(a2:Accusation)
                    WHERE a2.name <> a1.name
                    RETURN a2.name AS name, count(*) AS count
                    ORDER BY count DESC LIMIT 5
                    """,
                    name=acc_name,
                ).data()

                articles = session.run(
                    """
                    MATCH (a:Accusation {name: $name})-[:BASED_ON]->(art:Article)
                    RETURN art.article_id AS article_id
                    """,
                    name=acc_name,
                ).data()

                result = {
                    "name": acc_name,
                    "category": acc.get("category", ""),
                    "articles": [a["article_id"] for a in articles],
                    "related_accusations": [x["name"] for x in related],
                    "statistics": stat_dict,
                }
                if include_cases:
                    cases = session.run(
                        """
                        MATCH (cr:Criminal)-[:CHARGED_WITH]->(a:Accusation {name: $name})
                        MATCH (c:Case)-[:INVOLVES]->(cr)
                        RETURN DISTINCT c.case_id AS case_id, c.fact AS fact
                        LIMIT $limit
                        """,
                        name=acc_name,
                        limit=case_limit,
                    ).data()
                    result["cases"] = [
                        {"case_id": c["case_id"], "fact": (c["fact"] or "")[:200]}
                        for c in cases
                    ]
                return result
        except Exception as e:
            logger.error(f"Neo4j查询罪名失败: {e}")
            raise KGStoreError(f"Neo4j查询罪名失败: {e}")

    def query_article(
        self, article_id, include_cases: bool = False, case_limit: int = 10
    ) -> Optional[dict]:
        try:
            with self._session() as session:
                r = session.run(
                    "MATCH (a:Article {article_id: $aid}) RETURN a",
                    aid=str(article_id),
                ).single()
                if not r:
                    return None
                art = r["a"]

                acc_names = session.run(
                    "MATCH (acc:Accusation)-[:BASED_ON]->(:Article {article_id: $aid}) "
                    "RETURN DISTINCT acc.name AS name",
                    aid=str(article_id),
                ).data()
                case_count = session.run(
                    "MATCH (c:Case)-[:CITES]->(:Article {article_id: $aid}) "
                    "RETURN count(DISTINCT c) AS cnt",
                    aid=str(article_id),
                ).single()["cnt"]
                cases = []
                if include_cases:
                    all_cases = session.run(
                        "MATCH (c:Case)-[:CITES]->(:Article {article_id: $aid}) "
                        "RETURN DISTINCT c.case_id AS case_id, c.fact AS fact LIMIT $limit",
                        aid=str(article_id),
                        limit=case_limit,
                    ).data()
                    cases = all_cases

                return {
                    "article_id": art["article_id"],
                    "content": art.get("content", "") or LaneArticleDB().content(art["article_id"]),
                    "related_accusations": [x["name"] for x in acc_names],
                    "case_count": case_count,
                    **({"cases": [{"case_id": c["case_id"], "fact": (c["fact"] or "")[:200]} for c in cases]} if include_cases else {}),
                }
        except Exception as e:
            logger.error(f"Neo4j查询法条失败: {e}")
            raise KGStoreError(f"Neo4j查询法条失败: {e}")

    def query_sentence_distribution(self, accusation_name: str = None) -> dict:
        try:
            with self._session() as session:
                if accusation_name:
                    r = session.run(
                        """
                        MATCH (cr:Criminal)-[:CHARGED_WITH]->(a:Accusation {name: $name})
                        MATCH (cr)-[:SENTENCED_TO]->(s:Sentence)
                        WHERE s.type = 'imprisonment'
                        RETURN count(DISTINCT cr) AS total, avg(s.months) AS avg
                        """,
                        name=accusation_name,
                    ).single()
                    return {"total_defendants": r["total"], "avg_months": round(r["avg"] or 0, 1)}
                r = session.run(
                    "MATCH (s:Sentence) RETURN s.type AS type, count(*) AS count"
                ).data()
                types = {x["type"]: x["count"] for x in r}
                return {"total_defendants": types.get("imprisonment", 0), "distribution": types}
        except Exception as e:
            raise KGStoreError(f"Neo4j刑期分布失败: {e}")

    def query_related_accusations(self, accusation_name: str, limit: int = 10) -> list:
        try:
            with self._session() as session:
                r = session.run(
                    """
                    MATCH (a1:Accusation {name: $name})
                    MATCH (cr:Criminal)-[:CHARGED_WITH]->(a1)
                    MATCH (cr)-[:CHARGED_WITH]->(a2:Accusation)
                    WHERE a2.name <> a1.name
                    RETURN DISTINCT a2.name AS name LIMIT $limit
                    """,
                    name=accusation_name,
                    limit=limit,
                ).data()
                return [x["name"] for x in r]
        except Exception as e:
            raise KGStoreError(f"Neo4j相关罪名失败: {e}")

    def search_case_text(self, query: str, limit: int = 10) -> List[dict]:
        try:
            with self._session() as session:
                rows = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('case_fact_ft', $q) YIELD node, score
                    RETURN node.case_id AS case_id, node.fact AS fact, score
                    ORDER BY score DESC LIMIT $limit
                    """,
                    q=query,
                    limit=limit,
                ).data()
                if rows:
                    return [
                        {
                            "case_id": x["case_id"],
                            "fact": (x["fact"] or "")[:300],
                            "relevance": round(float(x["score"]), 2),
                        }
                        for x in rows
                    ]
                r = session.run(
                    "MATCH (c:Case) WHERE c.fact CONTAINS $q RETURN c.case_id AS case_id, c.fact AS fact LIMIT $limit",
                    q=query,
                    limit=limit,
                ).data()
                return [
                    {"case_id": x["case_id"], "fact": (x["fact"] or "")[:300], "relevance": 0.5}
                    for x in r
                ]
        except Exception as e:
            raise KGStoreError(f"Neo4j案例检索失败: {e}")


def build_store(driver=None, schema: LegalKGSchema = None):
    if driver is None:
        return InMemoryKGStore(schema)
    return Neo4jKGStore(driver, schema)