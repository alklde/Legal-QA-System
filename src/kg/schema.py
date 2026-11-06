"""法律知识图谱Schema定义（CAIL2024多人多罪模型）"""

from typing import Dict, Tuple


class LegalKGSchema:
    """法律知识图谱Schema定义

    节点关系采用多人多罪模型：
      (Case)-[:INVOLVES]->(Criminal)
      (Criminal)-[:CHARGED_WITH]->(Accusation)
      (Criminal)-[:SENTENCED_TO]->(Sentence)
      (Case)-[:CITES]->(Article)
      (Accusation)-[:BASED_ON]->(Article)
    """

    NODE_TYPES: Dict[str, list] = {
        "Case": ["case_id", "fact", "court_view"],
        "Criminal": ["name", "gender", "birth_year"],
        "Accusation": ["name", "category"],
        "Article": ["article_id", "content", "chapter"],
        "Sentence": ["type", "months", "fine"],
    }

    RELATION_TYPES: Dict[str, tuple] = {
        "INVOLVES": ("Case", "Criminal"),
        "CHARGED_WITH": ("Criminal", "Accusation"),
        "SENTENCED_TO": ("Criminal", "Sentence"),
        "CITES": ("Case", "Article"),
        "BASED_ON": ("Accusation", "Article"),
    }

    ACCUSATION_CATEGORIES = {
        "危害国家安全罪": ["间谍", "颠覆", "分裂", "叛国"],
        "危害公共安全罪": ["放火", "爆炸", "投毒", "交通肇事", "危险驾驶"],
        "破坏社会主义市场经济秩序罪": ["走私", "金融诈骗", "信用卡诈骗", "合同诈骗", "内幕交易"],
        "侵犯公民人身权利、民主权利罪": ["故意伤害", "故意杀人", "强奸", "非法拘禁"],
        "侵犯财产罪": ["盗窃", "抢劫", "诈骗", "抢夺", "职务侵占", "敲诈勒索"],
        "妨害社会管理秩序罪": ["赌博", "贩卖毒品", "寻衅滋事", "聚众斗殴"],
        "贪污贿赂罪": ["贪污", "受贿", "行贿", "挪用公款"],
        "渎职罪": ["滥用职权", "玩忽职守"],
        "其他犯罪": ["以危险方法危害公共安全"],
    }

    def __init__(self):
        self.validate()

    def validate(self):
        for rel_type, (start, end) in self.RELATION_TYPES.items():
            if start not in self.NODE_TYPES:
                raise ValueError(f"关系 {rel_type} 的起始节点类型 {start} 不存在")
            if end not in self.NODE_TYPES:
                raise ValueError(f"关系 {rel_type} 的结束节点类型 {end} 不存在")

    def get_node_attributes(self, node_type: str) -> list:
        return self.NODE_TYPES.get(node_type, [])

    def get_relation_types(self) -> list:
        return list(self.RELATION_TYPES.keys())

    def get_relation_endpoints(self, relation_type: str) -> Tuple[str, str]:
        return self.RELATION_TYPES.get(relation_type, (None, None))

    def is_valid_node_type(self, node_type: str) -> bool:
        return node_type in self.NODE_TYPES

    def is_valid_relation(self, relation_type: str, start_type: str, end_type: str) -> bool:
        expected = self.RELATION_TYPES.get(relation_type)
        if not expected:
            return False
        return expected[0] == start_type and expected[1] == end_type

    def categorize_accusation(self, accusation_name: str) -> str:
        for category, keywords in self.ACCUSATION_CATEGORIES.items():
            for keyword in keywords:
                if keyword in accusation_name:
                    return category
        return "其他犯罪"

    def to_dict(self) -> dict:
        return {
            "node_types": self.NODE_TYPES,
            "relation_types": {
                k: {"start": v[0], "end": v[1]}
                for k, v in self.RELATION_TYPES.items()
            },
        }

    def to_cypher(self) -> str:
        statements = []
        for node_type in ["Case", "Criminal", "Accusation", "Article", "Sentence"]:
            constraints = {
                "Case": "case_id",
                "Criminal": "name",
                "Accusation": "name",
                "Article": "article_id",
                "Sentence": "id",
            }
            if node_type in constraints:
                field = constraints[node_type]
                statements.append(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node_type}) "
                    f"REQUIRE n.{field} IS UNIQUE"
                )
        for node_type, fields in self.NODE_TYPES.items():
            for field in fields[:2]:
                statements.append(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:{node_type}) "
                    f"ON (n.{field})"
                )
        return "\n".join(statements)


legal_kg_schema = LegalKGSchema()