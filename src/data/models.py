"""数据模型定义（与CAIL数据结构对齐）"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Penalty:
    surveillance: int = 0
    detention: int = 0
    imprisonment: int = 0
    death_penalty: bool = False
    life_imprisonment: bool = False
    fine: int = 0
    fine_without_amount: bool = False

    def to_dict(self) -> dict:
        return {
            "surveillance": self.surveillance,
            "detention": self.detention,
            "imprisonment": self.imprisonment,
            "death_penalty": self.death_penalty,
            "life_imprisonment": self.life_imprisonment,
            "fine": self.fine,
            "fine_without_amount": self.fine_without_amount,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Penalty":
        return cls(
            surveillance=data.get("Surveillance", data.get("surveillance", 0)),
            detention=data.get("Detention", data.get("detention", 0)),
            imprisonment=data.get("Imprisonment", data.get("imprisonment", 0)),
            death_penalty=data.get("Death Penalty", data.get("death_penalty", False)),
            life_imprisonment=data.get(
                "Life Imprisonment", data.get("life_imprisonment", False)
            ),
            fine=data.get("Fine", data.get("fine", 0)),
            fine_without_amount=data.get(
                "Fine Without Amount", data.get("fine_without_amount", False)
            ),
        )

    @property
    def is_imprisonment(self) -> bool:
        return self.imprisonment > 0

    @property
    def months(self) -> int:
        return self.imprisonment or self.detention or self.surveillance


@dataclass
class Judgment:
    accusation: str
    standard_accusation: str
    related_law: List[str] = field(default_factory=list)
    penalty: Optional[Penalty] = None

    def to_dict(self) -> dict:
        return {
            "accusation": self.accusation,
            "standard_accusation": self.standard_accusation,
            "related_law": self.related_law,
            "penalty": self.penalty.to_dict() if self.penalty else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Judgment":
        penalty_data = data.get("Penalty", data.get("penalty", {}))
        law = data.get("Related Law", data.get("related_law", data.get("article", [])))
        if isinstance(law, str):
            law = [law]
        law = [str(x) for x in law]
        return cls(
            accusation=data.get("Accusation", data.get("accusation", "")),
            standard_accusation=data.get(
                "Standard Accusation", data.get("standard_accusation", "")
            ),
            related_law=law,
            penalty=Penalty.from_dict(penalty_data) if penalty_data else None,
        )


@dataclass
class Outcome:
    name: str
    judgments: List[Judgment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "judgments": [j.to_dict() for j in self.judgments],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Outcome":
        raw_judgments = data.get("Judgment", data.get("judgments", data.get("judgment", [])))
        if isinstance(raw_judgments, dict):
            raw_judgments = [raw_judgments]
        return cls(
            name=data.get("Name", data.get("name", "")),
            judgments=[Judgment.from_dict(j) for j in raw_judgments],
        )


@dataclass
class LegalCase:
    case_id: str
    fact: str
    court_view: str = ""
    defendants: List[str] = field(default_factory=list)
    outcomes: List[Outcome] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "fact": self.fact,
            "court_view": self.court_view,
            "defendants": self.defendants,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "metadata": self.metadata,
        }

    @property
    def accusations(self) -> List[str]:
        accs = set()
        for outcome in self.outcomes:
            for j in outcome.judgments:
                accs.add(j.standard_accusation or j.accusation)
        return list(accs)

    @property
    def articles(self) -> List[str]:
        arts = set()
        for outcome in self.outcomes:
            for j in outcome.judgments:
                arts.update(j.related_law)
        return list(arts)