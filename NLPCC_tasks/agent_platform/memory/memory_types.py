from dataclasses import dataclass, field
from typing import Optional

# Mapping from fund_id to sector label used as memory key.
# When a view is generated for a fund, its sector principles are also retrieved.
FUND_SECTOR_MAP: dict[str, str] = {
    # 金融板块
    "512880.SH": "金融_证券",
    "512800.SH": "金融_银行",
    "512070.SH": "金融_保险",
    # 科技板块
    "159995.SZ": "科技_半导体",
    "159819.SZ": "科技_AI",
    "515880.SH": "科技_通信",
    "159852.SZ": "科技_软件",
    "000688.SH": "科技_科创",
    # 医药板块
    "512010.SH": "医药_医药生物",
    "512170.SH": "医药_医疗保健",
    "159992.SZ": "医药_创新药",
    # 消费板块
    "515170.SH": "消费_食品饮料",
    "512690.SH": "消费_白酒",
    "000932.SH": "消费_中证消费",
    # 周期板块
    "512400.SH": "周期_有色金属",
    "000819.SH": "周期_有色金属",
    "515220.SH": "周期_煤炭",
    "159870.SZ": "周期_化工",
    "000928.SH": "周期_能源",
    # 新能源板块
    "000941.SH": "新能源",
    # 其他大类
    "512200.SH": "房地产",
    "399971.SZ": "传媒",
    "000300.SH": "宽基_沪深300",
    "000905.SH": "宽基_中证500",
    "399006.SZ": "宽基_创业板",
    "000012.SH": "债券_国债",
    "518880.SH": "黄金",
}


@dataclass
class RawObservation:
    """One day's view prediction vs actual outcome for a single fund."""
    date: str
    fund_id: str
    sector: str
    predicted_return: float   # expected_return from view (decimal, e.g. 0.012)
    actual_return: float      # actual pct_change / 100 (decimal)
    confidence: float         # confidence from view
    news_keywords: str        # brief summary of relevant news that day
    view_type: str            # "absolute" or "relative"

    @property
    def prediction_error(self) -> float:
        return self.actual_return - self.predicted_return

    @property
    def direction_correct(self) -> bool:
        if abs(self.predicted_return) < 1e-6:
            return False
        return (self.predicted_return > 0) == (self.actual_return > 0)


@dataclass
class SectorPrinciple:
    """A distilled, reusable rule for a sector or fund extracted via LLM reflection."""
    principle: str            # Human-readable rule + actionable suggestion
    confidence: float         # How reliable this rule is (0-1)
    evidence_count: int       # Number of supporting observations
    last_updated: str         # ISO date string

    def to_dict(self) -> dict:
        return {
            "principle": self.principle,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SectorPrinciple":
        return cls(
            principle=d["principle"],
            confidence=float(d.get("confidence", 0.5)),
            evidence_count=int(d.get("evidence_count", 1)),
            last_updated=d.get("last_updated", ""),
        )
