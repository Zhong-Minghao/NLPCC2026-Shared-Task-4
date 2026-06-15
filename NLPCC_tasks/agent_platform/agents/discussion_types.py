"""
Data classes for the multi-agent investment committee discussion mode.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DiscussionContext:
    """All context needed for one trading day's discussion."""
    date: str
    current_portfolio: Dict
    fund_pool: List[str]
    historical_prices: Dict
    technical_factors: Dict = field(default_factory=dict)
    news_data: List[Dict] = field(default_factory=list)
    market_data: Dict = field(default_factory=dict)


@dataclass
class PositionStatement:
    """BL agent's investment proposal for one discussion round."""
    proposed_trades: List[Dict]
    target_weights: Dict
    views: List[Dict]
    sentiment_analysis: Dict
    optimization_metrics: Dict
    reasoning_text: str
    bl_chain_of_thought: str
    intermediate_results: Dict
    concessions_made: List[str] = field(default_factory=list)
    round_number: int = 0


@dataclass
class RiskConcerns:
    """Risk agent's concerns and counter-proposals for one discussion round."""
    hard_blocks: List[Dict]      # Rule violations — non-negotiable
    soft_concerns: List[Dict]    # Negotiable risk concerns
    counter_proposal: List[Dict] # Risk agent's suggested alternative trades
    concerns_text: str
    risk_level: str
    requirements_for_approval: str
    round_number: int = 0


@dataclass
class DiscussionRound:
    """Record of one complete round: BL proposal + risk response."""
    round_number: int
    position_statement: PositionStatement
    risk_concerns: RiskConcerns
    timestamp: str


@dataclass
class ConsensusResult:
    """Final consensus reached after all discussion rounds."""
    final_trades: List[Dict]
    approved_trades: List[Dict]
    modified_trades: List[Dict]
    blocked_trades: List[Dict]
    consensus_reasoning: str
    bl_final_thesis: str
    risk_final_assessment: str
    discussion_rounds: int
    key_agreements: List[str]
    key_compromises: List[str]
    risk_level: str
    risk_summary: str
