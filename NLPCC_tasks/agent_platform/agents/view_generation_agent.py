"""
View Generation Agent for Black-Litterman Model

This agent generates structured investment views (opinions about expected returns)
that will be used as input to the Black-Litterman portfolio optimization model.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from loguru import logger

load_dotenv()

from agent_platform.utils import CustomJsonOutputParser
from agent_platform.agents.fund_info import FUND_INFO

# Import prompt templates
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_platform.prompts.view_generation_prompt import (
    VIEW_GENERATION_PROMPT,
    SIMPLIFIED_VIEW_GENERATION_PROMPT,
)


@dataclass
class View:
    """Investment view data structure."""
    fund_id: str
    view_type: str  # "absolute" or "relative"
    expected_return: float
    confidence: float  # 0-1
    reason: str
    benchmark_fund_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "fund_id": self.fund_id,
            "view_type": self.view_type,
            "expected_return": self.expected_return,
            "confidence": self.confidence,
            "reason": self.reason,
        }
        if self.benchmark_fund_id:
            result["benchmark_fund_id"] = self.benchmark_fund_id
        return result


class ViewValidator:
    """Validate and normalize investment views."""

    @staticmethod
    def validate_view(view: Dict, fund_pool: List[str]) -> bool:
        """Validate a single view."""
        fund_id = view.get("fund_id")
        if fund_id not in fund_pool:
            return False

        view_type = view.get("view_type", "absolute")
        if view_type not in ["absolute", "relative"]:
            return False

        expected_return = view.get("expected_return")
        if expected_return is None:
            return False
        if abs(expected_return) > 0.1:  # More than 10% daily return is unrealistic
            logger.warning(f"View for {fund_id} has extreme expected return: {expected_return}")
            # Still accept but cap it
            view["expected_return"] = max(min(expected_return, 0.05), -0.05)

        confidence = view.get("confidence", 0.5)
        if not (0 <= confidence <= 1):
            return False

        # For relative views, benchmark must exist
        if view_type == "relative":
            benchmark = view.get("benchmark_fund_id")
            if not benchmark or benchmark not in fund_pool:
                return False

        return True

    @staticmethod
    def normalize_views(views: List[Dict], fund_pool: List[str], max_views: int = 5) -> List[View]:
        """Normalize and filter views."""
        valid_views = []

        for v in views:
            if ViewValidator.validate_view(v, fund_pool):
                # Ensure minimum confidence
                v["confidence"] = max(v.get("confidence", 0.5), 0.1)
                valid_views.append(View(**v))

        # Limit number of views
        return valid_views[:max_views]


class ViewGenerationAgent:
    """Generate investment views using LLM."""

    def __init__(
        self,
        model_name: str = "deepseek-v4-pro",
        min_confidence: float = 0.3,
        max_views_per_day: int = 5,
        prompt_template: str = None,
    ):
        """
        Args:
            model_name: LLM model name
            min_confidence: Minimum confidence threshold for views
            max_views_per_day: Maximum number of views to generate per day
            prompt_template: Custom prompt template (optional)
        """
        self.llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            temperature=0.7,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        self.min_confidence = min_confidence
        self.max_views_per_day = max_views_per_day
        self.prompt_template = prompt_template or VIEW_GENERATION_PROMPT
        self.parser = CustomJsonOutputParser()

        # Track view history for performance analysis
        self.view_history: List[Dict] = []

    async def generate_views(
        self,
        date_to_decision: str,
        sentiment_analysis: Dict,
        historical_prices: Dict,
        current_portfolio: Dict,
        fund_pool: List[str],
        market_data: Optional[Dict] = None,
    ) -> List[View]:
        """
        Generate investment views based on market conditions.

        Args:
            date_to_decision: Current date for decision
            sentiment_analysis: Sentiment analysis from news
            historical_prices: Historical price data
            current_portfolio: Current portfolio holdings
            fund_pool: List of available fund IDs
            market_data: Additional market data (optional)

        Returns:
            List of View objects
        """
        # Build context for prompt
        funds_text = self._build_funds_text(fund_pool)
        holdings_text = self._build_holdings_text(current_portfolio)
        history_text = self._build_history_text(historical_prices, fund_pool)
        sentiment_summary = sentiment_analysis.get("summary", "无舆情分析")
        sentiment_details = json.dumps(
            sentiment_analysis.get("fund_analysis", {}),
            indent=2,
            ensure_ascii=False,
        )

        capital = current_portfolio.get("capital", 0)

        # Format prompt
        prompt = self.prompt_template.format(
            funds_text=funds_text,
            date_to_decision=date_to_decision,
            capital=capital,
            holdings_text=holdings_text,
            sentiment_summary=sentiment_summary,
            sentiment_details=sentiment_details,
            history_text=history_text,
        )

        # Call LLM
        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt),
                timeout=120,
            )

            result = json.loads(response.content)
            logger.info(f"LLM view generation result: {result}")

            # Parse views
            views_data = result.get("views", [])
            views = ViewValidator.normalize_views(
                views_data,
                fund_pool,
                self.max_views_per_day,
            )

            # Filter by minimum confidence
            views = [v for v in views if v.confidence >= self.min_confidence]

            logger.info(f"Generated {len(views)} valid views for {date_to_decision}")
            return views

        except asyncio.TimeoutError:
            logger.warning("View generation timed out")
            return []
        except Exception as e:
            logger.error(f"View generation failed: {e}")
            return []

    def views_to_matrices(
        self,
        views: List[View],
        fund_pool: List[str],
    ) -> tuple:
        """
        Convert views to BL model matrices.

        Args:
            views: List of View objects
            fund_pool: List of fund IDs

        Returns:
            Tuple of (fund_list, P, Q, confidences)
            - fund_list: Ordered list of fund IDs corresponding to matrix columns
            - P: View matrix (k x n)
            - Q: View returns (k,)
            - confidences: Confidence levels (k,)
        """
        if not views:
            return fund_pool, None, None, None

        # Use all funds in pool for matrix columns
        fund_to_idx = {fund: i for i, fund in enumerate(fund_pool)}
        n = len(fund_pool)
        k = len(views)

        import numpy as np

        P = np.zeros((k, n))
        Q = np.zeros(k)
        confidences = np.zeros(k)

        for i, view in enumerate(views):
            fund_idx = fund_to_idx.get(view.fund_id)
            if fund_idx is None:
                continue

            if view.view_type == "relative":
                bench_idx = fund_to_idx.get(view.benchmark_fund_id)
                if bench_idx is not None:
                    P[i, fund_idx] = 1
                    P[i, bench_idx] = -1
                else:
                    P[i, fund_idx] = 1
            else:  # absolute
                P[i, fund_idx] = 1

            Q[i] = view.expected_return
            confidences[i] = view.confidence

        return fund_pool, P, Q, confidences

    def _build_funds_text(self, fund_pool: List[str]) -> str:
        """Build funds description text."""
        lines = []
        for fund in fund_pool:
            info = FUND_INFO.get(fund, {})
            name = info.get("name", fund)
            scope = info.get("scope", "")
            meaning = info.get("meaning", "")
            lines.append(f"- {fund} ({name}): {scope} ({meaning})")
        return "\n".join(lines)

    def _build_holdings_text(self, current_portfolio: Dict) -> str:
        """Build current holdings text."""
        holdings = current_portfolio.get("holdings", {})
        if not holdings:
            return "  (空仓)"

        lines = []
        for fund_id, details in holdings.items():
            value = details.get("value", 0)
            price = details.get("price", 0)
            lines.append(f"- {fund_id}: 持仓价值 {value:.2f} 元 (当前价: {price:.2f})")
        return "\n".join(lines)

    def _build_history_text(self, historical_prices: Dict, fund_pool: List[str]) -> str:
        """Build historical price summary text."""
        lines = []

        for fund in fund_pool[:5]:  # Limit to top 5 for brevity
            prices = historical_prices.get(fund, [])
            if not prices:
                continue

            # Show last 3 days
            recent = prices[-3:]
            fund_line = f"{fund}: "
            price_changes = []

            for p in recent:
                date = p.get("date", "")
                pct = p.get("pct_change")
                if pct is not None:
                    price_changes.append(f"{date} 涨跌{pct:.2%}")

            if price_changes:
                fund_line += ", ".join(price_changes)
                lines.append(fund_line)

        if not lines:
            return "  (无历史数据)"

        return "\n".join(lines)

    def record_view_performance(self, date: str, views: List[View], actual_returns: Dict[str, float]):
        """Record view performance for tracking and improvement."""
        for view in views:
            actual_return = actual_returns.get(view.fund_id, 0)
            self.view_history.append({
                "date": date,
                "fund_id": view.fund_id,
                "expected_return": view.expected_return,
                "actual_return": actual_return,
                "confidence": view.confidence,
                "accuracy": 1 - abs(view.expected_return - actual_return) / max(abs(actual_return), 0.01),
            })

    def get_view_performance_summary(self, lookback_days: int = 30) -> str:
        """Get summary of recent view performance."""
        if not self.view_history:
            return "无历史观点记录"

        recent = self.view_history[-lookback_days*5:]  # Approximate

        if not recent:
            return "无近期观点记录"

        total_views = len(recent)
        accurate_views = sum(1 for v in recent if v.get("accuracy", 0) > 0.5)
        avg_accuracy = sum(v.get("accuracy", 0) for v in recent) / max(len(recent), 1)

        return (
            f"最近{len(recent)}个观点，准确率{accurate_views}/{total_views} "
            f"({accurate_views/max(total_views,1)*100:.1f}%), "
            f"平均准确度{avg_accuracy:.2%}"
        )
