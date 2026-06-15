"""
Technical Factor + Black-Litterman + ViewFlip Agent

Architecture:
  1. Risk parity (inverse-vol) as BL equilibrium prior  →  π = λ Σ w_RP
  2. Mechanical technical factor composite score         →  BL absolute views
  3. Rolling news sentiment (LLM-assisted)              →  ViewFlip confidence overlay

Factor scoring is fully deterministic (no LLM calls for view generation).
LLM is only used for news sentiment, which modulates view confidence via ViewFlip.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Any

from loguru import logger

from agent_platform.agents.advanced_agents import NewsProcessingAgent, SentimentAnalysisAgent
from agent_platform.agents.rolling_sentiment_tracker import RollingSentimentTracker
from agent_platform.portfolio_optimization import PortfolioOptimizer
from agent_platform.portfolio.weight_converter import WeightConverter


# ---------------------------------------------------------------------------
# Factor composite score configuration
# ---------------------------------------------------------------------------

# (factor_id, directional_sign, weight)
# sign=+1: higher raw value → more bullish
# sign=-1: higher raw value → more bearish (e.g. RSI > 50 is overbought)
FACTOR_CONFIGS = [
    ("momentum_60d_pct",          +1, 0.25),
    ("momentum_term_spread_pct",  +1, 0.10),
    ("rsi_20d",                   -1, 0.15),   # high RSI = overbought → negative
    ("bias_20d_pct",              -1, 0.10),   # far above MA → mean-revert negative
    ("macd_histogram",            +1, 0.20),
    ("volume_price_cov_rank",     +1, 0.10),
    ("amount_volatility_60d",     +1, 0.10),   # already negated in calculator → high=stable
]
# bollinger_width_pct is used as a confidence penalty, not a directional factor


class TechFactorScorer:
    """
    Convert raw technical factor values into BL-compatible view dicts.

    Steps:
    1. Cross-sectional Z-score normalization per factor across all funds
    2. Weighted composite score (FACTOR_CONFIGS)
    3. Bollinger-width confidence penalty (high volatility → low confidence)
    4. Filter by min_score_threshold and min_confidence
    """

    def __init__(
        self,
        base_magnitude: float = 0.015,
        min_score_threshold: float = 0.20,
        min_confidence: float = 0.25,
    ):
        """
        Args:
            base_magnitude: composite_score=1.0 maps to this daily expected return (e.g. 1.5%)
            min_score_threshold: |composite_score| must exceed this to generate a view
            min_confidence: minimum confidence for a view to be included
        """
        self.base_magnitude = base_magnitude
        self.min_score_threshold = min_score_threshold
        self.min_confidence = min_confidence

    def score_funds(
        self, tech_factors: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute composite score for each fund.

        Returns:
            {fund_id: {"score": float, "confidence": float, "expected_return": float}}
        """
        # Collect cross-sectional values per factor
        factor_vals: Dict[str, Dict[str, float]] = {}
        for factor_id, _, _ in FACTOR_CONFIGS:
            vals = {}
            for fund_id, data in tech_factors.items():
                v = data.get("factors", {}).get(factor_id)
                if v is not None and not math.isnan(v) and not math.isinf(v):
                    vals[fund_id] = float(v)
            factor_vals[factor_id] = vals

        # Z-score normalize each factor cross-sectionally
        z_scores: Dict[str, Dict[str, float]] = {}
        for factor_id, vals in factor_vals.items():
            if len(vals) < 2:
                z_scores[factor_id] = {}
                continue
            vs = list(vals.values())
            mean = sum(vs) / len(vs)
            variance = sum((v - mean) ** 2 for v in vs) / len(vs)
            std = variance ** 0.5
            if std < 1e-10:
                z_scores[factor_id] = {fid: 0.0 for fid in vals}
            else:
                z_scores[factor_id] = {fid: (v - mean) / std for fid, v in vals.items()}

        # Composite score per fund
        scores: Dict[str, Dict[str, float]] = {}
        for fund_id in tech_factors:
            weighted_sum = 0.0
            total_weight = 0.0
            for factor_id, sign, weight in FACTOR_CONFIGS:
                z = z_scores.get(factor_id, {}).get(fund_id)
                if z is not None:
                    weighted_sum += sign * weight * z
                    total_weight += weight

            if total_weight < 1e-10:
                composite = 0.0
            else:
                composite = weighted_sum / total_weight  # normalise so max ≈ 1

            # Bollinger-width confidence penalty: wide band = high uncertainty
            boll = tech_factors[fund_id].get("factors", {}).get("bollinger_width_pct")
            boll_penalty = 1.0
            if boll is not None and not math.isnan(boll):
                if boll > 5.0:
                    boll_penalty = max(0.4, 1.0 - (boll - 5.0) / 20.0)

            confidence = min(1.0, abs(composite)) * boll_penalty
            expected_return = composite * self.base_magnitude

            scores[fund_id] = {
                "score": round(composite, 4),
                "confidence": round(confidence, 4),
                "expected_return": round(expected_return, 6),
            }

        return scores

    def to_bl_views(self, scores: Dict[str, Dict[str, float]]) -> List[Dict]:
        """
        Convert composite scores to BL view dicts, filtering low-signal funds.

        Returns:
            List of view dicts compatible with PortfolioOptimizer.optimize_portfolio()
        """
        views = []
        for fund_id, s in scores.items():
            if abs(s["score"]) < self.min_score_threshold:
                continue
            if s["confidence"] < self.min_confidence:
                continue
            views.append({
                "fund_id": fund_id,
                "view_type": "absolute",
                "expected_return": s["expected_return"],
                "confidence": s["confidence"],
                "reason": f"tech_composite={s['score']:.3f}",
            })
        return views


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class TechBLViewflipAgent:
    """
    Portfolio agent that integrates:
      - Mechanical technical factor scoring (no LLM for view generation)
      - Black-Litterman with risk-parity (inverse-vol) equilibrium prior
      - Rolling news sentiment overlaid via ViewFlip confidence adjustment

    Args:
        agent_id: Identifier string for logging
        price_data_dir: Path to OHLCV CSV files (for TechnicalFactorCalculator)
        tau: BL uncertainty scaling parameter
        risk_aversion: Risk aversion λ for BL reverse-optimization prior
        lookback_days: Historical days for covariance estimation
        optimization_method: "max_sharpe" or "min_variance"
        risk_free_rate: Annual risk-free rate
        min_weight: Per-fund weight lower bound
        max_weight: Per-fund weight upper bound
        market_weight_method: BL prior weight method ("inverse_vol" = risk parity prior)
        base_magnitude: score=1.0 → this daily expected return (e.g. 0.015 = 1.5%/day)
        min_score_threshold: |composite_score| threshold to emit a view
        min_confidence: Minimum view confidence (post-ViewFlip)
        max_views_per_day: Cap on views passed to BL optimizer
        enable_rolling_sentiment: Toggle LLM news sentiment processing
        rolling_sentiment_days: EMA warm-up window (trading days)
        alpha_base: Base EMA update weight for rolling tracker
        no_news_decay: Daily decay toward neutral when no news
        view_flip_threshold: |rolling_score| must exceed this to apply ViewFlip
        asymmetric_factor: Same-direction confidence boost / opposite-direction penalty
        news_model_name: LLM model for news + sentiment analysis
    """

    def __init__(
        self,
        agent_id: str,
        price_data_dir: Path,
        tau: float = 0.05,
        risk_aversion: float = 3.0,
        lookback_days: int = 60,
        optimization_method: str = "max_sharpe",
        risk_free_rate: float = 0.03,
        min_weight: float = 0.0,
        max_weight: float = 0.4,
        market_weight_method: str = "inverse_vol",
        base_magnitude: float = 0.015,
        min_score_threshold: float = 0.20,
        min_confidence: float = 0.25,
        max_views_per_day: int = 10,
        enable_rolling_sentiment: bool = True,
        rolling_sentiment_days: int = 10,
        alpha_base: float = 0.30,
        no_news_decay: float = 0.05,
        view_flip_threshold: float = 0.25,
        asymmetric_factor: float = 0.50,
        news_model_name: str = "deepseek-v4-flash",
    ):
        self.agent_id = agent_id

        # Lazy import to avoid circular deps and heavy startup cost
        from agent_platform.backtest_tech_v2 import TechnicalFactorCalculator
        self.tech_calculator = TechnicalFactorCalculator(price_data_dir=Path(price_data_dir))
        self.tech_scorer = TechFactorScorer(
            base_magnitude=base_magnitude,
            min_score_threshold=min_score_threshold,
            min_confidence=min_confidence,
        )

        self.portfolio_optimizer = PortfolioOptimizer(
            tau=tau,
            risk_aversion=risk_aversion,
            lookback_days=lookback_days,
            risk_free_rate=risk_free_rate,
            optimization_method=optimization_method,
            min_weight=min_weight,
            max_weight=max_weight,
            market_weight_method=market_weight_method,
        )
        self.weight_converter = WeightConverter()

        self._max_views = max_views_per_day
        self._min_confidence = min_confidence
        self._view_flip_threshold = view_flip_threshold
        self._asymmetric_factor = asymmetric_factor

        # Rolling sentiment setup
        self.enable_rolling_sentiment = enable_rolling_sentiment
        self.rolling_tracker: Optional[RollingSentimentTracker] = None
        self._rs_alpha_base = alpha_base
        self._rs_no_news_decay = no_news_decay
        self._rs_view_flip_threshold = view_flip_threshold
        self._rs_base_magnitude = base_magnitude
        self._rs_asymmetric_factor = asymmetric_factor

        if enable_rolling_sentiment:
            self.news_agent = NewsProcessingAgent(model_name=news_model_name)
            self.sentiment_agent = SentimentAnalysisAgent(model_name=news_model_name)
        else:
            self.news_agent = None
            self.sentiment_agent = None

        logger.info(
            f"TechBLViewflipAgent {agent_id} initialized | "
            f"prior={market_weight_method} | method={optimization_method} | "
            f"sentiment={'on' if enable_rolling_sentiment else 'off'}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def make_decision(
        self,
        date_to_decision: str,
        news_data: List[Dict],
        historical_prices: Dict,
        current_portfolio: Dict,
        fund_pool: List[str],
    ) -> Dict:
        """
        Generate portfolio allocation for one trading day.

        Returns:
            {
              "final_decision": {trades, target_weights, optimization_metrics, reasoning},
              "intermediate_results": {tech_scores, tech_views, rolling_scores},
            }
        """
        logger.info(f"[{self.agent_id}] Decision for {date_to_decision} ...")

        # Step 1: Technical factor composite scores → BL views
        tech_factors = self.tech_calculator.calculate_for_funds(fund_pool, date_to_decision)
        scores = self.tech_scorer.score_funds(tech_factors)
        tech_views = self.tech_scorer.to_bl_views(scores)
        logger.info(f"  Tech: {len(tech_views)} views from {len(scores)} funds")

        # Step 2: Rolling sentiment overlay (ViewFlip)
        rolling_scores: Dict[str, float] = {}
        if self.enable_rolling_sentiment:
            rolling_scores = await self._update_rolling_sentiment(
                news_data, fund_pool, date_to_decision
            )
            if rolling_scores and tech_views:
                tech_views = self._apply_view_flip(tech_views, rolling_scores)
                logger.info(f"  After ViewFlip: {len(tech_views)} views remain")

        # Cap to max_views (keep highest |expected_return| × confidence)
        if len(tech_views) > self._max_views:
            tech_views.sort(
                key=lambda v: abs(v["expected_return"]) * v["confidence"], reverse=True
            )
            tech_views = tech_views[: self._max_views]

        # Step 3: BL optimization with risk-parity prior
        try:
            result = self.portfolio_optimizer.optimize_portfolio(
                historical_prices=historical_prices,
                views=tech_views,
                fund_pool=fund_pool,
            )
            logger.info(
                f"  BL: E[R]={result.expected_return:.4%} "
                f"Risk={result.expected_risk:.4%} Sharpe={result.sharpe_ratio:.4f} "
                f"views={result.views_count}"
            )
        except Exception as exc:
            logger.error(f"  BL optimization failed: {exc}")
            return self._get_hold_decision(current_portfolio, fund_pool, str(exc))

        # Step 4: Weights → trades
        trades = self.weight_converter.weights_to_trades(
            target_weights=result.weights,
            current_portfolio=current_portfolio,
            fund_pool=fund_pool,
        )

        return {
            "final_decision": {
                "reasoning": self._build_reasoning(scores, tech_views, result),
                "trades": trades,
                "target_weights": result.weights,
                "optimization_metrics": {
                    "expected_return": result.expected_return,
                    "expected_risk": result.expected_risk,
                    "sharpe_ratio": result.sharpe_ratio,
                    "views_count": result.views_count,
                },
            },
            "intermediate_results": {
                "tech_scores": scores,
                "tech_views": tech_views,
                "rolling_scores": rolling_scores,
            },
        }

    def warm_up_sentiment(
        self, historical_sentiments: List[Dict], fund_pool: List[str]
    ) -> None:
        """Pre-warm rolling sentiment tracker with historical data."""
        self.rolling_tracker = RollingSentimentTracker(
            fund_pool=fund_pool,
            alpha_base=self._rs_alpha_base,
            no_news_decay=self._rs_no_news_decay,
            view_flip_threshold=self._rs_view_flip_threshold,
            view_flip_return_threshold=0.005,
            base_magnitude=self._rs_base_magnitude,
            asymmetric_factor=self._rs_asymmetric_factor,
        )
        self.rolling_tracker.warm_up(historical_sentiments)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _update_rolling_sentiment(
        self,
        news_data: List[Dict],
        fund_pool: List[str],
        date: str,
    ) -> Dict[str, float]:
        """Process news, update rolling tracker, return rolling_scores."""
        if self.rolling_tracker is None:
            self.rolling_tracker = RollingSentimentTracker(
                fund_pool=fund_pool,
                alpha_base=self._rs_alpha_base,
                no_news_decay=self._rs_no_news_decay,
                view_flip_threshold=self._rs_view_flip_threshold,
                view_flip_return_threshold=0.005,
                base_magnitude=self._rs_base_magnitude,
                asymmetric_factor=self._rs_asymmetric_factor,
            )

        if not news_data:
            # No news today → decay only
            return self.rolling_tracker.update({}, date)

        processed_news = await self.news_agent.process_news_batch(news_data)
        sentiment = await self.sentiment_agent.analyze_sentiment(date, processed_news, fund_pool)
        rolling_scores = self.rolling_tracker.update(sentiment, date)

        summary = self.rolling_tracker.get_log_summary(date)
        logger.info(
            f"  Sentiment: bullish={len(summary['bullish'])} "
            f"bearish={len(summary['bearish'])} "
            f"reversal={summary['reversal_events']}"
        )
        return rolling_scores

    def _apply_view_flip(
        self, tech_views: List[Dict], rolling_scores: Dict[str, float]
    ) -> List[Dict]:
        """
        Adjust view confidence using rolling sentiment direction:
          - Same direction  → boost confidence by (1 + |rolling|*(1-asymmetric_factor))
          - Opposite dir    → penalty: confidence × asymmetric_factor
          - Weak sentiment  → no change
        Views whose confidence drops below min_confidence are dropped.
        """
        result = []
        for view in tech_views:
            fund_id = view["fund_id"]
            rolling = rolling_scores.get(fund_id, 0.0)

            if abs(rolling) < self._view_flip_threshold:
                result.append(view)
                continue

            conf = view["confidence"]
            ret = view["expected_return"]

            if rolling * ret >= 0:
                # Same direction → boost
                boost = 1.0 + abs(rolling) * (1.0 - self._asymmetric_factor)
                new_conf = min(1.0, conf * boost)
                result.append({**view, "confidence": round(new_conf, 4)})
                logger.debug(
                    f"  ViewFlip BOOST {fund_id}: conf {conf:.3f}→{new_conf:.3f} "
                    f"(rolling={rolling:.2f})"
                )
            else:
                # Opposite direction → penalty
                new_conf = conf * self._asymmetric_factor
                if new_conf >= self._min_confidence:
                    result.append({**view, "confidence": round(new_conf, 4)})
                    logger.debug(
                        f"  ViewFlip PENALTY {fund_id}: conf {conf:.3f}→{new_conf:.3f} "
                        f"(rolling={rolling:.2f})"
                    )
                else:
                    logger.debug(
                        f"  ViewFlip DROP {fund_id}: conf {new_conf:.3f} < min {self._min_confidence}"
                    )

        return result

    def _get_hold_decision(
        self, current_portfolio: Dict, fund_pool: List[str], reason: str
    ) -> Dict:
        trades = [
            {"fund_id": f, "action": "hold", "reason": reason} for f in fund_pool
        ]
        return {
            "final_decision": {
                "reasoning": f"Hold all: {reason}",
                "trades": trades,
                "target_weights": {},
                "optimization_metrics": {},
            },
            "intermediate_results": {},
        }

    def _build_reasoning(
        self,
        scores: Dict[str, Dict[str, float]],
        views: List[Dict],
        result,
    ) -> str:
        top_bull = sorted(
            [(fid, s["score"]) for fid, s in scores.items() if s["score"] > 0],
            key=lambda x: x[1], reverse=True,
        )[:3]
        top_bear = sorted(
            [(fid, s["score"]) for fid, s in scores.items() if s["score"] < 0],
            key=lambda x: x[1],
        )[:3]

        parts = [
            f"Tech BL ViewFlip | {len(views)} views | "
            f"E[R]={result.expected_return:.3%} Risk={result.expected_risk:.3%} "
            f"Sharpe={result.sharpe_ratio:.3f}",
        ]
        if top_bull:
            parts.append("Top bullish: " + ", ".join(f"{f}({s:.2f})" for f, s in top_bull))
        if top_bear:
            parts.append("Top bearish: " + ", ".join(f"{f}({s:.2f})" for f, s in top_bear))

        return " | ".join(parts)
