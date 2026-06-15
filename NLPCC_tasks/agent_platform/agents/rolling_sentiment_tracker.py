"""
Rolling Sentiment Tracker

Maintains a per-fund EMA (Exponential Moving Average) sentiment score
with asymmetric update rules to reduce portfolio turnover:

- Trend reinforcement has diminishing marginal returns
- Counter-trend signals are amplified (surprises move markets more)
- Silent days decay the score toward neutral
"""

from typing import Dict, List, Optional

from loguru import logger


class RollingSentimentTracker:
    """
    Per-fund rolling sentiment score with asymmetric EMA update.

    Score range: [-1.0, +1.0]
      +1.0 = persistently very positive narrative
      -1.0 = persistently very negative narrative
       0.0 = neutral / no history
    """

    def __init__(
        self,
        fund_pool: List[str],
        alpha_base: float = 0.30,
        no_news_decay: float = 0.05,
        view_flip_threshold: float = 0.25,
        view_flip_return_threshold: float = 0.01,
        base_magnitude: float = 0.015,
        asymmetric_factor: float = 0.50,
    ):
        """
        Args:
            fund_pool: List of fund IDs to track
            alpha_base: Base EMA update weight [0, 1]. Higher = more reactive to daily news.
            no_news_decay: Daily decay rate toward 0 when no news for a fund.
            view_flip_threshold: |rolling_score| must exceed this to be considered "directional".
            view_flip_return_threshold: LLM expected_return must exceed this (abs) to override rolling direction.
            base_magnitude: Maps rolling_score=1.0 to this daily expected return (e.g. 0.015 = 1.5%).
            asymmetric_factor: Controls same-direction alpha reduction. 0 = no asymmetry, 1 = fully asymmetric.
        """
        self.rolling_scores: Dict[str, float] = {f: 0.0 for f in fund_pool}
        self.alpha_base = alpha_base
        self.no_news_decay = no_news_decay
        self.view_flip_threshold = view_flip_threshold
        self.view_flip_return_threshold = view_flip_return_threshold
        self.base_magnitude = base_magnitude
        self.asymmetric_factor = asymmetric_factor
        self._history: List[Dict] = []

    # ------------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------------

    def _compute_daily_score(
        self, fund_id: str, fund_analysis: Dict
    ) -> Optional[float]:
        """Convert LLM per-fund sentiment output to continuous score [-1, 1].

        Returns None if the fund has no relevant news today.
        """
        fa = fund_analysis.get(fund_id)
        if not fa:
            return None

        direction_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        direction = direction_map.get(fa.get("sentiment", "neutral"), 0.0)
        confidence = float(fa.get("confidence", 0.5))
        return direction * confidence

    def _asymmetric_alpha(self, rolling: float, daily: float) -> float:
        """Compute effective EMA alpha using asymmetric update rule.

        Same direction → diminishing marginal returns (alpha reduced).
        Opposite direction → amplified reversal signal (alpha increased).
        """
        if rolling == 0.0 or daily == 0.0:
            return self.alpha_base

        same_direction = (rolling * daily) > 0
        trend_strength = min(abs(rolling), 1.0)

        if same_direction:
            return self.alpha_base * (1.0 - trend_strength * self.asymmetric_factor)
        else:
            return min(self.alpha_base * (1.0 + trend_strength), 0.8)

    def update(self, sentiment_analysis: Dict, date: str) -> Dict[str, float]:
        """Update rolling scores for all tracked funds.

        Args:
            sentiment_analysis: Output from SentimentAnalysisAgent.analyze_sentiment()
                                 Expected: {"fund_analysis": {fund_id: {sentiment, confidence}}}
            date: Current decision date string (for logging)

        Returns:
            Updated rolling_scores dict {fund_id: float}
        """
        fund_analysis = sentiment_analysis.get("fund_analysis", {})
        day_log = {}

        for fund_id in list(self.rolling_scores.keys()):
            old_rolling = self.rolling_scores[fund_id]
            daily_score = self._compute_daily_score(fund_id, fund_analysis)

            if daily_score is None:
                # No relevant news: decay toward zero
                new_rolling = old_rolling * (1.0 - self.no_news_decay)
                effective_alpha = None
                event = "no_news_decay"
            else:
                effective_alpha = self._asymmetric_alpha(old_rolling, daily_score)
                new_rolling = (1.0 - effective_alpha) * old_rolling + effective_alpha * daily_score
                event = "same_dir" if (old_rolling * daily_score) >= 0 else "reversal"

            # Clamp to [-1, 1]
            new_rolling = max(-1.0, min(1.0, new_rolling))
            self.rolling_scores[fund_id] = new_rolling

            day_log[fund_id] = {
                "old": round(old_rolling, 4),
                "daily": round(daily_score, 4) if daily_score is not None else None,
                "alpha": round(effective_alpha, 4) if effective_alpha is not None else None,
                "new": round(new_rolling, 4),
                "event": event,
            }

        self._history.append({"date": date, "updates": day_log})
        return dict(self.rolling_scores)

    # ------------------------------------------------------------------
    # Signal interpretation
    # ------------------------------------------------------------------

    def get_trend_directions(self) -> Dict[str, str]:
        """Return current trend direction for each fund."""
        result = {}
        for fund_id, score in self.rolling_scores.items():
            if score > self.view_flip_threshold:
                result[fund_id] = "bullish"
            elif score < -self.view_flip_threshold:
                result[fund_id] = "bearish"
            else:
                result[fund_id] = "neutral"
        return result

    def get_anchored_return(self, fund_id: str) -> float:
        """Map rolling score to a Python-anchored daily expected return.

        rolling_score=+1.0 → +base_magnitude (e.g. +1.5%/day)
        rolling_score=-0.5 → -0.5 × base_magnitude
        """
        return self.rolling_scores.get(fund_id, 0.0) * self.base_magnitude

    def should_override_view(self, fund_id: str, view_expected_return: float) -> bool:
        """Check whether a LLM-generated View should be overridden toward neutral.

        Returns True when:
          - Rolling score is strongly directional (|score| >= view_flip_threshold)
          - View direction contradicts rolling direction
          - View magnitude is small (< view_flip_return_threshold), i.e., LLM is uncertain
        """
        rolling = self.rolling_scores.get(fund_id, 0.0)
        if abs(rolling) < self.view_flip_threshold:
            return False
        if rolling * view_expected_return >= 0:
            return False
        return abs(view_expected_return) < self.view_flip_return_threshold

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    def warm_up(self, historical_sentiments: List[Dict]) -> None:
        """Pre-warm rolling scores using historical sentiment data.

        Args:
            historical_sentiments: List of {"date": str, "sentiment_analysis": Dict}
                                   sorted chronologically (oldest first).
        """
        if not historical_sentiments:
            logger.warning("[RollingTracker] warm_up called with empty historical_sentiments")
            return

        logger.info(f"[RollingTracker] Warming up with {len(historical_sentiments)} historical days...")
        for record in historical_sentiments:
            date = record.get("date", "unknown")
            sentiment = record.get("sentiment_analysis", {})
            self.update(sentiment, date)

        # Log final state after warm-up
        bullish = [f for f, d in self.get_trend_directions().items() if d == "bullish"]
        bearish = [f for f, d in self.get_trend_directions().items() if d == "bearish"]
        logger.info(
            f"[RollingTracker] Warm-up complete. "
            f"Bullish: {len(bullish)}, Bearish: {len(bearish)}, "
            f"Neutral: {len(self.rolling_scores) - len(bullish) - len(bearish)}"
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def get_log_summary(self, date: str) -> Dict:
        """Return a structured log dict for the current day."""
        directions = self.get_trend_directions()
        reversal_count = sum(
            1 for d in self._history[-1]["updates"].values()
            if d.get("event") == "reversal"
        ) if self._history else 0

        return {
            "date": date,
            "bullish": [f for f, d in directions.items() if d == "bullish"],
            "bearish": [f for f, d in directions.items() if d == "bearish"],
            "neutral": [f for f, d in directions.items() if d == "neutral"],
            "reversal_events": reversal_count,
            "top_scores": sorted(
                self.rolling_scores.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5],
        }

    def get_history(self) -> List[Dict]:
        """Return full update history (for reflection agent)."""
        return self._history

    def get_hyperparams(self) -> Dict:
        """Return all hyperparameters as a dict (for logging in output JSON)."""
        return {
            "alpha_base": self.alpha_base,
            "no_news_decay": self.no_news_decay,
            "view_flip_threshold": self.view_flip_threshold,
            "view_flip_return_threshold": self.view_flip_return_threshold,
            "base_magnitude": self.base_magnitude,
            "asymmetric_factor": self.asymmetric_factor,
        }
