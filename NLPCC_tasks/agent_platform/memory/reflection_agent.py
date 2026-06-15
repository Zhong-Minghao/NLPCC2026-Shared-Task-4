"""
ReflectionAgent: manages raw observation logging and periodic LLM reflection.

Flow:
  1. log_daily_outcome()   — called each day with views and actual returns
  2. should_reflect()      — returns True when interval + min observations met
  3. run_reflection()      — calls LLM, returns updated principles dict
"""

import asyncio
import json
import os
from collections import deque
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from loguru import logger

from agent_platform.memory.memory_types import FUND_SECTOR_MAP, RawObservation
from agent_platform.prompts.reflection_prompt import REFLECTION_PROMPT

RAW_OBSERVATION_WINDOW = 30   # keep at most this many days of raw observations
MIN_OBSERVATIONS_TO_REFLECT = 3
MAX_PRINCIPLES_PER_KEY = 3


class ReflectionAgent:
    def __init__(
        self,
        model_name: str,
        reflection_interval: int = 5,
    ):
        """
        Args:
            model_name: LLM model for reflection.
            reflection_interval: Minimum trading days between reflections.
        """
        self.llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            temperature=0.2,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        self.reflection_interval = reflection_interval

        # Rolling window of raw observations (most recent RAW_OBSERVATION_WINDOW days)
        self._observations: deque[RawObservation] = deque(maxlen=RAW_OBSERVATION_WINDOW)
        self._days_since_last_reflection: int = 0
        self._total_days_logged: int = 0
        # Index into _observations marking the start of "new" (unseen) observations.
        # Only observations from this index onward are sent to the LLM on each reflection,
        # avoiding re-processing data the LLM has already seen.
        self._new_observations_start: int = 0

    # ------------------------------------------------------------------
    # Step 1: Log daily outcome
    # ------------------------------------------------------------------

    def log_daily_outcome(
        self,
        date: str,
        views: List[Dict],
        actual_returns: Dict[str, float],
        news_summaries: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record what was predicted vs what actually happened for one trading day.

        Args:
            date: Trading date string "YYYY-MM-DD".
            views: List of view dicts (fund_id, expected_return, confidence, view_type, reason).
            actual_returns: {fund_id: actual_daily_return_decimal}.
            news_summaries: Optional {fund_id: brief_news_text} for context.
        """
        if news_summaries is None:
            news_summaries = {}

        for view in views:
            fund_id = view.get("fund_id")
            if not fund_id:
                continue
            actual = actual_returns.get(fund_id)
            if actual is None:
                continue

            obs = RawObservation(
                date=date,
                fund_id=fund_id,
                sector=FUND_SECTOR_MAP.get(fund_id, "其他"),
                predicted_return=float(view.get("expected_return", 0)),
                actual_return=float(actual),
                confidence=float(view.get("confidence", 0.5)),
                news_keywords=news_summaries.get(fund_id, view.get("reason", ""))[:200],
                view_type=view.get("view_type", "absolute"),
            )
            self._observations.append(obs)

        self._days_since_last_reflection += 1
        self._total_days_logged += 1
        logger.debug(
            f"ReflectionAgent: logged {len(views)} views for {date}, "
            f"total observations={len(self._observations)}"
        )

    # ------------------------------------------------------------------
    # Step 2: Check if reflection should run
    # ------------------------------------------------------------------

    def should_reflect(self) -> bool:
        return (
            self._days_since_last_reflection >= self.reflection_interval
            and len(self._observations) >= MIN_OBSERVATIONS_TO_REFLECT
        )

    # ------------------------------------------------------------------
    # Step 3: Run LLM reflection
    # ------------------------------------------------------------------

    async def run_reflection(
        self,
        existing_principles: Dict[str, List[dict]],
        timeout: int = 150,
    ) -> Dict[str, List[dict]]:
        """
        Call LLM to distill/update principles from recent observations.

        Returns:
            Dict[str, List[dict]] — updated_principles from LLM output,
            or empty dict on failure.
        """
        # Only send observations that are NEW since the last successful reflection.
        # This keeps prompt size proportional to reflection_interval (typically ~5 days
        # of views), not the full 30-day rolling window.
        all_obs = list(self._observations)
        new_obs = all_obs[self._new_observations_start:]
        if not new_obs:
            new_obs = all_obs  # fallback: send all if pointer is stale

        observations_text = self._build_observations_text(new_obs)
        existing_text = self._build_existing_principles_text(existing_principles)

        prompt = REFLECTION_PROMPT.format(
            window_days=len(new_obs),
            observations_text=observations_text,
            existing_principles_text=existing_text,
            max_principles=MAX_PRINCIPLES_PER_KEY,
        )

        logger.debug(
            f"ReflectionAgent: sending {len(new_obs)} new observations to LLM "
            f"(skipping {self._new_observations_start} already-processed)"
        )

        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt),
                timeout=timeout,
            )
            result = json.loads(response.content)
            updated = result.get("updated_principles", {})
            summary = result.get("reflection_summary", "")

            logger.info(
                f"ReflectionAgent: reflection complete — "
                f"{sum(len(v) for v in updated.values())} principles across "
                f"{len(updated)} keys. Summary: {summary}"
            )

            self._days_since_last_reflection = 0
            # Advance the pointer so next reflection only sees truly new observations.
            self._new_observations_start = len(all_obs)
            return updated

        except asyncio.TimeoutError:
            logger.warning("ReflectionAgent: reflection timed out, skipping")
        except Exception as e:
            logger.warning(f"ReflectionAgent: reflection failed: {e}")
        finally:
            # Always reset counter so we don't retry every single day after a failure.
            # Next reflection will trigger after another reflection_interval days.
            self._days_since_last_reflection = 0

        return {}

    # ------------------------------------------------------------------
    # Text builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_observations_text(observations: List[RawObservation]) -> str:
        if not observations:
            return "（无观测记录）"

        lines = []
        for obs in observations:
            direction = "✓" if obs.direction_correct else "✗"
            error_pct = obs.prediction_error * 100
            lines.append(
                f"[{obs.date}] {obs.fund_id}({obs.sector}) | "
                f"预测{obs.predicted_return*100:+.2f}% → 实际{obs.actual_return*100:+.2f}% "
                f"(误差{error_pct:+.2f}%, 方向{direction}) | "
                f"置信度{obs.confidence:.2f} | 新闻: {obs.news_keywords}"
            )
        return "\n".join(lines)

    @staticmethod
    def _build_existing_principles_text(
        existing: Dict[str, List[dict]],
    ) -> str:
        if not existing:
            return "（暂无已有规律）"

        lines = []
        for key, principles in existing.items():
            lines.append(f"[{key}]")
            for p in principles:
                lines.append(
                    f"  - {p.get('principle', '')} "
                    f"(可信度{p.get('confidence', 0):.2f}, "
                    f"依据{p.get('evidence_count', 0)}条)"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Accessors for debugging / testing
    # ------------------------------------------------------------------

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    def get_observations(self) -> List[RawObservation]:
        return list(self._observations)
