"""
Sentiment Preloader

Preloads historical daily sentiment scores before the backtest starts,
so RollingSentimentTracker can be warm-started with 2024 data when
running the 2025 test-set backtest.

Results are cached to a JSON file to avoid redundant LLM calls on re-runs.
"""

import asyncio
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger

from server_platform.app.core.data_loader import get_data_loader


def _fund_pool_hash(fund_pool: List[str]) -> str:
    return hashlib.md5(",".join(sorted(fund_pool)).encode()).hexdigest()[:8]


def _load_cache(cache_file: str) -> Dict:
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"[SentimentPreloader] Loaded cache: {len(data)} entries from {cache_file}")
            return data
        except Exception as e:
            logger.warning(f"[SentimentPreloader] Failed to load cache {cache_file}: {e}")
    return {}


def _save_cache(cache_file: str, cache: Dict) -> None:
    if not cache_file:
        return
    try:
        tmp = cache_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, cache_file)
        logger.debug(f"[SentimentPreloader] Cache saved: {len(cache)} entries")
    except Exception as e:
        logger.error(f"[SentimentPreloader] Failed to save cache: {e}")


async def preload_historical_sentiment(
    fund_pool: List[str],
    backtest_start_date: str,
    lookback_days: int,
    news_sources: List[str],
    top_rank: int,
    pre_k_days: int,
    news_agent,
    sentiment_agent,
    cache_file: Optional[str] = "sentiment_preload_cache.json",
) -> List[Dict]:
    """
    Preload historical daily sentiment scores for each trading day in
    [backtest_start_date - lookback_days, backtest_start_date - 1].

    Each entry in the returned list is:
        {"date": "YYYY-MM-DD", "sentiment_analysis": {...}}
    sorted chronologically (oldest first).

    Results are cached keyed by (date, sources, top_rank, fund_pool_hash)
    so that re-runs skip already-computed days.

    Args:
        fund_pool: List of fund IDs
        backtest_start_date: Backtest start date "YYYY-MM-DD"
        lookback_days: Number of trading days to look back for warm-up
        news_sources: List of news source names (e.g. ["caixin", "tiantian"])
        top_rank: Max news ranking to include
        pre_k_days: How many days of news per decision day
        news_agent: NewsProcessingAgent instance (for summarization)
        sentiment_agent: SentimentAnalysisAgent instance (for per-fund scoring)
        cache_file: Path to persist results (None = no cache)

    Returns:
        List of {"date", "sentiment_analysis"} dicts, oldest first
    """
    data_loader = get_data_loader()
    pool_hash = _fund_pool_hash(fund_pool)
    sources_key = "_".join(sorted(news_sources))

    # Calculate trading days to cover
    start_dt = datetime.strptime(backtest_start_date, "%Y-%m-%d")
    # Use 1.6× calendar-day buffer to account for weekends + holidays
    hist_start_dt = start_dt - timedelta(days=int(lookback_days * 1.6) + 5)
    hist_start_int = int(hist_start_dt.strftime("%Y%m%d"))
    backtest_start_int = int(backtest_start_date.replace("-", ""))

    trading_days = data_loader.get_trading_dates(hist_start_int, backtest_start_int - 1)
    # Keep only the last `lookback_days` trading days
    trading_days = trading_days[-lookback_days:]

    if not trading_days:
        logger.warning("[SentimentPreloader] No trading days found for warm-up range")
        return []

    logger.info(
        f"[SentimentPreloader] Will process {len(trading_days)} days "
        f"({trading_days[0]} → {trading_days[-1]})"
    )

    # Load cache
    cache = _load_cache(cache_file)
    results = []
    new_entries = 0

    for date_int in trading_days:
        date_str = datetime.strptime(str(date_int), "%Y%m%d").strftime("%Y-%m-%d")
        cache_key = f"{date_int}_{sources_key}_{top_rank}_{pool_hash}"

        if cache_key in cache:
            results.append({
                "date": date_str,
                "sentiment_analysis": cache[cache_key],
            })
            continue

        # Fetch and process news for this day
        try:
            news_data = data_loader.get_news(
                sources=news_sources,
                current_date=date_int,
                top_rank=top_rank,
                pre_k_days=pre_k_days,
            )

            if not news_data:
                sentiment = {
                    "overall_sentiment": "neutral",
                    "fund_analysis": {},
                    "summary": "无新闻数据",
                }
            else:
                processed_news = await news_agent.process_news_batch(news_data)
                sentiment = await sentiment_agent.analyze_sentiment(
                    date_str, processed_news, fund_pool
                )

            cache[cache_key] = sentiment
            results.append({"date": date_str, "sentiment_analysis": sentiment})
            new_entries += 1

            # Save cache periodically
            if new_entries % 5 == 0:
                _save_cache(cache_file, cache)

        except Exception as e:
            logger.warning(f"[SentimentPreloader] Failed to process {date_str}: {e}")
            results.append({
                "date": date_str,
                "sentiment_analysis": {
                    "overall_sentiment": "neutral",
                    "fund_analysis": {},
                    "summary": f"处理失败: {e}",
                },
            })

    # Final cache save
    if new_entries > 0:
        _save_cache(cache_file, cache)
        logger.info(f"[SentimentPreloader] Computed {new_entries} new days, {len(trading_days) - new_entries} from cache")

    return results
