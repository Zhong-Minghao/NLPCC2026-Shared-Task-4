#!/usr/bin/env python3
"""
score_news_for_ic.py

IC检验第一步：读取新闻，对各行业ETF进行情感打分，输出可复用的JSON格式文件。

两种模式:
  --mode daily     : 日聚合打分（SentimentAnalysisAgent，每天全部新闻→每ETF一个分）
  --mode per_news  : 单条新闻打分（PerNewsETFScoringAgent，每篇文章→每ETF的relevance+score）
  --mode both      : 两种都跑（默认）

输出（--output-dir 下）:
  daily_scores.json        模式A：{date: {etf: {sentiment, confidence, daily_score, reason}}}
  per_news_scores.jsonl    模式B：每行一条新闻的评分结果

用法示例:
  cd NLPCC_tasks
  python agent_platform/score_news_for_ic.py --start 2024-01-02 --end 2024-02-28 --mode both
  python agent_platform/score_news_for_ic.py --start 2024-01-02 --end 2025-12-31 --mode daily
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import DATA_DIRS
from server_platform.app.core.data_loader import init_data_loader
from agent_platform.agents.advanced_agents import NewsProcessingAgent, SentimentAnalysisAgent
from agent_platform.agents.per_news_etf_scoring_agent import PerNewsETFScoringAgent

INDUSTRY_FUND_POOL = [
    "512880.SH", "512800.SH", "512070.SH", "159995.SZ",
    "159819.SZ", "515880.SH", "159852.SZ", "512010.SH",
    "512170.SH", "159992.SZ", "515170.SH", "512690.SH",
    "512400.SH", "515220.SH", "159870.SZ", "512200.SH",
]

DIRECTION_VALUE = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


# ── JSON/JSONL helpers ─────────────────────────────────────────────────────────

def _atomic_write(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _load_daily_scores(path: str) -> Dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load daily_scores from {path}: {e}")
    return {"metadata": {}, "dates": {}}


def _load_per_news_processed_dates(path: str) -> Set[str]:
    """Return set of date strings already written to JSONL."""
    processed = set()
    if not os.path.exists(path):
        return processed
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    processed.add(rec.get("date", ""))
    except Exception as e:
        logger.warning(f"Failed to scan per_news_scores JSONL: {e}")
    return processed


def _append_per_news_records(path: str, records: List[Dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# ── Scoring helpers ─────────────────────────────────────────────────────────────

def _parse_daily_sentiment(sentiment_analysis: Dict, fund_pool: List[str]) -> Dict:
    """Convert SentimentAnalysisAgent output → {etf: {sentiment, confidence, daily_score, reason}}"""
    fund_analysis = sentiment_analysis.get("fund_analysis", {})
    etf_scores = {}
    for etf in fund_pool:
        entry = fund_analysis.get(etf, {})
        sentiment = entry.get("sentiment", "neutral")
        confidence = float(entry.get("confidence", 0.5))
        direction_val = DIRECTION_VALUE.get(sentiment, 0.0)
        etf_scores[etf] = {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "daily_score": round(direction_val * confidence, 4),
            "reason": entry.get("reason", ""),
        }
    return etf_scores


# ── Main scoring loop ──────────────────────────────────────────────────────────

async def run_scoring(
    start_date: str,
    end_date: str,
    mode: str,
    output_dir: str,
    sources: List[str],
    top_rank: int,
    pre_k_days: int,
    model: str,
    news_model: str,
    max_workers: int,
    save_every: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    daily_path = os.path.join(output_dir, "daily_scores.json")
    per_news_path = os.path.join(output_dir, "per_news_scores.jsonl")

    # Init data loader
    data_loader = init_data_loader(
        str(DATA_DIRS["PRICE_DATA"]),
        str(DATA_DIRS["NEWS_DATA"]),
    )

    # Load existing data
    daily_store = _load_daily_scores(daily_path)
    if not daily_store.get("metadata"):
        daily_store["metadata"] = {
            "sources": sources,
            "top_rank": top_rank,
            "pre_k_days": pre_k_days,
            "etf_pool": INDUSTRY_FUND_POOL,
            "model": model,
        }

    per_news_processed_dates: Set[str] = set()
    if mode in ("per_news", "both"):
        per_news_processed_dates = _load_per_news_processed_dates(per_news_path)

    # Build trading date list
    start_int = int(start_date.replace("-", ""))
    end_int = int(end_date.replace("-", ""))
    trading_dates = data_loader.get_trading_dates(start_int, end_int)
    logger.info(f"Trading days to process: {len(trading_dates)} ({trading_dates[0]} → {trading_dates[-1]})")

    # Init agents
    news_agent = NewsProcessingAgent(model_name=news_model or model)
    sentiment_agent = SentimentAnalysisAgent(model_name=model)
    per_news_agent = PerNewsETFScoringAgent(model_name=model) if mode in ("per_news", "both") else None

    unsaved_count = 0

    for i, date_int in enumerate(trading_dates):
        date_str = datetime.strptime(str(date_int), "%Y%m%d").strftime("%Y-%m-%d")
        need_daily = (mode in ("daily", "both")) and (date_str not in daily_store["dates"])
        need_per_news = (mode in ("per_news", "both")) and (date_str not in per_news_processed_dates)

        if not need_daily and not need_per_news:
            logger.debug(f"[{date_str}] already processed, skipping")
            continue

        logger.info(f"[{date_str}] processing ({i+1}/{len(trading_dates)})...")

        # Fetch and summarize news
        try:
            news_data = data_loader.get_news(
                sources=sources,
                current_date=date_int,
                top_rank=top_rank,
                pre_k_days=pre_k_days,
            )
        except Exception as e:
            logger.error(f"[{date_str}] get_news failed: {e}")
            news_data = []

        if news_data:
            try:
                processed_news = await news_agent.process_news_batch(news_data, max_workers=max_workers)
            except Exception as e:
                logger.error(f"[{date_str}] process_news_batch failed: {e}")
                processed_news = []
        else:
            processed_news = []

        logger.info(f"[{date_str}] {len(news_data)} news items → {len(processed_news)} summarized")

        # ── Mode A: daily aggregate ───────────────────────────────────────────
        if need_daily:
            try:
                sentiment = await sentiment_agent.analyze_sentiment(
                    date_str, processed_news, INDUSTRY_FUND_POOL
                )
                daily_store["dates"][date_str] = {
                    "overall_sentiment": sentiment.get("overall_sentiment", "neutral"),
                    "summary": sentiment.get("summary", ""),
                    "etf_scores": _parse_daily_sentiment(sentiment, INDUSTRY_FUND_POOL),
                }
                logger.info(f"[{date_str}] Daily aggregate done. overall={sentiment.get('overall_sentiment')}")
            except Exception as e:
                logger.error(f"[{date_str}] Daily sentiment failed: {e}")
                daily_store["dates"][date_str] = {
                    "overall_sentiment": "neutral",
                    "summary": f"error: {e}",
                    "etf_scores": {
                        etf: {"sentiment": "neutral", "confidence": 0.5, "daily_score": 0.0, "reason": ""}
                        for etf in INDUSTRY_FUND_POOL
                    },
                }

        # ── Mode B: per-news ──────────────────────────────────────────────────
        if need_per_news and per_news_agent:
            per_news_records = []
            for news_item in processed_news:
                try:
                    etf_scores = await per_news_agent.score_article(news_item, INDUSTRY_FUND_POOL)
                    per_news_records.append({
                        "date": date_str,
                        "source": news_item.get("source", ""),
                        "ranking": news_item.get("ranking", 0),
                        "title": news_item.get("title", ""),
                        "summary": news_item.get("summary", ""),
                        "etf_scores": etf_scores,
                    })
                except Exception as e:
                    logger.warning(f"[{date_str}] per_news scoring failed for article: {e}")

            if per_news_records:
                _append_per_news_records(per_news_path, per_news_records)
            per_news_processed_dates.add(date_str)
            logger.info(f"[{date_str}] Per-news done: {len(per_news_records)} articles scored")

        unsaved_count += 1
        if unsaved_count % save_every == 0:
            if mode in ("daily", "both"):
                _atomic_write(daily_path, daily_store)
                logger.info(f"Auto-saved daily_scores.json ({len(daily_store['dates'])} dates)")

    # Final save
    if mode in ("daily", "both"):
        _atomic_write(daily_path, daily_store)
        logger.info(f"Saved daily_scores.json: {len(daily_store['dates'])} dates total")

    logger.info("Scoring complete.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Score news articles for IC analysis")
    parser.add_argument("--start", default="2024-01-02", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--mode", choices=["daily", "per_news", "both"], default="both",
        help="Scoring mode: daily aggregate, per-news, or both"
    )
    parser.add_argument("--output-dir", default="ic_scores", help="Output directory")
    parser.add_argument("--sources", nargs="+", default=["caixin"], help="News sources")
    parser.add_argument("--top-rank", type=int, default=20, help="Max news ranking to include")
    parser.add_argument("--pre-k-days", type=int, default=1, help="Days of news lookback per date")
    parser.add_argument("--model", default="deepseek-v4-flash", help="LLM model for scoring")
    parser.add_argument("--news-model", default=None, help="LLM model for news summarization (default: same as --model)")
    parser.add_argument("--max-workers", type=int, default=5, help="Concurrent LLM calls")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N days")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_scoring(
        start_date=args.start,
        end_date=args.end,
        mode=args.mode,
        output_dir=args.output_dir,
        sources=args.sources,
        top_rank=args.top_rank,
        pre_k_days=args.pre_k_days,
        model=args.model,
        news_model=args.news_model,
        max_workers=args.max_workers,
        save_every=args.save_every,
    ))
