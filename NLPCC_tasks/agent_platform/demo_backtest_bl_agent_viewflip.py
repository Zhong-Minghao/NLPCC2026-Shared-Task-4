#!/usr/bin/env python3
"""
Black-Litterman Agent Backtest Demo (with Rolling Sentiment / ViewFlip)

Extends demo_backtest_bl_agent.py with:
  - RollingSentimentTracker: 10-day EMA per-fund sentiment with asymmetric update
  - Sentiment preloading: warm-starts tracker with historical data before backtest
  - ViewFlip filter: Python soft-constraint prevents noisy LLM direction flips
  - All rolling-sentiment hyperparameters are exposed as CLI args and logged
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

from loguru import logger

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent_platform.agents.black_litterman_agent import BlackLittermanAgent
from agent_platform.client.platform_client import PlatformClient
from agent_platform.backtest_risk_parity_baseline import preload_historical_prices
from agent_platform.sentiment_preloader import preload_historical_sentiment
from config import AGENT_PLATFORM, DATA_DIRS
from server_platform.app.core.data_loader import init_data_loader
from server_platform.app.models.backtest import AgentDecision

# Fund pools
MAJOR_FUND_POOL = [
    "000300.SH", "000905.SH", "399006.SZ", "000688.SH",
    "000932.SH", "000941.SH", "399971.SZ", "000819.SH",
    "000928.SH", "000012.SH", "518880.SH",
]

INDUSTRY_FUND_POOL = [
    "512880.SH", "512800.SH", "512070.SH", "159995.SZ",
    "159819.SZ", "515880.SH", "159852.SZ", "512010.SH",
    "512170.SH", "159992.SZ", "515170.SH", "512690.SH",
    "512400.SH", "515220.SH", "159870.SZ", "512200.SH",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Black-Litterman agent backtest with rolling sentiment (ViewFlip)"
    )
    parser.add_argument("--track", choices=["macro", "sector"], default="sector")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--news-model", default=None)
    parser.add_argument("--start-date", default="2025-01-02")
    parser.add_argument("--end-date", default="2025-02-28")
    parser.add_argument("--initial-capital", type=float, default=100000)

    # BL parameters
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--risk-aversion", type=float, default=3.0)
    parser.add_argument("--vol-lookback", type=int, default=60)
    parser.add_argument("--optimization-method", default="max_sharpe")
    parser.add_argument("--risk-free-rate", type=float, default=0.03)
    parser.add_argument("--max-weight", type=float, default=0.4)

    # View generation parameters
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--max-views", type=int, default=5)

    # Other parameters
    parser.add_argument("--lookback-days", type=int, default=5)
    parser.add_argument("--top-rank", type=int, default=20)
    parser.add_argument("--pre-k-days", type=int, default=1)
    parser.add_argument("--history-days", type=int, default=5)
    parser.add_argument("--enable-risk-management", action="store_true")

    # Rolling sentiment parameters
    parser.add_argument("--enable-rolling-sentiment", action="store_true", default=True)
    parser.add_argument("--no-rolling-sentiment", dest="enable_rolling_sentiment", action="store_false",
                        help="Disable rolling sentiment tracker (for ablation)")
    parser.add_argument("--rolling-sentiment-days", type=int, default=10,
                        help="Warm-up lookback days for rolling tracker")
    parser.add_argument("--alpha-base", type=float, default=0.30,
                        help="Base EMA update weight [0,1]")
    parser.add_argument("--no-news-decay", type=float, default=0.05,
                        help="Daily decay rate when no news for a fund")
    parser.add_argument("--view-flip-threshold", type=float, default=0.25,
                        help="|rolling_score| threshold for directional signal")
    parser.add_argument("--view-flip-return-threshold", type=float, default=0.01,
                        help="LLM expected_return must exceed this to override rolling direction")
    parser.add_argument("--base-magnitude", type=float, default=0.015,
                        help="rolling_score=1.0 maps to this daily expected return")
    parser.add_argument("--asymmetric-factor", type=float, default=0.50,
                        help="Same-direction alpha reduction factor (0=none, 1=full)")
    parser.add_argument("--sentiment-preload-cache",
                        default="sentiment_preload_cache.json",
                        help="Path to cache file for preloaded sentiment scores")

    # Server config
    parser.add_argument("--username", default=AGENT_PLATFORM.get("AGENT_USERNAME", "bl_viewflip_agent"))
    parser.add_argument("--password", default=AGENT_PLATFORM.get("AGENT_PASSWORD", "bl_password"))
    parser.add_argument("--base-url", default=AGENT_PLATFORM.get("BASE_URL", "http://localhost:6207"))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output", default=None)

    return parser.parse_args()


def build_config(args):
    fund_pool = MAJOR_FUND_POOL if args.track == "macro" else INDUSTRY_FUND_POOL
    default_results_dir = (
        "backtest_results_macro_bl_viewflip"
        if args.track == "macro"
        else "backtest_results_sector_bl_viewflip"
    )
    return {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_capital": args.initial_capital,
        "fund_pool": fund_pool,
        "agents": [{"name": args.username, "prompt": "Black-Litterman ViewFlip Agent"}],
        "news_sources": ["caixin", "tiantian", "sinafinance", "tencent"],
        "lookback_days": args.lookback_days,
        "top_rank": args.top_rank,
        "pre_k_days": args.pre_k_days,
        "view_platform_trading_history_days": args.history_days,
        "decision_model_name": args.model,
        "results_dir": args.results_dir or default_results_dir,
    }


def build_output_path(args, session_id):
    if args.output:
        return Path(args.output)
    out_dir = Path(project_root) / "agent_platform" / "demo_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    tag = "viewflip" if args.enable_rolling_sentiment else "noflip"
    return out_dir / f"{args.track}_bl_{tag}_{args.model}_{session_id}_{timestamp}.json"


async def run_demo_backtest(args):
    """Run BL agent backtest with rolling sentiment tracker."""
    fund_pool = MAJOR_FUND_POOL if args.track == "macro" else INDUSTRY_FUND_POOL
    config = build_config(args)

    # Initialize DataLoader
    init_data_loader(
        price_data_dir=str(DATA_DIRS["PRICE_DATA"]),
        news_data_dir=str(DATA_DIRS["NEWS_DATA"]),
    )
    logger.info("DataLoader initialized")

    # Initialize client
    client = PlatformClient(base_url=args.base_url)
    client.register(args.username, args.password)
    client.login(args.username, args.password)

    # Preload historical prices for covariance calculation
    logger.info(f"Preloading historical prices (lookback: {args.vol_lookback} days)...")
    preloaded_prices = preload_historical_prices(
        fund_pool=fund_pool,
        backtest_start_date=args.start_date,
        lookback_days=args.vol_lookback,
    )
    min_preloaded = min(len(v) for v in preloaded_prices.values()) if preloaded_prices else 0
    logger.info(f"Preloaded {min_preloaded} days of price data")

    # Initialize BL Agent with rolling sentiment params
    agent = BlackLittermanAgent(
        agent_id=f"{args.track}_bl_viewflip_agent",
        view_model_name=args.model,
        news_model_name=args.news_model or args.model,
        tau=args.tau,
        risk_aversion=args.risk_aversion,
        lookback_days=args.vol_lookback,
        optimization_method=args.optimization_method,
        risk_free_rate=args.risk_free_rate,
        min_weight=0.0,
        max_weight=args.max_weight,
        min_confidence=args.min_confidence,
        max_views_per_day=args.max_views,
        enable_risk_management=args.enable_risk_management,
        # Rolling sentiment
        enable_rolling_sentiment=args.enable_rolling_sentiment,
        rolling_sentiment_days=args.rolling_sentiment_days,
        alpha_base=args.alpha_base,
        no_news_decay=args.no_news_decay,
        view_flip_threshold=args.view_flip_threshold,
        view_flip_return_threshold=args.view_flip_return_threshold,
        base_magnitude=args.base_magnitude,
        asymmetric_factor=args.asymmetric_factor,
    )

    # Preload historical sentiment for warm-up
    if args.enable_rolling_sentiment:
        logger.info(
            f"Preloading historical sentiment for warm-up "
            f"({args.rolling_sentiment_days} trading days before {args.start_date})..."
        )
        historical_sentiments = await preload_historical_sentiment(
            fund_pool=fund_pool,
            backtest_start_date=args.start_date,
            lookback_days=args.rolling_sentiment_days,
            news_sources=config["news_sources"],
            top_rank=args.top_rank,
            pre_k_days=args.pre_k_days,
            news_agent=agent.news_agent,
            sentiment_agent=agent.sentiment_agent,
            cache_file=args.sentiment_preload_cache,
        )
        agent.warm_up_sentiment(historical_sentiments, fund_pool)
        logger.info(f"Sentiment warm-up complete: {len(historical_sentiments)} days")

    # Start backtest
    start_response = client.start_backtest(config)
    session_id = start_response["session_id"]
    data = start_response.get("data")

    if not data:
        raise RuntimeError("Failed to get initial backtest data.")

    logger.info(f"Started session {session_id} | track={args.track} | model={args.model}")
    logger.info(
        f"Rolling sentiment: enabled={args.enable_rolling_sentiment}, "
        f"alpha={args.alpha_base}, decay={args.no_news_decay}, "
        f"flip_thresh={args.view_flip_threshold}, flip_ret_thresh={args.view_flip_return_threshold}"
    )

    trading_days = 0
    total_views = 0
    total_flip_overrides = 0

    while True:
        trading_days += 1
        current_date = data.get("date", "Unknown")

        # Get portfolio status
        portfolio = client.get_backtest_status(session_id)

        # Merge preloaded + backtest prices
        backtest_prices = client.get_historical_prices(
            session_id, lookback_days=max(args.lookback_days, args.vol_lookback)
        ).get("historical_prices", {})

        historical_prices = {}
        for fund_id in fund_pool:
            preloaded = preloaded_prices.get(fund_id, [])
            backtest = backtest_prices.get(fund_id, [])
            if preloaded and backtest:
                if preloaded[-1].get("date") == (backtest[0].get("date") if backtest else None):
                    preloaded = preloaded[:-1]
            combined = preloaded + backtest
            historical_prices[fund_id] = combined[-args.vol_lookback:]

        market_data = data.get("market_data", {})

        try:
            decision_result = await agent.make_decision(
                date_to_decision=data["date"],
                news_data=data["news"],
                historical_prices=historical_prices,
                current_portfolio=portfolio,
                market_data=market_data,
                fund_pool=config["fund_pool"],
                view_platform_trading_history_days=config["view_platform_trading_history_days"],
            )

            final_decision = decision_result["final_decision"]
            trades = [
                t for t in final_decision.get("trades", [])
                if t.get("action") != "hold"
            ]

            intermediate = decision_result.get("intermediate_results", {})
            views = intermediate.get("views", [])
            total_views += len(views)

            if trades:
                logger.info(
                    f"Day {trading_days} ({current_date}): "
                    f"{len(views)} views, {len(trades)} trades"
                )
                for trade in trades:
                    action = trade.get("action")
                    fund_id = trade.get("fund_id")
                    if action == "buy":
                        logger.info(f"  BUY  {fund_id}: {trade.get('amount', 0):.0f} CNY")
                    elif action == "sell":
                        logger.info(f"  SELL {fund_id}: {trade.get('percentage', 0):.1%}")

                opt_metrics = final_decision.get("optimization_metrics", {})
                if opt_metrics:
                    logger.info(
                        f"  Portfolio: E[Return]={opt_metrics.get('expected_return', 0):.4%}, "
                        f"Risk={opt_metrics.get('expected_risk', 0):.4%}, "
                        f"Sharpe={opt_metrics.get('sharpe_ratio', 0):.4f}"
                    )

                agent_decision = AgentDecision(
                    decision=final_decision,
                    reasoning=final_decision.get("reasoning", ""),
                    chain_of_thought=str(final_decision.get("chain_of_thought", "")),
                )
                client.submit_trade_with_decision(session_id, trades, agent_decision)

        except Exception as exc:
            logger.error(f"Decision failed on {current_date}: {exc}")
            logger.error(traceback.format_exc())

        data = client.get_next_day_data(session_id)
        if data.get("message") == "Backtest finished":
            break

        await asyncio.sleep(0.1)

    # Final results
    final_results = client.get_backtest_results(session_id)
    output_path = build_output_path(args, session_id)

    # Collect rolling tracker history for reflection agent
    rolling_history = []
    if args.enable_rolling_sentiment and agent.rolling_tracker is not None:
        rolling_history = agent.rolling_tracker.get_history()

    output_data = {
        **final_results,
        "session_id": session_id,
        "backtest_config": config,
        "bl_params": {
            "tau": args.tau,
            "risk_aversion": args.risk_aversion,
            "vol_lookback": args.vol_lookback,
            "optimization_method": args.optimization_method,
            "min_confidence": args.min_confidence,
            "max_views": args.max_views,
            "enable_risk_management": args.enable_risk_management,
        },
        "rolling_sentiment_params": {
            "enable": args.enable_rolling_sentiment,
            "rolling_sentiment_days": args.rolling_sentiment_days,
            "alpha_base": args.alpha_base,
            "no_news_decay": args.no_news_decay,
            "view_flip_threshold": args.view_flip_threshold,
            "view_flip_return_threshold": args.view_flip_return_threshold,
            "base_magnitude": args.base_magnitude,
            "asymmetric_factor": args.asymmetric_factor,
        },
        "decision_stats": {
            "trading_days": trading_days,
            "total_views": total_views,
            "avg_views_per_day": total_views / max(trading_days, 1),
        },
        "rolling_tracker_history": rolling_history,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2, ensure_ascii=False, default=str)

    perf = final_results.get("performance", {})
    logger.info(f"\n{'='*60}")
    logger.info(f"BL ViewFlip Agent Backtest Results ({args.track} track)")
    logger.info(f"{'='*60}")
    logger.info(f"Total Return:  {perf.get('total_return', 0)*100:.2f}%")
    logger.info(f"Sharpe Ratio:  {perf.get('sharpe_ratio', 0):.4f}")
    logger.info(f"Max Drawdown:  {perf.get('max_drawdown', 0)*100:.2f}%")
    logger.info(f"Trading Days:  {trading_days}")
    logger.info(f"Total Views:   {total_views}")
    logger.info(f"Rolling sentiment: {'enabled' if args.enable_rolling_sentiment else 'disabled'}")
    logger.info(f"Results saved to: {output_path}")


def main():
    args = parse_args()

    log_dir = Path(project_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"demo_backtest_bl_viewflip_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logger.add(str(log_path), level="INFO")

    logger.info(
        f"Running BL ViewFlip backtest | track={args.track} | model={args.model} | "
        f"period={args.start_date}~{args.end_date}"
    )

    try:
        asyncio.run(run_demo_backtest(args))
    except Exception as exc:
        logger.error(f"BL ViewFlip backtest failed: {exc}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
