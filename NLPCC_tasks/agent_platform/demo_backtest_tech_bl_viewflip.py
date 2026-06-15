#!/usr/bin/env python3
"""
Tech-BL-ViewFlip Backtest Demo

Combines three strategies:
  1. Technical factor mechanical composite scoring → BL absolute views
  2. Risk-parity (inverse-vol) as BL equilibrium prior  [replaces equal-weight default]
  3. Rolling news sentiment → ViewFlip confidence overlay on tech views

No LLM is used for view generation; views are fully deterministic from price data.
LLM is only called for news sentiment analysis (optional, --enable-rolling-sentiment).
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

from agent_platform.agents.tech_bl_agent import TechBLViewflipAgent
from agent_platform.client.platform_client import PlatformClient
from agent_platform.backtest_risk_parity_baseline import preload_historical_prices
from agent_platform.sentiment_preloader import preload_historical_sentiment
from config import AGENT_PLATFORM, DATA_DIRS
from server_platform.app.core.data_loader import init_data_loader
from server_platform.app.models.backtest import AgentDecision

# Fund pools (identical to other demo scripts)
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
        description="Backtest: Technical factor scoring + BL (risk-parity prior) + ViewFlip"
    )
    parser.add_argument("--track", choices=["macro", "sector"], default="sector")
    parser.add_argument("--start-date", default="2025-01-02")
    parser.add_argument("--end-date", default="2025-02-28")
    parser.add_argument("--initial-capital", type=float, default=100000)

    # BL / optimizer parameters
    parser.add_argument("--tau", type=float, default=0.05,
                        help="BL uncertainty scaling (smaller = prior dominates)")
    parser.add_argument("--risk-aversion", type=float, default=3.0,
                        help="Risk aversion λ for equilibrium prior π = λ Σ w")
    parser.add_argument("--vol-lookback", type=int, default=60,
                        help="Days of history for covariance estimation")
    parser.add_argument("--optimization-method", default="max_sharpe",
                        choices=["max_sharpe", "min_variance"])
    parser.add_argument("--risk-free-rate", type=float, default=0.03)
    parser.add_argument("--max-weight", type=float, default=0.4,
                        help="Maximum per-fund weight")
    parser.add_argument("--market-weight-method", default="inverse_vol",
                        choices=["inverse_vol", "equal_weight"],
                        help="BL prior weight method: 'inverse_vol'=risk-parity prior")

    # Technical factor scoring parameters
    parser.add_argument("--base-magnitude", type=float, default=0.015,
                        help="composite_score=1.0 → this expected daily return (e.g. 0.015=1.5%%)")
    parser.add_argument("--min-score-threshold", type=float, default=0.20,
                        help="|composite_score| below this → no view generated")
    parser.add_argument("--min-confidence", type=float, default=0.25,
                        help="Minimum view confidence (pre- and post-ViewFlip)")
    parser.add_argument("--max-views", type=int, default=10,
                        help="Max views passed to BL optimizer per day")

    # Rolling sentiment / ViewFlip parameters
    parser.add_argument("--enable-rolling-sentiment", action="store_true", default=True,
                        help="Enable LLM news sentiment + ViewFlip (default: on)")
    parser.add_argument("--no-rolling-sentiment", dest="enable_rolling_sentiment",
                        action="store_false",
                        help="Disable sentiment overlay (pure technical mode)")
    parser.add_argument("--news-model", default=None,
                        help="LLM for news/sentiment (default: same as --model)")
    parser.add_argument("--model", default="deepseek-v4-flash",
                        help="Default LLM model (used for sentiment if --news-model unset)")
    parser.add_argument("--rolling-sentiment-days", type=int, default=10,
                        help="Historical days to warm-start rolling tracker")
    parser.add_argument("--alpha-base", type=float, default=0.30,
                        help="Rolling tracker EMA update weight")
    parser.add_argument("--no-news-decay", type=float, default=0.05,
                        help="Daily rolling score decay when no news")
    parser.add_argument("--view-flip-threshold", type=float, default=0.25,
                        help="|rolling_score| must exceed this to trigger ViewFlip")
    parser.add_argument("--asymmetric-factor", type=float, default=0.50,
                        help="Same-direction confidence boost / opposite-direction penalty")
    parser.add_argument("--sentiment-preload-cache",
                        default="sentiment_preload_cache_tech_bl.json",
                        help="Cache file path for preloaded sentiment scores")

    # Platform connection
    parser.add_argument("--username",
                        default=AGENT_PLATFORM.get("AGENT_USERNAME", "tech_bl_viewflip_agent"))
    parser.add_argument("--password",
                        default=AGENT_PLATFORM.get("AGENT_PASSWORD", "bl_password"))
    parser.add_argument("--base-url",
                        default=AGENT_PLATFORM.get("BASE_URL", "http://localhost:6207"))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output", default=None,
                        help="Override output JSON path")

    # Backtest platform data parameters
    parser.add_argument("--lookback-days", type=int, default=5,
                        help="Days of price history returned per API call")
    parser.add_argument("--top-rank", type=int, default=20)
    parser.add_argument("--pre-k-days", type=int, default=1)
    parser.add_argument("--history-days", type=int, default=5,
                        help="Trading history days passed to platform")

    return parser.parse_args()


def build_config(args):
    fund_pool = MAJOR_FUND_POOL if args.track == "macro" else INDUSTRY_FUND_POOL
    default_results_dir = (
        "backtest_results_macro_tech_bl_viewflip"
        if args.track == "macro"
        else "backtest_results_sector_tech_bl_viewflip"
    )
    return {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_capital": args.initial_capital,
        "fund_pool": fund_pool,
        "agents": [{"name": args.username, "prompt": "Tech-BL-ViewFlip Agent"}],
        "news_sources": ["caixin", "tiantian", "sinafinance", "tencent"],
        "lookback_days": args.lookback_days,
        "top_rank": args.top_rank,
        "pre_k_days": args.pre_k_days,
        "view_platform_trading_history_days": args.history_days,
        "results_dir": args.results_dir or default_results_dir,
    }


def build_output_path(args, session_id):
    if args.output:
        return Path(args.output)
    out_dir = Path(project_root) / "agent_platform" / "demo_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prior_tag = "rp" if args.market_weight_method == "inverse_vol" else "eq"
    sent_tag = "sent" if args.enable_rolling_sentiment else "nosent"
    return out_dir / (
        f"{args.track}_tech_bl_{prior_tag}_{sent_tag}_{session_id}_{timestamp}.json"
    )


async def run_demo_backtest(args):
    """Main backtest loop."""
    fund_pool = MAJOR_FUND_POOL if args.track == "macro" else INDUSTRY_FUND_POOL
    config = build_config(args)
    price_data_dir = Path(DATA_DIRS["PRICE_DATA"])

    # Initialize DataLoader (needed by server-side price APIs)
    init_data_loader(
        price_data_dir=str(DATA_DIRS["PRICE_DATA"]),
        news_data_dir=str(DATA_DIRS["NEWS_DATA"]),
    )
    logger.info("DataLoader initialized")

    # Platform client
    client = PlatformClient(base_url=args.base_url)
    client.register(args.username, args.password)
    client.login(args.username, args.password)

    # Preload historical prices for BL covariance calculation
    logger.info(f"Preloading historical prices (lookback={args.vol_lookback}d)...")
    preloaded_prices = preload_historical_prices(
        fund_pool=fund_pool,
        backtest_start_date=args.start_date,
        lookback_days=args.vol_lookback,
    )
    min_preloaded = min(len(v) for v in preloaded_prices.values()) if preloaded_prices else 0
    logger.info(f"Preloaded {min_preloaded} days of prices per fund")

    # Initialize agent
    news_model = args.news_model or args.model
    agent = TechBLViewflipAgent(
        agent_id=f"{args.track}_tech_bl_viewflip",
        price_data_dir=price_data_dir,
        tau=args.tau,
        risk_aversion=args.risk_aversion,
        lookback_days=args.vol_lookback,
        optimization_method=args.optimization_method,
        risk_free_rate=args.risk_free_rate,
        min_weight=0.0,
        max_weight=args.max_weight,
        market_weight_method=args.market_weight_method,
        base_magnitude=args.base_magnitude,
        min_score_threshold=args.min_score_threshold,
        min_confidence=args.min_confidence,
        max_views_per_day=args.max_views,
        enable_rolling_sentiment=args.enable_rolling_sentiment,
        rolling_sentiment_days=args.rolling_sentiment_days,
        alpha_base=args.alpha_base,
        no_news_decay=args.no_news_decay,
        view_flip_threshold=args.view_flip_threshold,
        asymmetric_factor=args.asymmetric_factor,
        news_model_name=news_model,
    )

    # Warm-start rolling sentiment tracker with historical data
    if args.enable_rolling_sentiment:
        logger.info(
            f"Preloading historical sentiment for warm-up "
            f"({args.rolling_sentiment_days} days before {args.start_date})..."
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

    # Start backtest session
    start_response = client.start_backtest(config)
    session_id = start_response["session_id"]
    data = start_response.get("data")

    if not data:
        raise RuntimeError("Failed to get initial backtest data from platform.")

    logger.info(
        f"Session {session_id} started | track={args.track} | "
        f"prior={args.market_weight_method} | sentiment={'on' if args.enable_rolling_sentiment else 'off'}"
    )

    trading_days = 0
    total_views = 0
    daily_stats = []  # for JSON output

    while True:
        trading_days += 1
        current_date = data.get("date", "Unknown")

        portfolio = client.get_backtest_status(session_id)

        # Merge preloaded prices with backtest prices for covariance
        backtest_prices = client.get_historical_prices(
            session_id, lookback_days=max(args.lookback_days, args.vol_lookback)
        ).get("historical_prices", {})

        historical_prices = {}
        for fund_id in fund_pool:
            preloaded = preloaded_prices.get(fund_id, [])
            backtest = backtest_prices.get(fund_id, [])
            # Deduplicate boundary day
            if preloaded and backtest:
                if preloaded[-1].get("date") == (backtest[0].get("date") if backtest else None):
                    preloaded = preloaded[:-1]
            historical_prices[fund_id] = (preloaded + backtest)[-args.vol_lookback:]

        try:
            decision_result = await agent.make_decision(
                date_to_decision=current_date,
                news_data=data.get("news", []),
                historical_prices=historical_prices,
                current_portfolio=portfolio,
                fund_pool=fund_pool,
            )

            final_decision = decision_result["final_decision"]
            intermediate = decision_result.get("intermediate_results", {})

            trades = [t for t in final_decision.get("trades", []) if t.get("action") != "hold"]
            views_today = intermediate.get("tech_views", [])
            total_views += len(views_today)

            # Log and submit
            if trades:
                opt = final_decision.get("optimization_metrics", {})
                logger.info(
                    f"Day {trading_days} ({current_date}): "
                    f"{len(views_today)} views, {len(trades)} trades | "
                    f"E[R]={opt.get('expected_return', 0):.4%} "
                    f"Sharpe={opt.get('sharpe_ratio', 0):.4f}"
                )
                for t in trades:
                    if t.get("action") == "buy":
                        logger.info(f"  BUY  {t['fund_id']}: {t.get('amount', 0):.0f} CNY")
                    elif t.get("action") == "sell":
                        logger.info(f"  SELL {t['fund_id']}: {t.get('percentage', 0):.1%}")

                agent_decision = AgentDecision(
                    decision=final_decision,
                    reasoning=final_decision.get("reasoning", ""),
                    chain_of_thought="",
                )
                client.submit_trade_with_decision(session_id, trades, agent_decision)

            daily_stats.append({
                "date": current_date,
                "views_count": len(views_today),
                "trades_count": len(trades),
                "optimization_metrics": final_decision.get("optimization_metrics", {}),
            })

        except Exception as exc:
            logger.error(f"Decision failed on {current_date}: {exc}")
            logger.error(traceback.format_exc())

        data = client.get_next_day_data(session_id)
        if data.get("message") == "Backtest finished":
            break

        await asyncio.sleep(0.05)

    # Collect results
    final_results = client.get_backtest_results(session_id)
    output_path = build_output_path(args, session_id)

    rolling_history = []
    if args.enable_rolling_sentiment and agent.rolling_tracker is not None:
        rolling_history = agent.rolling_tracker.get_history()

    output_data = {
        **final_results,
        "session_id": session_id,
        "backtest_config": config,
        "strategy_params": {
            "tau": args.tau,
            "risk_aversion": args.risk_aversion,
            "vol_lookback": args.vol_lookback,
            "optimization_method": args.optimization_method,
            "market_weight_method": args.market_weight_method,
            "base_magnitude": args.base_magnitude,
            "min_score_threshold": args.min_score_threshold,
            "min_confidence": args.min_confidence,
            "max_views": args.max_views,
        },
        "rolling_sentiment_params": {
            "enable": args.enable_rolling_sentiment,
            "rolling_sentiment_days": args.rolling_sentiment_days,
            "alpha_base": args.alpha_base,
            "no_news_decay": args.no_news_decay,
            "view_flip_threshold": args.view_flip_threshold,
            "asymmetric_factor": args.asymmetric_factor,
        },
        "decision_stats": {
            "trading_days": trading_days,
            "total_views": total_views,
            "avg_views_per_day": total_views / max(trading_days, 1),
        },
        "daily_stats": daily_stats,
        "rolling_tracker_history": rolling_history,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=2, ensure_ascii=False, default=str)

    perf = final_results.get("performance", {})
    logger.info(f"\n{'='*60}")
    logger.info(f"Tech-BL-ViewFlip Backtest Results ({args.track})")
    logger.info(f"{'='*60}")
    logger.info(f"Total Return:    {perf.get('total_return', 0)*100:.2f}%")
    logger.info(f"Ann. Return:     {perf.get('annualized_return', 0)*100:.2f}%")
    logger.info(f"Sharpe Ratio:    {perf.get('sharpe_ratio', 0):.4f}")
    logger.info(f"Max Drawdown:    {perf.get('max_drawdown', 0)*100:.2f}%")
    logger.info(f"Trading Days:    {trading_days}")
    logger.info(f"Avg Views/Day:   {total_views/max(trading_days,1):.1f}")
    logger.info(f"Prior:           {args.market_weight_method}")
    logger.info(f"Sentiment:       {'enabled' if args.enable_rolling_sentiment else 'disabled'}")
    logger.info(f"Results saved →  {output_path}")


def main():
    args = parse_args()

    log_dir = Path(project_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"tech_bl_viewflip_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logger.add(str(log_path), level="INFO")

    logger.info(
        f"Tech-BL-ViewFlip | track={args.track} | "
        f"prior={args.market_weight_method} | sentiment={'on' if args.enable_rolling_sentiment else 'off'} | "
        f"period={args.start_date}~{args.end_date}"
    )

    try:
        asyncio.run(run_demo_backtest(args))
    except Exception as exc:
        logger.error(f"Backtest failed: {exc}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
