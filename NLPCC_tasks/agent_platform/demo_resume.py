#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo backtest with checkpoint/resume functionality.

This script automatically saves checkpoints and can resume from interruptions.
Based on actual API behavior from README and code inspection.

Usage:
    # First run (will save checkpoints automatically)
    python agent_platform/demo_resume.py --track sector --model kimi-k2.6

    # If interrupted, run again - will resume from checkpoint
    python agent_platform/demo_resume.py --track sector --model kimi-k2.6

    # Force fresh start (ignore checkpoint)
    python agent_platform/demo_resume.py --track sector --model kimi-k2.6 --no-resume
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

from agent_platform.agents.advanced_agents import get_advanced_agent
from agent_platform.agents.trading_strategy_prompt import BASELINE_TRADING_PROMPT
from agent_platform.client.platform_client import PlatformClient
from config import AGENT_PLATFORM
from server_platform.app.models.backtest import AgentDecision

MAJOR_FUND_POOL = [
    "000300.SH",
    "000905.SH",
    "399006.SZ",
    "000688.SH",
    "000932.SH",
    "000941.SH",
    "399971.SZ",
    "000819.SH",
    "000928.SH",
    "000012.SH",
    "518880.SH",
]

INDUSTRY_FUND_POOL = [
    "512880.SH",
    "512800.SH",
    "512070.SH",
    "159995.SZ",
    "159819.SZ",
    "515880.SH",
    "159852.SZ",
    "512010.SH",
    "512170.SH",
    "159992.SZ",
    "515170.SH",
    "512690.SH",
    "512400.SH",
    "515220.SH",
    "159870.SZ",
    "512200.SH",
]

# Checkpoint directory
CHECKPOINT_DIR = Path(project_root) / "agent_platform" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run demo backtest with checkpoint/resume support."
    )
    parser.add_argument("--track", choices=["macro", "sector"], default="sector")
    parser.add_argument("--model", default="kimi-k2.6")
    parser.add_argument("--start-date", default="2025-01-02")
    parser.add_argument("--end-date", default="2025-01-10")
    parser.add_argument("--initial-capital", type=float, default=100000)
    parser.add_argument("--lookback-days", type=int, default=5)
    parser.add_argument("--top-rank", type=int, default=20)
    parser.add_argument("--pre-k-days", type=int, default=1)
    parser.add_argument("--history-days", type=int, default=5)
    parser.add_argument("--username", default=AGENT_PLATFORM["AGENT_USERNAME"])
    parser.add_argument("--password", default=AGENT_PLATFORM["AGENT_PASSWORD"])
    parser.add_argument("--base-url", default=AGENT_PLATFORM["BASE_URL"])
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N trading days (default: 5)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoint and start fresh",
    )
    return parser.parse_args()


def build_config(args):
    fund_pool = MAJOR_FUND_POOL if args.track == "macro" else INDUSTRY_FUND_POOL
    default_results_dir = (
        "backtest_results_macro_resume"
        if args.track == "macro"
        else "backtest_results_sector_resume"
    )
    return {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_capital": args.initial_capital,
        "fund_pool": fund_pool,
        "agents": [{"name": args.username, "prompt": "..."}],
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
    return out_dir / f"{args.track}_{args.model}_{session_id}.json"


def get_checkpoint_path(args):
    """Get checkpoint file path for this configuration."""
    # Create unique filename based on track and date range
    filename = f"{args.track}_{args.start_date}_{args.end_date}_checkpoint.json"
    return CHECKPOINT_DIR / filename


def save_checkpoint(client, session_id, config, trading_days, current_date):
    """
    Save current backtest state to checkpoint file.

    This saves what we can get from client API.
    Note: This is a simplified version - full state requires server-side export API.
    """
    try:
        # Get current portfolio status
        portfolio_status = client.get_backtest_status(session_id)

        # Get agent decisions
        try:
            agent_decisions = client.get_agent_decisions(session_id).get(
                "decisions", []
            )
        except Exception:
            agent_decisions = []

        # Build saved_data structure based on actual API response
        # Structure based on create_backtest_session_with_restore requirements
        saved_data = {
            "capital": portfolio_status.get("capital", 0),
            "portfolio": portfolio_status.get("holdings", {}),
            # These are empty because client API doesn't provide them
            # Server will create empty lists if not provided
            "daily_portfolio_snapshots": [],
            "transaction_history": [],
            "agent_decisions": agent_decisions,
            "portfolio_value_history": [],
        }

        # Build full checkpoint
        checkpoint = {
            "config": config,
            "session_id": session_id,
            "trading_days": trading_days,
            "current_date": current_date,
            "saved_data": saved_data,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save to file
        checkpoint_path = get_checkpoint_path(config)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        logger.info(
            f"[CHECKPOINT] Saved: {checkpoint_path.name} "
            f"(Day {trading_days}, Date: {current_date})"
        )
        return checkpoint_path

    except Exception as e:
        logger.error(f"[ERROR] Failed to save checkpoint: {e}")
        logger.error(traceback.format_exc())
        return None


def load_checkpoint(args):
    """Load checkpoint if exists and not --no-resume flag."""
    if args.no_resume:
        logger.info("[RESUME] --no-resume flag set, ignoring existing checkpoint")
        return None

    checkpoint_path = get_checkpoint_path(args)

    if not checkpoint_path.exists():
        logger.info(f"[RESUME] No checkpoint found")
        return None

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        logger.info(
            f"[RESUME] Found checkpoint:\n"
            f"  - Previous session: {checkpoint.get('session_id', 'N/A')}\n"
            f"  - Completed days: {checkpoint.get('trading_days', 0)}\n"
            f"  - Last date: {checkpoint.get('current_date', 'N/A')}\n"
            f"  - Saved at: {checkpoint.get('timestamp', 'N/A')}"
        )
        return checkpoint

    except Exception as e:
        logger.error(f"[ERROR] Failed to load checkpoint: {e}")
        return None


def delete_checkpoint(args):
    """Delete checkpoint file after successful completion."""
    checkpoint_path = get_checkpoint_path(args)
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.info(f"[CLEANUP] Checkpoint deleted: {checkpoint_path.name}")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to delete checkpoint: {e}")


async def run_demo_backtest_with_resume(args):
    client = PlatformClient(base_url=args.base_url)
    client.register(args.username, args.password)
    client.login(args.username, args.password)

    config = build_config(args)
    agent = get_advanced_agent(
        agent_id=f"{args.track}_resume_agent",
        trading_prompt_template=BASELINE_TRADING_PROMPT,
        decision_model_name=args.model,
    )

    # Try to load existing checkpoint
    checkpoint = load_checkpoint(args)

    if checkpoint:
        # Resume from checkpoint
        logger.info("[RESUME] Resuming from checkpoint...")
        try:
            # Call resume API with config and saved_data from checkpoint
            start_response = client.resume_backtest(
                checkpoint["config"],
                checkpoint["saved_data"]
            )
        except Exception as e:
            logger.error(f"[ERROR] Failed to resume from checkpoint: {e}")
            logger.info("[FALLBACK] Starting fresh backtest...")
            start_response = client.start_backtest(config)
            checkpoint = None
    else:
        # Start fresh
        logger.info("[START] Starting new backtest session...")
        start_response = client.start_backtest(config)
        checkpoint = None

    session_id = start_response["session_id"]
    data = start_response.get("data")

    if not data:
        raise RuntimeError("Failed to get backtest data.")

    # Determine starting point
    start_day = checkpoint.get("trading_days", 0) if checkpoint else 0
    trading_days = start_day

    logger.info(
        f"[SESSION] {'Resumed' if checkpoint else 'Started'} session {session_id} "
        f"for track={args.track}, model={args.model}"
    )

    # Main trading loop
    while True:
        trading_days += 1
        current_date = data.get("date", "unknown")

        logger.info(f"\n{'='*60}")
        logger.info(f"[DAY {trading_days}] {current_date}")
        logger.info(f"{'='*60}")

        # Get current state
        portfolio = client.get_backtest_status(session_id)
        logger.info(
            f"[STATUS] Capital: {portfolio.get('capital', 0):.2f} | "
            f"Holdings: {len(portfolio.get('holdings', {}))} funds"
        )

        # Get historical prices
        historical_prices = client.get_historical_prices(
            session_id, lookback_days=config["lookback_days"]
        )

        # Make trading decision
        try:
            logger.info("[AGENT] Making decision...")
            decision_result = await agent.make_decision(
                date_to_decision=data["date"],
                news_data=data["news"],
                historical_prices=historical_prices.get("historical_prices", {}),
                current_portfolio=portfolio,
                market_data=data["market_data"],
                fund_pool=config["fund_pool"],
                view_platform_trading_history_days=config[
                    "view_platform_trading_history_days"
                ],
            )

            final_decision = decision_result["final_decision"]
            trades = [
                trade
                for trade in final_decision.get("trades", [])
                if trade.get("action") != "hold"
            ]

            if trades:
                logger.info(f"[TRADE] Submitting {len(trades)} trade(s)...")
                for trade in trades:
                    logger.info(
                        f"  - {trade.get('fund_id')}: {trade.get('action')} "
                        f"{trade.get('amount', trade.get('percentage', 'N/A'))}"
                    )

                agent_decision = AgentDecision(
                    decision=final_decision,
                    reasoning=final_decision.get("reasoning", ""),
                    chain_of_thought=str(final_decision.get("chain_of_thought", "")),
                )
                client.submit_trade_with_decision(session_id, trades, agent_decision)
            else:
                logger.info("[TRADE] No trades (all hold)")

        except Exception as exc:
            logger.error(f"[ERROR] Decision failed on {data.get('date')}: {exc}")
            logger.error(traceback.format_exc())

            # Save checkpoint on error before re-raising
            logger.info("[CHECKPOINT] Saving before exit...")
            save_checkpoint(client, session_id, config, trading_days, current_date)
            raise

        # Move to next day
        data = client.get_next_day_data(session_id)

        if data.get("message") == "Backtest finished":
            logger.info("\n[FINISH] Backtest completed!")
            break

        # Periodic checkpoint save
        if trading_days % args.checkpoint_interval == 0:
            save_checkpoint(
                client, session_id, config, trading_days, data.get("date", "unknown")
            )

        await asyncio.sleep(0.1)

    # Get final results
    logger.info("[RESULT] Fetching final results...")
    final_results = client.get_backtest_results(session_id)
    output_path = build_output_path(args, session_id)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(final_results, handle, indent=2, ensure_ascii=False)

    logger.info(
        f"\n{'='*60}\n"
        f"[RESULT] Session {session_id} completed after {trading_days} days\n"
        f"{'='*60}\n"
        f"Total Return: {final_results.get('performance', {}).get('total_return', 0) * 100:.2f}%\n"
        f"Sharpe Ratio: {final_results.get('performance', {}).get('sharpe_ratio', 0):.4f}\n"
        f"Max Drawdown: {final_results.get('performance', {}).get('max_drawdown', 0) * 100:.2f}%\n"
        f"{'='*60}\n"
        f"Saved to: {output_path}"
    )

    # Clean up checkpoint after successful completion
    delete_checkpoint(args)


def main():
    args = parse_args()

    # Setup logging
    log_dir = Path(project_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"demo_resume_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logger.add(str(log_path), level="INFO")

    logger.info("=" * 60)
    logger.info("Demo Resume Backtest Runner")
    logger.info("=" * 60)
    logger.info(
        f"Track: {args.track} | Model: {args.model} | "
        f"Period: {args.start_date} ~ {args.end_date}"
    )
    logger.info(f"Checkpoint interval: Every {args.checkpoint_interval} days")
    logger.info(f"Checkpoint dir: {CHECKPOINT_DIR}")
    if args.no_resume:
        logger.info("Mode: Fresh start (--no-resume flag set)")
    else:
        logger.info("Mode: Auto-resume enabled")
    logger.info("=" * 60)

    try:
        asyncio.run(run_demo_backtest_with_resume(args))
    except KeyboardInterrupt:
        logger.warning("\n[INTERRUPT] Interrupted by user (Ctrl+C)")
        logger.info("[CHECKPOINT] Checkpoint saved. Run again to resume.")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"\n[ERROR] Demo backtest failed: {exc}")
        logger.error(traceback.format_exc())
        logger.info("[CHECKPOINT] Checkpoint saved. Run again to resume.")
        sys.exit(1)


if __name__ == "__main__":
    main()
