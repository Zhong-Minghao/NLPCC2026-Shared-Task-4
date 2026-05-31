#!/usr/bin/env python3
"""
Black-Litterman Agent + Risk Management Backtest Demo

Pipeline: News → Sentiment → BL Views → BL Optimization → Trades → Risk Management Agent
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from loguru import logger

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent_platform.agents.advanced_agents import RiskManagementAgent
from agent_platform.agents.black_litterman_agent import BlackLittermanAgent
from agent_platform.backtest_risk_parity_baseline import preload_historical_prices
from agent_platform.client.platform_client import PlatformClient
from config import AGENT_PLATFORM, DATA_DIRS
from server_platform.app.core.data_loader import init_data_loader
from server_platform.app.models.backtest import AgentDecision

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


def _clean_trade_for_server(trade: Dict) -> Dict:
    """Strip fields that violate server validation rules (buy must not have percentage, etc.)."""
    action = trade.get("action", "")
    cleaned = {
        "fund_id": trade.get("fund_id"),
        "action": action,
        "reason": trade.get("reason", ""),
    }
    if action == "buy":
        cleaned["amount"] = trade.get("amount", 0)
    elif action == "sell":
        cleaned["percentage"] = trade.get("percentage", 0)
    elif action == "hold":
        if "amount" in trade:
            cleaned["amount"] = trade["amount"]
        if "percentage" in trade:
            cleaned["percentage"] = trade["percentage"]
    return cleaned


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Black-Litterman + Risk Management agent backtest"
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

    # Risk management parameters
    parser.add_argument("--min-holding-days", type=int, default=7,
                        help="Minimum holding period enforced by the risk management agent")
    parser.add_argument("--max-concentration", type=float, default=0.4,
                        help="Maximum single-position concentration (0.0-1.0)")

    # Other parameters
    parser.add_argument("--lookback-days", type=int, default=5)
    parser.add_argument("--top-rank", type=int, default=20)
    parser.add_argument("--pre-k-days", type=int, default=1)
    parser.add_argument("--history-days", type=int, default=5)

    # Server config
    parser.add_argument("--username", default=AGENT_PLATFORM.get("AGENT_USERNAME", "bl_mgmt_agent"))
    parser.add_argument("--password", default=AGENT_PLATFORM.get("AGENT_PASSWORD", "bl_mgmt_password"))
    parser.add_argument("--base-url", default=AGENT_PLATFORM.get("BASE_URL", "http://localhost:6207"))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output", default=None)

    return parser.parse_args()


def build_config(args):
    fund_pool = MAJOR_FUND_POOL if args.track == "macro" else INDUSTRY_FUND_POOL
    default_results_dir = (
        "backtest_results_macro_bl_mgmt"
        if args.track == "macro"
        else "backtest_results_sector_bl_mgmt"
    )
    return {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_capital": args.initial_capital,
        "fund_pool": fund_pool,
        "agents": [{"name": args.username, "prompt": "Black-Litterman + Risk Management Agent"}],
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
    return out_dir / f"{args.track}_bl_mgmt_{args.model}_{session_id}_{timestamp}.json"


async def run_demo_backtest(args):
    """Run Black-Litterman + Risk Management agent backtest."""
    fund_pool = MAJOR_FUND_POOL if args.track == "macro" else INDUSTRY_FUND_POOL
    config = build_config(args)

    init_data_loader(
        price_data_dir=str(DATA_DIRS["PRICE_DATA"]),
        news_data_dir=str(DATA_DIRS["NEWS_DATA"]),
    )
    logger.info("DataLoader initialized")

    client = PlatformClient(base_url=args.base_url)
    client.register(args.username, args.password)
    client.login(args.username, args.password)

    logger.info(f"Preloading historical prices (lookback: {args.vol_lookback} days)...")
    preloaded_prices = preload_historical_prices(
        fund_pool=fund_pool,
        backtest_start_date=args.start_date,
        lookback_days=args.vol_lookback,
    )
    min_preloaded = min(len(v) for v in preloaded_prices.values()) if preloaded_prices else 0
    logger.info(f"Preloaded {min_preloaded} days of historical data")

    # BL agent with internal risk management disabled — we handle it externally
    bl_agent = BlackLittermanAgent(
        agent_id=f"{args.track}_bl_agent",
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
        enable_risk_management=False,
    )

    # Standalone risk management agent (step 5 of the pipeline)
    risk_management_agent = RiskManagementAgent(
        model_name=args.model,
        min_holding_days=args.min_holding_days,
        max_position_concentration=args.max_concentration,
    )

    logger.info(f"Risk Management Settings:")
    logger.info(f"  - Min holding period: {args.min_holding_days} days")
    logger.info(f"  - Max position concentration: {args.max_concentration:.0%}")

    start_response = client.start_backtest(config)
    session_id = start_response["session_id"]
    data = start_response.get("data")

    if not data:
        raise RuntimeError("Failed to get initial backtest data.")

    logger.info(
        f"Started session {session_id} for track={args.track}, model={args.model}"
    )
    logger.info(
        f"BL parameters: tau={args.tau}, method={args.optimization_method}, "
        f"vol_lookback={args.vol_lookback}"
    )

    trading_days = 0
    total_views = 0
    risk_stats = {
        "total_trades_proposed": 0,
        "total_trades_approved": 0,
        "total_trades_modified": 0,
        "total_trades_blocked": 0,
    }

    while True:
        trading_days += 1
        current_date = data.get("date", "Unknown")

        portfolio = client.get_backtest_status(session_id)

        backtest_prices = client.get_historical_prices(
            session_id, lookback_days=max(args.lookback_days, args.vol_lookback)
        ).get("historical_prices", {})

        # Merge preloaded and backtest prices
        historical_prices = {}
        for fund_id in fund_pool:
            preloaded = preloaded_prices.get(fund_id, [])
            backtest = backtest_prices.get(fund_id, [])

            if preloaded and backtest:
                last_pre_date = preloaded[-1].get("date")
                if last_pre_date == backtest[0].get("date"):
                    preloaded = preloaded[:-1]

            combined = preloaded + backtest
            historical_prices[fund_id] = combined[-args.vol_lookback:]

        market_data = data.get("market_data", {})

        try:
            # Steps 1-4: News → Sentiment → BL Views → BL Optimization → Trades
            decision_result = await bl_agent.make_decision(
                date_to_decision=data["date"],
                news_data=data["news"],
                historical_prices=historical_prices,
                current_portfolio=portfolio,
                market_data=market_data,
                fund_pool=config["fund_pool"],
                view_platform_trading_history_days=config["view_platform_trading_history_days"],
            )

            final_decision = decision_result["final_decision"]
            intermediate = decision_result.get("intermediate_results", {})
            views = intermediate.get("views", [])
            total_views += len(views)

            proposed_trades = [
                t for t in final_decision.get("trades", [])
                if t.get("action") != "hold"
            ]
            risk_stats["total_trades_proposed"] += len(proposed_trades)

            # Step 5: Risk Management Agent
            if proposed_trades:
                logger.info(f"🛡️ [Risk Management] 评估 {len(proposed_trades)} 笔交易...")
                sentiment_analysis = intermediate.get("sentiment_analysis", {})
                risk_result = await risk_management_agent.evaluate_trades(
                    proposed_trades=proposed_trades,
                    current_portfolio=portfolio,
                    sentiment_analysis=sentiment_analysis,
                    current_date=current_date,
                )

                approved_trades = risk_result.get("approved_trades", [])
                modified_trades = risk_result.get("modified_trades", [])
                blocked_trades = risk_result.get("blocked_trades", [])

                risk_stats["total_trades_approved"] += len(approved_trades)
                risk_stats["total_trades_modified"] += len(modified_trades)
                risk_stats["total_trades_blocked"] += len(blocked_trades)

                if modified_trades:
                    logger.warning(f"  🛡️ 调整 {len(modified_trades)} 笔交易")
                    for mt in modified_trades:
                        logger.warning(f"    - {mt['fund_id']}: {mt.get('reason', 'N/A')}")
                if blocked_trades:
                    logger.warning(f"  🛡️ 阻止 {len(blocked_trades)} 笔交易")
                    for bt in blocked_trades:
                        logger.warning(f"    - {bt['fund_id']}: {bt.get('reason', 'N/A')}")

                logger.info(
                    f"  最终执行: {len(approved_trades) + len(modified_trades)} 笔交易 "
                    f"| 风险摘要: {risk_result.get('risk_summary', 'N/A')}"
                )

                # Merge approved and modified trades into final list
                risk_managed_trades = approved_trades + [
                    {k: v for k, v in mt.items()
                     if k in ["fund_id", "action", "amount", "percentage", "reason"]}
                    for mt in modified_trades
                ]

                final_decision["trades"] = risk_managed_trades
                final_decision["risk_management"] = {
                    "approved": len(approved_trades),
                    "modified": len(modified_trades),
                    "blocked": len(blocked_trades),
                    "risk_summary": risk_result.get("risk_summary", ""),
                }
            else:
                risk_managed_trades = []

            trades_to_submit = [
                _clean_trade_for_server(t)
                for t in final_decision.get("trades", [])
                if t.get("action") != "hold"
            ]

            if trades_to_submit:
                logger.info(
                    f"Day {trading_days} ({current_date}): "
                    f"{len(views)} views, {len(trades_to_submit)} trades after risk management"
                )

                for trade in trades_to_submit:
                    action = trade.get("action")
                    fund_id = trade.get("fund_id")
                    if action == "buy":
                        logger.info(f"  BUY {fund_id}: {trade.get('amount', 0):.0f} CNY")
                    elif action == "sell":
                        logger.info(f"  SELL {fund_id}: {trade.get('percentage', 0):.1%}")

                opt_metrics = final_decision.get("optimization_metrics", {})
                if opt_metrics:
                    logger.info(
                        f"  Portfolio: E[Return]={opt_metrics.get('expected_return', 0):.4%}, "
                        f"Risk={opt_metrics.get('expected_risk', 0):.4%}, "
                        f"Sharpe={opt_metrics.get('sharpe_ratio', 0):.4f}"
                    )

                risk_mgmt_info = final_decision.get("risk_management", {})
                if risk_mgmt_info:
                    logger.info(
                        f"  风控: 批准{risk_mgmt_info.get('approved', 0)} "
                        f"调整{risk_mgmt_info.get('modified', 0)} "
                        f"阻止{risk_mgmt_info.get('blocked', 0)}"
                    )

                agent_decision = AgentDecision(
                    decision=final_decision,
                    reasoning=final_decision.get("reasoning", ""),
                    chain_of_thought=str(final_decision.get("chain_of_thought", "")),
                )
                client.submit_trade_with_decision(session_id, trades_to_submit, agent_decision)

        except Exception as exc:
            logger.error(f"Decision failed on {current_date}: {exc}")
            logger.error(traceback.format_exc())

        data = client.get_next_day_data(session_id)
        if data.get("message") == "Backtest finished":
            break

        await asyncio.sleep(0.1)

    final_results = client.get_backtest_results(session_id)
    output_path = build_output_path(args, session_id)

    # Compute approval rates
    total_proposed = risk_stats["total_trades_proposed"]
    risk_stats_output = {
        **risk_stats,
        "approval_rate": risk_stats["total_trades_approved"] / max(total_proposed, 1),
        "modification_rate": risk_stats["total_trades_modified"] / max(total_proposed, 1),
        "block_rate": risk_stats["total_trades_blocked"] / max(total_proposed, 1),
    }

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
        },
        "risk_management_params": {
            "min_holding_days": args.min_holding_days,
            "max_concentration": args.max_concentration,
        },
        "decision_stats": {
            "trading_days": trading_days,
            "total_views": total_views,
            "avg_views_per_day": total_views / max(trading_days, 1),
        },
        "risk_statistics": risk_stats_output,
        "decision_history": bl_agent.get_decision_history(),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2, ensure_ascii=False)

    perf = final_results.get("performance", {})
    logger.info(f"\n{'='*60}")
    logger.info(f"Black-Litterman + Risk Management Backtest ({args.track} track)")
    logger.info(f"{'='*60}")
    logger.info(f"Total Return:  {perf.get('total_return', 0)*100:.2f}%")
    logger.info(f"Sharpe Ratio:  {perf.get('sharpe_ratio', 0):.4f}")
    logger.info(f"Max Drawdown:  {perf.get('max_drawdown', 0)*100:.2f}%")
    logger.info(f"Trading Days:  {trading_days}")
    logger.info(f"Total Views:   {total_views}")
    logger.info(f"Risk Management Statistics:")
    logger.info(f"  - Proposed:  {risk_stats['total_trades_proposed']}")
    logger.info(
        f"  - Approved:  {risk_stats['total_trades_approved']} "
        f"({risk_stats_output['approval_rate']*100:.1f}%)"
    )
    logger.info(
        f"  - Modified:  {risk_stats['total_trades_modified']} "
        f"({risk_stats_output['modification_rate']*100:.1f}%)"
    )
    logger.info(
        f"  - Blocked:   {risk_stats['total_trades_blocked']} "
        f"({risk_stats_output['block_rate']*100:.1f}%)"
    )
    logger.info(f"Results saved to: {output_path}")


def main():
    args = parse_args()

    log_dir = Path(project_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"demo_backtest_bl_mgmt_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logger.add(str(log_path), level="INFO")

    logger.info(
        f"Running BL + Risk Management backtest: track={args.track}, model={args.model}, "
        f"period={args.start_date}~{args.end_date}"
    )
    logger.info(
        f"BL: tau={args.tau}, method={args.optimization_method} | "
        f"RiskMgmt: min_holding={args.min_holding_days}d, max_conc={args.max_concentration:.0%}"
    )

    try:
        asyncio.run(run_demo_backtest(args))
    except Exception as exc:
        logger.error(f"BL+RiskMgmt backtest failed: {exc}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
