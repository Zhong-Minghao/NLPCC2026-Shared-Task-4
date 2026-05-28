#!/usr/bin/env python3
"""
Risk Parity + Risk Management Agent Demo Backtest Runner.

This demo integrates:
1. Risk Parity baseline weights
2. 4-Agent system: News → Sentiment → Trading → Risk Management
3. Risk Management Agent enforces holding periods and controls trading frequency
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent_platform.agents.advanced_agents import (
    NewsProcessingAgent,
    SentimentAnalysisAgent,
    TradingStrategyAgent,
    RiskManagementAgent,
)
from agent_platform.agents.fund_info import FUND_INFO
from agent_platform.agents.trading_strategy_prompt import RISK_PARITY_TRADING_PROMPT
from agent_platform.client.platform_client import PlatformClient
from agent_platform.utils import CustomJsonOutputParser
from config import AGENT_PLATFORM, DATA_DIRS
from server_platform.app.core.data_loader import get_data_loader, init_data_loader
from server_platform.app.models.backtest import AgentDecision


def _clean_trade_for_server(trade: Dict) -> Dict:
    """Clean trade dict to match server validation requirements.

    Server requires:
    - buy: amount must be set, percentage must be None/not present
    - sell: percentage must be set, amount must be None/not present
    """
    action = trade.get("action", "")
    cleaned = {
        "fund_id": trade.get("fund_id"),
        "action": action,
        "reason": trade.get("reason", ""),
    }

    if action == "buy":
        cleaned["amount"] = trade.get("amount", 0)
        # Do NOT include percentage for buy trades
    elif action == "sell":
        cleaned["percentage"] = trade.get("percentage", 0)
        # Do NOT include amount for sell trades
    elif action == "hold":
        # For hold, keep optional fields if present
        if "amount" in trade:
            cleaned["amount"] = trade["amount"]
        if "percentage" in trade:
            cleaned["percentage"] = trade["percentage"]

    return cleaned


def _clean_nan_value(value):
    """Convert NaN, inf, -inf values to None for JSON serialization."""
    if value is None:
        return None
    try:
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value) if isinstance(value, (int, float, np.number)) else value
    except (TypeError, ValueError):
        return None


def preload_historical_prices(
    fund_pool: List[str],
    backtest_start_date: str,
    lookback_days: int,
) -> Dict[str, List[Dict]]:
    """
    Preload historical prices before backtest starts.

    This ensures we have enough historical data (e.g., 60 days) for risk parity
    calculation, even when the backtest starts in 2025 and data goes back to 2024.
    """
    data_loader = get_data_loader()

    # Calculate the date range we need
    start_dt = datetime.strptime(backtest_start_date, "%Y-%m-%d")
    # Go back lookback_days trading days (approx lookback_days * 1.4 calendar days)
    hist_start_dt = start_dt - timedelta(days=int(lookback_days * 1.4) + 10)
    hist_start_int = int(hist_start_dt.strftime("%Y%m%d"))
    backtest_start_int = int(backtest_start_date.replace("-", ""))

    logger.info(f"Preloading historical prices from {hist_start_dt.strftime('%Y-%m-%d')} to {backtest_start_date}")

    result = {}
    for fund_id in fund_pool:
        df = data_loader._get_price_df(fund_id)
        if df is None:
            logger.warning(f"No data available for {fund_id}")
            result[fund_id] = []
            continue

        # Filter data for the date range we need (before backtest start)
        fund_data = df[
            (df.index >= hist_start_int) & (df.index < backtest_start_int)
        ].reset_index()

        records = []
        for _, row in fund_data.iterrows():
            date_int = int(row["date"])
            records.append({
                "date": datetime.strptime(str(date_int), "%Y%m%d").strftime("%Y-%m-%d"),
                "date_int": date_int,
                "open": _clean_nan_value(row.get("open")),
                "close": _clean_nan_value(row.get("close")),
                "high": _clean_nan_value(row.get("high")),
                "low": _clean_nan_value(row.get("low")),
                "pct_change": _clean_nan_value(row.get("pctchange")),
            })

        result[fund_id] = records
        logger.info(f"  {fund_id}: loaded {len(records)} days of historical data")

    return result

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


class RiskParityCalculator:
    """Calculate risk parity weights from historical returns."""

    def __init__(
        self,
        lookback_days: int = 60,
        min_data_points: int = 20,
    ):
        self.lookback_days = lookback_days
        self.min_data_points = min_data_points

    def calculate_weights(
        self,
        historical_prices: Dict[str, List[Dict]],
        fund_pool: List[str],
    ) -> Dict[str, float]:
        """Calculate risk parity weights based on historical volatility."""
        # Build returns data for each fund
        returns_data = {}
        for fund_id in fund_pool:
            prices = historical_prices.get(fund_id, [])
            if len(prices) < self.min_data_points:
                continue

            # Extract closes and calculate daily returns
            closes = []
            for p in prices[-self.lookback_days:]:
                close = p.get("close")
                if close is not None and not math.isnan(close):
                    closes.append(close)

            if len(closes) < 2:
                continue

            # Calculate daily returns
            returns = []
            for i in range(1, len(closes)):
                ret = (closes[i] - closes[i - 1]) / closes[i - 1]
                if not math.isnan(ret) and not math.isinf(ret):
                    returns.append(ret)

            if len(returns) >= self.min_data_points:
                returns_data[fund_id] = returns

        if not returns_data:
            logger.warning("Insufficient data for risk parity, falling back to equal weights")
            return {f: 1.0 / len(fund_pool) for f in fund_pool}

        # Calculate volatility (standard deviation) for each asset
        volatilities = {}
        for fund_id, returns in returns_data.items():
            vol = np.std(returns) if len(returns) > 0 else 1.0
            volatilities[fund_id] = max(vol, 1e-6)

        # Risk parity: weight inversely proportional to volatility
        inv_vols = {f: 1.0 / v for f, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        weights = {f: v / total_inv_vol for f, v in inv_vols.items()}

        # For funds with no data, assign small residual weight
        allocated = sum(weights.values())
        unallocated = [f for f in fund_pool if f not in weights]
        if unallocated:
            if allocated < 0.99:
                residual_weight = (1.0 - allocated) / len(unallocated)
                for f in unallocated:
                    weights[f] = residual_weight
            else:
                # Proportionally reduce existing weights to make room
                scale_factor = 0.99 / allocated
                for f in weights:
                    weights[f] *= scale_factor
                residual_weight = 0.01 / len(unallocated)
                for f in unallocated:
                    weights[f] = residual_weight

        return weights


class RiskParityWithManagementAgent:
    """Trading Agent with 4-Agent System: News → Sentiment → Trading → Risk Management."""

    def __init__(
        self,
        agent_id: str,
        trading_prompt_template: str = None,
        decision_model_name: str = "deepseek-v4-pro",
        news_model_name: str = "deepseek-v4-flash",
        vol_lookback: int = 60,
        min_holding_days: int = 7,
        max_position_concentration: float = 0.4,
    ):
        self.agent_id = agent_id
        self.news_agent = NewsProcessingAgent(
            model_name=news_model_name or decision_model_name
        )
        self.sentiment_agent = SentimentAnalysisAgent(model_name=decision_model_name)
        self.trading_agent = TradingStrategyAgent(
            prompt_template=trading_prompt_template or RISK_PARITY_TRADING_PROMPT,
            model_name=decision_model_name,
        )
        self.risk_management_agent = RiskManagementAgent(
            model_name=decision_model_name,
            min_holding_days=min_holding_days,
            max_position_concentration=max_position_concentration,
        )
        self.rp_calculator = RiskParityCalculator(lookback_days=vol_lookback)
        self.vol_lookback = vol_lookback

        self.decision_history = []
        self.trading_history = []
        self.platform_trading_history = []
        self.prev_rp_weights = {}

        # Risk management statistics
        self.risk_stats = {
            "total_trades_proposed": 0,
            "total_trades_approved": 0,
            "total_trades_modified": 0,
            "total_trades_blocked": 0,
        }

    def _format_rp_weights(self, weights: Dict[str, float], fund_pool: List[str]) -> str:
        """Format risk parity weights for display in prompt."""
        lines = []
        for fund_id in fund_pool:
            weight = weights.get(fund_id, 0.0)
            fund_name = FUND_INFO.get(fund_id, {}).get('name', 'Unknown')
            lines.append(f"  - {fund_id} ({fund_name}): {weight:.2%}")
        return "\n".join(lines)

    async def make_decision(
        self,
        date_to_decision: str,
        news_data: List[Dict],
        historical_prices: Dict,
        current_portfolio: Dict,
        market_data: Dict,
        fund_pool: List[str],
        view_platform_trading_history_days: int = 5,
    ) -> Dict:
        """Complete 4-agent decision flow with risk management."""

        logger.info(f"🤖 {self.agent_id} 开始4-Agent决策流程（新闻→舆情→交易→风控）...")

        # 0. Calculate risk parity weights
        logger.info("📊 计算风险平价基准权重...")
        current_rp_weights = self.rp_calculator.calculate_weights(
            historical_prices, fund_pool
        )
        logger.info(f"  本期RP权重已计算 (lookback={self.vol_lookback}天)")

        # Format weights for prompt
        current_rp_weights_text = self._format_rp_weights(current_rp_weights, fund_pool)
        prev_rp_weights_text = self._format_rp_weights(self.prev_rp_weights, fund_pool)

        # 1. News processing
        logger.info("📰 [Agent 1/4] 新闻处理中...")
        processed_news = await self.news_agent.process_news_batch(news_data)
        logger.info(f"  处理完成: {len(processed_news)}/{len(news_data)} 条新闻")

        # 2. Sentiment analysis
        logger.info("🎯 [Agent 2/4] 舆情分析中...")
        sentiment_analysis = await self.sentiment_agent.analyze_sentiment(
            date_to_decision, processed_news, fund_pool
        )
        logger.info(
            f"  舆情结果: {sentiment_analysis.get('overall_sentiment', 'unknown')}"
        )

        # 3. Trading decision with risk parity weights
        logger.info("💹 [Agent 3/4] 交易决策中（结合风险平价基准）...")
        trading_decision = await self.trading_agent.make_trading_decision_with_rp(
            date_to_decision,
            sentiment_analysis,
            historical_prices,
            current_portfolio,
            market_data,
            fund_pool,
            self.trading_history,
            self.platform_trading_history,
            view_platform_trading_history_days,
            current_rp_weights,
            current_rp_weights_text,
            prev_rp_weights_text,
            self.vol_lookback,
        )
        proposed_trades = trading_decision.get("trades", [])
        logger.info(f"  生成 {len(proposed_trades)} 个交易指令")
        self.risk_stats["total_trades_proposed"] += len(proposed_trades)

        # 4. Risk management evaluation
        logger.info("🛡️ [Agent 4/4] 风险管理评估中...")
        risk_result = await self.risk_management_agent.evaluate_trades(
            proposed_trades=proposed_trades,
            current_portfolio=current_portfolio,
            sentiment_analysis=sentiment_analysis,
            current_date=date_to_decision,
            current_rp_weights=current_rp_weights,
        )

        approved_trades = risk_result.get("approved_trades", [])
        modified_trades = risk_result.get("modified_trades", [])
        blocked_trades = risk_result.get("blocked_trades", [])

        # Update statistics
        self.risk_stats["total_trades_approved"] += len(approved_trades)
        self.risk_stats["total_trades_modified"] += len(modified_trades)
        self.risk_stats["total_trades_blocked"] += len(blocked_trades)

        # Log risk management actions
        if modified_trades:
            logger.warning(f"  🛡️ 调整 {len(modified_trades)} 笔交易")
            for mt in modified_trades:
                logger.warning(f"    - {mt['fund_id']}: {mt.get('reason', 'N/A')}")
        if blocked_trades:
            logger.warning(f"  🛡️ 阻止 {len(blocked_trades)} 笔交易")
            for bt in blocked_trades:
                logger.warning(f"    - {bt['fund_id']}: {bt.get('reason', 'N/A')}")

        logger.info(f"  最终执行: {len(approved_trades)} 笔交易 | 风险摘要: {risk_result.get('risk_summary', 'N/A')}")

        # Combine approved and modified trades for final execution
        final_trades = approved_trades + [
            {k: v for k, v in mt.items() if k in ["fund_id", "action", "amount", "percentage", "reason"]}
            for mt in modified_trades
        ]

        # Update trading decision with final trades
        final_decision = trading_decision.copy()
        final_decision["trades"] = final_trades
        final_decision["risk_management"] = {
            "approved": len(approved_trades),
            "modified": len(modified_trades),
            "blocked": len(blocked_trades),
            "risk_summary": risk_result.get("risk_summary", ""),
        }

        # Store current weights as previous for next iteration
        self.prev_rp_weights = current_rp_weights

        # Record decision history
        decision_record = {
            "date": current_portfolio.get("date", "Unknown"),
            "rp_weights": current_rp_weights,
            "processed_news_count": len(processed_news),
            "sentiment_analysis": sentiment_analysis,
            "trading_decision": trading_decision,
            "risk_management_result": risk_result,
            "final_decision": final_decision,
            "portfolio_value": current_portfolio.get("total_value", 0),
        }
        self.decision_history.append(decision_record)
        self.trading_history.append(
            {
                decision_record["date"]: final_trades
            }
        )

        return {
            "final_decision": final_decision,
            "intermediate_results": {
                "rp_weights": current_rp_weights,
                "processed_news": processed_news,
                "sentiment_analysis": sentiment_analysis,
                "risk_management": risk_result,
            },
        }

    def get_decision_history(self) -> List[Dict]:
        """Get decision history."""
        return self.decision_history

    def get_risk_statistics(self) -> Dict:
        """Get risk management statistics."""
        total = self.risk_stats["total_trades_proposed"]
        if total > 0:
            return {
                **self.risk_stats,
                "approval_rate": self.risk_stats["total_trades_approved"] / total,
                "modification_rate": self.risk_stats["total_trades_modified"] / total,
                "block_rate": self.risk_stats["total_trades_blocked"] / total,
            }
        return self.risk_stats

    def clear_history(self):
        """Clear history."""
        self.decision_history = []
        self.prev_rp_weights = {}
        self.risk_stats = {
            "total_trades_proposed": 0,
            "total_trades_approved": 0,
            "total_trades_modified": 0,
            "total_trades_blocked": 0,
        }


# Patch TradingStrategyAgent to support risk parity weights
async def make_trading_decision_with_rp(
    self,
    date_to_decision: str,
    sentiment_analysis: Dict,
    historical_prices: Dict,
    current_portfolio: Dict,
    market_data: Dict,
    fund_pool: List[str],
    trading_history: List[Dict],
    platform_trading_history: List[Dict],
    view_platform_trading_history_days: int = 5,
    current_rp_weights: Dict = None,
    current_rp_weights_text: str = "",
    prev_rp_weights_text: str = "",
    vol_lookback: int = 60,
) -> Dict:
    """Make trading decision with risk parity weights integration."""

    # Build fund info text
    funds_text = "\n".join(
        [
            f"- {fund} ({FUND_INFO.get(fund, {}).get('name', 'Unknown')}): {FUND_INFO.get(fund, {}).get('scope', 'N/A')}。 ({FUND_INFO.get(fund, {}).get('meaning', 'Unknown')})"
            for fund in fund_pool
        ]
    )

    logger.info(f"current_portfolio {current_portfolio}")
    # Format holdings
    holdings = current_portfolio.get("holdings", {})
    capital = current_portfolio.get("capital", 0)
    holdings_text = "\n".join(
        [
            f"- {fund}: 持仓价值 {details['value']:.2f} 元 (当前价: {details['price']:.2f})"
            for fund, details in holdings.items()
        ]
    )

    # Format historical prices
    history_text = ""
    for fund, prices in historical_prices.items():
        if prices:
            history_text += f"{fund} 最近{len(prices)}天:\n"
            for price in prices[-4:]:
                close_price = price.get("close", "N/A")
                pct_change = price.get("pct_change", "N/A")
                if close_price is None:
                    close_price = "N/A"
                if pct_change is None:
                    pct_change = "N/A"
                else:
                    pct_change = f"{pct_change}%"
                history_text += f"  {price['date']}: 开{price.get('open', 'N/A')} 收{close_price} 涨跌{pct_change}\n"
            history_text += "\n"

    # Format trading history
    history_trading = ""
    if platform_trading_history:
        trades_by_date = {}
        for trade in platform_trading_history:
            date = trade.get("date")
            if date not in trades_by_date:
                trades_by_date[date] = []
            trades_by_date[date].append(trade)

        sorted_dates = sorted(trades_by_date.keys(), reverse=True)
        recent_dates = sorted_dates[:view_platform_trading_history_days]

        day_trade_strings = []
        for date in sorted(recent_dates):
            trades_for_day = trades_by_date[date]
            trade_lines = []
            for trade in trades_for_day:
                trade_str = f"{trade.get('date')} {trade.get('fund_id')} {trade.get('action')}"
                if trade.get("action") == "buy":
                    trade_str += f" amount: {trade.get('amount', 0):.2f}"
                elif trade.get("action") == "sell":
                    trade_str += f" percentage: {trade.get('percentage', 0):.2%}, amount_sold: {trade.get('amount_sold', 0):.2f}"
                trade_lines.append(trade_str)
            day_trade_strings.append("\n".join(trade_lines))

        history_trading = "\n\n".join(day_trade_strings)

    # Use default values if RP weights not provided
    if current_rp_weights is None:
        current_rp_weights = {}
    if not current_rp_weights_text:
        current_rp_weights_text = "  (暂无数据)"
    if not prev_rp_weights_text:
        prev_rp_weights_text = "  (暂无数据)"

    prompt = self.prompt_template.format(
        funds_text=funds_text,
        date_to_decision=date_to_decision,
        capital=capital,
        holdings_text=holdings_text if holdings_text else "  (空仓)",
        history_trading=history_trading if history_trading else "  (无历史交易)",
        current_rp_weights=current_rp_weights_text,
        prev_rp_weights=prev_rp_weights_text,
        vol_lookback=vol_lookback,
        sentiment_summary=sentiment_analysis.get("summary", "无舆情分析"),
        sentiment_details=json.dumps(
            sentiment_analysis.get("fund_analysis", {}),
            indent=2,
            ensure_ascii=False,
        ),
        history_text=history_text if history_text else "  (无历史价格)",
    )

    try:
        for i in range(5):
            try:
                response = await self.llm.ainvoke(prompt)
                decision = self.parser.parse(response.content)
                logger.info(f"LLM Agent decision: {decision}")
                return decision
            except Exception as e:
                logger.exception(f"Parser failed on attempt {i + 1}/5: {e}")
                if i == 4:
                    raise
    except Exception as e:
        logger.error(f"决策生成失败，采取保守策略")
        return {
            "reasoning": "决策生成失败，采取保守策略",
            "chain_of_thought": f"系统错误: {str(e)}",
            "trades": [
                {"fund_id": fund, "action": "hold", "reason": "系统错误，保守持有"}
                for fund in holdings.keys()
            ],
            "risk_assessment": "高风险-系统错误",
        }


# Monkey patch the method
TradingStrategyAgent.make_trading_decision_with_rp = make_trading_decision_with_rp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run risk parity + risk management agent demo backtest"
    )
    parser.add_argument("--track", choices=["macro", "sector"], default="sector")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--start-date", default="2025-01-02")
    parser.add_argument("--end-date", default="2025-01-31")
    parser.add_argument("--initial-capital", type=float, default=100000)
    parser.add_argument("--lookback-days", type=int, default=5)
    parser.add_argument("--top-rank", type=int, default=20)
    parser.add_argument("--pre-k-days", type=int, default=1)
    parser.add_argument("--history-days", type=int, default=5)
    parser.add_argument("--vol-lookback", type=int, default=60,
                        help="Days to look back for volatility calculation in risk parity")
    parser.add_argument("--min-holding-days", type=int, default=7,
                        help="Minimum holding period in days")
    parser.add_argument("--max-concentration", type=float, default=0.4,
                        help="Maximum position concentration (0.0-1.0)")
    parser.add_argument("--username", default=AGENT_PLATFORM["AGENT_USERNAME"])
    parser.add_argument("--password", default=AGENT_PLATFORM["AGENT_PASSWORD"])
    parser.add_argument("--base-url", default=AGENT_PLATFORM["BASE_URL"])
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def build_config(args):
    fund_pool = MAJOR_FUND_POOL if args.track == "macro" else INDUSTRY_FUND_POOL
    default_results_dir = (
        "backtest_results_macro_rp_mgmt_agent"
        if args.track == "macro"
        else "backtest_results_sector_rp_mgmt_agent"
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
    return out_dir / f"rp_mgmt_agent_{args.track}_{args.model}_{session_id}.json"


async def run_demo_backtest(args):
    # Initialize DataLoader (required when using get_data_loader() directly)
    init_data_loader(
        price_data_dir=str(DATA_DIRS["PRICE_DATA"]),
        news_data_dir=str(DATA_DIRS["NEWS_DATA"]),
    )
    logger.info("DataLoader initialized")

    client = PlatformClient(base_url=args.base_url)
    client.register(args.username, args.password)
    client.login(args.username, args.password)

    config = build_config(args)
    fund_pool = config["fund_pool"]

    # Preload historical prices before backtest starts
    # This ensures we have 60 days of data even when backtest starts in 2025
    preloaded_prices = preload_historical_prices(
        fund_pool=fund_pool,
        backtest_start_date=args.start_date,
        lookback_days=args.vol_lookback,
    )

    # Check if we got enough data
    min_preloaded = min(len(v) for v in preloaded_prices.values()) if preloaded_prices else 0
    if min_preloaded < args.vol_lookback:
        logger.warning(
            f"Preloaded only {min_preloaded} days, expected {args.vol_lookback} days. "
            f"Risk parity calculations may be unreliable until sufficient data accumulates."
        )

    agent = RiskParityWithManagementAgent(
        agent_id=f"{args.track}_rp_mgmt_agent",
        trading_prompt_template=RISK_PARITY_TRADING_PROMPT,
        decision_model_name=args.model,
        vol_lookback=args.vol_lookback,
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
    logger.info(f"Risk Parity: vol_lookback={args.vol_lookback} days")

    trading_days = 0
    while True:
        trading_days += 1
        portfolio = client.get_backtest_status(session_id)
        server_historical_prices = client.get_historical_prices(
            session_id, lookback_days=config["lookback_days"]
        )

        # Merge preloaded data with server data for full history
        merged_historical_prices = {}
        for fund_id in fund_pool:
            preloaded = preloaded_prices.get(fund_id, [])
            server_data = server_historical_prices.get("historical_prices", {}).get(fund_id, [])

            # Remove duplicate if preloaded last date == server first date
            if preloaded and server_data:
                last_pre_date = preloaded[-1].get("date")
                first_server_date = server_data[0].get("date") if server_data else None
                if last_pre_date == first_server_date:
                    preloaded = preloaded[:-1]

            # Take the last vol_lookback days from combined data
            combined = preloaded + server_data
            merged_historical_prices[fund_id] = combined[-args.vol_lookback:]

        # Log merged data count on first day
        if trading_days == 1:
            min_merged = min(len(v) for v in merged_historical_prices.values()) if merged_historical_prices else 0
            logger.info(f"Merged historical data: minimum {min_merged} days across {len(merged_historical_prices)} funds")

        try:
            decision_result = await agent.make_decision(
                date_to_decision=data["date"],
                news_data=data["news"],
                historical_prices=merged_historical_prices,
                current_portfolio=portfolio,
                market_data=data["market_data"],
                fund_pool=config["fund_pool"],
                view_platform_trading_history_days=config[
                    "view_platform_trading_history_days"
                ],
            )

            final_decision = decision_result["final_decision"]
            trades = [
                _clean_trade_for_server(trade)
                for trade in final_decision.get("trades", [])
                if trade.get("action") != "hold"
            ]

            if trades:
                agent_decision = AgentDecision(
                    decision=final_decision,
                    reasoning=final_decision.get("reasoning", ""),
                    chain_of_thought=str(final_decision.get("chain_of_thought", "")),
                )
                client.submit_trade_with_decision(session_id, trades, agent_decision)

                # Log RP weights for reference
                rp_weights = decision_result["intermediate_results"].get("rp_weights", {})
                logger.info(f"  RP基准权重: {json.dumps(rp_weights, ensure_ascii=False)}")

                # Log risk management info
                risk_mgmt = final_decision.get("risk_management", {})
                if risk_mgmt:
                    logger.info(f"  风控: 批准{risk_mgmt.get('approved', 0)} "
                              f"调整{risk_mgmt.get('modified', 0)} "
                              f"阻止{risk_mgmt.get('blocked', 0)}")

        except Exception as exc:
            logger.error(f"Decision failed on {data.get('date')}: {exc}")
            logger.error(traceback.format_exc())

        data = client.get_next_day_data(session_id)
        if data.get("message") == "Backtest finished":
            break

        await asyncio.sleep(0.1)

    final_results = client.get_backtest_results(session_id)

    # Add decision history and risk statistics to results
    final_results["decision_history"] = agent.get_decision_history()
    final_results["risk_statistics"] = agent.get_risk_statistics()

    output_path = build_output_path(args, session_id)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(final_results, handle, indent=2, ensure_ascii=False)

    risk_stats = agent.get_risk_statistics()
    logger.info(
        f"Finished session {session_id} after {trading_days} trading days. "
        f"Return={final_results.get('performance', {}).get('total_return', 0) * 100:.2f}%"
    )
    logger.info(f"Risk Management Statistics:")
    logger.info(f"  - Proposed: {risk_stats.get('total_trades_proposed', 0)}")
    logger.info(f"  - Approved: {risk_stats.get('total_trades_approved', 0)} "
               f"({risk_stats.get('approval_rate', 0)*100:.1f}%)")
    logger.info(f"  - Modified: {risk_stats.get('total_trades_modified', 0)} "
               f"({risk_stats.get('modification_rate', 0)*100:.1f}%)")
    logger.info(f"  - Blocked: {risk_stats.get('total_trades_blocked', 0)} "
               f"({risk_stats.get('block_rate', 0)*100:.1f}%)")
    logger.info(f"Saved final results to {output_path}")


def main():
    args = parse_args()
    log_dir = Path(project_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"demo_rp_mgmt_agent_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logger.add(str(log_path), level="INFO")

    logger.info(
        f"Running risk parity + risk management agent demo backtest with track={args.track}, model={args.model}, "
        f"period={args.start_date}~{args.end_date}, vol_lookback={args.vol_lookback}, "
        f"min_holding_days={args.min_holding_days}"
    )
    try:
        asyncio.run(run_demo_backtest(args))
    except Exception as exc:
        logger.error(f"Demo backtest failed: {exc}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
