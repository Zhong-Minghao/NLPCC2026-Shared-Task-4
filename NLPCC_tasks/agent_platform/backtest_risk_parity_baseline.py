#!/usr/bin/env python3
"""
Risk Parity Backtest - Non-agent baseline for comparison.

This script implements a risk parity strategy without using LLM agents.
Risk parity allocates capital inversely proportional to risk (volatility):
w_i = (1/σ_i) / Σ(1/σ_j) for simple volatility parity

Use this to establish a performance baseline for comparison with agent-based strategies.
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent_platform.agents.fund_info import FUND_INFO
from server_platform.app.core.backtest import BacktestSession, create_backtest_session
from server_platform.app.core.data_loader import DataLoader, get_data_loader, init_data_loader
from config import DATA_DIRS


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

    Args:
        fund_pool: List of fund IDs
        backtest_start_date: Backtest start date in 'YYYY-MM-DD' format
        lookback_days: Number of days to look back

    Returns:
        Dict mapping fund_id to list of historical price data
    """
    data_loader = get_data_loader()

    # Calculate the date range we need
    start_dt = datetime.strptime(backtest_start_date, "%Y-%m-%d")
    # Go back lookback_days trading days (approx lookback_days * 1.4 calendar days)
    # Using 1.4 to account for weekends and holidays
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


# Same fund pools as backtest_tech_v1.py
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
        method: str = "volatility_parity",
    ):
        """
        Args:
            lookback_days: Number of days to look back for volatility calculation
            min_data_points: Minimum data points required for volatility calculation
            method: "volatility_parity" for simple 1/vol weighting,
                    "equal_weight" for equal weight baseline
        """
        self.lookback_days = lookback_days
        self.min_data_points = min_data_points
        self.method = method

    def calculate_weights(
        self,
        historical_prices: Dict[str, List[Dict]],
        fund_pool: List[str],
    ) -> Dict[str, float]:
        """
        Calculate risk parity weights based on historical volatility.

        Args:
            historical_prices: Dict mapping fund_id to list of price data
            fund_pool: List of fund IDs to consider

        Returns:
            Dict mapping fund_id to weight (sums to 1.0)
        """
        if self.method == "equal_weight":
            return {f: 1.0 / len(fund_pool) for f in fund_pool}

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


class CovarianceRiskParityCalculator(RiskParityCalculator):
    """
    Advanced risk parity using full covariance matrix.

    Solves for weights such that each asset contributes equal risk to the portfolio.
    Risk contribution of asset i: RC_i = w_i * (Σw)_i
    """

    def __init__(
        self,
        lookback_days: int = 60,
        min_data_points: int = 20,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        super().__init__(lookback_days, min_data_points, method="covariance_parity")
        self.max_iter = max_iter
        self.tol = tol

    def calculate_weights(
        self,
        historical_prices: Dict[str, List[Dict]],
        fund_pool: List[str],
    ) -> Dict[str, float]:
        """
        Calculate risk parity weights using covariance matrix.
        Uses iterative algorithm to solve for equal risk contributions.
        """
        # Build returns matrix
        returns_dict = {}
        valid_funds = []

        for fund_id in fund_pool:
            prices = historical_prices.get(fund_id, [])
            if len(prices) < self.min_data_points:
                continue

            closes = []
            for p in prices[-self.lookback_days:]:
                close = p.get("close")
                if close is not None and not math.isnan(close):
                    closes.append(close)

            if len(closes) < 2:
                continue

            returns = []
            for i in range(1, len(closes)):
                ret = (closes[i] - closes[i - 1]) / closes[i - 1]
                if not math.isnan(ret) and not math.isinf(ret):
                    returns.append(ret)

            if len(returns) >= self.min_data_points:
                returns_dict[fund_id] = returns[-self.min_data_points:]
                valid_funds.append(fund_id)

        if len(valid_funds) < 2:
            logger.warning("Insufficient funds for covariance parity, using volatility parity")
            return super().calculate_weights(historical_prices, fund_pool)

        # Build returns DataFrame
        min_len = min(len(r) for r in returns_dict.values())
        returns_array = np.array([r[-min_len:] for r in returns_dict.values()])

        # Calculate covariance matrix
        cov_matrix = np.cov(returns_array)

        # Ensure positive definiteness
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvalues < 1e-8):
            logger.warning("Covariance matrix not positive definite, adding regularization")
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6

        # Solve for risk parity weights using iterative algorithm
        n = len(valid_funds)
        weights = np.ones(n) / n

        for iteration in range(self.max_iter):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib

            # Target risk contribution
            target_contrib = portfolio_vol / n

            # Update weights
            new_weights = np.zeros(n)
            for i in range(n):
                if marginal_contrib[i] > 1e-10:
                    new_weights[i] = weights[i] * (target_contrib / risk_contrib[i])

            # Normalize and clip
            new_weights = np.clip(new_weights, 1e-6, None)
            new_weights = new_weights / new_weights.sum()

            # Check convergence
            if np.max(np.abs(new_weights - weights)) < self.tol:
                break

            weights = new_weights

        # Build output dictionary
        weights_dict = {valid_funds[i]: float(weights[i]) for i in range(n)}

        # Handle unallocated funds
        allocated = sum(weights_dict.values())
        unallocated = [f for f in fund_pool if f not in weights_dict]
        if unallocated:
            residual_weight = (1.0 - allocated) / len(unallocated)
            for f in unallocated:
                weights_dict[f] = residual_weight

        return weights_dict


def rebalance_to_target_weights(
    session: BacktestSession,
    target_weights: Dict[str, float],
    tolerance: float = 0.02,
) -> List[Dict]:
    """
    Generate trades to rebalance portfolio to target weights.

    Args:
        session: BacktestSession instance
        target_weights: Dict mapping fund_id to target weight
        tolerance: Minimum deviation to trigger trade (as fraction of portfolio value)

    Returns:
        List of trade dictionaries
    """
    trades = []
    portfolio_status = session.get_portfolio_status()
    current_value = portfolio_status["total_value"]
    current_holdings = portfolio_status["holdings"]
    available_cash = session.capital

    for fund_id, target_weight in target_weights.items():
        current_value_in_fund = current_holdings.get(fund_id, {}).get("value", 0)
        target_value_in_fund = current_value * target_weight

        diff = target_value_in_fund - current_value_in_fund

        if abs(diff) < current_value * tolerance:
            trades.append({
                "fund_id": fund_id,
                "action": "hold",
                "reason": f"Within {tolerance:.1%} tolerance (current: {current_value_in_fund/current_value:.2%}, target: {target_weight:.2%})"
            })
        elif diff > 0:
            # Need to buy
            required_cash = diff * 1.0001  # Account for commission
            if available_cash >= required_cash:
                trades.append({
                    "fund_id": fund_id,
                    "action": "buy",
                    "amount": diff,
                    "reason": f"Rebalance to {target_weight:.2%} weight (current: {current_value_in_fund/current_value:.2%})"
                })
                available_cash -= diff
            else:
                # Scale down to fit available cash
                scaled_amount = available_cash / 1.0001 * 0.99
                if scaled_amount > current_value * 0.005:  # Minimum trade size
                    trades.append({
                        "fund_id": fund_id,
                        "action": "buy",
                        "amount": scaled_amount,
                        "reason": f"Partial rebalance to {target_weight:.2%} (cash constrained)"
                    })
                    available_cash = 0
        else:
            # Need to sell
            sell_percentage = abs(diff) / max(current_value_in_fund, 1e-6)
            sell_percentage = min(sell_percentage, 1.0)
            trades.append({
                "fund_id": fund_id,
                "action": "sell",
                "percentage": sell_percentage,
                "reason": f"Rebalance to {target_weight:.2%} weight (current: {current_value_in_fund/current_value:.2%})"
            })

    return trades


def run_risk_parity_backtest(args):
    """Run risk parity backtest without LLM agent."""

    fund_pool = MAJOR_FUND_POOL if args.track == "macro" else INDUSTRY_FUND_POOL

    config = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_capital": args.initial_capital,
        "fund_pool": fund_pool,
        "agents": [{"name": f"risk_parity_{args.method}", "prompt": f"Risk parity ({args.method}) baseline"}],
        "news_sources": [],
        "lookback_days": max(args.lookback_days, args.vol_lookback),
        "pre_k_days": args.pre_k_days,
        "results_dir": f"backtest_results_{args.track}_risk_parity_{args.method}",
    }

    logger.info(f"Starting Risk Parity backtest for {args.track} track")
    logger.info(f"Period: {args.start_date} to {args.end_date}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Rebalance frequency: every {args.rebalance_freq} days")
    logger.info(f"Volatility lookback: {args.vol_lookback} days")

    # Initialize DataLoader (required when using BacktestSession directly)
    init_data_loader(
        price_data_dir=str(DATA_DIRS["PRICE_DATA"]),
        news_data_dir=str(DATA_DIRS["NEWS_DATA"]),
    )
    logger.info("DataLoader initialized")

    # Create backtest session
    session_id = create_backtest_session(config)
    session = BacktestSession(session_id, config)

    # Initialize risk parity calculator
    if args.method == "covariance_parity":
        rp_calculator = CovarianceRiskParityCalculator(lookback_days=args.vol_lookback)
    else:
        rp_calculator = RiskParityCalculator(
            lookback_days=args.vol_lookback,
            method=args.method,
        )

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

    # Start backtest
    day_data = session.start()

    trading_days = 0
    rebalance_count = 0
    last_weights = {}

    # Track cumulative historical prices during the backtest
    cumulative_prices = {fund: list(preloaded_prices.get(fund, [])) for fund in fund_pool}

    while not session.is_finished:
        trading_days += 1
        current_date = day_data["date"]

        # Get historical prices for risk calculation
        # Note: data_loader.get_historical_prices only returns data within the backtest period
        backtest_prices = session.data_loader.get_historical_prices(
            fund_ids=fund_pool,
            current_date=session.current_date,
            lookback_days=args.vol_lookback + args.pre_k_days,
        )

        # Merge preloaded data with backtest data for full history
        historical_prices = {}
        for fund_id in fund_pool:
            # Combine: preloaded (before backtest) + backtest data (during backtest)
            preloaded = preloaded_prices.get(fund_id, [])
            backtest = backtest_prices.get(fund_id, [])

            # Remove the last entry from preloaded if it's the same as first in backtest (avoid dup)
            if preloaded and backtest:
                last_pre_date = preloaded[-1].get("date")
                first_back_date = backtest[0].get("date") if backtest else None
                if last_pre_date == first_back_date:
                    preloaded = preloaded[:-1]

            # Take the last vol_lookback days from combined data
            combined = preloaded + backtest
            historical_prices[fund_id] = combined[-args.vol_lookback:]

        # Check if we should rebalance
        should_rebalance = (trading_days % args.rebalance_freq == 0)

        if should_rebalance:
            # Calculate risk parity weights
            target_weights = rp_calculator.calculate_weights(historical_prices, fund_pool)
            last_weights = target_weights

            # Generate rebalancing trades
            trades = rebalance_to_target_weights(
                session,
                target_weights,
                tolerance=args.tolerance,
            )

            # Filter out hold trades
            active_trades = [t for t in trades if t.get("action") != "hold"]

            if active_trades:
                rebalance_count += 1
                logger.info(
                    f"{current_date}: Rebalance #{rebalance_count} with {len(active_trades)} trades"
                )
                logger.debug(f"  Target weights: {target_weights}")

                # Submit trades
                session.submit_trades(
                    active_trades,
                    agent_decision={
                        "strategy": f"risk_parity_{args.method}",
                        "weights": target_weights,
                        "reasoning": f"Risk parity rebalance based on {args.vol_lookback}-day "
                                   f"{'covariance matrix' if args.method == 'covariance_parity' else 'volatility'}",
                        "rebalance_number": rebalance_count,
                    },
                )
            else:
                logger.debug(f"{current_date}: Portfolio within tolerance, no trades needed")

        # Move to next day
        day_data = session.next_day()
        if day_data is None:
            break

    # Get final results
    results = session.get_results()

    # Save results
    output_dir = Path(project_root) / "agent_platform" / "demo_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"risk_parity_{args.method}_{args.track}_{session_id}.json"

    # 直接展开 results 内容到顶层，与 demo_backtest 输出格式保持一致
    output_data = {
        # results 中的核心字段直接放在顶层
        **results,
        # 额外的策略参数放在顶层
        "session_id": session_id,
        "backtest_config": config,
        "strategy_params": {
            "method": args.method,
            "vol_lookback": args.vol_lookback,
            "rebalance_freq": args.rebalance_freq,
            "tolerance": args.tolerance,
            "rebalance_count": rebalance_count,
            "trading_days": trading_days,
        },
        "final_weights": last_weights,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    perf = results["performance"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Risk Parity Backtest Results ({args.track} track, {args.method})")
    logger.info(f"{'='*60}")
    logger.info(f"Total Return: {perf['total_return']*100:.2f}%")
    logger.info(f"Final Portfolio Value: {perf['final_portfolio_value']:.2f}")
    logger.info(f"Annualized Return: {perf['annualized_return']*100:.2f}%")
    logger.info(f"Trading Days: {trading_days}")
    logger.info(f"Rebalances: {rebalance_count}")
    logger.info(f"Results saved to: {output_path}")

    return results, output_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run risk parity baseline backtest without LLM agent"
    )
    parser.add_argument("--track", choices=["macro", "sector"], default="sector")
    parser.add_argument("--start-date", default="2025-01-02")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--initial-capital", type=float, default=100000)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--pre-k-days", type=int, default=1)
    parser.add_argument(
        "--vol-lookback",
        type=int,
        default=60,
        help="Days to look back for volatility calculation",
    )
    parser.add_argument(
        "--rebalance-freq",
        type=int,
        default=5,
        help="Rebalance every N trading days",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Weight deviation tolerance (default: 2%%)",
    )
    parser.add_argument(
        "--method",
        choices=["volatility_parity", "covariance_parity", "equal_weight"],
        default="volatility_parity",
        help="Risk parity calculation method",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    log_dir = Path(project_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"backtest_risk_parity_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logger.add(str(log_path), level="INFO")

    logger.info(
        f"Running risk parity backtest with track={args.track}, method={args.method}, "
        f"period={args.start_date}~{args.end_date}"
    )

    try:
        run_risk_parity_backtest(args)
    except Exception as exc:
        logger.error(f"Risk parity backtest failed: {exc}")
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
