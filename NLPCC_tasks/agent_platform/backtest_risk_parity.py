#!/usr/bin/env python3
"""
风险平价（Risk Parity）基准回测脚本
不使用 LLM Agent，按逆波动率权重每 N 天再平衡一次。

算法：
    w_i = (1/σ_i) / Σ(1/σ_j)
    σ_i = 过去 vol_window 个交易日日收益率的标准差（log returns）

用法：
    python agent_platform/backtest_risk_parity.py \\
        --track sector \\
        --start-date 2025-01-02 --end-date 2025-12-31 \\
        --vol-window 20 --rebalance-freq 5

后续规划：将此脚本计算出的目标权重作为结构化输入喂给 LLM Agent，
         使智能体在新闻驱动决策之外有量化锚点。
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from loguru import logger

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent_platform.client.platform_client import PlatformClient
from config import AGENT_PLATFORM
from server_platform.app.models.backtest import AgentDecision


# ── 基金池（与 demo_backtest.py 保持一致）───────────────────────────────────
MAJOR_FUND_POOL = [
    "000300.SH", "000905.SH", "399006.SZ", "000688.SH", "000932.SH",
    "000941.SH", "399971.SZ", "000819.SH", "000928.SH", "000012.SH", "518880.SH",
]

INDUSTRY_FUND_POOL = [
    "512880.SH", "512800.SH", "512070.SH", "159995.SZ", "159819.SZ",
    "515880.SH", "159852.SZ", "512010.SH", "512170.SH", "159992.SZ",
    "515170.SH", "512690.SH", "512400.SH", "515220.SH", "159870.SZ", "512200.SH",
]


# ── 参数解析 ─────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Risk Parity baseline — no LLM, inverse-volatility weights."
    )
    parser.add_argument("--track", choices=["macro", "sector"], default="sector")
    parser.add_argument("--start-date", default="2025-01-02")
    parser.add_argument("--end-date",   default="2025-12-31")
    parser.add_argument("--initial-capital", type=float, default=100_000)
    parser.add_argument(
        "--lookback-days", type=int, default=30,
        help="向服务器请求的历史价格天数（建议 ≥ vol-window + 5）",
    )
    parser.add_argument(
        "--vol-window", type=int, default=20,
        help="计算波动率时使用的收益率窗口（交易日数）",
    )
    parser.add_argument(
        "--rebalance-freq", type=int, default=5,
        help="每隔多少交易日再平衡一次（1=每日，5=每周）",
    )
    parser.add_argument(
        "--min-trade", type=float, default=200,
        help="低于此金额的调仓操作将被忽略，避免频繁小额交易",
    )
    parser.add_argument(
        "--vol-floor", type=float, default=0.001,
        help="波动率下限，防止低波动标的被过度集中配置",
    )
    parser.add_argument("--username", default=AGENT_PLATFORM["AGENT_USERNAME"])
    parser.add_argument("--password", default=AGENT_PLATFORM["AGENT_PASSWORD"])
    parser.add_argument("--base-url",  default=AGENT_PLATFORM["BASE_URL"])
    parser.add_argument("--output",    default=None, help="结果 JSON 输出路径（可选）")
    return parser.parse_args()


# ── 核心算法 ─────────────────────────────────────────────────────────────────
def calc_risk_parity_weights(
    historical_prices: dict,
    fund_pool: list,
    vol_window: int,
    vol_floor: float,
) -> dict:
    """
    逆波动率权重（简化风险平价）。

    参数
    ----
    historical_prices : {fund_id: [{date, open, close, high, low, volume}, ...]}
        服务器返回的历史价格字典（当日 close 被隐藏，只有 open）
    fund_pool         : 参与配置的基金代码列表
    vol_window        : 使用最近多少天的收益率计算标准差
    vol_floor         : 波动率下限

    返回
    ----
    {fund_id: weight}，权重之和为 1.0
    """
    inv_vols = {}

    for fund_id in fund_pool:
        raw = historical_prices.get(fund_id, [])
        # 按日期升序排列，只保留有有效 close 的历史日（排除今日，今日 close=None）
        closes = [
            float(p["close"])
            for p in sorted(raw, key=lambda x: x.get("date", ""))
            if p.get("close") is not None
        ]

        if len(closes) < 2:
            # 历史不足：先给等权占位，日后会被替换
            inv_vols[fund_id] = 1.0
            continue

        # 取最近 vol_window 根收盘价计算对数收益率
        recent = np.array(closes[-vol_window:], dtype=float)
        log_ret = np.diff(np.log(recent))

        vol = float(np.std(log_ret, ddof=1)) if len(log_ret) >= 2 else vol_floor
        inv_vols[fund_id] = 1.0 / max(vol, vol_floor)

    total = sum(inv_vols.values())
    if total <= 0:
        n = len(fund_pool)
        return {f: 1.0 / n for f in fund_pool}

    return {f: inv_vols[f] / total for f in fund_pool}


def build_rebalance_trades(
    target_weights: dict,
    portfolio: dict,
    fund_pool: list,
    min_trade: float,
) -> list:
    """
    生成从当前持仓到目标权重的调仓指令。
    卖出指令排在买入指令之前，确保卖出腾出的资金可用于后续买入。

    portfolio 结构（来自 client.get_backtest_status）:
        {
          "capital":     float,                      # 可用现金
          "portfolio":   {fund_id: float},            # 各基金持仓市值
          "total_value": float,                       # 总资产
        }

    返回
    ----
    [{"fund_id": ..., "action": "sell", "percentage": ...},
     {"fund_id": ..., "action": "buy",  "amount": ...}, ...]
    """
    total_value = float(portfolio.get("total_value", 0))
    if total_value <= 0:
        return []

    holdings: dict = portfolio.get("portfolio", {})
    capital = float(portfolio.get("capital", 0))

    sells = []
    buy_needs = []   # 暂存需要买入的 (fund_id, diff)
    freed_cash = 0.0  # 预估卖出后释放的现金

    for fund_id in fund_pool:
        target_val  = target_weights.get(fund_id, 0.0) * total_value
        current_val = float(holdings.get(fund_id, 0.0))
        diff = target_val - current_val

        if diff < -min_trade:
            # 减仓：按比例卖出
            sell_pct = min(1.0, (-diff) / max(current_val, 1e-9))
            sells.append({
                "fund_id": fund_id,
                "action": "sell",
                "percentage": round(sell_pct, 6),
            })
            freed_cash += current_val * sell_pct

        elif diff > min_trade:
            buy_needs.append((fund_id, diff))

    # 可用现金 = 当前现金 + 预估卖出所得
    available = capital + freed_cash
    total_need = sum(d for _, d in buy_needs)

    # 按比例缩放买入量（如果现金不足以全部买入）
    scale = min(1.0, available / total_need) if total_need > 0 else 0.0

    buys = []
    for fund_id, need in buy_needs:
        amount = round(need * scale, 2)
        if amount >= min_trade:
            buys.append({"fund_id": fund_id, "action": "buy", "amount": amount})

    return sells + buys   # 卖出在前！


# ── 配置 & 路径 ───────────────────────────────────────────────────────────────
def build_config(args, fund_pool: list) -> dict:
    results_dir = (
        "backtest_results_macro_risk_parity"
        if args.track == "macro"
        else "backtest_results_sector_risk_parity"
    )
    return {
        "start_date":      args.start_date,
        "end_date":        args.end_date,
        "initial_capital": args.initial_capital,
        "fund_pool":       fund_pool,
        "agents":          [{"name": args.username, "prompt": "risk_parity_baseline"}],
        "news_sources":    [],          # 不需要新闻
        "lookback_days":   args.lookback_days,
        "results_dir":     results_dir,
    }


def build_output_path(args, session_id: str) -> Path:
    if args.output:
        return Path(args.output)
    out_dir = Path(project_root) / "agent_platform" / "demo_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"risk_parity_{args.track}_{session_id}.json"


# ── 主回测循环 ────────────────────────────────────────────────────────────────
def run_backtest(args):
    client = PlatformClient(base_url=args.base_url)
    client.register(args.username, args.password)

    fund_pool = MAJOR_FUND_POOL if args.track == "macro" else INDUSTRY_FUND_POOL
    config    = build_config(args, fund_pool)

    start_resp = client.start_backtest(config)
    session_id = start_resp["session_id"]
    data       = start_resp.get("data")
    if not data:
        raise RuntimeError("服务器未返回初始数据，请检查服务器是否正常运行。")

    logger.info(
        f"Session {session_id} 已启动 | track={args.track} | "
        f"{args.start_date} ~ {args.end_date} | "
        f"再平衡周期={args.rebalance_freq} 天 | 波动率窗口={args.vol_window} 天"
    )

    trading_days = 0
    current_weights: dict = {}

    while True:
        trading_days += 1
        current_date = data.get("date", "?")

        # ── 获取当前持仓与历史行情 ──────────────────────────────────────
        portfolio   = client.get_backtest_status(session_id)
        hist_resp   = client.get_historical_prices(session_id, lookback_days=args.lookback_days)
        hist_prices = hist_resp.get("historical_prices", {})

        # ── 判断是否再平衡 ────────────────────────────────────────────
        is_rebalance_day = (trading_days == 1) or (trading_days % args.rebalance_freq == 1)

        if is_rebalance_day:
            try:
                current_weights = calc_risk_parity_weights(
                    hist_prices, fund_pool, args.vol_window, args.vol_floor
                )
                trades = build_rebalance_trades(
                    current_weights, portfolio, fund_pool, args.min_trade
                )

                if trades:
                    weights_log = {f: f"{w:.4f}" for f, w in current_weights.items()}
                    agent_decision = AgentDecision(
                        decision={
                            "strategy": "risk_parity",
                            "target_weights": current_weights,
                            "trades": trades,
                        },
                        reasoning=(
                            f"第 {trading_days} 交易日再平衡。"
                            f"逆波动率权重（vol_window={args.vol_window} 天）"
                        ),
                        chain_of_thought=json.dumps(weights_log, ensure_ascii=False),
                    )
                    client.submit_trade_with_decision(session_id, trades, agent_decision)

                    n_sell = sum(1 for t in trades if t["action"] == "sell")
                    n_buy  = sum(1 for t in trades if t["action"] == "buy")
                    logger.info(
                        f"Day {trading_days:3d} ({current_date}) [再平衡] "
                        f"卖出 {n_sell} 笔 买入 {n_buy} 笔 | "
                        f"总资产 ≈ ¥{portfolio.get('total_value', 0):,.0f}"
                    )
                else:
                    logger.info(
                        f"Day {trading_days:3d} ({current_date}) [再平衡] "
                        f"偏差在阈值内，无需调仓"
                    )

            except Exception as exc:
                logger.error(f"Day {trading_days} ({current_date}) 再平衡异常: {exc}")
                logger.error(traceback.format_exc())

        else:
            logger.debug(
                f"Day {trading_days:3d} ({current_date}) 持仓不变（非再平衡日）"
            )

        # ── 推进到下一个交易日 ────────────────────────────────────────
        data = client.get_next_day_data(session_id)
        if data.get("message") == "Backtest finished":
            break

    # ── 收尾 ──────────────────────────────────────────────────────────────
    final_results = client.get_backtest_results(session_id)
    output_path   = build_output_path(args, session_id)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    perf = final_results.get("performance", {})
    logger.info(
        f"回测完成 | {trading_days} 个交易日 | "
        f"收益率={perf.get('total_return', 0) * 100:.2f}% | "
        f"年化={perf.get('annualized_return', 0) * 100:.2f}%"
    )
    logger.info(f"结果已保存至 {output_path}")
    return final_results


# ── 入口 ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    log_dir  = Path(project_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"risk_parity_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logger.add(str(log_path), level="INFO")

    logger.info(
        f"启动风险平价回测 | track={args.track} | "
        f"{args.start_date} ~ {args.end_date} | "
        f"vol_window={args.vol_window} | rebalance_freq={args.rebalance_freq}"
    )
    try:
        run_backtest(args)
    except Exception as exc:
        logger.error(f"回测失败: {exc}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
