#!/usr/bin/env python3
"""
Demo backtest runner with technical factors.

This file keeps the original server/client workflow from demo_backtest.py, and
adds a technical factor snapshot to the agent's daily decision input.
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_openai import ChatOpenAI
from loguru import logger

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent_platform.agents.advanced_agents import NewsProcessingAgent, SentimentAnalysisAgent
from agent_platform.agents.fund_info import FUND_INFO
from agent_platform.client.platform_client import PlatformClient
from agent_platform.utils import CustomJsonOutputParser
from config import AGENT_PLATFORM, DATA_DIRS
from dataset.price_data.price_normalizer import load_standardized_price_csv
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


TECH_FACTOR_INFO = {
    "momentum_60d_pct": {
        "name": "60日综合动量",
        "meaning": "最近60个已完成交易日的累计收益率（约3个月），不考虑近期短期波动。",
        "interpretation": "正值代表行业处于长期上升趋势，负值代表长期弱势。A股行业轮动中最核心的驱动力之一，长周期动量能有效捕捉风格切换，但与短期动量结合使用效果更佳。",
    },
    "momentum_term_spread_pct": {
        "name": "动量期限差",
        "meaning": "长期动量（60日）减去短期动量（10日）的差值。",
        "interpretation": "正值代表长期趋势强于短期热度（趋势健康），负值代表短期涨幅过大已透支长期动能（拥挤信号）。该因子能有效区分“真趋势”与“短期炒作”，分组收益差异显著。",
    },
    "rsi_20d": {
        "name": "20日相对强弱指数",
        "meaning": "基于最近20个已完成交易日计算RSI，公式：平均上涨幅度/(平均上涨幅度+平均下跌幅度)*100。",
        "interpretation": "数值>70为超买区，短期有回调压力；<30为超卖区，可能反弹。但在强趋势行业中RSI可长期维持80以上，此时不盲目看空，需结合动量趋势判断。",
    },
    "bias_20d_pct": {
        "name": "20日乖离率",
        "meaning": "上一交易日收盘价相对20日简单移动平均线的偏离百分比，公式：(close/ma20 - 1)*100。",
        "interpretation": "正值越大说明价格远高于中期均线（获利盘丰厚），负值越大说明深跌于均线之下。极端乖离（如>15%或<-15%）常伴随均值回归，适度乖离（3%~8%）则趋势延续性较好。",
    },
    "amount_volatility_60d": {
        "name": "60日成交金额波动率",
        "meaning": "最近60个已完成交易日成交金额（万元）的标准差，计算后取相反数使用。",
        "interpretation": "数值越高（即原始波动率越低）代表行业量能稳定、情绪平稳，适合持有；数值越低代表成交忽大忽小，市场分歧剧烈。低成交波动的行业组合超额收益显著，是优选信号。",
    },
    "volume_price_cov_rank": {
        "name": "量价协方差排名",
        "meaning": "过去20日，每日收盘价排名与每日成交量排名的协方差（或斯皮尔曼相关系数）。",
        "interpretation": "负值代表“价涨量缩”或“价跌量放”（背离），上涨动能衰竭；正值代表“价涨量增”或“价跌量缩”（确认），趋势可信。行业轮动中优先选择量价同向的行业，避开背离的行业。",
    },
    "turnover_zscore": {
        "name": "截面换手率拥挤度",
        "meaning": "上一交易日行业换手率，相对于全市场所有行业的均值与标准差，计算Z-score： (x - mean)/std。",
        "interpretation": "Z-score > 1.5 代表交易极度拥挤，后续踩踏风险高，应主动规避；Z-score < -1 代表无人问津，但可能左侧布局。拥挤度因子不用于优选，只用于剔除过热行业，能显著改善动量策略回撤。",
    },
    "bollinger_width_pct": {
        "name": "布林带宽度",
        "meaning": "布林带（参数20,2）的上轨减下轨，再除以中轨得到的百分比，即 (upper - lower)/middle * 100。",
        "interpretation": "宽度持续收缩至历史低位（如20日最小值），预示即将变盘；宽度从低位开始扩张，往往对应趋势启动。在行业轮动中，可辅助判断行业是否进入“波动收缩-突破”的敏感阶段。",
    },
    "macd_histogram": {
        "name": "MACD柱状线",
        "meaning": "基于收盘价计算MACD（12,26,9），公式：DIF = EMA12 - EMA26，DEA = DIF的9日EMA，柱状线 = (DIF - DEA)*2。",
        "interpretation": "柱状线从负转正（突破零轴）是趋势转强信号，从正转负是转弱信号；柱状线持续放大代表动能加速，缩小代表动能衰减。行业层面，MACD零轴上方的行业组合显著优于下方。"
    }
}


TECHNICAL_TRADING_PROMPT = """
你是一个专业的日频率量化交易员。你的任务是根据市场舆情、技术面因子、历史价格和当前持仓情况，做出明智的投资决策。
请注意手续费，你的目的是关注市场信号，争取长期获利，同时不要过于频繁地响应噪声。

**核心交易规则**:
1. **买入 (Buy)**: 你需要决定投入多少资金 `amount`，不需要关注份额。
2. **卖出 (Sell)**: 你需要决定卖出当前持有基金的百分比 `percentage`。例如 `percentage: 0.5` 表示卖出某基金持仓的50%。
3. **持有 (Hold)**: 不进行任何操作。

**可投资基金及其核心意义**:
{funds_text}

**技术面因子说明**:
{technical_factor_info}

**今天是{date_to_decision}，你当前的投资组合状态**:
- **可用现金**: {capital:.2f} 元
- **当前持仓**:
{holdings_text}

**你最近几个交易日被平台确认的成功交易**:
{history_trading}

**市场舆情分析**:
- **整体摘要**: {sentiment_summary}
- **详细舆情**: {sentiment_details}

**当前技术面因子快照**:
下面是每个基金在当前决策日可获得的信息计算出的技术因子。因子只使用已完成交易日的 OHLCV，以及当前交易日开盘价；`null` 表示历史数据不足或该字段不可用。
{technical_factor_text}

**历史价格走势 (最近几个交易日)**:
{history_text}

**交易成本与要求**:
- 所有交易（买入和卖出）手续费均为 **0.01%**，手续费平台会自动计算，从 `amount` 中扣除，不需要你额外计算。
- 非常重要：这次交易卖出基金的现金，不会立刻回到你的现金中，绝对不能用于当前交易的买入，买入必须使用现金 `capital`，否则会导致交易失败。

**决策要求**:
1. **综合分析**: 同时结合新闻舆情、技术面因子、历史价格和当前持仓，形成投资逻辑。
2. **理解因子**: 理解技术因子的含义。趋势、均线、波动率、回撤、量能如果互相冲突，要说明你更信任哪类信号。
3. **控制换手**: 查看最近几次历史交易，不要因为单个噪声因子过度交易。
4. **明确指令**: 给出具体买卖指令。买入时指定 `amount`，不要超出现金量；卖出时指定 `percentage`。
5. **详细推理**: 在 `reasoning` 字段中解释你为什么做出这些决策。

**输出格式 (必须是严格的JSON)**:
{{
    "reasoning": "在这里详细说明你的决策逻辑，必须包含新闻和技术因子的综合判断。",
    "chain_of_thought": "简要描述你的分析步骤。",
    "trades": [
        {{
            "fund_id": "基金代码",
            "action": "buy",
            "amount": 10000,
            "reason": "为什么买入这个基金，以及为什么是这个金额。"
        }},
        {{
            "fund_id": "基金代码",
            "action": "sell",
            "percentage": 0.5,
            "reason": "为什么卖出这个基金，以及为什么是这个比例。"
        }},
        {{
            "fund_id": "基金代码",
            "action": "hold",
            "reason": "为什么选择持有。"
        }}
    ],
    "risk_assessment": "对当前决策的风险进行评估。"
}}

请根据以上信息，给出你的专业投资决策。你的输出应当只包含 JSON：
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a demo backtest with news plus technical factors."
    )
    parser.add_argument("--track", choices=["macro", "sector"], default="sector")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--start-date", default="2025-01-02")
    parser.add_argument("--end-date", default="2025-01-31")
    parser.add_argument("--initial-capital", type=float, default=100000)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--top-rank", type=int, default=20)
    parser.add_argument("--pre-k-days", type=int, default=1)
    parser.add_argument("--history-days", type=int, default=5)
    parser.add_argument("--tech-short-window", type=int, default=5)
    parser.add_argument("--tech-medium-window", type=int, default=10)
    parser.add_argument("--tech-long-window", type=int, default=20)
    parser.add_argument("--username", default=AGENT_PLATFORM["AGENT_USERNAME"])
    parser.add_argument("--password", default=AGENT_PLATFORM["AGENT_PASSWORD"])
    parser.add_argument("--base-url", default=AGENT_PLATFORM["BASE_URL"])
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def build_config(args):
    fund_pool = MAJOR_FUND_POOL if args.track == "macro" else INDUSTRY_FUND_POOL
    default_results_dir = (
        "backtest_results_macro_tech_v1"
        if args.track == "macro"
        else "backtest_results_sector_tech_v1"
    )
    min_lookback = max(args.lookback_days, args.tech_long_window + 1)
    return {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_capital": args.initial_capital,
        "fund_pool": fund_pool,
        "agents": [{"name": args.username, "prompt": "news + technical factors"}],
        "news_sources": ["caixin", "tiantian", "sinafinance", "tencent"],
        "lookback_days": min_lookback,
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
    return out_dir / f"tech_v1_{args.track}_{args.model}_{session_id}.json"


def _date_to_int(date_value: Any) -> int:
    if isinstance(date_value, int):
        return date_value
    return int(str(date_value).replace("-", ""))


def _date_to_str(date_int: int) -> str:
    date_str = str(date_int)
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


def _pct(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return (numerator / denominator - 1) * 100


def _cumulative_return(pct_changes: pd.Series, window: int) -> Optional[float]:
    values = pd.to_numeric(pct_changes, errors="coerce").dropna().tail(window)
    if len(values) < window:
        return None
    cumulative = 1.0
    for pct_change in values:
        cumulative *= 1 + float(pct_change) / 100
    return (cumulative - 1) * 100


def format_technical_factor_info() -> str:
    lines = []
    for factor_id, info in TECH_FACTOR_INFO.items():
        lines.append(
            f"- {factor_id}（{info['name']}）: {info['meaning']} 解读：{info['interpretation']}"
        )
    return "\n".join(lines)


class TechnicalFactorCalculator:
    """Calculate leakage-safe technical factors from local OHLCV CSV files."""

    def __init__(
        self,
        price_data_dir: Path,
        short_window: int = 5,
        medium_window: int = 10,
        long_window: int = 20,
    ):
        self.price_data_dir = Path(price_data_dir)
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.price_cache: Dict[str, pd.DataFrame] = {}

    def _load_price_df(self, fund_id: str) -> Optional[pd.DataFrame]:
        if fund_id in self.price_cache:
            return self.price_cache[fund_id]

        filepath = self.price_data_dir / f"{fund_id}.csv"
        if not filepath.exists():
            filepath = self.price_data_dir / f"{fund_id}_demo.csv"
        if not filepath.exists():
            logger.warning(f"Price file not found for {fund_id}: {filepath}")
            return None

        try:
            frame = load_standardized_price_csv(str(filepath), encoding="utf-8")
            frame["date"] = pd.to_numeric(frame["date"], errors="coerce").astype("Int64")
            frame = frame.dropna(subset=["date"]).copy()
            frame["date"] = frame["date"].astype(int)
            frame = frame.sort_values("date").set_index("date")
            self.price_cache[fund_id] = frame
            return frame
        except Exception as exc:
            logger.error(f"Failed to load price data for {fund_id}: {exc}")
            return None

    def calculate_for_funds(self, fund_ids: List[str], current_date: Any) -> Dict[str, Dict[str, Any]]:
        date_int = _date_to_int(current_date)
        results = {fund_id: self.calculate_one(fund_id, date_int) for fund_id in fund_ids}

        # Cross-sectional turnover z-score: pop the raw amount proxy and normalize
        raw_amounts: Dict[str, float] = {}
        for fid, res in results.items():
            raw = res["factors"].pop("_amount_raw", None)
            if raw is not None:
                raw_amounts[fid] = raw

        if len(raw_amounts) >= 2:
            vals = list(raw_amounts.values())
            mean_a = sum(vals) / len(vals)
            std_a = (sum((v - mean_a) ** 2 for v in vals) / len(vals)) ** 0.5
            for fid, raw in raw_amounts.items():
                if std_a > 0:
                    results[fid]["factors"]["turnover_zscore"] = round((raw - mean_a) / std_a, 4)

        return results

    def calculate_one(self, fund_id: str, current_date: int) -> Dict[str, Any]:
        frame = self._load_price_df(fund_id)
        if frame is None or frame.empty:
            return self._empty_result(fund_id, current_date)

        completed = frame[frame.index < current_date].copy()
        if completed.empty:
            return self._empty_result(fund_id, current_date)

        recent_60 = completed.tail(60)
        recent_20 = completed.tail(20)
        _empty_s: pd.Series = pd.Series(dtype=float)

        closes_all = pd.to_numeric(completed.get("close", _empty_s), errors="coerce").dropna()
        pct_all = pd.to_numeric(completed.get("pctchange", _empty_s), errors="coerce").dropna()
        closes_20 = pd.to_numeric(recent_20.get("close", _empty_s), errors="coerce").dropna()
        pct_20 = pd.to_numeric(recent_20.get("pctchange", _empty_s), errors="coerce").dropna()
        amounts_60 = pd.to_numeric(recent_60.get("amount", _empty_s), errors="coerce").dropna()

        # 1. momentum_60d_pct: 60-day cumulative return
        momentum_60d = _cumulative_return(pct_all, 60)

        # 2. momentum_term_spread_pct: cumret(60) - cumret(10)
        cumret_10 = _cumulative_return(pct_all, 10)
        term_spread = (
            momentum_60d - cumret_10
            if momentum_60d is not None and cumret_10 is not None
            else None
        )

        # 3. rsi_20d: simple-average RSI — avg_gain / (avg_gain + avg_loss) * 100
        rsi_20 = None
        if len(pct_20) >= 20:
            gains = pct_20[pct_20 > 0]
            losses = pct_20[pct_20 < 0]
            avg_gain = float(gains.mean()) if not gains.empty else 0.0
            avg_loss = abs(float(losses.mean())) if not losses.empty else 0.0
            total = avg_gain + avg_loss
            if total > 0:
                rsi_20 = avg_gain / total * 100

        # 4. bias_20d_pct: (close / MA20 - 1) * 100
        bias_20d = None
        if len(closes_20) >= 20:
            ma20 = float(closes_20.mean())
            prev_close = _safe_float(closes_20.iloc[-1])
            if prev_close is not None and ma20 != 0:
                bias_20d = (prev_close / ma20 - 1) * 100

        # 5. amount_volatility_60d: negate std so higher = more stable
        amount_vol = None
        if len(amounts_60) >= 60:
            amount_vol = -float(amounts_60.std(ddof=0))

        # 6. volume_price_cov_rank: Spearman rank corr(close, volume) over 20 days
        vp_corr = None
        c_20 = pd.to_numeric(recent_20.get("close", _empty_s), errors="coerce")
        v_20 = pd.to_numeric(recent_20.get("volume", _empty_s), errors="coerce")
        df_cv = pd.DataFrame({"c": c_20, "v": v_20}).dropna()
        if len(df_cv) >= 20:
            vp_corr = _safe_float(df_cv["c"].rank().corr(df_cv["v"].rank()))

        # 7. turnover_zscore: cross-sectional, computed in calculate_for_funds
        last_amount_raw = _safe_float(amounts_60.iloc[-1]) if not amounts_60.empty else None

        # 8. bollinger_width_pct: (upper - lower) / middle * 100 = (4 * std20 / MA20) * 100
        boll_width = None
        if len(closes_20) >= 20:
            ma20 = float(closes_20.mean())
            std20 = float(closes_20.std(ddof=0))
            if ma20 != 0:
                boll_width = 4 * std20 / ma20 * 100

        # 9. macd_histogram: MACD(12,26,9), uses all completed closes for better EMA accuracy
        macd_hist = None
        if len(closes_all) >= 35:
            ema12 = closes_all.ewm(span=12, adjust=False).mean()
            ema26 = closes_all.ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9, adjust=False).mean()
            macd_hist = _safe_float(((dif - dea) * 2).iloc[-1])

        factors = {
            "momentum_60d_pct": _round_or_none(momentum_60d),
            "momentum_term_spread_pct": _round_or_none(term_spread),
            "rsi_20d": _round_or_none(rsi_20),
            "bias_20d_pct": _round_or_none(bias_20d),
            "amount_volatility_60d": _round_or_none(amount_vol),
            "volume_price_cov_rank": _round_or_none(vp_corr),
            "turnover_zscore": None,
            "bollinger_width_pct": _round_or_none(boll_width),
            "macd_histogram": _round_or_none(macd_hist),
            "_amount_raw": last_amount_raw,
        }

        return {
            "date": _date_to_str(current_date),
            "fund_name": FUND_INFO.get(fund_id, {}).get("name", fund_id),
            "factors": factors,
        }

    def _empty_result(self, fund_id: str, current_date: int) -> Dict[str, Any]:
        return {
            "date": _date_to_str(current_date),
            "fund_name": FUND_INFO.get(fund_id, {}).get("name", fund_id),
            "factors": {factor_id: None for factor_id in TECH_FACTOR_INFO},
        }


class TechnicalTradingStrategyAgent:
    """Trading strategy agent that receives technical factors in addition to news."""

    def __init__(
        self,
        prompt_template: str = TECHNICAL_TRADING_PROMPT,
        model_name: str = "",
    ):
        self.llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            temperature=1,
        )
        self.parser = CustomJsonOutputParser()
        self.prompt_template = prompt_template
        logger.info(f"decision_model is {model_name}")

    async def make_trading_decision(
        self,
        date_to_decision: str,
        sentiment_analysis: Dict[str, Any],
        historical_prices: Dict[str, List[Dict[str, Any]]],
        current_portfolio: Dict[str, Any],
        fund_pool: List[str],
        trading_history: List[Dict[str, Any]],
        platform_trading_history: List[Dict[str, Any]],
        technical_factors: Dict[str, Dict[str, Any]],
        view_platform_trading_history_days: int = 3,
    ) -> Dict[str, Any]:
        funds_text = "\n".join(
            [
                f"- {fund} ({FUND_INFO.get(fund, {}).get('name', 'Unknown')}): "
                f"{FUND_INFO.get(fund, {}).get('scope', 'N/A')}。"
                f"({FUND_INFO.get(fund, {}).get('meaning', 'Unknown')})"
                for fund in fund_pool
            ]
        )

        holdings = current_portfolio.get("holdings", {})
        capital = current_portfolio.get("capital", 0)
        holdings_text = self._format_holdings(holdings)
        history_text = self._format_price_history(historical_prices)
        history_trading = self._format_platform_trading_history(
            platform_trading_history, view_platform_trading_history_days
        )
        technical_factor_text = json.dumps(technical_factors, indent=2, ensure_ascii=False)

        prompt = self.prompt_template.format(
            funds_text=funds_text,
            technical_factor_info=format_technical_factor_info(),
            date_to_decision=date_to_decision,
            capital=capital,
            holdings_text=holdings_text if holdings_text else "  (空仓)",
            history_trading=history_trading if history_trading else "  (无历史交易)",
            sentiment_summary=sentiment_analysis.get("summary", "无舆情分析"),
            sentiment_details=json.dumps(
                sentiment_analysis.get("fund_analysis", {}),
                indent=2,
                ensure_ascii=False,
            ),
            technical_factor_text=technical_factor_text,
            history_text=history_text if history_text else "  (无历史价格)",
        )

        try:
            for attempt in range(5):
                try:
                    response = await self.llm.ainvoke(prompt)
                    decision = self.parser.parse(response.content)
                    logger.info(f"LLM Agent decision: {decision}")
                    return decision
                except Exception as exc:
                    logger.exception(f"Parser failed on attempt {attempt + 1}/5: {exc}")
                    if attempt == 4:
                        raise
        except Exception as exc:
            logger.error("Decision generation failed; using conservative hold strategy.")
            return {
                "reasoning": "决策生成失败，采取保守持有策略。",
                "chain_of_thought": f"系统错误: {str(exc)}",
                "trades": [
                    {"fund_id": fund, "action": "hold", "reason": "系统错误，保守持有。"}
                    for fund in holdings.keys()
                ],
                "risk_assessment": "高风险：系统错误。",
            }

    def _format_holdings(self, holdings: Dict[str, Any]) -> str:
        lines = []
        for fund, details in holdings.items():
            value = _safe_float(details.get("value"))
            price = _safe_float(details.get("price"))
            value_text = f"{value:.2f}" if value is not None else "N/A"
            price_text = f"{price:.2f}" if price is not None else "N/A"
            lines.append(f"- {fund}: 持仓价值 {value_text} 元 (当前价: {price_text})")
        return "\n".join(lines)

    def _format_price_history(self, historical_prices: Dict[str, List[Dict[str, Any]]]) -> str:
        history_text = ""
        for fund, prices in historical_prices.items():
            if not prices:
                continue
            history_text += f"{fund} 最近{len(prices)}条价格记录\n"
            for price in prices[-5:]:
                close_price = price.get("close")
                pct_change = price.get("pct_change")
                close_text = "N/A" if close_price is None else close_price
                pct_text = "N/A" if pct_change is None else f"{pct_change}%"
                history_text += (
                    f"  {price['date']}: 开{price.get('open', 'N/A')} "
                    f"收{close_text} 涨跌{pct_text}\n"
                )
            history_text += "\n"
        return history_text

    def _format_platform_trading_history(
        self,
        platform_trading_history: List[Dict[str, Any]],
        view_days: int,
    ) -> str:
        if not platform_trading_history:
            return ""

        trades_by_date: Dict[str, List[Dict[str, Any]]] = {}
        for trade in platform_trading_history:
            date = trade.get("date")
            trades_by_date.setdefault(date, []).append(trade)

        sorted_dates = sorted(trades_by_date.keys(), reverse=True)
        recent_dates = sorted_dates[:view_days]
        day_trade_strings = []
        for date in sorted(recent_dates):
            trade_lines = []
            for trade in trades_by_date[date]:
                trade_str = f"{trade.get('date')} {trade.get('fund_id')} {trade.get('action')}"
                if trade.get("action") == "buy":
                    trade_str += f" amount: {trade.get('amount', 0):.2f}"
                elif trade.get("action") == "sell":
                    trade_str += (
                        f" percentage: {trade.get('percentage', 0):.2%}, "
                        f"amount_sold: {trade.get('amount_sold', 0):.2f}"
                    )
                trade_lines.append(trade_str)
            day_trade_strings.append("\n".join(trade_lines))
        return "\n\n".join(day_trade_strings)


class TechnicalAdvancedTradingAgent:
    """Coordinates news summarization, sentiment analysis, and technical trading decisions."""

    def __init__(
        self,
        agent_id: str,
        decision_model_name: str,
        news_model_name: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.news_agent = NewsProcessingAgent(model_name=news_model_name or decision_model_name)
        self.sentiment_agent = SentimentAnalysisAgent(model_name=decision_model_name)
        self.trading_agent = TechnicalTradingStrategyAgent(model_name=decision_model_name)
        self.decision_history: List[Dict[str, Any]] = []
        self.trading_history: List[Dict[str, Any]] = []
        self.platform_trading_history: List[Dict[str, Any]] = []

    async def make_decision(
        self,
        date_to_decision: str,
        news_data: List[Dict[str, Any]],
        historical_prices: Dict[str, List[Dict[str, Any]]],
        current_portfolio: Dict[str, Any],
        fund_pool: List[str],
        technical_factors: Dict[str, Dict[str, Any]],
        view_platform_trading_history_days: int = 5,
    ) -> Dict[str, Any]:
        logger.info(f"{self.agent_id} starts news + technical factor decision flow.")

        processed_news = await self.news_agent.process_news_batch(news_data)
        logger.info(f"Processed news: {len(processed_news)}/{len(news_data)}")

        sentiment_analysis = await self.sentiment_agent.analyze_sentiment(
            date_to_decision, processed_news, fund_pool
        )
        logger.info(f"Sentiment: {sentiment_analysis.get('overall_sentiment', 'unknown')}")

        trading_decision = await self.trading_agent.make_trading_decision(
            date_to_decision=date_to_decision,
            sentiment_analysis=sentiment_analysis,
            historical_prices=historical_prices,
            current_portfolio=current_portfolio,
            fund_pool=fund_pool,
            trading_history=self.trading_history,
            platform_trading_history=self.platform_trading_history,
            technical_factors=technical_factors,
            view_platform_trading_history_days=view_platform_trading_history_days,
        )

        decision_record = {
            "date": current_portfolio.get("date", date_to_decision),
            "processed_news_count": len(processed_news),
            "sentiment_analysis": sentiment_analysis,
            "technical_factors": technical_factors,
            "trading_decision": trading_decision,
            "portfolio_value": current_portfolio.get("total_value", 0),
        }
        self.decision_history.append(decision_record)
        self.trading_history.append(
            {
                decision_record["date"]: trading_decision.get("trades", [])
            }
        )
        return {
            "final_decision": trading_decision,
            "intermediate_results": {
                "processed_news": processed_news,
                "sentiment_analysis": sentiment_analysis,
                "technical_factors": technical_factors,
            },
        }

    def update_platform_trading_history(self, transaction_history: List[Dict[str, Any]]) -> None:
        self.platform_trading_history = transaction_history or []


def get_technical_agent(
    agent_id: str,
    decision_model_name: str,
    news_model_name: Optional[str] = None,
) -> TechnicalAdvancedTradingAgent:
    return TechnicalAdvancedTradingAgent(
        agent_id=agent_id,
        decision_model_name=decision_model_name,
        news_model_name=news_model_name,
    )


async def run_tech_backtest(args):
    client = PlatformClient(base_url=args.base_url)
    client.register(args.username, args.password)
    client.login(args.username, args.password)

    config = build_config(args)
    calculator = TechnicalFactorCalculator(
        price_data_dir=DATA_DIRS["PRICE_DATA"],
        short_window=args.tech_short_window,
        medium_window=args.tech_medium_window,
        long_window=args.tech_long_window,
    )
    agent = get_technical_agent(
        agent_id=f"{args.track}_tech_v1_agent",
        decision_model_name=args.model,
    )

    start_response = client.start_backtest(config)
    session_id = start_response["session_id"]
    data = start_response.get("data")

    if not data:
        raise RuntimeError("Failed to get initial backtest data.")

    logger.info(
        f"Started tech_v1 session {session_id} for track={args.track}, model={args.model}"
    )

    trading_days = 0
    while True:
        trading_days += 1
        portfolio = client.get_backtest_status(session_id)
        historical_prices_response = client.get_historical_prices(
            session_id, lookback_days=config["lookback_days"]
        )
        historical_prices = historical_prices_response.get("historical_prices", {})
        technical_factors = calculator.calculate_for_funds(
            fund_ids=config["fund_pool"],
            current_date=data["date"],
        )

        try:
            decision_result = await agent.make_decision(
                date_to_decision=data["date"],
                news_data=data["news"],
                historical_prices=historical_prices,
                current_portfolio=portfolio,
                fund_pool=config["fund_pool"],
                technical_factors=technical_factors,
                view_platform_trading_history_days=config["view_platform_trading_history_days"],
            )

            final_decision = decision_result["final_decision"]
            trades = [
                trade
                for trade in final_decision.get("trades", [])
                if trade.get("action") != "hold"
            ]
            decision_payload = {
                **final_decision,
                "technical_factors_snapshot": technical_factors,
            }
            agent_decision = AgentDecision(
                decision=decision_payload,
                reasoning=final_decision.get("reasoning", ""),
                chain_of_thought=str(final_decision.get("chain_of_thought", "")),
            )
            trade_response = client.submit_trade_with_decision(session_id, trades, agent_decision)
            agent.update_platform_trading_history(trade_response.get("transaction_history", []))
        except Exception as exc:
            logger.error(f"Decision failed on {data.get('date')}: {exc}")
            logger.error(traceback.format_exc())

        data = client.get_next_day_data(session_id)
        if data.get("message") == "Backtest finished":
            break

        await asyncio.sleep(0.1)

    final_results = client.get_backtest_results(session_id)
    output_path = build_output_path(args, session_id)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(final_results, handle, indent=2, ensure_ascii=False)

    logger.info(
        f"Finished tech_v1 session {session_id} after {trading_days} trading days. "
        f"Return={final_results.get('performance', {}).get('total_return', 0) * 100:.2f}%"
    )
    logger.info(f"Saved final results to {output_path}")


def main():
    args = parse_args()
    log_dir = Path(project_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"backtest_tech_v1_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logger.add(str(log_path), level="INFO")

    logger.info(
        f"Running tech_v1 backtest with track={args.track}, model={args.model}, "
        f"period={args.start_date}~{args.end_date}"
    )
    try:
        asyncio.run(run_tech_backtest(args))
    except Exception as exc:
        logger.error(f"Tech v1 backtest failed: {exc}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
