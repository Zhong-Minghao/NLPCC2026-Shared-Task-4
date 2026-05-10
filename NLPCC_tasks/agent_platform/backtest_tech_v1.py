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
    "open_gap_pct": {
        "name": "当日开盘跳空幅度",
        "meaning": "当前交易日开盘价相对上一交易日收盘价的涨跌幅，只使用当前日开盘价，不使用当前日收盘价。",
        "interpretation": "正值说明隔夜资金情绪偏强，负值说明开盘承压；单日跳空容易反复，需要结合新闻和趋势确认。",
    },
    "ret_1d_pct": {
        "name": "上一交易日收益率",
        "meaning": "最近一个已完成交易日的涨跌幅。",
        "interpretation": "反映短期惯性或回撤压力，极端单日涨跌后要警惕追涨杀跌。",
    },
    "momentum_5d_pct": {
        "name": "5日动量",
        "meaning": "最近5个已完成交易日的累计收益。",
        "interpretation": "正值代表短线趋势向上，负值代表短线走弱；适合判断短期资金方向。",
    },
    "momentum_20d_pct": {
        "name": "20日动量",
        "meaning": "最近20个已完成交易日的累计收益。",
        "interpretation": "衡量中短期趋势强弱，比5日动量更稳定，但反应更慢。",
    },
    "ma5_bias_pct": {
        "name": "5日均线偏离",
        "meaning": "上一交易日收盘价相对5日均线的偏离幅度。",
        "interpretation": "正值说明价格位于短期均线上方，负值说明低于短期均线；偏离过大时可能有回归风险。",
    },
    "ma20_bias_pct": {
        "name": "20日均线偏离",
        "meaning": "上一交易日收盘价相对20日均线的偏离幅度。",
        "interpretation": "用于判断资产是否处于中期趋势上方或下方。",
    },
    "volatility_10d_pct": {
        "name": "10日波动率",
        "meaning": "最近10个已完成交易日收益率的标准差。",
        "interpretation": "数值越高代表价格波动越剧烈，仓位应更谨慎；数值较低代表走势相对平稳。",
    },
    "drawdown_20d_pct": {
        "name": "20日回撤",
        "meaning": "上一交易日收盘价相对近20日最高收盘价的回撤幅度。",
        "interpretation": "越接近0说明接近阶段高位，越负说明回撤越深；可辅助识别趋势修复或弱势资产。",
    },
    "volume_ratio_5d": {
        "name": "5日量能比",
        "meaning": "上一交易日成交量相对最近5个已完成交易日平均成交量的比例。",
        "interpretation": "大于1说明放量，小于1说明缩量；放量上涨更可信，放量下跌风险更高。",
    },
    "range_5d_pct": {
        "name": "5日平均振幅",
        "meaning": "最近5个已完成交易日的日内振幅均值，振幅=(high-low)/close。",
        "interpretation": "衡量近期交易分歧和日内波动，振幅高时更容易出现假突破或快速回撤。",
    },
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
        return {
            fund_id: self.calculate_one(fund_id, date_int)
            for fund_id in fund_ids
        }

    def calculate_one(self, fund_id: str, current_date: int) -> Dict[str, Any]:
        frame = self._load_price_df(fund_id)
        if frame is None or frame.empty:
            return self._empty_result(fund_id, current_date)

        completed = frame[frame.index < current_date].copy()
        if completed.empty:
            return self._empty_result(fund_id, current_date)

        current_row = None
        if current_date in frame.index:
            current_row = frame.loc[current_date]
            if isinstance(current_row, pd.DataFrame):
                current_row = current_row.iloc[0]

        recent = completed.tail(max(self.long_window, self.medium_window, self.short_window))
        closes = pd.to_numeric(recent.get("close"), errors="coerce").dropna()
        pct_changes = pd.to_numeric(recent.get("pctchange"), errors="coerce").dropna()
        highs = pd.to_numeric(recent.get("high"), errors="coerce")
        lows = pd.to_numeric(recent.get("low"), errors="coerce")
        volumes = pd.to_numeric(recent.get("volume"), errors="coerce").dropna()

        prev_close = _safe_float(closes.iloc[-1]) if not closes.empty else None
        current_open = _safe_float(current_row.get("open")) if current_row is not None else None

        ma_short = _safe_float(closes.tail(self.short_window).mean()) if len(closes) >= self.short_window else None
        ma_long = _safe_float(closes.tail(self.long_window).mean()) if len(closes) >= self.long_window else None

        vol_values = pct_changes.tail(self.medium_window)
        volatility = _safe_float(vol_values.std(ddof=0)) if len(vol_values) >= self.medium_window else None

        drawdown = None
        long_closes = closes.tail(self.long_window)
        if len(long_closes) >= self.long_window and prev_close is not None:
            high_close = _safe_float(long_closes.max())
            drawdown = _pct(prev_close, high_close)

        volume_ratio = None
        recent_volumes = volumes.tail(self.short_window)
        if len(recent_volumes) >= self.short_window:
            prev_volume = _safe_float(recent_volumes.iloc[-1])
            avg_volume = _safe_float(recent_volumes.mean())
            if prev_volume is not None and avg_volume not in (None, 0):
                volume_ratio = prev_volume / avg_volume

        range_5d = None
        if len(recent) >= self.short_window:
            ranges = ((highs - lows) / pd.to_numeric(recent.get("close"), errors="coerce") * 100).dropna()
            range_tail = ranges.tail(self.short_window)
            if len(range_tail) >= self.short_window:
                range_5d = _safe_float(range_tail.mean())

        factors = {
            "open_gap_pct": _round_or_none(_pct(current_open, prev_close)),
            "ret_1d_pct": _round_or_none(_safe_float(pct_changes.iloc[-1]) if not pct_changes.empty else None),
            "momentum_5d_pct": _round_or_none(_cumulative_return(pct_changes, self.short_window)),
            "momentum_20d_pct": _round_or_none(_cumulative_return(pct_changes, self.long_window)),
            "ma5_bias_pct": _round_or_none(_pct(prev_close, ma_short)),
            "ma20_bias_pct": _round_or_none(_pct(prev_close, ma_long)),
            "volatility_10d_pct": _round_or_none(volatility),
            "drawdown_20d_pct": _round_or_none(drawdown),
            "volume_ratio_5d": _round_or_none(volume_ratio),
            "range_5d_pct": _round_or_none(range_5d),
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
