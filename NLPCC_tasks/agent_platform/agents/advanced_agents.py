import asyncio
import json
import os
import re
import sys
from typing import Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from loguru import logger

# from luluai.langchain_contrib.chat_models.openai import EFundChatModel

load_dotenv()

# 添加项目根目录到Python路径
# agents/advanced_agents.py 相对于项目根目录的路径是两层 parent (agents -> agent_platform -> 项目根目录)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 此时 project_root = agent_platform 目录
sys.path.insert(0, project_root)

# 直接用缓存的fund info, 也可以通过client get_fund_info实现
from agent_platform.utils import CustomJsonOutputParser

from .fund_info import FUND_INFO
from .trading_strategy_prompt import (
    BASELINE_TRADING_PROMPT,
    NONEWS_TRADING_PROMPT,
    RISK_MANAGEMENT_PROMPT,
    RISK_CONCERNS_PROMPT,
    CONSENSUS_SUMMARY_PROMPT,
)
from agent_platform.agents.discussion_types import (
    ConsensusResult,
    DiscussionContext,
    DiscussionRound,
    PositionStatement,
    RiskConcerns,
)

# 全局缓存变量
GLOBAL_NEWS_CACHE = {}
# 正在处理中的任务，用于防止重复请求 {cache_key: Future}
PROCESSING_TASKS = {}
# project_root 已经是 agent_platform 目录，所以直接用 project_root 保存 cache
CACHE_FILE_PATH = os.path.join(project_root, "news_summary_cache.json")
CACHE_LOADED = False
SAVE_COUNTER = 0


def load_global_cache():
    global GLOBAL_NEWS_CACHE, CACHE_LOADED
    if CACHE_LOADED:
        return

    if os.path.exists(CACHE_FILE_PATH):
        try:
            with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                GLOBAL_NEWS_CACHE = json.load(f)
            logger.info(f"Loaded {len(GLOBAL_NEWS_CACHE)} items from news cache.")
        except Exception as e:
            logger.warning(f"Failed to load news cache: {e}")
    CACHE_LOADED = True


def _save_cache_sync():
    try:
        # 确保目录存在
        cache_dir = os.path.dirname(CACHE_FILE_PATH)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Created cache directory: {cache_dir}")

        # 使用原子写入：先写临时文件，再重命名
        temp_file = CACHE_FILE_PATH + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(GLOBAL_NEWS_CACHE, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, CACHE_FILE_PATH)
    except Exception as e:
        logger.error(f"Failed to save news cache: {e}")


async def save_global_cache():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _save_cache_sync)


class NewsProcessingAgent:
    """新闻处理Agent - 负责新闻摘要提取"""

    # Model concurrency limits
    MODEL_CONCURRENCY_LIMITS = {
        "deepseek-v4-flash": 2500,
        "deepseek-v4-pro": 500,
        "gpt-4": 100,
        "gpt-3.5-turbo": 500,
        "claude-opus-4": 100,
        "default": 25,
    }

    def __init__(self, model_name="EFundGPT-pro"):
        self.llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            temperature=1,   # EFundGPT 原来是0.1
            model_kwargs={   # kimi-k2.6允许json输出
                "response_format": {"type": "json_object"}
            }
        )
        # Set max_workers based on model
        self.max_workers = self.MODEL_CONCURRENCY_LIMITS.get(
            model_name,
            self.MODEL_CONCURRENCY_LIMITS["default"]
        )
        load_global_cache()  # avoid duplicate requests to save tokens

    async def extract_news_summary(self, news_item: Dict) -> Dict:
        """提取单条新闻摘要"""
        global SAVE_COUNTER

        # Generate cache key: THEDATE+TITLE+APP_TYPE+RANKING
        cache_key = f"{news_item.get('THEDATE', 'N/A')}_{news_item.get('TITLE', 'N/A')}_{news_item.get('APP_TYPE', 'N/A')}_{news_item.get('RANKING', 'N/A')}"

        # 1. Check Global Cache
        if cache_key in GLOBAL_NEWS_CACHE:
            logger.debug(f"  - [Cache Hit] {cache_key}")
            summary = GLOBAL_NEWS_CACHE[cache_key]
            return self._format_result(news_item, summary)

        # 2. Check if currently processing (In-flight Deduplication)
        if cache_key in PROCESSING_TASKS:
            logger.debug(f"  - [Waiting for In-flight] {cache_key}")
            try:
                # Add timeout to prevent infinite waiting
                summary = await asyncio.wait_for(
                    PROCESSING_TASKS[cache_key], timeout=120
                )
                return self._format_result(news_item, summary)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for in-flight task: {cache_key}")
                return self._format_result(news_item, "处理超时 (In-flight wait)")
            except Exception as e:
                logger.warning(f"Pending task failed for {cache_key}: {e}")
                return self._format_result(news_item, f"In-flight处理失败: {e}")

        # 3. Create new processing task
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        PROCESSING_TASKS[cache_key] = future

        try:
            prompt = f"""
请将以下金融新闻提取为非常简短的摘要（1-2句话），只保留核心信息，以json格式返回：

**原始新闻**:
时间：{news_item.get("THEDATE", "无日期")}
标题: {news_item.get("TITLE", "无标题")}
来源: {news_item.get("APP_TYPE", "未知")}
排名: {news_item.get("RANKING", "N/A")}
内容: {news_item.get("CONTENT", "无内容")}

**要求**:
1. 提取最核心的市场影响信息
2. 用1-2句话概括，非常简短精炼
3. 如果是无关的市场噪音，返回"无关键信息"
4. 只返回摘要内容，不要额外解释

**摘要**:
"""
            # Add retry logic for LLM call
            retry_count = 5
            base_delay = 2
            response = None
            last_error = None

            for i in range(retry_count):
                try:
                    # Add timeout for LLM call
                    response = await asyncio.wait_for(
                        self.llm.ainvoke(prompt), timeout=60
                    )
                    # Validate response
                    if response and hasattr(response, "content") and response.content:
                        break
                    else:
                        logger.warning(
                            f"Empty response from LLM (attempt {i + 1}/{retry_count}) for {cache_key}"
                        )
                        response = None
                except asyncio.TimeoutError:
                    last_error = f"Timeout after 45s"
                    logger.warning(
                        f"News summary timeout (attempt {i + 1}/{retry_count}) for {cache_key}"
                    )
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"News summary failed (attempt {i + 1}/{retry_count}) for {cache_key}: {e}"
                    )

                if i < retry_count - 1:
                    delay = base_delay * (2**i)  # Exponential backoff
                    await asyncio.sleep(delay)

            if (
                response is None
                or not hasattr(response, "content")
                or response.content is None
            ):
                raise last_error or Exception(
                    "Failed to generate summary after all retries"
                )

            summary = response.content.strip()

            # 清理摘要内容
            summary = re.sub(r'["\']', "", summary)
            if len(summary) > 500:
                summary = summary[:497] + "..."

            logger.debug(f"  - [New Summary] {cache_key}")

            # Update Global Cache
            GLOBAL_NEWS_CACHE[cache_key] = summary

            # Resolve waiting tasks
            if not future.done():
                future.set_result(summary)

            # Periodically save cache (e.g., every 1000 new items)
            SAVE_COUNTER += 1
            if SAVE_COUNTER % 1000 == 0:
                # Fire and forget save to avoid waiting
                asyncio.create_task(save_global_cache())

            return self._format_result(news_item, summary)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"News processing error {cache_key}: {e}")

            # Set exception for waiting tasks
            if not future.done():
                future.set_exception(Exception(error_msg))

            return self._format_result(news_item, f"处理失败: {error_msg[:100]}")

        finally:
            # Cleanup pending task safely
            if cache_key in PROCESSING_TASKS and PROCESSING_TASKS[cache_key] is future:
                del PROCESSING_TASKS[cache_key]

    def _format_result(self, news_item, summary):
        return {
            "thedate": news_item.get("THEDATE", "无日期"),
            "title": news_item.get("TITLE", ""),
            "source": news_item.get("APP_TYPE", ""),
            "ranking": news_item.get("RANKING", 0),
            "summary": summary,
            "original_content": news_item.get("CONTENT", "")[:100] + "...",
        }

    async def process_news_batch(
        self, news_list: List[Dict], max_workers: int = None
    ) -> List[Dict]:
        """批量处理新闻摘要（并发）"""
        if not news_list:
            return []

        # Use instance default if not specified
        if max_workers is None:
            max_workers = self.max_workers

        semaphore = asyncio.Semaphore(max_workers)

        async def sem_task(news_item):
            async with semaphore:
                return await self.extract_news_summary(news_item)

        tasks = [sem_task(news_item) for news_item in news_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤处理失败的结果
        processed_news = []
        for result in results:
            if isinstance(result, Dict) and "summary" in result:
                processed_news.append(result)
        return processed_news


class SentimentAnalysisAgent:
    """舆情分析Agent - 负责市场舆情判断"""

    def __init__(
        self,
        model_name="EFundGPT-pro",
    ):
        self.llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            temperature=1,   # EFundGPT 原来是0.1
        )
        self.parser = CustomJsonOutputParser()

    async def analyze_sentiment(
        self,
        date_to_decision: str,
        processed_news: List[Dict],
        fund_pool: List[str],
        llm_retry: int = 5,
    ) -> Dict:
        """分析新闻舆情对基金的影响"""
        if not processed_news:
            return {
                "overall_sentiment": "neutral",
                "fund_analysis": {},
                "summary": "无相关新闻信息",
            }

        # 构建基金列表文本
        funds_text = "\n".join(
            [
                f"- {fund} ({FUND_INFO.get(fund, {}).get('name', 'Unknown')}): {FUND_INFO.get(fund, {}).get('scope', 'N/A')}。 ({FUND_INFO.get(fund, {}).get('meaning', 'Unknown')})"
                for fund in fund_pool
            ]
        )

        # 构建处理后的新闻文本
        news_text = "\n\n".join(
            [
                f"{news.get('thedate', '无日期')} 【{news['source']}】排名{news['ranking']}: {news['title']}\n摘要: {news['summary']}"
                for news in processed_news
            ]
        )
        output_formatter = {
            "overall_sentiment": "positive/negative/neutral",
            "fund_analysis": {
                "基金代码": {
                    "sentiment": "positive/negative/neutral",
                    "reason": "简要原因",
                    "confidence": 0.8,
                }
            },
            "summary": "整体市场舆情摘要",
        }

        prompt = f"""
你是一个专业的金融市场舆情分析师。请分析以下新闻对指定投资基金的影响，用于后续生成今日的交易指令，今天是{date_to_decision}：

**可投资基金列表**:
{funds_text}

**处理后的新闻摘要**（共{len(processed_news)}条）:
{news_text}

**分析要求**:
1. 判断新闻对哪些基金可能有正面/负面/中性影响，注意这些新闻并不都是今天，综合时间线，考虑市场的可能的行情
2. 排除完全无关的新闻内容
3. 对每个基金，给出整体的舆情判断（positive/negative/neutral）
4. 提供简要的市场舆情状态和预测，考虑不同天舆情的变化存在的因素
5. 如果没有相关信息，明确说明"无相关信息"

**输出格式**（为严格的JSON）:
{output_formatter}

请给出你的输出：
"""

        try:
            for i in range(llm_retry):
                try:
                    response = await asyncio.wait_for(
                        self.llm.ainvoke(
                            prompt,
                            response_format={"type": "json_object"}  # kimi-k2.6直接支持json输出
                        ), timeout=120
                    )
                    # logger.debug(f"analyze agent input: {prompt}")
                    # process response
                    # analysis = self.parser.parse(response.content)
                    analysis = json.loads(response.content)    # 保证是json对象，简化解析
                    logger.info(f"LLM analysis: {analysis}")
                    return analysis
                except asyncio.TimeoutError:
                    logger.warning(f"LLM call timeout on attempt {i + 1}/{llm_retry}")
                    if i == llm_retry - 1:
                        raise
                except Exception as e:
                    logger.warning(f"Parser failed on attempt {i + 1}/{llm_retry}: {e}")
                    if i == llm_retry - 1:
                        raise e
        except Exception as e:
            logger.error(f"analyze_sentiment failed after {llm_retry} attempts: {e}")
            return {
                "overall_sentiment": "neutral",
                "fund_analysis": {},
                "summary": f"舆情分析失败: {str(e)}",
            }


class TradingStrategyAgent:
    """行情分析Agent - 负责投资决策"""

    def __init__(
        self,
        prompt_template: str = BASELINE_TRADING_PROMPT,
        model_name="",
    ):

        self.llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            temperature=1,   # EFundGPT 原来是0.1
        )
        self.parser = CustomJsonOutputParser()
        self.prompt_template = prompt_template
        logger.info(f"decision_model is {model_name}")

    async def make_trading_decision(
        self,
        date_to_decision: str,
        sentiment_analysis: Dict,
        historical_prices: Dict,
        current_portfolio: Dict,
        market_data: Dict,
        fund_pool: List[str],
        trading_history: List[Dict],
        platform_trading_history: List[Dict],
        view_platform_trading_history_days: int = 3,
    ) -> Dict:
        """基于舆情和历史行情做出交易决策"""

        # 构建基金信息文本
        funds_text = "\n".join(
            [
                f"- {fund} ({FUND_INFO.get(fund, {}).get('name', 'Unknown')}): {FUND_INFO.get(fund, {}).get('scope', 'N/A')}。 ({FUND_INFO.get(fund, {}).get('meaning', 'Unknown')})"
                for fund in fund_pool
            ]
        )
        logger.info(f"current_portfolio {current_portfolio}")
        # 格式化持仓信息
        holdings = current_portfolio.get("holdings", {})
        capital = current_portfolio.get("capital", 0)
        # portfolio_date = current_portfolio.get("date", "Unknown")
        holdings_text = "\n".join(
            [
                f"- {fund}: 持仓价值 {details['value']:.2f} 元 (当前价: {details['price']:.2f})"
                for fund, details in holdings.items()
            ]
        )

        # 格式化历史价格数据
        history_text = ""
        for fund, prices in historical_prices.items():
            if prices:
                history_text += f"{fund} 最近{len(prices)}天:\n"
                for price in prices[-4:]:  # 显示最近4天
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

        history_trading = ""
        if platform_trading_history:
            # Group trades by date
            trades_by_date = {}
            for trade in platform_trading_history:
                date = trade.get("date")
                if date not in trades_by_date:
                    trades_by_date[date] = []
                trades_by_date[date].append(trade)

            # Get the last 3 dates with trades
            sorted_dates = sorted(trades_by_date.keys(), reverse=True)
            recent_dates = sorted_dates[:view_platform_trading_history_days]

            # Build the history string, sorted chronologically
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

        prompt = self.prompt_template.format(
            funds_text=funds_text,
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
            history_text=history_text if history_text else "  (无历史价格)",
        )

        try:
            for i in range(5):
                try:
                    # logger.debug(f"trading agent prompt: {prompt}")
                    response = await self.llm.ainvoke(prompt)
                    decision = self.parser.parse(response.content)
                    logger.info(f"LLM Agent decision: {decision}")
                    return decision
                except Exception as e:
                    logger.exception(f"Parser failed on attempt {i + 1}/5: {e}")
                    if i == 4:
                        raise  # OutputParserException already contains LLM output
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


class AdvancedTradingAgent:
    """高级交易Agent - 协调三个专业Agent"""

    def __init__(
        self,
        agent_id: str,
        trading_prompt_template: str = None,
        decision_model_name="EFundGPT-max",
        news_model_name=None,
    ):
        self.agent_id = agent_id
        # Use decision_model_name for news if news_model_name not specified
        self.news_agent = NewsProcessingAgent(
            model_name=news_model_name or decision_model_name
        )
        self.sentiment_agent = SentimentAnalysisAgent(model_name=decision_model_name)
        self.trading_agent = TradingStrategyAgent(
            prompt_template=trading_prompt_template or BASELINE_TRADING_PROMPT,
            model_name=decision_model_name,
        )
        self.decision_history = []
        self.trading_history = []
        self.platform_trading_history = []

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
        """完整的决策流程"""

        logger.info(f"🤖 {self.agent_id} 开始决策流程...")

        # 1. 新闻处理阶段
        logger.info("📰 新闻处理中...")
        processed_news = await self.news_agent.process_news_batch(news_data)
        logger.info(f"  处理完成: {len(processed_news)}/{len(news_data)} 条新闻")

        # 2. 舆情分析阶段
        logger.info("🎯 舆情分析中...")
        sentiment_analysis = await self.sentiment_agent.analyze_sentiment(
            date_to_decision, processed_news, fund_pool
        )
        logger.info(
            f"  舆情结果: {sentiment_analysis.get('overall_sentiment', 'unknown')}"
        )

        # 3. 交易决策阶段
        logger.info("💹 交易决策中...")
        trading_decision = await self.trading_agent.make_trading_decision(
            date_to_decision,
            sentiment_analysis,
            historical_prices,
            current_portfolio,
            market_data,
            fund_pool,
            self.trading_history,
            self.platform_trading_history,
            view_platform_trading_history_days,
        )
        logger.info(f"  生成 {len(trading_decision.get('trades', []))} 个交易指令")
        logger.info(
            f"  [Model Decision]: {json.dumps(trading_decision, indent=2, ensure_ascii=False)}"
        )

        # 记录决策历史
        decision_record = {
            "date": current_portfolio.get("date", "Unknown"),
            "processed_news_count": len(processed_news),
            "sentiment_analysis": sentiment_analysis,
            "trading_decision": trading_decision,
            "portfolio_value": current_portfolio.get("total_value", 0),
        }
        self.decision_history.append(decision_record)
        self.trading_history.append(
            {
                decision_record["date"]: decision_record["trading_decision"].get(
                    "trades", []
                )
            }
        )
        return {
            "final_decision": trading_decision,
            "intermediate_results": {
                "processed_news": processed_news,
                "sentiment_analysis": sentiment_analysis,
            },
        }

    def get_decision_history(self) -> List[Dict]:
        """获取决策历史"""
        return self.decision_history

    def clear_history(self):
        """清空历史记录"""
        self.decision_history = []


class TradeHistoryTracker:
    """Track trading history and position P&L for risk management."""

    def __init__(self):
        self.trades_by_fund = {}  # fund_id -> list of trade records
        self.position_tracker = {}  # fund_id -> {entry_date, entry_value, current_value, shares}
        self.daily_pnl = {}  # date -> total_pnl

    def record_trade(self, fund_id: str, action: str, amount: float, date: str,
                     price: float = None, percentage: float = None):
        """Record a trade execution."""
        if fund_id not in self.trades_by_fund:
            self.trades_by_fund[fund_id] = []

        trade_record = {
            "date": date,
            "action": action,
            "amount": amount,
            "price": price,
            "percentage": percentage,
        }
        self.trades_by_fund[fund_id].append(trade_record)

        # Update position tracker for buy trades
        if action == "buy":
            self.position_tracker[fund_id] = {
                "entry_date": date,
                "entry_value": amount,
                "entry_price": price or 0,
                "current_value": amount,
            }
        elif action == "sell" and percentage is not None:
            # For sells, we update entry date if fully sold or adjust if partial
            if fund_id in self.position_tracker:
                if percentage >= 0.99:  # Essentially full sell
                    del self.position_tracker[fund_id]
                # Partial sell keeps the original entry date

    def update_position_value(self, fund_id: str, current_value: float, current_price: float):
        """Update current position value for P&L calculation."""
        if fund_id in self.position_tracker:
            self.position_tracker[fund_id]["current_value"] = current_value
            self.position_tracker[fund_id]["current_price"] = current_price

    def get_holding_days(self, fund_id: str, current_date: str) -> int:
        """Get days held for current position."""
        if fund_id not in self.position_tracker:
            return 0

        entry_date_str = self.position_tracker[fund_id]["entry_date"]
        try:
            from datetime import datetime
            entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d")
            curr_date = datetime.strptime(current_date, "%Y-%m-%d")
            return (curr_date - entry_date).days
        except Exception:
            return 0

    def get_pnl_percentage(self, fund_id: str) -> float:
        """Calculate unrealized P&L percentage."""
        if fund_id not in self.position_tracker:
            return 0.0

        pos = self.position_tracker[fund_id]
        if pos.get("entry_value", 0) > 0:
            return (pos["current_value"] - pos["entry_value"]) / pos["entry_value"]
        return 0.0

    def get_trades_for_fund(self, fund_id: str) -> List[Dict]:
        """Get all trades for a specific fund."""
        return self.trades_by_fund.get(fund_id, [])

    def get_last_trade_date(self, fund_id: str) -> str:
        """Get the date of the last trade for a fund."""
        trades = self.trades_by_fund.get(fund_id, [])
        if trades:
            return trades[-1]["date"]
        return None

    def get_all_positions_pnl(self, current_date: str = None) -> List[Dict]:
        """Get P&L for all current positions."""
        result = []
        for fund_id in self.position_tracker.keys():
            pnl_pct = self.get_pnl_percentage(fund_id)
            holding_days = 0
            if current_date:
                holding_days = self.get_holding_days(fund_id, current_date)

            pos = self.position_tracker[fund_id]
            result.append({
                "fund_id": fund_id,
                "pnl_percentage": pnl_pct,
                "holding_days": holding_days,
                "current_value": pos.get("current_value", 0),
            })
        return result


class RiskManagementAgent:
    """Risk Management Agent - Enforces trading rules and manages portfolio risk.

    Uses a hybrid approach:
    - Rule-based for clear violations (holding period, daily loss limits)
    - LLM-based for complex decisions (trade size adjustments, exception handling)
    """

    def __init__(
        self,
        model_name: str = "deepseek-v4-pro",
        min_holding_days: int = 7,
        daily_loss_limit: float = 0.05,  # 5% daily loss limit
        max_position_concentration: float = 0.4,  # 40% max single position
        prompt_template: str = None,
    ):
        self.llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            temperature=0.1,
        )
        self.parser = CustomJsonOutputParser()
        self.min_holding_days = min_holding_days
        self.daily_loss_limit = daily_loss_limit
        self.max_position_concentration = max_position_concentration
        self.prompt_template = prompt_template or RISK_MANAGEMENT_PROMPT
        self.history_tracker = TradeHistoryTracker()

        # Track portfolio value history for drawdown calculation
        self.portfolio_value_history = []

    def _clean_trade_for_server(self, trade: Dict) -> Dict:
        """Clean trade dict to match server validation requirements.

        Server requires:
        - buy: amount must be set, percentage must be None/not present
        - sell: percentage must be set, amount must be None/not present
        """
        action = trade.get("action", "")
        cleaned = {
            "fund_id": trade.get("fund_id"),
            "action": action,
        }

        if action == "buy":
            cleaned["amount"] = trade.get("amount", 0)
            cleaned["reason"] = trade.get("reason", "")
        elif action == "sell":
            cleaned["percentage"] = trade.get("percentage", 0)
            cleaned["reason"] = trade.get("reason", "")
        elif action == "hold":
            cleaned["reason"] = trade.get("reason", "")
            if "amount" in trade:
                cleaned["amount"] = trade["amount"]
            if "percentage" in trade:
                cleaned["percentage"] = trade["percentage"]

        return cleaned

    async def evaluate_trades(
        self,
        proposed_trades: List[Dict],
        current_portfolio: Dict,
        sentiment_analysis: Dict,
        current_date: str,
        current_rp_weights: Dict = None,
    ) -> Dict:
        """Evaluate and potentially modify proposed trades based on risk rules.

        Returns:
            dict with:
                - approved_trades: list of approved trades
                - modified_trades: list of modified trades with reasons
                - blocked_trades: list of blocked trades with reasons
                - risk_summary: summary of risk assessment
        """
        logger.info(f"🛡️ [风险管理] 开始评估 {len(proposed_trades)} 个拟交易...")

        approved_trades = []
        modified_trades = []
        blocked_trades = []

        # Update position tracker with current portfolio values
        holdings = current_portfolio.get("holdings", {})
        for fund_id, details in holdings.items():
            self.history_tracker.update_position_value(
                fund_id, details.get("value", 0), details.get("price", 0)
            )

        # Track portfolio value for drawdown
        total_value = current_portfolio.get("total_value", 0)
        self.portfolio_value_history.append({"date": current_date, "value": total_value})

        # Rule-based layer (fast path)
        for trade in proposed_trades:
            fund_id = trade.get("fund_id")
            action = trade.get("action")

            if action == "hold":
                continue

            # Check 1: Minimum holding period for sells
            if action == "sell":
                holding_days = self.history_tracker.get_holding_days(fund_id, current_date)
                if holding_days < self.min_holding_days:
                    # Check for negative sentiment exception
                    fund_sentiment = sentiment_analysis.get("fund_analysis", {}).get(fund_id, {})
                    sentiment = fund_sentiment.get("sentiment", "neutral")

                    if sentiment != "negative":
                        # Block the sell due to minimum holding period
                        blocked_trades.append({
                            "fund_id": fund_id,
                            "action": action,
                            "reason": f"持有仅{holding_days}天，未达到最小持有期{self.min_holding_days}天",
                            "holding_days": holding_days,
                            "min_holding_days": self.min_holding_days,
                        })
                        logger.info(f"  🛡️ 阻止卖出 {fund_id}: 持有{holding_days}天 < {self.min_holding_days}天")
                        continue

            # Check 2: Position concentration for buys
            if action == "buy":
                current_holding = holdings.get(fund_id, {}).get("value", 0)
                proposed_amount = trade.get("amount", 0)
                new_value = current_holding + proposed_amount
                if total_value > 0 and new_value / total_value > self.max_position_concentration:
                    # Modify trade to respect concentration limit
                    max_allowed = total_value * self.max_position_concentration - current_holding
                    if max_allowed > 0:
                        original_amount = proposed_amount
                        trade["amount"] = min(proposed_amount, max_allowed)
                        modified_trades.append({
                            "fund_id": fund_id,
                            "original_action": "buy",
                            "original_amount": original_amount,
                            "new_action": "buy",
                            "new_amount": trade["amount"],
                            "reason": f"单一持仓超过{self.max_position_concentration:.0%}限制",
                        })
                        logger.info(f"  🛡️ 调整买入 {fund_id}: {original_amount:.0f} -> {trade['amount']:.0f}")
                    else:
                        blocked_trades.append({
                            "fund_id": fund_id,
                            "action": action,
                            "reason": f"当前持仓已达到{self.max_position_concentration:.0%}集中度限制",
                        })
                        logger.info(f"  🛡️ 阻止买入 {fund_id}: 超过集中度限制")
                        continue

            approved_trades.append(trade)

        # LLM-based layer for complex decisions
        if approved_trades or modified_trades:
            llm_adjustment = await self._llm_risk_adjustment(
                approved_trades,
                modified_trades,
                current_portfolio,
                sentiment_analysis,
                current_date,
                current_rp_weights,
            )
            if llm_adjustment.get("adjusted_trades"):
                # Clean the LLM-returned trades to match server format
                approved_trades = [
                    self._clean_trade_for_server(t)
                    for t in llm_adjustment["adjusted_trades"]
                ]
            if llm_adjustment.get("risk_summary"):
                risk_summary = llm_adjustment["risk_summary"]
            else:
                risk_summary = self._generate_risk_summary(
                    approved_trades, modified_trades, blocked_trades
                )
        else:
            risk_summary = self._generate_risk_summary(
                approved_trades, modified_trades, blocked_trades
            )

        # Record approved trades (clean format for tracker)
        for trade in approved_trades:
            fund_id = trade.get("fund_id")
            action = trade.get("action")
            if action == "buy":
                price = holdings.get(fund_id, {}).get("price", None)
                self.history_tracker.record_trade(
                    fund_id, action, trade.get("amount", 0), current_date, price=price
                )
            elif action == "sell":
                self.history_tracker.record_trade(
                    fund_id, action, 0, current_date, percentage=trade.get("percentage", 0)
                )

        return {
            "approved_trades": approved_trades,
            "modified_trades": modified_trades,
            "blocked_trades": blocked_trades,
            "risk_summary": risk_summary,
        }

    async def _llm_risk_adjustment(
        self,
        approved_trades: List[Dict],
        modified_trades: List[Dict],
        current_portfolio: Dict,
        sentiment_analysis: Dict,
        current_date: str,
        current_rp_weights: Dict = None,
    ) -> Dict:
        """Use LLM for complex risk decisions using RISK_MANAGEMENT_PROMPT template."""

        holdings = current_portfolio.get("holdings", {})
        capital = current_portfolio.get("capital", 0)
        total_value = current_portfolio.get("total_value", 0)

        # Format holdings text
        holdings_text = "\n".join([
            f"- {fund_id}: 持仓价值 {details['value']:.2f} 元 (当前价: {details['price']:.2f})"
            for fund_id, details in holdings.items()
        ]) if holdings else "  (空仓)"

        # Format proposed trades
        proposed_trades = approved_trades + modified_trades
        trades_text = json.dumps(proposed_trades, ensure_ascii=False, indent=2)

        # Format trading history from tracker
        trading_history_list = []
        for fund_id in list(self.history_tracker.trades_by_fund.keys())[:10]:  # Last 10 funds
            trades = self.history_tracker.get_trades_for_fund(fund_id)
            if trades:
                last_trade = trades[-1]
                trading_history_list.append(
                    f"{last_trade['date']} {fund_id} {last_trade['action']} "
                    f"amount:{last_trade.get('amount', 0)} "
                    f"percentage:{last_trade.get('percentage', 0)}"
                )
        trading_history = "\n".join(trading_history_list) if trading_history_list else "  (无交易历史)"

        # Format sentiment analysis
        sentiment_text = json.dumps(
            sentiment_analysis.get('fund_analysis', {}),
            ensure_ascii=False,
            indent=2
        )

        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown()

        # Format position P&L
        position_pnl_list = []
        all_pnl = self.history_tracker.get_all_positions_pnl()
        for pnl_info in all_pnl:
            fund_id = pnl_info["fund_id"]
            pnl_pct = pnl_info["pnl_percentage"]
            holding_days = pnl_info["holding_days"]
            position_pnl_list.append(
                f"- {fund_id}: P&L {pnl_pct:.2%}, 持有{holding_days}天"
            )
        position_pnl = "\n".join(position_pnl_list) if position_pnl_list else "  (无持仓)"

        # Build prompt using template
        # Note: Pass raw float values for template formatting
        prompt = self.prompt_template.format(
            current_date=current_date,
            total_value=total_value,
            capital=capital,
            holdings_text=holdings_text,
            proposed_trades=trades_text,
            trading_history=trading_history,
            sentiment_analysis=sentiment_text,
            max_drawdown=max_drawdown,  # Pass raw float, template formats it
            min_holding_days=self.min_holding_days,
            max_concentration=self.max_position_concentration,  # Pass raw float
            daily_loss_limit=self.daily_loss_limit,  # Pass raw float
            position_pnl=position_pnl,
        )

        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt, response_format={"type": "json_object"}),
                timeout=60
            )
            result = json.loads(response.content)
            logger.info(f"  🛡️ LLM风险评估: {result.get('risk_level', 'unknown')}")
            return result
        except Exception as e:
            logger.warning(f"LLM风险评估失败: {e}")
            return {"adjusted_trades": approved_trades, "risk_summary": "规则基础风险评估"}

    def _calculate_max_drawdown(self) -> float:
        """Calculate current maximum drawdown."""
        if len(self.portfolio_value_history) < 2:
            return 0.0

        values = [h["value"] for h in self.portfolio_value_history]
        peak = max(values)
        current = values[-1]
        if peak > 0:
            return (peak - current) / peak
        return 0.0

    def _generate_risk_summary(
        self, approved: List[Dict], modified: List[Dict], blocked: List[Dict]
    ) -> str:
        """Generate a text summary of risk assessment."""
        parts = []
        if approved:
            parts.append(f"批准{len(approved)}笔交易")
        if modified:
            parts.append(f"调整{len(modified)}笔交易")
        if blocked:
            parts.append(f"阻止{len(blocked)}笔交易")

        summary = ", ".join(parts) if parts else "无交易"
        summary += f" | 最大回撤: {self._calculate_max_drawdown():.2%}"
        return summary

    def get_position_pnl(self, fund_id: str) -> Dict:
        """Get P&L information for a position."""
        pnl_pct = self.history_tracker.get_pnl_percentage(fund_id)
        holding_days = 0
        if fund_id in self.history_tracker.position_tracker:
            holding_days = self.history_tracker.get_holding_days(
                fund_id,
                self.history_tracker.position_tracker[fund_id].get("current_date", "")
            )

        return {
            "fund_id": fund_id,
            "pnl_percentage": pnl_pct,
            "holding_days": holding_days,
            "current_value": self.history_tracker.position_tracker.get(fund_id, {}).get(
                "current_value", 0
            ),
        }

    def get_all_positions_pnl(self) -> List[Dict]:
        """Get P&L for all current positions."""
        return [
            self.get_position_pnl(fund_id)
            for fund_id in self.history_tracker.position_tracker.keys()
        ]

    # ------------------------------------------------------------------ #
    #  Discussion-mode methods (committee deliberation)                   #
    # ------------------------------------------------------------------ #

    def _apply_rule_based_checks(
        self,
        proposed_trades: List[Dict],
        current_portfolio: Dict,
        sentiment_analysis: Dict,
        current_date: str,
    ):
        """
        Apply hard trading rules and return (hard_blocks, soft_concerns).

        hard_blocks: trades that must be blocked (rule violation with no waiver).
        soft_concerns: trades that can be negotiated (e.g. amount reduction).
        """
        holdings = current_portfolio.get("holdings", {})
        total_value = current_portfolio.get("total_value", 0)
        hard_blocks = []
        soft_concerns = []

        for trade in proposed_trades:
            fund_id = trade.get("fund_id")
            action = trade.get("action")
            if action == "hold":
                continue

            if action == "sell":
                holding_days = self.history_tracker.get_holding_days(fund_id, current_date)
                if holding_days < self.min_holding_days:
                    fund_sentiment = sentiment_analysis.get("fund_analysis", {}).get(fund_id, {})
                    if fund_sentiment.get("sentiment", "neutral") != "negative":
                        hard_blocks.append({
                            "fund_id": fund_id,
                            "action": action,
                            "rule": "min_holding_period",
                            "reason": (
                                f"持有仅{holding_days}天，未达到最小持有期{self.min_holding_days}天"
                            ),
                        })

            if action == "buy":
                current_holding = holdings.get(fund_id, {}).get("value", 0)
                proposed_amount = trade.get("amount", 0)
                new_value = current_holding + proposed_amount
                if total_value > 0 and new_value / total_value > self.max_position_concentration:
                    max_allowed = total_value * self.max_position_concentration - current_holding
                    if max_allowed <= 0:
                        hard_blocks.append({
                            "fund_id": fund_id,
                            "action": action,
                            "rule": "position_concentration",
                            "reason": (
                                f"当前持仓已达到{self.max_position_concentration:.0%}集中度限制"
                            ),
                        })
                    else:
                        soft_concerns.append({
                            "fund_id": fund_id,
                            "action": action,
                            "concern": (
                                f"买入后持仓将超过{self.max_position_concentration:.0%}集中度限制"
                            ),
                            "suggested_modification": f"建议买入不超过{max_allowed:.0f}元",
                            "severity": "high",
                        })

        return hard_blocks, soft_concerns

    async def generate_risk_concerns(
        self,
        position_statement: PositionStatement,
        context: DiscussionContext,
        discussion_history: List[DiscussionRound],
    ) -> RiskConcerns:
        """
        Generate initial risk concerns for the BL agent's position statement.
        Layer 1: rule-based hard blocks.
        Layer 2: LLM-based soft concerns and counter-proposals.
        """
        holdings = context.current_portfolio.get("holdings", {})
        capital = context.current_portfolio.get("capital", 0)
        total_value = context.current_portfolio.get("total_value", 0)

        for fund_id, details in holdings.items():
            self.history_tracker.update_position_value(
                fund_id, details.get("value", 0), details.get("price", 0)
            )
        self.portfolio_value_history.append({"date": context.date, "value": total_value})

        hard_blocks, rule_soft = self._apply_rule_based_checks(
            position_statement.proposed_trades,
            context.current_portfolio,
            position_statement.sentiment_analysis,
            context.date,
        )

        holdings_text = "\n".join([
            f"- {fid}: 持仓价值 {d.get('value', 0):.2f} 元"
            for fid, d in holdings.items()
        ]) if holdings else "  (空仓)"

        position_statement_text = (
            f"投资逻辑: {position_statement.reasoning_text}\n"
            f"拟交易: {json.dumps(position_statement.proposed_trades, ensure_ascii=False)}\n"
            f"目标权重: {json.dumps({k: f'{v:.2%}' for k, v in position_statement.target_weights.items()}, ensure_ascii=False)}\n"
            f"预期收益: {position_statement.optimization_metrics.get('expected_return', 0):.4%}, "
            f"Sharpe: {position_statement.optimization_metrics.get('sharpe_ratio', 0):.4f}"
        )

        max_drawdown = self._calculate_max_drawdown()
        all_pnl = self.history_tracker.get_all_positions_pnl(current_date=context.date)
        pnl_text = "\n".join([
            f"- {p['fund_id']}: P&L {p['pnl_percentage']:.2%}, 持有{p['holding_days']}天"
            for p in all_pnl
        ]) if all_pnl else "  (无持仓)"
        risk_metrics_text = f"最大回撤: {max_drawdown:.2%}\n持仓P&L:\n{pnl_text}"

        tech_factors = context.technical_factors
        if tech_factors:
            tech_text = "\n".join([
                f"- {fid}: 动量60d={f.get('factors', {}).get('momentum_60d_pct', 'N/A')}, "
                f"RSI20={f.get('factors', {}).get('rsi_20d', 'N/A')}"
                for fid, f in list(tech_factors.items())[:5]
            ])
        else:
            tech_text = "  (无技术因子数据)"

        prior_concerns_section = ""
        if discussion_history:
            last = discussion_history[-1]
            prior_concerns_section = (
                f"\n**上轮风险关切**: {last.risk_concerns.concerns_text[:300]}\n"
                f"**投资委员会的回应**: {last.position_statement.reasoning_text[:200]}\n"
                f"**已做出的让步**: {last.position_statement.concessions_made}\n"
                f"**请评估上述关切是否已被解决，并提出新的关切（如有）**\n"
            )

        prompt = RISK_CONCERNS_PROMPT.format(
            min_holding_days=self.min_holding_days,
            max_concentration=self.max_position_concentration,
            daily_loss_limit=self.daily_loss_limit,
            current_date=context.date,
            max_drawdown=max_drawdown,
            position_statement_text=position_statement_text,
            risk_metrics_text=risk_metrics_text,
            prior_concerns_section=prior_concerns_section,
            technical_factors_text=tech_text,
        )

        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt, response_format={"type": "json_object"}),
                timeout=60,
            )
            result = json.loads(response.content)
        except Exception as e:
            logger.warning(f"风险关切LLM生成失败: {e}")
            result = {
                "hard_blocks": [],
                "soft_concerns": [],
                "counter_proposal": [],
                "concerns_text": "规则检查完成，无重大关切",
                "risk_level": "low",
                "requirements_for_approval": "无特殊要求",
            }

        llm_hard = result.get("hard_blocks", [])
        all_hard_blocks = hard_blocks.copy()
        for lb in llm_hard:
            if not any(
                b["fund_id"] == lb.get("fund_id") and b["action"] == lb.get("action")
                for b in all_hard_blocks
            ):
                all_hard_blocks.append(lb)

        all_soft = rule_soft + result.get("soft_concerns", [])

        return RiskConcerns(
            hard_blocks=all_hard_blocks,
            soft_concerns=all_soft,
            counter_proposal=result.get("counter_proposal", []),
            concerns_text=result.get("concerns_text", ""),
            risk_level=result.get("risk_level", "medium"),
            requirements_for_approval=result.get("requirements_for_approval", ""),
        )

    async def evaluate_revised_proposal(
        self,
        revised_statement: PositionStatement,
        prior_concerns: RiskConcerns,
        context: DiscussionContext,
        discussion_history: List[DiscussionRound],
    ) -> RiskConcerns:
        """
        Evaluate the BL agent's revised proposal.
        The discussion_history (which now includes the round that produced
        prior_concerns) provides context for the LLM.
        """
        return await self.generate_risk_concerns(
            position_statement=revised_statement,
            context=context,
            discussion_history=discussion_history,
        )

    async def reach_consensus(
        self,
        final_statement: PositionStatement,
        final_concerns: RiskConcerns,
        context: DiscussionContext,
        discussion_history: List[DiscussionRound],
    ) -> ConsensusResult:
        """
        Finalize the discussion: apply remaining hard blocks, accept/modify
        the rest, and generate a merged consensus narrative.
        """
        holdings = context.current_portfolio.get("holdings", {})
        blocked_keys = {
            (b.get("fund_id"), b.get("action")) for b in final_concerns.hard_blocks
        }

        approved = []
        modified = []
        blocked = []

        for trade in final_statement.proposed_trades:
            key = (trade.get("fund_id"), trade.get("action"))
            if key in blocked_keys:
                blocked.append(trade)
                continue

            counter = next(
                (cp for cp in final_concerns.counter_proposal
                 if cp.get("fund_id") == trade.get("fund_id")),
                None,
            )
            if counter and (
                (trade.get("action") == "buy" and counter.get("amount") is not None
                 and abs(counter.get("amount", trade.get("amount", 0)) - trade.get("amount", 0)) > 1)
                or
                (trade.get("action") == "sell" and counter.get("percentage") is not None
                 and abs(counter.get("percentage", trade.get("percentage", 0)) - trade.get("percentage", 0)) > 0.01)
            ):
                merged = {**trade}
                if counter.get("amount") is not None and trade.get("action") == "buy":
                    merged["amount"] = counter["amount"]
                if counter.get("percentage") is not None and trade.get("action") == "sell":
                    merged["percentage"] = counter["percentage"]
                merged["reason"] = (
                    f"{trade.get('reason', '')} [风险调整: {counter.get('reason', '')}]"
                )
                modified.append(self._clean_trade_for_server(merged))
            else:
                approved.append(self._clean_trade_for_server(trade))

        for trade in approved + modified:
            fund_id = trade.get("fund_id")
            action = trade.get("action")
            if action == "buy":
                price = holdings.get(fund_id, {}).get("price", None)
                self.history_tracker.record_trade(
                    fund_id, action, trade.get("amount", 0), context.date, price=price
                )
            elif action == "sell":
                self.history_tracker.record_trade(
                    fund_id, action, 0, context.date, percentage=trade.get("percentage", 0)
                )

        bl_final_thesis = (
            discussion_history[-1].position_statement.reasoning_text
            if discussion_history else final_statement.reasoning_text
        )
        risk_final_assessment = final_concerns.concerns_text

        discussion_summary = "\n".join([
            f"第{r.round_number}轮: 关切={r.risk_concerns.concerns_text[:80]}... "
            f"让步={r.position_statement.concessions_made}"
            for r in discussion_history[-2:]
        ])
        final_trades_text = json.dumps(approved + modified, ensure_ascii=False, indent=2)

        prompt = CONSENSUS_SUMMARY_PROMPT.format(
            rounds=len(discussion_history),
            discussion_summary=discussion_summary or "(首轮即达成共识)",
            final_trades_text=final_trades_text,
            bl_final_thesis=bl_final_thesis[:300],
            risk_final_assessment=risk_final_assessment[:300],
        )

        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt, response_format={"type": "json_object"}),
                timeout=60,
            )
            result = json.loads(response.content)
        except Exception as e:
            logger.warning(f"共识摘要LLM生成失败: {e}")
            result = {
                "consensus_reasoning": f"经过{len(discussion_history)}轮讨论，委员会达成共识",
                "key_agreements": [],
                "key_compromises": final_statement.concessions_made,
                "risk_level": final_concerns.risk_level,
                "risk_summary": final_concerns.requirements_for_approval,
            }

        return ConsensusResult(
            final_trades=approved + modified,
            approved_trades=approved,
            modified_trades=modified,
            blocked_trades=blocked,
            consensus_reasoning=result.get("consensus_reasoning", ""),
            bl_final_thesis=bl_final_thesis,
            risk_final_assessment=risk_final_assessment,
            discussion_rounds=len(discussion_history),
            key_agreements=result.get("key_agreements", []),
            key_compromises=result.get("key_compromises", []),
            risk_level=result.get("risk_level", final_concerns.risk_level),
            risk_summary=result.get("risk_summary", ""),
        )


class SillyTradingAgent:
    """无新闻输入的Agent - 仅使用历史数据进行交易决策"""

    def __init__(
        self,
        agent_id: str,
        trading_prompt_template: str = None,
        decision_model_name="EFundGPT-max",
    ):
        self.agent_id = agent_id
        # 使用NONEWS_TRADING_PROMPT模板，忽略新闻数据
        self.trading_agent = TradingStrategyAgent(
            prompt_template=trading_prompt_template or NONEWS_TRADING_PROMPT,
            model_name=decision_model_name,
        )
        self.decision_history = []
        self.trading_history = []
        self.platform_trading_history = []

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
        """完整的决策流程 - 忽略新闻数据，仅使用历史价格"""

        logger.info(f"🤖 {self.agent_id} 开始无新闻决策流程...")

        # 跳过新闻处理和舆情分析阶段，直接使用空数据
        sentiment_analysis = {
            "overall_sentiment": "neutral",
            "fund_analysis": {},
            "summary": "无新闻输入，仅基于历史价格分析",
        }

        # 交易决策阶段
        logger.info("💹 基于历史价格的交易决策中...")
        trading_decision = await self.trading_agent.make_trading_decision(
            date_to_decision,
            sentiment_analysis,
            historical_prices,
            current_portfolio,
            market_data,
            fund_pool,
            self.trading_history,
            self.platform_trading_history,
            view_platform_trading_history_days,
        )
        logger.info(f"  生成 {len(trading_decision.get('trades', []))} 个交易指令")
        logger.info(
            f"  [Model Decision]: {json.dumps(trading_decision, indent=2, ensure_ascii=False)}"
        )

        # 记录决策历史
        decision_record = {
            "date": current_portfolio.get("date", "Unknown"),
            "processed_news_count": 0,  # 无新闻处理
            "sentiment_analysis": sentiment_analysis,
            "trading_decision": trading_decision,
            "portfolio_value": current_portfolio.get("total_value", 0),
        }
        self.decision_history.append(decision_record)
        self.trading_history.append(
            {
                decision_record["date"]: decision_record["trading_decision"].get(
                    "trades", []
                )
            }
        )
        return {
            "final_decision": trading_decision,
            "intermediate_results": {
                "processed_news": [],  # 返回空新闻列表
                "sentiment_analysis": sentiment_analysis,
            },
        }

    def get_decision_history(self) -> List[Dict]:
        """获取决策历史"""
        return self.decision_history

    def clear_history(self):
        """清空历史记录"""
        self.decision_history = []


# 全局Agent实例
advanced_agent = None


def get_advanced_agent(
    agent_id: str = "advanced_agent",
    trading_prompt_template: str = None,
    decision_model_name="EFundGPT-max",
    news_model_name=None,
) -> AdvancedTradingAgent:
    """获取高级Agent实例 (每次返回新实例以支持并发回测)"""
    # 在并发回测场景下，每个session需要独立的Agent实例来维护自己的history
    return AdvancedTradingAgent(
        agent_id,
        trading_prompt_template=trading_prompt_template,
        decision_model_name=decision_model_name,
        news_model_name=news_model_name,
    )
