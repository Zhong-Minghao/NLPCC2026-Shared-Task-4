"""
Black-Litterman Agent

Integrates view generation, Black-Litterman optimization, and weight conversion
to provide a complete portfolio management solution.
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, TYPE_CHECKING

from langchain_openai import ChatOpenAI
from loguru import logger

from agent_platform.agents.view_generation_agent import ViewGenerationAgent, View
from agent_platform.agents.advanced_agents import (
    NewsProcessingAgent,
    SentimentAnalysisAgent,
)
from agent_platform.agents.rolling_sentiment_tracker import RollingSentimentTracker
from agent_platform.portfolio_optimization import PortfolioOptimizer, OptimizationResult
from agent_platform.portfolio.weight_converter import WeightConverter
from agent_platform.agents.advanced_agents import RiskManagementAgent
from agent_platform.agents.discussion_types import PositionStatement
from agent_platform.memory.memory_bank import MemoryBank
from agent_platform.memory.reflection_agent import ReflectionAgent

if TYPE_CHECKING:
    from agent_platform.agents.discussion_types import (
        DiscussionContext,
        DiscussionRound,
        RiskConcerns,
    )


class BlackLittermanAgent:
    """
    Black-Litterman portfolio management agent.

    This agent integrates:
    1. News processing and sentiment analysis
    2. View generation using LLM
    3. Black-Litterman portfolio optimization
    4. Weight-to-trade conversion
    5. Risk management
    """

    def __init__(
        self,
        agent_id: str,
        view_model_name: str = "deepseek-v4-pro",
        news_model_name: str = "deepseek-v4-flash",
        # Black-Litterman parameters
        tau: float = 0.05,
        risk_aversion: float = 3.0,
        lookback_days: int = 60,
        # Optimization parameters
        optimization_method: str = "max_sharpe",
        risk_free_rate: float = 0.03,
        min_weight: float = 0.0,
        max_weight: float = 0.4,
        # View generation parameters
        min_confidence: float = 0.3,
        max_views_per_day: int = 5,
        # Risk management parameters
        enable_risk_management: bool = True,
        min_holding_days: int = 7,
        max_position_concentration: float = 0.4,
        daily_loss_limit: float = 0.05,
        # Memory / reflection parameters
        enable_memory: bool = False,
        memory_file: Optional[str] = None,
        reflection_interval: int = 5,
        reflection_model_name: Optional[str] = None,
        # Rolling sentiment parameters
        enable_rolling_sentiment: bool = True,
        rolling_sentiment_days: int = 10,
        alpha_base: float = 0.30,
        no_news_decay: float = 0.05,
        view_flip_threshold: float = 0.25,
        view_flip_return_threshold: float = 0.01,
        base_magnitude: float = 0.015,
        asymmetric_factor: float = 0.50,
        # Turnover reduction parameters
        turnover_penalty: float = 0.0,
        enable_view_persistence: bool = True,
    ):
        """
        Args:
            agent_id: Agent identifier
            view_model_name: Model for view generation
            news_model_name: Model for news processing
            tau: BL scaling parameter
            risk_aversion: Risk aversion coefficient
            lookback_days: Days for covariance calculation
            optimization_method: "max_sharpe" or "min_variance"
            risk_free_rate: Annual risk-free rate
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint
            min_confidence: Minimum view confidence
            max_views_per_day: Maximum views per day
            enable_risk_management: Enable risk management checks
            min_holding_days: Minimum holding period for sells
            max_position_concentration: Maximum single position
            daily_loss_limit: Maximum daily loss limit
            enable_memory: Enable reflection & memory mechanism
            memory_file: Path for persistent memory (None = session-only)
            reflection_interval: Trading days between LLM reflections
            reflection_model_name: Model for reflection (defaults to view_model_name)
            enable_rolling_sentiment: Enable rolling sentiment tracker
            rolling_sentiment_days: Warm-up lookback window (days)
            alpha_base: Base EMA update weight
            no_news_decay: Daily decay when no news for a fund
            view_flip_threshold: |rolling_score| threshold for "directional" signal
            view_flip_return_threshold: LLM expected_return must exceed this to override rolling direction
            base_magnitude: rolling_score=1.0 maps to this daily expected return
            asymmetric_factor: Same-direction alpha reduction factor
            turnover_penalty: L1 penalty coefficient λ on weight changes vs prev period (0=off)
            enable_view_persistence: Inject yesterday's views into prompt for LLM consistency
        """
        self.agent_id = agent_id
        self.model_name = view_model_name
        self.enable_memory = enable_memory
        self.llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model=view_model_name,
            temperature=0.1,
        )

        # Initialize sub-components
        self.news_agent = NewsProcessingAgent(model_name=news_model_name)
        self.sentiment_agent = SentimentAnalysisAgent(model_name=view_model_name)
        self.view_agent = ViewGenerationAgent(
            model_name=view_model_name,
            min_confidence=min_confidence,
            max_views_per_day=max_views_per_day,
        )
        self.portfolio_optimizer = PortfolioOptimizer(
            tau=tau,
            risk_aversion=risk_aversion,
            lookback_days=lookback_days,
            risk_free_rate=risk_free_rate,
            optimization_method=optimization_method,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        self.weight_converter = WeightConverter()

        # Optional risk management
        self.enable_risk_management = enable_risk_management
        if enable_risk_management:
            self.risk_manager = RiskManagementAgent(
                model_name=view_model_name,
                min_holding_days=min_holding_days,
                max_position_concentration=max_position_concentration,
                daily_loss_limit=daily_loss_limit,
            )

        # Optional memory / reflection
        if enable_memory:
            self.memory_bank = MemoryBank(persistence_path=memory_file)
            self.reflection_agent = ReflectionAgent(
                model_name=reflection_model_name or view_model_name,
                reflection_interval=reflection_interval,
            )
        else:
            self.memory_bank = None
            self.reflection_agent = None

        # Rolling sentiment tracker (fund_pool set lazily or via warm_up_sentiment)
        self.enable_rolling_sentiment = enable_rolling_sentiment
        self._rolling_alpha_base = alpha_base
        self._rolling_no_news_decay = no_news_decay
        self._rolling_view_flip_threshold = view_flip_threshold
        self._rolling_view_flip_return_threshold = view_flip_return_threshold
        self._rolling_base_magnitude = base_magnitude
        self._rolling_asymmetric_factor = asymmetric_factor
        self.rolling_tracker: Optional[RollingSentimentTracker] = None

        # Turnover reduction
        self.turnover_penalty = turnover_penalty
        self.enable_view_persistence = enable_view_persistence
        self.prev_target_weights: Dict[str, float] = {}
        self.prev_views: Dict[str, Dict] = {}  # {fund_id: {expected_return, confidence, date}}

        # Track history
        self.decision_history: List[Dict] = []
        self.view_history: List[Dict] = []

        logger.info(
            f"Initialized BlackLittermanAgent {agent_id} with "
            f"tau={tau}, method={optimization_method}, "
            f"memory={'on' if enable_memory else 'off'}"
        )

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
        """
        Make investment decision using Black-Litterman framework.

        Args:
            date_to_decision: Current date for decision
            news_data: List of news items
            historical_prices: Historical price data
            current_portfolio: Current portfolio status
            market_data: Additional market data
            fund_pool: List of available fund IDs
            view_platform_trading_history_days: Days of trading history to view

        Returns:
            Dict with final_decision and intermediate_results
        """
        logger.info(f"🤖 {self.agent_id} 开始 Black-Litterman 决策流程...")

        # Step 0: Inject historical principles into market_data for view_agent
        if self.enable_memory and self.memory_bank and not self.memory_bank.is_empty():
            principles = self.memory_bank.get_relevant_memories(fund_pool)
            if principles:
                if market_data is None:
                    market_data = {}
                market_data = dict(market_data)
                market_data["historical_principles"] = principles
                logger.info(
                    f"  记忆库注入 {sum(len(v) for v in principles.values())} 条规律"
                    f"（涉及 {len(principles)} 个key）"
                )

        # Step 1: Process news
        logger.info("📰 处理新闻...")
        processed_news = await self.news_agent.process_news_batch(news_data)
        logger.info(f"  处理完成: {len(processed_news)}/{len(news_data)} 条新闻")

        # Step 2: Analyze sentiment
        logger.info("🎯 分析舆情...")
        sentiment_analysis = await self.sentiment_agent.analyze_sentiment(
            date_to_decision, processed_news, fund_pool
        )
        logger.info(f"  舆情结果: {sentiment_analysis.get('overall_sentiment', 'unknown')}")

        # Step 2.5: Update rolling sentiment tracker
        if self.enable_rolling_sentiment:
            if self.rolling_tracker is None:
                self.rolling_tracker = RollingSentimentTracker(
                    fund_pool=fund_pool,
                    alpha_base=self._rolling_alpha_base,
                    no_news_decay=self._rolling_no_news_decay,
                    view_flip_threshold=self._rolling_view_flip_threshold,
                    view_flip_return_threshold=self._rolling_view_flip_return_threshold,
                    base_magnitude=self._rolling_base_magnitude,
                    asymmetric_factor=self._rolling_asymmetric_factor,
                )
            rolling_scores = self.rolling_tracker.update(sentiment_analysis, date_to_decision)
            trend_directions = self.rolling_tracker.get_trend_directions()
            augmented_sentiment = {
                **sentiment_analysis,
                "rolling_scores": rolling_scores,
                "trend_directions": trend_directions,
                "anchored_returns": {
                    f: self.rolling_tracker.get_anchored_return(f) for f in fund_pool
                },
            }
            log_summary = self.rolling_tracker.get_log_summary(date_to_decision)
            logger.info(
                f"📊 [RollingTracker] bullish={log_summary['bullish']} "
                f"bearish={log_summary['bearish']} "
                f"reversal_events={log_summary['reversal_events']}"
            )
        else:
            augmented_sentiment = sentiment_analysis

        # Step 2.8: Inject yesterday's views for LLM consistency (view persistence)
        if self.enable_view_persistence and self.prev_views:
            if market_data is None:
                market_data = {}
            market_data = dict(market_data)
            market_data["previous_views"] = self.prev_views
            logger.info(f"  注入昨日观点 {len(self.prev_views)} 条（观点持久化）")

        # Step 3: Generate views
        logger.info("💭 生成投资观点...")
        views = await self.view_agent.generate_views(
            date_to_decision=date_to_decision,
            sentiment_analysis=augmented_sentiment,
            historical_prices=historical_prices,
            current_portfolio=current_portfolio,
            fund_pool=fund_pool,
            market_data=market_data,
        )

        if not views:
            logger.warning("  无有效观点，使用保守策略")
            return self._get_hold_decision(current_portfolio, fund_pool, "无有效观点")

        # Step 3.5: Apply view flip filter (soft constraint from rolling sentiment)
        if self.enable_rolling_sentiment and self.rolling_tracker is not None:
            views = self._apply_view_flip_filter(views)
            if not views:
                logger.warning("  ViewFlip过滤后无有效观点，使用保守策略")
                return self._get_hold_decision(current_portfolio, fund_pool, "ViewFlip过滤后无有效观点")

        logger.info(f"  生成 {len(views)} 个观点")
        for view in views:
            logger.info(
                f"    - {view.fund_id}: {view.view_type} "
                f"预期{view.expected_return:.2%}, 置信度{view.confidence:.2f}"
            )

        # Step 4: Optimize portfolio
        logger.info("📊 Black-Litterman 优化...")
        try:
            optimization_result = self.portfolio_optimizer.optimize_portfolio(
                historical_prices=historical_prices,
                views=[v.to_dict() for v in views],
                fund_pool=fund_pool,
                prev_weights=self.prev_target_weights if self.prev_target_weights else None,
                turnover_penalty=self.turnover_penalty,
            )

            target_weights = optimization_result.weights
            logger.info(f"  最优权重: {self._format_weights(target_weights)}")
            logger.info(
                f"  预期收益: {optimization_result.expected_return:.4%}, "
                f"预期风险: {optimization_result.expected_risk:.4%}, "
                f"Sharpe: {optimization_result.sharpe_ratio:.4f}"
            )

        except Exception as e:
            logger.error(f"  优化失败: {e}")
            return self._get_hold_decision(current_portfolio, fund_pool, f"优化失败: {e}")

        # Step 5: Convert weights to trades
        logger.info("🔄 转换权重为交易指令...")
        trades = self.weight_converter.weights_to_trades(
            target_weights=target_weights,
            current_portfolio=current_portfolio,
            fund_pool=fund_pool,
        )

        active_trades = [t for t in trades if t.get("action") != "hold"]
        logger.info(f"  生成 {len(active_trades)} 个交易指令")

        # Step 6: Risk management (optional)
        if self.enable_risk_management and active_trades:
            logger.info("🛡️ 风险管理检查...")

            # Get trading history from platform
            platform_trading_history = current_portfolio.get("trading_history", [])

            risk_result = await self.risk_manager.evaluate_trades(
                proposed_trades=active_trades,
                current_portfolio=current_portfolio,
                sentiment_analysis=sentiment_analysis,
                current_date=date_to_decision,
            )

            approved_trades = risk_result.get("approved_trades", [])
            blocked_trades = risk_result.get("blocked_trades", [])

            logger.info(f"  风险检查: 批准{len(approved_trades)}, 阻止{len(blocked_trades)}")

            # Combine approved trades with holds
            final_trades = approved_trades + [
                t for t in trades if t.get("action") == "hold"
            ]
        else:
            final_trades = trades

        # Build final decision
        final_decision = {
            "reasoning": self._build_reasoning(
                views, optimization_result, sentiment_analysis
            ),
            "chain_of_thought": self._build_chain_of_thought(
                sentiment_analysis, views, optimization_result
            ),
            "trades": final_trades,
            "risk_assessment": "中等风险",
            "target_weights": target_weights,
            "optimization_metrics": {
                "expected_return": optimization_result.expected_return,
                "expected_risk": optimization_result.expected_risk,
                "sharpe_ratio": optimization_result.sharpe_ratio,
                "views_count": optimization_result.views_count,
            },
        }

        # Record decision
        self._record_decision(
            date_to_decision, views, optimization_result, final_decision
        )

        # Update persistent state for next day's decision
        self.prev_target_weights = dict(target_weights)
        self.prev_views = {
            v.fund_id: {
                "expected_return": v.expected_return,
                "confidence": v.confidence,
                "date": date_to_decision,
            }
            for v in views
        }

        logger.info(f"✅ 决策完成: {len(final_trades)} 个指令")

        return {
            "final_decision": final_decision,
            "intermediate_results": {
                "processed_news": processed_news,
                "sentiment_analysis": sentiment_analysis,
                "views": [v.to_dict() for v in views],
                "optimization_result": {
                    "weights": optimization_result.weights,
                    "expected_return": optimization_result.expected_return,
                    "expected_risk": optimization_result.expected_risk,
                    "sharpe_ratio": optimization_result.sharpe_ratio,
                },
            },
        }

    def warm_up_sentiment(
        self, historical_sentiments: List[Dict], fund_pool: List[str]
    ) -> None:
        """Pre-warm rolling sentiment tracker with historical data.

        Call this after agent init and before the backtest loop starts.

        Args:
            historical_sentiments: List of {"date", "sentiment_analysis"} dicts (oldest first)
            fund_pool: Full list of fund IDs for this backtest track
        """
        self.rolling_tracker = RollingSentimentTracker(
            fund_pool=fund_pool,
            alpha_base=self._rolling_alpha_base,
            no_news_decay=self._rolling_no_news_decay,
            view_flip_threshold=self._rolling_view_flip_threshold,
            view_flip_return_threshold=self._rolling_view_flip_return_threshold,
            base_magnitude=self._rolling_base_magnitude,
            asymmetric_factor=self._rolling_asymmetric_factor,
        )
        self.rolling_tracker.warm_up(historical_sentiments)

    def _apply_view_flip_filter(self, views: List[View]) -> List[View]:
        """Apply rolling-sentiment soft constraint to LLM-generated views.

        Two rules:
        1. Direction contradiction + small LLM magnitude → override expected_return
           to Python-anchored value (aligned with rolling direction)
        2. Direction agrees (or rolling is weak) → override magnitude only,
           keep LLM direction but use rolling-anchored magnitude
        """
        result = []
        tracker = self.rolling_tracker

        for view in views:
            fund_id = view.fund_id
            rolling = tracker.rolling_scores.get(fund_id, 0.0)
            anchored = tracker.get_anchored_return(fund_id)

            if tracker.should_override_view(fund_id, view.expected_return):
                # Rolling direction contradicts LLM + LLM magnitude is small → override
                logger.info(
                    f"  🔄 ViewFlip override: {fund_id} "
                    f"LLM={view.expected_return:.3%} vs rolling={rolling:.2f} "
                    f"→ anchored={anchored:.3%}"
                )
                view.expected_return = anchored
                view.confidence = max(0.1, min(0.5, abs(rolling)))
                view.reason += f" [rolling覆盖: 10d_score={rolling:.2f}]"

            elif abs(rolling) >= tracker.view_flip_threshold and rolling * view.expected_return >= 0:
                # Same direction: replace LLM magnitude with Python-anchored magnitude
                view.expected_return = anchored
                view.confidence = max(0.1, min(0.9, abs(rolling)))

            result.append(view)

        # Filter out views with near-zero return or below min confidence
        min_conf = self.view_agent.min_confidence
        return [v for v in result if abs(v.expected_return) > 1e-4 and v.confidence >= min_conf]

    def _get_hold_decision(self, current_portfolio: Dict, fund_pool: List[str], reason: str) -> Dict:
        """Generate a hold-all decision."""
        holdings = current_portfolio.get("holdings", {}).keys()

        trades = []
        for fund_id in fund_pool:
            if fund_id in holdings:
                trades.append({
                    "fund_id": fund_id,
                    "action": "hold",
                    "reason": reason,
                })
            else:
                trades.append({
                    "fund_id": fund_id,
                    "action": "hold",
                    "reason": "当前无持仓",
                })

        final_decision = {
            "reasoning": f"保守持有: {reason}",
            "chain_of_thought": f"由于{reason}，选择保持当前持仓不变",
            "trades": trades,
            "risk_assessment": "低风险",
        }

        return {
            "final_decision": final_decision,
            "intermediate_results": {},
        }

    def _build_reasoning(
        self,
        views: List[View],
        optimization_result: OptimizationResult,
        sentiment_analysis: Dict,
    ) -> str:
        """Build reasoning text for final decision."""
        parts = [
            f"基于{sentiment_analysis.get('summary', '市场舆情')}，",
            f"生成{len(views)}个投资观点。",
            f"通过Black-Litterman模型优化得到Sharpe比率{optimization_result.sharpe_ratio:.2f}的投资组合。",
        ]

        return "".join(parts)

    def _build_chain_of_thought(
        self,
        sentiment_analysis: Dict,
        views: List[View],
        optimization_result: OptimizationResult,
    ) -> str:
        """Build chain of thought text for final decision."""
        parts = []

        # Sentiment
        parts.append(f"1. 舆情分析: {sentiment_analysis.get('overall_sentiment', 'neutral')}")

        # Views
        parts.append(f"2. 投资观点:")
        for i, view in enumerate(views, 1):
            parts.append(
                f"   {i}. {view.fund_id} {view.view_type} "
                f"预期{view.expected_return:.2%} (置信度{view.confidence:.2f})"
            )

        # Optimization
        parts.append(
            f"3. 组合优化: 预期收益{optimization_result.expected_return:.4%}, "
            f"风险{optimization_result.expected_risk:.4%}, "
            f"Sharpe{optimization_result.sharpe_ratio:.4f}"
        )

        return "\n".join(parts)

    def _format_weights(self, weights: Dict[str, float]) -> str:
        """Format weights for logging."""
        top_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        return ", ".join([f"{k}:{v:.2%}" for k, v in top_weights])

    def _record_decision(
        self,
        date: str,
        views: List[View],
        optimization_result: OptimizationResult,
        final_decision: Dict,
    ):
        """Record decision for tracking."""
        self.decision_history.append({
            "date": date,
            "views": [v.to_dict() for v in views],
            "weights": optimization_result.weights,
            "sharpe_ratio": optimization_result.sharpe_ratio,
            "trades": [t for t in final_decision.get("trades", []) if t.get("action") != "hold"],
        })

    async def generate_position_statement(
        self,
        date_to_decision: str,
        news_data: List[Dict],
        historical_prices: Dict,
        current_portfolio: Dict,
        market_data: Dict,
        fund_pool: List[str],
    ) -> PositionStatement:
        """
        Run the full BL pipeline and return a PositionStatement for
        use in the discussion-mode committee workflow.

        Identical to make_decision steps 1-5 but skips internal risk
        management and returns a structured PositionStatement instead
        of the standard make_decision dict.
        """
        from agent_platform.agents.trading_strategy_prompt import BL_POSITION_STATEMENT_PROMPT

        logger.info(f"🗣️ {self.agent_id} 生成初始立场陈述 (委员会讨论模式)...")

        processed_news = await self.news_agent.process_news_batch(news_data)
        sentiment_analysis = await self.sentiment_agent.analyze_sentiment(
            date_to_decision, processed_news, fund_pool
        )
        views = await self.view_agent.generate_views(
            date_to_decision=date_to_decision,
            sentiment_analysis=sentiment_analysis,
            historical_prices=historical_prices,
            current_portfolio=current_portfolio,
            fund_pool=fund_pool,
            market_data=market_data,
        )

        if not views:
            return PositionStatement(
                proposed_trades=[],
                target_weights={},
                views=[],
                sentiment_analysis=sentiment_analysis,
                optimization_metrics={},
                reasoning_text="无有效投资观点，建议持仓不变",
                bl_chain_of_thought="无有效观点，保守策略",
                intermediate_results={"processed_news": processed_news, "sentiment_analysis": sentiment_analysis},
            )

        try:
            optimization_result = self.portfolio_optimizer.optimize_portfolio(
                historical_prices=historical_prices,
                views=[v.to_dict() for v in views],
                fund_pool=fund_pool,
            )
        except Exception as e:
            logger.error(f"  BL优化失败: {e}")
            return PositionStatement(
                proposed_trades=[],
                target_weights={},
                views=[v.to_dict() for v in views],
                sentiment_analysis=sentiment_analysis,
                optimization_metrics={},
                reasoning_text=f"组合优化失败: {e}",
                bl_chain_of_thought=f"优化失败: {e}",
                intermediate_results={"processed_news": processed_news, "sentiment_analysis": sentiment_analysis},
            )

        target_weights = optimization_result.weights
        trades = self.weight_converter.weights_to_trades(
            target_weights=target_weights,
            current_portfolio=current_portfolio,
            fund_pool=fund_pool,
        )
        active_trades = [t for t in trades if t.get("action") != "hold"]

        # Build reasoning text via LLM for a richer position statement
        holdings = current_portfolio.get("holdings", {})
        capital = current_portfolio.get("capital", 0)
        total_value = current_portfolio.get("total_value", 0)
        holdings_text = "\n".join([
            f"- {fid}: 持仓价值 {d.get('value', 0):.2f} 元"
            for fid, d in holdings.items()
        ]) if holdings else "  (空仓)"
        target_weights_text = json.dumps(
            {k: f"{v:.2%}" for k, v in target_weights.items()},
            ensure_ascii=False,
        )
        views_text = "\n".join([
            f"- {v.fund_id}: {v.view_type} 预期{v.expected_return:.2%} 置信度{v.confidence:.2f} - {v.reason}"
            for v in views
        ])
        proposed_trades_text = json.dumps(active_trades, ensure_ascii=False, indent=2)

        prompt = BL_POSITION_STATEMENT_PROMPT.format(
            current_date=date_to_decision,
            total_value=total_value,
            capital=capital,
            holdings_text=holdings_text,
            target_weights_text=target_weights_text,
            expected_return=optimization_result.expected_return,
            expected_risk=optimization_result.expected_risk,
            sharpe_ratio=optimization_result.sharpe_ratio,
            views_count=len(views),
            views_text=views_text,
            sentiment_summary=sentiment_analysis.get("summary", ""),
            proposed_trades_text=proposed_trades_text,
        )

        reasoning_text = self._build_reasoning(views, optimization_result, sentiment_analysis)
        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt, response_format={"type": "json_object"}),
                timeout=60,
            )
            result = json.loads(response.content)
            reasoning_text = result.get("reasoning_text", reasoning_text)
        except Exception as e:
            logger.warning(f"  立场陈述LLM调用失败: {e}，使用默认推理文本")

        return PositionStatement(
            proposed_trades=active_trades,
            target_weights=target_weights,
            views=[v.to_dict() for v in views],
            sentiment_analysis=sentiment_analysis,
            optimization_metrics={
                "expected_return": optimization_result.expected_return,
                "expected_risk": optimization_result.expected_risk,
                "sharpe_ratio": optimization_result.sharpe_ratio,
                "views_count": optimization_result.views_count,
            },
            reasoning_text=reasoning_text,
            bl_chain_of_thought=self._build_chain_of_thought(
                sentiment_analysis, views, optimization_result
            ),
            intermediate_results={
                "processed_news": processed_news,
                "sentiment_analysis": sentiment_analysis,
                "views": [v.to_dict() for v in views],
                "optimization_result": {
                    "weights": optimization_result.weights,
                    "expected_return": optimization_result.expected_return,
                    "expected_risk": optimization_result.expected_risk,
                    "sharpe_ratio": optimization_result.sharpe_ratio,
                },
            },
        )

    async def respond_to_risk_concerns(
        self,
        concerns: "RiskConcerns",
        current_statement: PositionStatement,
        context: "DiscussionContext",
        discussion_history: List["DiscussionRound"],
    ) -> PositionStatement:
        """
        Receive the risk agent's concerns and produce a revised PositionStatement.
        Does not re-run BL math — only makes semantic adjustments to trade
        sizes/actions in response to the risk agent's arguments.
        """
        from agent_platform.agents.trading_strategy_prompt import BL_RESPONSE_TO_CONCERNS_PROMPT

        holdings = context.current_portfolio.get("holdings", {})
        capital = context.current_portfolio.get("capital", 0)
        total_value = context.current_portfolio.get("total_value", 0)

        holdings_text = "\n".join([
            f"- {fid}: 持仓价值 {d.get('value', 0):.2f} 元"
            for fid, d in holdings.items()
        ]) if holdings else "  (空仓)"

        original_proposal_text = (
            f"投资逻辑: {current_statement.reasoning_text}\n"
            f"拟交易: {json.dumps(current_statement.proposed_trades, ensure_ascii=False)}"
        )

        counter_proposal_text = (
            json.dumps(concerns.counter_proposal, ensure_ascii=False)
            if concerns.counter_proposal else "(无反提案)"
        )

        history_parts = []
        for r in discussion_history[-2:]:
            history_parts.append(
                f"第{r.round_number}轮:\n"
                f"  BL方案: {r.position_statement.reasoning_text[:100]}\n"
                f"  风控关切: {r.risk_concerns.concerns_text[:100]}\n"
                f"  让步: {r.position_statement.concessions_made}"
            )
        discussion_history_text = "\n".join(history_parts) if history_parts else "(无历史讨论)"

        round_number = len(discussion_history) + 1
        prompt = BL_RESPONSE_TO_CONCERNS_PROMPT.format(
            round_number=round_number,
            original_proposal_text=original_proposal_text,
            risk_concerns_text=concerns.concerns_text,
            counter_proposal_text=counter_proposal_text,
            discussion_history_text=discussion_history_text,
            total_value=total_value,
            capital=capital,
            holdings_text=holdings_text,
        )

        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt, response_format={"type": "json_object"}),
                timeout=60,
            )
            result = json.loads(response.content)
        except Exception as e:
            logger.warning(f"  BL回应风控关切失败: {e}，保持原方案")
            result = {
                "response_to_concerns": "维持原方案",
                "concessions_made": [],
                "maintained_positions": [],
                "revised_trades": current_statement.proposed_trades,
                "revised_reasoning_text": current_statement.reasoning_text,
            }

        revised_trades = result.get("revised_trades") or current_statement.proposed_trades

        return PositionStatement(
            proposed_trades=revised_trades,
            target_weights=current_statement.target_weights,
            views=current_statement.views,
            sentiment_analysis=current_statement.sentiment_analysis,
            optimization_metrics=current_statement.optimization_metrics,
            reasoning_text=result.get("revised_reasoning_text", current_statement.reasoning_text),
            bl_chain_of_thought=(
                current_statement.bl_chain_of_thought
                + f"\n[修订{round_number}] {result.get('response_to_concerns', '')[:200]}"
            ),
            intermediate_results=current_statement.intermediate_results,
            concessions_made=result.get("concessions_made", []),
        )

    async def record_outcome(
        self,
        date: str,
        views: List[Dict],
        actual_returns: Dict[str, float],
        news_summaries: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record actual market outcomes for a past trading day and optionally
        trigger LLM reflection to update the memory bank.

        Call this at the start of each new trading day (when yesterday's
        close prices become available in historical_prices).

        Args:
            date: The trading date that just ended (yesterday).
            views: The views list that was generated for that date.
            actual_returns: {fund_id: actual_return_decimal} for that date.
            news_summaries: Optional {fund_id: news_text} for richer context.
        """
        if not self.enable_memory or self.reflection_agent is None:
            return

        self.reflection_agent.log_daily_outcome(
            date=date,
            views=views,
            actual_returns=actual_returns,
            news_summaries=news_summaries,
        )

        if self.reflection_agent.should_reflect():
            logger.info("🔍 触发反思机制，开始提炼记忆规律...")
            try:
                updated = await self.reflection_agent.run_reflection(
                    existing_principles=self.memory_bank.get_all_principles(),
                )
                if updated:
                    self.memory_bank.update_principles(updated)
                    logger.info(
                        f"  记忆库更新完成，共 "
                        f"{sum(len(v) for v in self.memory_bank.get_all_principles().values())} 条规律"
                    )
                    if self.memory_bank.persistence_path:
                        self.memory_bank.save_to_disk()
            except Exception as e:
                logger.warning(f"  反思失败，跳过本次更新: {e}")

    def get_decision_history(self) -> List[Dict]:
        """Get decision history."""
        return self.decision_history

    def clear_history(self):
        """Clear decision history."""
        self.decision_history = []
