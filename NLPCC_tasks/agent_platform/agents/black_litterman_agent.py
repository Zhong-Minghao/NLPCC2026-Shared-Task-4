"""
Black-Litterman Agent

Integrates view generation, Black-Litterman optimization, and weight conversion
to provide a complete portfolio management solution.
"""

import asyncio
import os
from typing import Dict, List, Optional

from loguru import logger

from agent_platform.agents.view_generation_agent import ViewGenerationAgent, View
from agent_platform.agents.advanced_agents import (
    NewsProcessingAgent,
    SentimentAnalysisAgent,
)
from agent_platform.portfolio_optimization import PortfolioOptimizer, OptimizationResult
from agent_platform.portfolio.weight_converter import WeightConverter
from agent_platform.agents.advanced_agents import RiskManagementAgent


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
        """
        self.agent_id = agent_id

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

        # Track history
        self.decision_history: List[Dict] = []
        self.view_history: List[Dict] = []

        logger.info(
            f"Initialized BlackLittermanAgent {agent_id} with "
            f"tau={tau}, method={optimization_method}"
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

        # Step 3: Generate views
        logger.info("💭 生成投资观点...")
        views = await self.view_agent.generate_views(
            date_to_decision=date_to_decision,
            sentiment_analysis=sentiment_analysis,
            historical_prices=historical_prices,
            current_portfolio=current_portfolio,
            fund_pool=fund_pool,
            market_data=market_data,
        )

        if not views:
            logger.warning("  无有效观点，使用保守策略")
            return self._get_hold_decision(current_portfolio, fund_pool, "无有效观点")

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

    def get_decision_history(self) -> List[Dict]:
        """Get decision history."""
        return self.decision_history

    def clear_history(self):
        """Clear decision history."""
        self.decision_history = []
