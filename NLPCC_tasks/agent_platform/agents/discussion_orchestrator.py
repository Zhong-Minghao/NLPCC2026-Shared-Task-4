"""
Investment Committee Discussion Orchestrator

Replaces the sequential BL → Risk pipeline with a multi-round deliberative
discussion where both agents argue, revise, and reach a consensus signal.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from agent_platform.agents.discussion_types import (
    ConsensusResult,
    DiscussionContext,
    DiscussionRound,
    PositionStatement,
    RiskConcerns,
)

if TYPE_CHECKING:
    from agent_platform.agents.black_litterman_agent import BlackLittermanAgent
    from agent_platform.agents.advanced_agents import RiskManagementAgent


class DiscussionOrchestrator:
    """
    Orchestrates a multi-round investment committee discussion between the BL
    decision agent and the risk management agent.

    Each round:
      1. BL agent presents/revises its proposal (PositionStatement)
      2. Risk agent responds with concerns (RiskConcerns)

    Rounds continue until convergence or max_rounds is reached, then
    reach_consensus() is called to produce the final executable trades.
    """

    def __init__(
        self,
        bl_agent: "BlackLittermanAgent",
        risk_agent: "RiskManagementAgent",
        max_rounds: int = 2,
        convergence_threshold: float = 0.8,
    ):
        self.bl_agent = bl_agent
        self.risk_agent = risk_agent
        self.max_rounds = max(1, min(max_rounds, 3))
        self.convergence_threshold = convergence_threshold
        self.discussion_history: List[DiscussionRound] = []

    async def run_discussion(self, context: DiscussionContext) -> Dict:
        """
        Run the full multi-round discussion and return a dict in the same
        schema as BlackLittermanAgent.make_decision(), so existing downstream
        code (trade cleaning, server submission) works without changes.
        """
        self.discussion_history = []

        # Round 1: BL agent generates initial position statement
        logger.info("🗣️ [委员会] 第1轮 - 投资端陈述初始方案...")
        try:
            statement = await self.bl_agent.generate_position_statement(
                date_to_decision=context.date,
                news_data=context.news_data,
                historical_prices=context.historical_prices,
                current_portfolio=context.current_portfolio,
                market_data=context.market_data,
                fund_pool=context.fund_pool,
            )
        except Exception as e:
            logger.error(f"  [委员会] 投资端初始方案生成失败: {e}，回退到顺序模式")
            return await self._fallback_sequential(context)

        if not statement.proposed_trades:
            logger.info("  [委员会] 无拟交易，直接返回持仓方案")
            return self._build_output(statement, None, None)

        # Round 1: Risk agent generates initial concerns
        logger.info("🛡️ [委员会] 第1轮 - 风控端提出关切...")
        try:
            concerns = await self.risk_agent.generate_risk_concerns(
                position_statement=statement,
                context=context,
                discussion_history=[],
            )
        except Exception as e:
            logger.error(f"  [委员会] 风控端关切生成失败: {e}，使用原方案")
            concerns = RiskConcerns(
                hard_blocks=[], soft_concerns=[], counter_proposal=[],
                concerns_text="风控评估失败", risk_level="medium",
                requirements_for_approval="",
            )

        statement.round_number = 1
        concerns.round_number = 1
        self.discussion_history.append(DiscussionRound(
            round_number=1,
            position_statement=statement,
            risk_concerns=concerns,
            timestamp=datetime.now().isoformat(),
        ))
        logger.info(
            f"  [委员会] 第1轮完成: 硬性阻止={len(concerns.hard_blocks)}, "
            f"软性关切={len(concerns.soft_concerns)}, 风险={concerns.risk_level}"
        )

        # Rounds 2..max_rounds
        for round_num in range(2, self.max_rounds + 1):
            if self._is_converged(concerns, self.discussion_history[-2].risk_concerns if len(self.discussion_history) > 1 else None):
                logger.info(f"  [委员会] 提前收敛，跳过第{round_num}轮")
                break

            logger.info(f"🗣️ [委员会] 第{round_num}轮 - 投资端回应关切...")
            try:
                revised = await self.bl_agent.respond_to_risk_concerns(
                    concerns=concerns,
                    current_statement=statement,
                    context=context,
                    discussion_history=self.discussion_history,
                )
                revised.round_number = round_num
            except Exception as e:
                logger.warning(f"  [委员会] 第{round_num}轮BL回应失败: {e}，保持原方案")
                break

            logger.info(f"🛡️ [委员会] 第{round_num}轮 - 风控端重新评估...")
            prior_concerns = concerns
            try:
                concerns = await self.risk_agent.evaluate_revised_proposal(
                    revised_statement=revised,
                    prior_concerns=prior_concerns,
                    context=context,
                    discussion_history=self.discussion_history,
                )
                concerns.round_number = round_num
            except Exception as e:
                logger.warning(f"  [委员会] 第{round_num}轮风控评估失败: {e}")
                concerns = prior_concerns

            statement = revised
            self.discussion_history.append(DiscussionRound(
                round_number=round_num,
                position_statement=statement,
                risk_concerns=concerns,
                timestamp=datetime.now().isoformat(),
            ))
            logger.info(
                f"  [委员会] 第{round_num}轮完成: 让步={statement.concessions_made}, "
                f"硬性阻止={len(concerns.hard_blocks)}, 风险={concerns.risk_level}"
            )

        # Reach consensus
        logger.info(f"🤝 [委员会] 达成共识 (共{len(self.discussion_history)}轮)...")
        try:
            consensus = await self.risk_agent.reach_consensus(
                final_statement=statement,
                final_concerns=concerns,
                context=context,
                discussion_history=self.discussion_history,
            )
        except Exception as e:
            logger.error(f"  [委员会] 共识生成失败: {e}，使用回退逻辑")
            consensus = self._fallback_consensus(statement, concerns)

        logger.info(
            f"✅ [委员会] 共识达成: {len(consensus.final_trades)} 笔交易, "
            f"批准={len(consensus.approved_trades)}, "
            f"修改={len(consensus.modified_trades)}, "
            f"阻止={len(consensus.blocked_trades)}"
        )
        return self._build_output(statement, concerns, consensus)

    def _is_converged(
        self,
        concerns: RiskConcerns,
        prior_concerns: Optional[RiskConcerns],
    ) -> bool:
        if concerns.hard_blocks:
            return False
        high_severity = [c for c in concerns.soft_concerns if c.get("severity") == "high"]
        if not high_severity:
            return True
        if prior_concerns is None:
            return False
        prior_high = [c for c in prior_concerns.soft_concerns if c.get("severity") == "high"]
        if not prior_high:
            return True
        resolved_rate = 1.0 - len(high_severity) / max(len(prior_high), 1)
        return resolved_rate >= self.convergence_threshold

    def _fallback_consensus(
        self,
        statement: PositionStatement,
        concerns: RiskConcerns,
    ) -> ConsensusResult:
        """Simple consensus without LLM: apply hard blocks, pass rest."""
        blocked_ids = {
            (b.get("fund_id"), b.get("action")) for b in concerns.hard_blocks
        }
        approved = []
        blocked = []
        for trade in statement.proposed_trades:
            key = (trade.get("fund_id"), trade.get("action"))
            if key in blocked_ids:
                blocked.append(trade)
            else:
                approved.append(trade)
        return ConsensusResult(
            final_trades=approved,
            approved_trades=approved,
            modified_trades=[],
            blocked_trades=blocked,
            consensus_reasoning=f"经过{len(self.discussion_history)}轮讨论",
            bl_final_thesis=statement.reasoning_text,
            risk_final_assessment=concerns.concerns_text,
            discussion_rounds=len(self.discussion_history),
            key_agreements=[],
            key_compromises=statement.concessions_made,
            risk_level=concerns.risk_level,
            risk_summary=concerns.requirements_for_approval,
        )

    async def _fallback_sequential(self, context: DiscussionContext) -> Dict:
        """Fall back to sequential make_decision + evaluate_trades on error."""
        logger.warning("  [委员会] 回退到顺序模式")
        decision_result = await self.bl_agent.make_decision(
            date_to_decision=context.date,
            news_data=context.news_data,
            historical_prices=context.historical_prices,
            current_portfolio=context.current_portfolio,
            market_data=context.market_data,
            fund_pool=context.fund_pool,
        )
        final_decision = decision_result["final_decision"]
        proposed = [t for t in final_decision.get("trades", []) if t.get("action") != "hold"]
        if proposed:
            sentiment = decision_result.get("intermediate_results", {}).get("sentiment_analysis", {})
            risk_result = await self.risk_agent.evaluate_trades(
                proposed_trades=proposed,
                current_portfolio=context.current_portfolio,
                sentiment_analysis={
                    **sentiment,
                    "technical_factors": context.technical_factors,
                },
                current_date=context.date,
            )
            final_decision["trades"] = (
                risk_result.get("approved_trades", []) + risk_result.get("modified_trades", [])
            )
            final_decision["risk_management"] = {
                "approved": len(risk_result.get("approved_trades", [])),
                "modified": len(risk_result.get("modified_trades", [])),
                "blocked": len(risk_result.get("blocked_trades", [])),
                "risk_summary": risk_result.get("risk_summary", ""),
            }
        final_decision["discussion_mode"] = False
        return decision_result

    def _build_output(
        self,
        statement: PositionStatement,
        concerns: Optional[RiskConcerns],
        consensus: Optional[ConsensusResult],
    ) -> Dict:
        """Convert ConsensusResult to the same dict schema as make_decision()."""
        if consensus is None:
            return {
                "final_decision": {
                    "reasoning": statement.reasoning_text,
                    "chain_of_thought": statement.bl_chain_of_thought,
                    "trades": statement.proposed_trades,
                    "risk_assessment": "无交易",
                    "target_weights": statement.target_weights,
                    "optimization_metrics": statement.optimization_metrics,
                    "discussion_mode": True,
                    "discussion_rounds": 0,
                    "key_agreements": [],
                    "key_compromises": [],
                },
                "intermediate_results": statement.intermediate_results,
            }

        history_summary = [
            {
                "round": r.round_number,
                "concerns_text": r.risk_concerns.concerns_text[:200],
                "concessions_made": r.position_statement.concessions_made,
                "risk_level": r.risk_concerns.risk_level,
            }
            for r in self.discussion_history
        ]

        return {
            "final_decision": {
                "reasoning": consensus.consensus_reasoning,
                "chain_of_thought": (
                    statement.bl_chain_of_thought
                    + f"\n[委员会讨论 {len(self.discussion_history)} 轮]\n"
                    + f"共识: {consensus.key_agreements}\n"
                    + f"让步: {consensus.key_compromises}"
                ),
                "trades": consensus.final_trades,
                "risk_assessment": consensus.risk_level,
                "target_weights": statement.target_weights,
                "optimization_metrics": statement.optimization_metrics,
                "discussion_mode": True,
                "discussion_rounds": consensus.discussion_rounds,
                "key_agreements": consensus.key_agreements,
                "key_compromises": consensus.key_compromises,
                "bl_final_thesis": consensus.bl_final_thesis,
                "risk_final_assessment": consensus.risk_final_assessment,
                "risk_management": {
                    "approved": len(consensus.approved_trades),
                    "modified": len(consensus.modified_trades),
                    "blocked": len(consensus.blocked_trades),
                    "risk_summary": consensus.risk_summary,
                },
            },
            "intermediate_results": {
                **statement.intermediate_results,
                "discussion_history": history_summary,
            },
        }
