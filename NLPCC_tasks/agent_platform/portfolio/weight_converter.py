"""
Weight Converter Module

Converts optimal portfolio weights to trade instructions compatible with the backtest engine.
"""

from typing import Dict, List

from loguru import logger


class WeightConverter:
    """Convert optimal weights to trade instructions."""

    def __init__(
        self,
        transaction_cost: float = 0.0001,  # 0.01%
        rebalance_tolerance: float = 0.02,  # 2%
        min_trade_amount: float = 1000,
        min_trade_percentage: float = 0.01,  # 1%
    ):
        """
        Args:
            transaction_cost: Transaction cost rate (0.01% = 0.0001)
            rebalance_tolerance: Minimum weight deviation to trigger trade
            min_trade_amount: Minimum trade amount in CNY
            min_trade_percentage: Minimum trade percentage for sells
        """
        self.transaction_cost = transaction_cost
        self.rebalance_tolerance = rebalance_tolerance
        self.min_trade_amount = min_trade_amount
        self.min_trade_percentage = min_trade_percentage

    def weights_to_trades(
        self,
        target_weights: Dict[str, float],
        current_portfolio: Dict,
        fund_pool: List[str],
    ) -> List[Dict]:
        """
        Convert target weights to trade list.

        Args:
            target_weights: Dict mapping fund_id to target weight
            current_portfolio: Current portfolio status from backtest engine
            fund_pool: List of fund IDs

        Returns:
            List of trade dicts with keys: fund_id, action, amount/percentage, reason
        """
        trades = []
        total_value = current_portfolio.get("total_value", 0)
        capital = current_portfolio.get("capital", 0)
        holdings = current_portfolio.get("holdings", {})

        if total_value <= 0:
            logger.warning(f"Invalid portfolio value: {total_value}")
            return [{"fund_id": f, "action": "hold", "reason": "Invalid portfolio value"} for f in fund_pool]

        # Calculate remaining capital after accounting for transaction costs
        available_cash = capital

        for fund_id in fund_pool:
            target_weight = target_weights.get(fund_id, 0)
            current_value = holdings.get(fund_id, {}).get("value", 0)
            current_weight = current_value / total_value if total_value > 0 else 0

            weight_diff = target_weight - current_weight

            if abs(weight_diff) < self.rebalance_tolerance:
                trades.append({
                    "fund_id": fund_id,
                    "action": "hold",
                    "reason": f"权重偏差{weight_diff:.2%}在容差范围内",
                })
            elif weight_diff > 0:
                # Need to buy
                target_value = total_value * target_weight
                buy_amount = target_value - current_value

                # Account for transaction cost
                required_cash = buy_amount * (1 + self.transaction_cost)

                if required_cash <= available_cash and buy_amount >= self.min_trade_amount:
                    trades.append({
                        "fund_id": fund_id,
                        "action": "buy",
                        "amount": buy_amount,
                        "reason": f"调增至目标权重{target_weight:.2%} (当前: {current_weight:.2%})",
                    })
                    available_cash -= required_cash
                elif buy_amount >= self.min_trade_amount:
                    # Scale down to fit available cash
                    scaled_amount = (available_cash / (1 + self.transaction_cost)) * 0.99
                    if scaled_amount >= self.min_trade_amount:
                        trades.append({
                            "fund_id": fund_id,
                            "action": "buy",
                            "amount": scaled_amount,
                            "reason": f"部分调增至{target_weight:.2%} (现金约束)",
                        })
                        available_cash = 0
                    else:
                        trades.append({
                            "fund_id": fund_id,
                            "action": "hold",
                            "reason": f"资金不足，保持当前权重{current_weight:.2%}",
                        })
                else:
                    trades.append({
                        "fund_id": fund_id,
                        "action": "hold",
                        "reason": f"目标金额{buy_amount:.0f}低于最小交易额",
                    })
            else:
                # Need to sell
                if current_value > 0:
                    sell_percentage = abs(weight_diff) / max(current_weight, 1e-6)
                    sell_percentage = min(sell_percentage, 1.0)

                    if sell_percentage >= self.min_trade_percentage:
                        trades.append({
                            "fund_id": fund_id,
                            "action": "sell",
                            "percentage": sell_percentage,
                            "reason": f"调减至目标权重{target_weight:.2%} (当前: {current_weight:.2%})",
                        })
                    else:
                        trades.append({
                            "fund_id": fund_id,
                            "action": "hold",
                            "reason": f"需要卖出{sell_percentage:.2%}低于最小比例",
                        })
                else:
                    trades.append({
                        "fund_id": fund_id,
                        "action": "hold",
                        "reason": "当前无持仓",
                    })

        # Log summary
        buy_count = sum(1 for t in trades if t.get("action") == "buy")
        sell_count = sum(1 for t in trades if t.get("action") == "sell")
        hold_count = sum(1 for t in trades if t.get("action") == "hold")

        logger.info(
            f"Weight conversion: {buy_count} buys, {sell_count} sells, {hold_count} holds"
        )

        return trades

    def calculate_rebalance_cost(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        total_value: float,
    ) -> float:
        """
        Calculate estimated transaction cost for rebalancing.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            total_value: Total portfolio value

        Returns:
            Estimated transaction cost
        """
        total_cost = 0

        for fund_id, target_weight in target_weights.items():
            current_weight = current_weights.get(fund_id, 0)
            weight_diff = abs(target_weight - current_weight)

            if weight_diff > self.rebalance_tolerance:
                trade_value = total_value * weight_diff
                total_cost += trade_value * self.transaction_cost

        return total_cost

    def adjust_for_transaction_costs(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
        total_value: float,
    ) -> Dict[str, float]:
        """
        Adjust target weights to account for transaction costs.

        This reduces the total turnover to stay within transaction cost budget.

        Args:
            target_weights: Target weights before adjustment
            current_weights: Current portfolio weights
            total_value: Total portfolio value

        Returns:
            Adjusted target weights
        """
        adjusted = target_weights.copy()

        # Calculate total turnover
        total_turnover = sum(
            abs(target_weights.get(f, 0) - current_weights.get(f, 0))
            for f in set(target_weights) | set(current_weights)
        )

        if total_turnover > 0.5:  # More than 50% turnover
            # Scale down adjustments
            scale_factor = 0.5 / total_turnover

            for fund_id in adjusted:
                current = current_weights.get(fund_id, 0)
                target = adjusted[fund_id]
                diff = target - current
                adjusted[fund_id] = current + diff * scale_factor

            # Renormalize
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {f: v / total for f, v in adjusted.items()}

        return adjusted

    def validate_trades(self, trades: List[Dict], capital: float) -> List[Dict]:
        """
        Validate trades against capital constraints.

        Args:
            trades: List of trade dicts
            capital: Available capital

        Returns:
            Validated trades list
        """
        validated = []
        total_buy_amount = 0

        for trade in trades:
            if trade.get("action") == "buy":
                amount = trade.get("amount", 0)
                if total_buy_amount + amount <= capital * 1.001:  # Small buffer
                    validated.append(trade)
                    total_buy_amount += amount
                else:
                    # Adjust amount or convert to hold
                    remaining = capital - total_buy_amount
                    if remaining >= self.min_trade_amount:
                        validated.append({
                            **trade,
                            "amount": remaining,
                            "reason": f"{trade.get('reason', '')} (调整金额)",
                        })
                        total_buy_amount = capital
                    else:
                        validated.append({
                            "fund_id": trade["fund_id"],
                            "action": "hold",
                            "reason": "资金不足",
                        })
            else:
                validated.append(trade)

        return validated
