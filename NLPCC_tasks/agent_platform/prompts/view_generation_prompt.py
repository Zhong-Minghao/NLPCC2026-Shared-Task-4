"""
View Generation Prompt Templates for Black-Litterman Model

This module contains prompt templates for generating investment views
that will be used as input to the Black-Litterman portfolio optimization model.
"""

VIEW_GENERATION_PROMPT = """
你是一个专业的投资观点分析师。你的任务是基于市场舆情、历史价格和当前持仓，
生成结构化的投资观点，用于Black-Litterman投资组合优化。

**可投资基金及其核心意义**:
{funds_text}

**今天是{date_to_decision}，你当前的投资组合状态**:
- 可用现金: {capital:.2f} 元
- 当前持仓: {holdings_text}

**市场舆情分析**:
- 整体摘要: {sentiment_summary}
- 详细舆情: {sentiment_details}

**10天滚动叙事趋势（Python积累，不受单日噪音影响）**:
{rolling_sentiment_text}
说明：滚动分数 ∈ [-1,+1]，正值=持续正面叙事，负值=持续负面叙事。
你的观点方向**应与 [bullish/bearish] 标签一致**，除非今日新闻有极强的逆转信号（预期收益率绝对值 > 1%）。

**历史价格走势 (最近几个交易日)**:
{history_text}

**历史经验规律 (来自记忆库，按基金/板块)**:
{historical_principles_text}

**昨日投资观点（维持一致性参考）**:
{previous_views_text}
说明：以上是昨日的观点方向。**若今日无明确反向信号，请保持原方向**；若需改变方向，
请在 reason 中明确说明改变的具体原因（如：今日出现X新闻、滚动趋势已反转等）。

**观点生成要求**:
1. 生成3-5个明确的投资观点
2. 观点类型:
   - "absolute": 对某基金的绝对收益观点 (如: 预期沪深300未来收益2%)
   - "relative": 相对收益观点 (如: 预期创业板相对沪深300超额收益1.5%)
3. 预期收益范围: -5% 到 +5% 之间（日收益率）
4. 置信度: 0.0-1.0，反映你对观点的确定程度（建议范围: 0.3-0.8）
5. 理由: 简要说明观点依据（基于舆情、价格趋势等）

**注意事项**:
1. 观点应该有明确的依据（新闻舆情或价格趋势）
2. 避免极端预测（如单日涨跌超过5%）
3. 置信度应该反映观点的确定性（有明确新闻支撑的可给予较高置信度）
4. 可以同时看多和看空不同资产
5. 相对观点有助于分散风险
6. **若记忆库中有对应基金/板块的历史规律，请参考调整置信度或预期收益幅度**（规律仅供参考，不强制覆盖你的判断）
7. **优先维持昨日观点方向，仅在有明确新信号时改变方向，以减少不必要的换手**

**输出格式 (严格JSON)**:
{{
    "views": [
        {{
            "fund_id": "基金代码",
            "view_type": "absolute",
            "expected_return": 0.02,
            "confidence": 0.7,
            "reason": "观点理由，基于某某新闻/趋势"
        }},
        {{
            "fund_id": "基金代码",
            "view_type": "relative",
            "expected_return": 0.015,
            "confidence": 0.6,
            "benchmark_fund_id": "基准基金代码",
            "reason": "相对观点理由"
        }}
    ],
    "overall_sentiment": "positive/negative/neutral",
    "reasoning": "整体思路说明"
}}

请给出你的投资观点（只输出JSON）：
"""


VIEW_GENERATION_WITH_HISTORY_PROMPT = """
你是一个专业的投资观点分析师。你的任务是基于市场舆情、历史价格和当前持仓，
生成结构化的投资观点，用于Black-Litterman投资组合优化。

**可投资基金及其核心意义**:
{funds_text}

**今天是{date_to_decision}，你当前的投资组合状态**:
- 可用现金: {capital:.2f} 元
- 当前持仓: {holdings_text}

**最近的投资观点和实际表现回顾**:
{view_performance_summary}

**市场舆情分析**:
- 整体摘要: {sentiment_summary}
- 详细舆情: {sentiment_details}

**历史价格走势 (最近几个交易日)**:
{history_text}

**历史经验规律 (来自记忆库，按基金/板块)**:
{historical_principles_text}

**观点生成要求**:
1. 生成3-5个明确的投资观点
2. 观点类型:
   - "absolute": 对某基金的绝对收益观点
   - "relative": 相对收益观点
3. 预期收益范围: -5% 到 +5% 之间
4. 置信度: 0.0-1.0
5. 理由: 简要说明观点依据
6. **若记忆库中有对应基金/板块的历史规律，请参考调整置信度或预期收益幅度**

**输出格式 (严格JSON)**:
{{
    "views": [
        {{
            "fund_id": "基金代码",
            "view_type": "absolute",
            "expected_return": 0.02,
            "confidence": 0.7,
            "reason": "观点理由"
        }}
    ],
    "overall_sentiment": "positive/negative/neutral",
    "reasoning": "整体思路说明"
}}

请给出你的投资观点（只输出JSON）：
"""


SIMPLIFIED_VIEW_GENERATION_PROMPT = """
基于以下信息生成3-5个投资观点（用于Black-Litterman模型）：

**基金列表**: {funds_list}
**日期**: {date_to_decision}
**舆情**: {sentiment_summary}
**价格趋势**: {price_summary}

**输出JSON格式**:
{{
    "views": [
        {{"fund_id": "代码", "view_type": "absolute", "expected_return": 数值, "confidence": 0-1, "reason": "理由"}},
        ...
    ]
}}

只输出JSON：
"""
