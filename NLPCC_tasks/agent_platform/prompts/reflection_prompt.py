"""
Reflection prompt template for the Black-Litterman memory system.

The LLM is given recent raw observations (predicted vs actual returns)
and existing principles, and asked to distill/update actionable rules
organized by fund or sector key.
"""

REFLECTION_PROMPT = """
你是一个专业的量化投资策略分析师，负责对AI投资决策进行系统性复盘。

**任务**：基于最近 {window_days} 个交易日的观点预测与实际表现，提炼可复用的投资规律，
更新各板块/基金的经验库，帮助未来做出更准确的判断。

---

**最近观测记录（预测 vs 实际）**：
{observations_text}

---

**当前已有的经验规律**（如有）：
{existing_principles_text}

---

**分析要求**：

1. **归因分析**：对每个出现预测偏差的基金/板块，分析偏差模式：
   - 是否系统性高估/低估某类新闻信号的影响？
   - 方向判断正确但幅度偏差大？还是方向本身就判断错误？
   - 是否与特定市场状态（整体涨跌、成交量）相关？

2. **规律提炼**：按基金代码或板块（如"科技_半导体"、"金融_证券"）归类，
   每个key最多输出 {max_principles} 条规律，保留最重要、最具可操作性的。

3. **规律格式要求**：
   - 规律要**可操作**，必须包含明确建议，例如：
     - "XXX板块政策利好新闻的实际涨幅约为预期的60%，建议将置信度下调0.15"
     - "XXX板块在市场整体下跌时，往往会放大跌幅，建议同步下调预期收益20%"
   - 不要写泛泛而谈的规律（如"市场波动较大"）
   - 已有规律若有足够新证据支持则保留，若被新证据否定则删除或修正

4. **置信度评估**：每条规律给出0-1的可信度（基于支持观测数量和一致性）

---

**输出格式（严格JSON，不要有注释）**：
{{
    "updated_principles": {{
        "基金代码或板块key": [
            {{
                "principle": "规律描述 + 具体可操作建议",
                "confidence": 0.7,
                "evidence_count": 6
            }}
        ]
    }},
    "reflection_summary": "本次复盘的核心发现，2-3句话总结"
}}

只输出JSON，不要有其他内容：
"""
