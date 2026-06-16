"""
PerNewsETFScoringAgent

对单条新闻评估其与各行业ETF的相关性和影响方向，用于IC检验的单条新闻信号生成。

输出：relevance（相关度）+ direction（方向）+ confidence（置信度）+ score（综合分）
"""
import asyncio
import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from loguru import logger

load_dotenv()

from .fund_info import FUND_INFO

DIRECTION_VALUE = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


class PerNewsETFScoringAgent:
    """对单条新闻给出对每个ETF的 relevance × direction × confidence 评分"""

    def __init__(self, model_name: str = "EFundGPT-pro"):
        self.llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            temperature=1,
        )

    async def score_article(
        self,
        news: Dict,
        fund_pool: List[str],
        llm_retry: int = 3,
    ) -> Dict:
        """
        对单条（已摘要）新闻，给出对各ETF的评分。

        Args:
            news: processed news dict，含 title/summary/source/thedate/ranking
            fund_pool: ETF代码列表

        Returns:
            {
                etf_code: {
                    "relevance": float,  # 0-1 新闻与该行业的相关程度
                    "direction": str,    # positive / negative / neutral
                    "confidence": float, # 0-1 方向判断的置信度
                    "score": float,      # relevance × direction_value × confidence
                }
            }
        """
        if not news.get("summary") and not news.get("title"):
            return {etf: _null_score() for etf in fund_pool}

        funds_text = "\n".join([
            f"- {etf} ({FUND_INFO.get(etf, {}).get('name', etf)}): "
            f"{FUND_INFO.get(etf, {}).get('scope', '')}"
            for etf in fund_pool
        ])

        prompt = f"""你是一个专业的金融分析师。请评估以下单条新闻对各行业ETF的相关性和影响方向。

**新闻信息**:
日期: {news.get('thedate', '未知')}
来源: {news.get('source', '未知')}
标题: {news.get('title', '')}
摘要: {news.get('summary', '')}

**行业ETF列表**:
{funds_text}

**评估规则**:
1. relevance（相关度）: 0.0=完全无关, 1.0=高度相关，代表该新闻直接影响该行业板块
2. direction（方向）: positive=利好 / negative=利空 / neutral=中性或不确定
3. confidence（置信度）: 对direction判断的确定程度，0.0-1.0
4. 若relevance < 0.2，则direction固定为neutral，confidence固定为0.0
5. 只输出JSON，不要解释文字

**输出格式（严格JSON，覆盖所有ETF代码）**:
{{
    "512880.SH": {{"relevance": 0.0, "direction": "neutral", "confidence": 0.0}},
    ...
}}
"""
        for attempt in range(llm_retry):
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(prompt, response_format={"type": "json_object"}),
                    timeout=60,
                )
                raw = json.loads(response.content)
                return _parse_scores(raw, fund_pool)
            except asyncio.TimeoutError:
                logger.warning(
                    f"[PerNewsETFScoringAgent] Timeout attempt {attempt + 1}/{llm_retry} "
                    f"title='{news.get('title', '')[:40]}'"
                )
                if attempt == llm_retry - 1:
                    return {etf: _null_score() for etf in fund_pool}
            except Exception as e:
                logger.warning(
                    f"[PerNewsETFScoringAgent] Error attempt {attempt + 1}/{llm_retry}: {e}"
                )
                if attempt == llm_retry - 1:
                    return {etf: _null_score() for etf in fund_pool}

        return {etf: _null_score() for etf in fund_pool}


def _parse_scores(raw: Dict, fund_pool: List[str]) -> Dict:
    scores = {}
    for etf in fund_pool:
        entry = raw.get(etf, {})
        relevance = float(entry.get("relevance", 0.0))
        direction = entry.get("direction", "neutral")
        confidence = float(entry.get("confidence", 0.0))
        direction_val = DIRECTION_VALUE.get(direction, 0.0)
        scores[etf] = {
            "relevance": round(min(max(relevance, 0.0), 1.0), 4),
            "direction": direction,
            "confidence": round(min(max(confidence, 0.0), 1.0), 4),
            "score": round(relevance * direction_val * confidence, 4),
        }
    return scores


def _null_score() -> Dict:
    return {"relevance": 0.0, "direction": "neutral", "confidence": 0.0, "score": 0.0}
