# AI 交易智能体框架详解（NLPCC2026 Shared Task 4）

本文基于仓库当前 Starter Kit 代码，系统梳理比赛中的“AI 投资顾问 Agent”是如何运行的，重点覆盖：

1. 如何调用 Agent
2. 给 Agent 喂了哪些数据
3. Agent 必须输出什么
4. 平台如何执行、记账、评估
5. 可扩展点与常见踩坑

---

## 1. 总体架构：Server 平台 + Agent 客户端

整套系统是一个典型的“回测服务端 + 决策客户端”双端架构：

- **Server 侧（FastAPI）**：负责管理 session、控制交易日推进、提供防泄露数据、执行交易撮合、记录结果。
- **Agent 侧（Python Client + LLM）**：负责读取当天可见数据，调用大模型生成交易决策，再把指令回传给 Server。

入口上：

- 服务端入口：`start_server.py`（启动 FastAPI 并初始化 DataLoader）。
- Agent 端示例入口：`agent_platform/demo_backtest.py`（单次回测 demo）。

---

## 2. 先说你最关心的：怎么“调用 Agent”

以 `demo_backtest.py` 为例，调用链如下：

1. 构造配置 `build_config(args)`（时间区间、资金、基金池、新闻源、lookback 等）。
2. `client.start_backtest(config)` 启动回测会话，拿到 `session_id` + 当天初始数据。
3. `get_advanced_agent(...)` 创建一个 `AdvancedTradingAgent` 实例。
4. 每个交易日循环：
   - 拉取组合状态：`get_backtest_status`
   - 拉取防泄露历史行情：`get_historical_prices`
   - 调用 `agent.make_decision(...)` 产出交易 JSON
   - 过滤掉 `hold` 后，通过 `submit_trade_with_decision` 提交交易
   - `get_next_day_data` 推进到下一日
5. 回测结束后 `get_backtest_results` 拉最终结果并落盘。

你可以把它理解为：

> Server 是环境（Environment），Agent 是策略（Policy），每天做一次 observe → think → act。

---

## 3. Agent 内部不是一个模型，而是三段式流水线

`AdvancedTradingAgent` 由三个子 Agent 串联：

1. **NewsProcessingAgent**（新闻摘要）
2. **SentimentAnalysisAgent**（舆情/基金影响分析）
3. **TradingStrategyAgent**（最终交易决策）

### 3.1 NewsProcessingAgent

输入：当天可见新闻列表（多源）

处理：

- 对每条新闻调用 LLM 做 1-2 句摘要。
- 有全局缓存：`GLOBAL_NEWS_CACHE`（键是日期+标题+源+排名），避免重复耗 token。
- 有并发去重：同一新闻如果同时被处理，后续协程等待同一 future。
- 有重试和超时机制。

输出：`processed_news`，每条包含 `thedate/title/source/ranking/summary` 等。

### 3.2 SentimentAnalysisAgent

输入：

- `processed_news`
- `fund_pool`（可投基金）
- 决策日期

处理：

- 给 LLM 一个严格 JSON 模板，要求输出：
  - `overall_sentiment`
  - `fund_analysis`（逐基金正/负/中性、原因、置信度）
  - `summary`
- 用 `CustomJsonOutputParser` 解析文本为 JSON（支持去 markdown code fence、去注释、`ast.literal_eval`）。

输出：结构化舆情分析 JSON。

### 3.3 TradingStrategyAgent

输入：

- 舆情分析结果
- 历史行情（防泄露）
- 当前持仓（capital + holdings）
- 可投资基金池
- 最近若干天平台成交历史（提示 Agent 不要过度交易）

处理：

- 把以上信息拼装为交易 prompt（`BASELINE_TRADING_PROMPT`）。
- 明确约束：
  - buy 用 `amount`
  - sell 用 `percentage`
  - 手续费 0.01%
  - **当天卖出资金不能用于当天买入**（防止指令失败）
- LLM 输出严格 JSON，经 parser 解析。

输出：最终决策 JSON（`reasoning`, `chain_of_thought`, `trades`, `risk_assessment`）。

---

## 4. “给 Agent 看了什么数据”——输入字段全拆开

## 4.1 每日主输入（来自 `session.start()` / `session.get_day_data()`）

Server 在每天给的数据包含：

- `date`
- `market_data`（这里是按 `pre_k_days` 取的历史切片）
- `news`
- `portfolio`
- `is_finished`

但 demo 实际上还会额外调用：

- `get_backtest_status`（拿更全组合状态）
- `get_historical_prices(lookback_days)`（标准防泄露行情接口）

所以 Agent 的“实际可见输入”是上述两部分组合。

## 4.2 行情数据（防 future leakage）

`DataLoader.get_historical_prices(...)` 的关键规则：

- 返回过去 `lookback_days - 1` 个交易日的完整 OHLC 等字段。
- 对**当前决策日**，只给 `open`。
- 当前日 `close/high/low/change/pct_change` 都是 `None`。

这意味着 Agent 无法直接看到“今天收盘结果”，从机制上阻断了日内未来信息泄露。

## 4.3 新闻数据（时间截断 + 排名截断）

`DataLoader.get_news(...)` 的规则：

- 新闻源默认：`caixin/tiantian/sinafinance/tencent`
- 窗口：按交易日回看 `pre_k_days`
- 截断：对当日新闻只保留 **15:00 前** 发布
- 热度：保留 `RANKING <= top_rank`
- 最终按排名排序

即：Agent 只能看到“截至决策时点可获得”的热点新闻。

## 4.4 组合与历史成交

Agent prompt 还显式喂了：

- 可用现金 `capital`
- 当前持仓（每个 fund 的持仓价值 + 当前价）
- 最近若干日已成交记录（按日期聚合）

这部分非常关键：它让模型知道“现金约束”和“自己最近干过什么”。

---

## 5. “Agent 要输出什么数据”——输出协议

在策略 prompt 中，要求输出严格 JSON：

- `reasoning`：决策理由
- `chain_of_thought`：逐步思考过程（示例里显式要求）
- `trades`: 列表，每条为
  - buy: `{fund_id, action:"buy", amount, reason}`
  - sell: `{fund_id, action:"sell", percentage, reason}`
  - hold: `{fund_id, action:"hold", reason}`
- `risk_assessment`

demo 在提交前会过滤 `hold`，只把 buy/sell 发送给交易接口。

---

## 6. 平台如何校验并执行交易

服务端交易模型 `Trade` 有 schema 约束：

- buy 必须有 `amount` 且 `percentage` 必须为空
- sell 必须有 `percentage` 且 `amount` 必须为空
- sell 的 `percentage` 必须在 `(0,1]`

执行引擎 `BacktestSession.submit_trades()` 的关键逻辑：

1. 先把已有持仓按当日涨跌更新（`_update_holdings_value`）。
2. 交易执行顺序：**先 buy，再 sell**。
   - 这与注释中的比赛规则一致：当日卖出现金不能回流到当日买入。
3. 手续费：`commission = 0.0001 * amount`（买卖同费率）。
4. buy 时扣现金，给持仓价值增加 `amount - commission`。
5. sell 时按持仓价值 * 百分比卖出，回笼现金 `value_to_sell - commission`。
6. 每笔交易都会记录 success/fail 原因。

---

## 7. 回测状态如何推进（一天一天走）

- `start()`：记录初始组合价值并返回首日数据。
- 每日流程（典型）：
  1. Agent 拿数据做决策
  2. 提交交易
  3. `next_day()` 推到下个交易日
- 到 `end_date` 时 `finish()`：
  - 最后再结算一次持仓
  - 记录最终净值
  - 计算收益指标并保存结果 JSON

---

## 8. 结果输出与评估字段

`get_results()` 返回结果中核心字段：

- `performance`：
  - `total_return`
  - `final_portfolio_value`
  - `annualized_return`
- `portfolio_value_history`
- `transaction_history`
- `agent_decisions`
- `daily_portfolio_snapshots`
- `fund_performance`

此外 API 层还提供可视化用接口：

- portfolio history
- trading signals
- benchmark/csi300 对比序列

---

## 9. 接口级数据交互（你写 Agent 必看的 I/O）

常用 API：

- `POST /api/backtest/start`
- `POST /api/backtest/{session_id}/trade`
- `GET /api/backtest/{session_id}/status`
- `GET /api/backtest/{session_id}/historical_prices`
- `POST /api/backtest/{session_id}/news`
- `GET /api/backtest/{session_id}/next_day`
- `GET /api/backtest/results/{session_id}`

你可以把每日 I/O 记成：

- 输入：`news + historical_prices + portfolio/status + (optional market_data)`
- 输出：`trades[]`（buy/sell）+ 可选 agent_decision 解释字段

---

## 10. 逐字段 I/O 样例（按“一天决策”完整走一遍）

下面给你一个“可以直接照着实现”的最小闭环样例：

- 场景日期：`2024-03-18`
- 目标：演示 **Server -> Agent 输入字段**、**Agent -> Server 输出字段**、以及 **执行后回传字段**。

> 注意：下面是“字段级示例”，数值仅用于说明结构和约束，不代表真实收益。

### 10.1 `start_backtest` 请求（你先把配置发给平台）

```json
{
  "start_date": "2024-01-02",
  "end_date": "2024-06-28",
  "initial_capital": 1000000,
  "fund_pool": ["512010.SH", "515220.SH", "159992.SZ"],
  "pre_k_days": 20,
  "news_sources": ["caixin", "sinafinance", "tencent"],
  "news_top_n": 30,
  "model_name": "gpt-4o-mini"
}
```

字段说明（最关键的几个）：

- `fund_pool`: Agent 可交易标的全集（后续 trade 的 `fund_id` 必须来自这里）。
- `pre_k_days`: Server 每天向后看的历史窗口大小。
- `news_top_n`: 当天新闻会先按热度排序再截断。

### 10.2 `start_backtest` 响应（平台回给你的首日可见数据）

```json
{
  "session_id": "bt_9f2c...",
  "current_date": "2024-01-02",
  "is_finished": false,
  "portfolio": {
    "capital": 1000000,
    "holdings": {}
  },
  "news": [...],
  "market_data": {...}
}
```

你至少要保存两个东西：

- `session_id`（后续所有接口都要带）
- `current_date`（用于日志、缓存 key、结果对齐）

### 10.3 某交易日拉取输入：`status + historical_prices + news`

#### A) `GET /status`（账户约束）

```json
{
  "date": "2024-03-18",
  "portfolio": {
    "capital": 624300.55,
    "holdings": {
      "512010.SH": {"value": 210000.11, "current_price": 1.023},
      "515220.SH": {"value": 165699.34, "current_price": 0.978}
    }
  },
  "recent_transactions": [
    {"date": "2024-03-14", "fund_id": "512010.SH", "action": "buy", "amount": 80000},
    {"date": "2024-03-15", "fund_id": "515220.SH", "action": "sell", "percentage": 0.2}
  ]
}
```

#### B) `GET /historical_prices?lookback_days=20`（价格特征 + 防泄露）

```json
{
  "date": "2024-03-18",
  "historical_prices": {
    "512010.SH": [
      {"date": "2024-03-15", "open": 1.018, "high": 1.025, "low": 1.012, "close": 1.023, "pct_change": 0.39},
      {"date": "2024-03-18", "open": 1.021, "high": null, "low": null, "close": null, "pct_change": null}
    ],
    "515220.SH": [
      {"date": "2024-03-15", "open": 0.981, "high": 0.984, "low": 0.975, "close": 0.978, "pct_change": -0.31},
      {"date": "2024-03-18", "open": 0.976, "high": null, "low": null, "close": null, "pct_change": null}
    ]
  }
}
```

逐字段重点：

- 历史日：`open/high/low/close/pct_change` 完整。
- 当日：只给 `open`；`close/high/low/pct_change = null`。

#### C) `POST /news`（新闻输入）

```json
{
  "date": "2024-03-18",
  "news": [
    {
      "publish_time": "2024-03-18 09:12:00",
      "title": "算力产业链景气延续",
      "source": "caixin",
      "ranking": 3,
      "content": "..."
    },
    {
      "publish_time": "2024-03-18 14:47:00",
      "title": "新能源板块成交放量",
      "source": "sinafinance",
      "ranking": 8,
      "content": "..."
    }
  ]
}
```

逐字段重点：

- 同日 15:00 之后新闻默认不在可见集。
- `ranking` 越小热度越高，通常先被纳入 prompt。

### 10.4 Agent 内部中间输出（建议你保留日志）

`AdvancedTradingAgent` 的中间产物建议最少记录两层：

1. `processed_news`（摘要后）
2. `sentiment_result`（逐基金情绪）

示例：

```json
{
  "sentiment_result": {
    "overall_sentiment": "neutral_to_positive",
    "fund_analysis": {
      "512010.SH": {"sentiment": "positive", "confidence": 0.74, "reason": "算力链条新闻密集且偏正面"},
      "515220.SH": {"sentiment": "neutral", "confidence": 0.58, "reason": "短期波动较大，缺乏新增催化"},
      "159992.SZ": {"sentiment": "negative", "confidence": 0.63, "reason": "板块估值压缩预期"}
    },
    "summary": "市场风险偏好修复，但分化明显"
  }
}
```

### 10.5 Agent 最终输出（提交前 JSON 协议）

```json
{
  "reasoning": "算力相关情绪偏正，组合现金充足；新能源信号中性，先降仓控制回撤。",
  "chain_of_thought": "1) 资金约束检查 2) 新闻与价格共振检查 3) 风险预算后生成交易",
  "trades": [
    {"fund_id": "512010.SH", "action": "buy", "amount": 120000, "reason": "情绪与近5日趋势同向"},
    {"fund_id": "515220.SH", "action": "sell", "percentage": 0.25, "reason": "降低波动敞口"},
    {"fund_id": "159992.SZ", "action": "hold", "reason": "等待明确信号"}
  ],
  "risk_assessment": "单标的仓位不超过35%，保留至少40%现金应对波动"
}
```

逐字段硬约束（最容易错）：

- `action=buy` 必须给 `amount`，不能给 `percentage`。
- `action=sell` 必须给 `percentage`（`0 < percentage <= 1`），不能给 `amount`。
- `action=hold` 不会被 demo 提交到交易接口（会先被过滤）。

### 10.6 提交到交易接口（demo 过滤 hold 之后）

最终提交 payload（示例）：

```json
[
  {"fund_id": "512010.SH", "action": "buy", "amount": 120000, "reason": "情绪与近5日趋势同向"},
  {"fund_id": "515220.SH", "action": "sell", "percentage": 0.25, "reason": "降低波动敞口"}
]
```

### 10.7 平台执行回传（你要消费的结果字段）

```json
{
  "date": "2024-03-18",
  "execution_results": [
    {"fund_id": "512010.SH", "action": "buy", "success": true, "executed_amount": 120000, "commission": 12.0},
    {"fund_id": "515220.SH", "action": "sell", "success": true, "executed_percentage": 0.25, "commission": 4.14}
  ],
  "portfolio_after_trade": {
    "capital": 545984.41,
    "holdings": {
      "512010.SH": {"value": 329988.11},
      "515220.SH": {"value": 124274.50}
    }
  }
}
```

你在工程上至少要做三件事：

1. 把 `success=false` 的交易原因打日志（常见是现金不足/字段不合法）。
2. 用 `portfolio_after_trade` 覆盖本地状态，避免“本地账本”和平台不一致。
3. 保存“输入快照 + 模型输出 + 执行结果”三联日志，便于复盘与调参。

---

## 11. 这个框架的设计优点

1. **防泄露规则在平台层保证**：不是靠选手自觉。
2. **Agent 模块化**：新闻摘要/舆情/交易策略可独立替换。
3. **支持断点恢复和日志追溯**：适合长周期回测与复盘。
4. **交易语义简单统一**：buy=金额、sell=持仓比例。

---

## 12. 你在比赛中最该优化的点（实战建议）

1. **减少无效交易**：手续费虽小，但日频高换手会侵蚀收益。
2. **改进新闻选择机制**：不是所有 Top 新闻都和基金池相关。
3. **Prompt 加入资金管理约束**：例如单标的最大仓位、最小交易阈值。
4. **把“平台成交历史”用好**：控制追涨杀跌和反复打脸。
5. **输出鲁棒 JSON**：避免 parser 失败触发保守兜底（全 hold）。

---

## 13. 一句话总结

这个 Starter Kit 的本质是：

> 在严格防未来泄露的数据口径下，让 LLM Agent 每天基于“新闻 + 历史价格 + 当前仓位”输出结构化交易指令，并由统一回测引擎执行和评估。

如果你愿意，我下一步可以再给你一版“参赛可落地”的升级蓝图：

- 从当前三段式 Agent 升级到“多专家投票 + 风控裁决”
- 每日特征工程模板（可直接喂模型）
- 回测日志自动诊断脚本（定位亏损日的原因）
