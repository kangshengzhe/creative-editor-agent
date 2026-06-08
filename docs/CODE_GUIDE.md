# 代码导读 — 45 分钟理解整个项目

> 按"一条请求从进来到出去"的数据流顺序阅读，不要逐文件看。

---

## 全局架构图（先记住这张）

```
用户提交 Brief (JSON)
    │
    ▼
┌─ api/gateway.py ─────────────────────┐
│  校验字段 → 分配 request_id → 持久化  │
└──────────────────────────────────────┘
    │
    ▼
┌─ orchestrator/orchestrator.py ───────┐
│  加载平台规格 → 调 Generator → 并行   │
│  处理每个候选 → 检查够不够 3 个 →     │
│  不够就补充生成 → 最后排序            │
└──────────────────────────────────────┘
    │ 对每个候选调用 ↓
    ▼
┌─ orchestrator/pipeline.py ───────────┐
│  Compliance首检 → Localization →     │
│  Keyword嵌入 → Compliance复检 →      │
│  CTA评分                             │
└──────────────────────────────────────┘
    │
    ▼
┌─ orchestrator/composite_scorer.py ───┐
│  综合评分 = 0.5×合规 + 0.25×关键词    │
│           + 0.25×CTA                 │
│  排序 → 输出 AB_Ranking              │
└──────────────────────────────────────┘
```

---

## 阅读顺序（按调用链走 6 个核心函数）

### 第 1 站：demo_full.py → main()

**作用**：组装所有工具实例，调用 handle_request

**关键代码**（伪代码简化）：
```python
llm = RealLLMClient()                    # 从 .env 读 token
compliance_checker = ComplianceChecker()  # 本地词典模式
orchestrator = Orchestrator(              # 把 5 个工具注入
    creative_generator, compliance_checker,
    localization_tool, keyword_embedder, cta_optimizer
)
result = await handle_request(brief, orchestrator)
```

**你要理解的**：所有工具通过构造函数注入，不是全局变量。

---

### 第 2 站：api/handler.py → handle_request()

**作用**：顶层入口，串联校验→编排→错误处理

**关键逻辑**：
```python
async def handle_request(raw_input, orchestrator):
    request_id, brief, warnings = await parse_and_validate(raw_input)
    ranking = await orchestrator.orchestrate(brief, request_id, warnings)
    return {"status_code": 200, "body": ranking.model_dump()}
    # 如果任何一步抛异常 → 捕获 → 转成 ErrorResponse
```

**你要理解的**：4 种异常各对应一个 HTTP 状态码（400/502/503/503）。

---

### 第 3 站：api/gateway.py → parse_and_validate()

**作用**：校验 JSON + 分配 request_id + 持久化原始输入

**关键逻辑**：
```python
async def parse_and_validate(raw_input):
    request_id = "req_<时间戳>_<序号>"
    _persist_raw_input(request_id, raw_input)  # 先存，再校验
    data = json.loads(raw_input)               # JSON 解析
    _check_required_fields(data)               # 必填字段
    _check_enum_fields(data)                   # 枚举值合法性
    _check_topic_length(data)                  # 1-200 字符
    data = _truncate_keywords(data, warnings)  # >20 截断
    brief = Creative_Brief.model_validate(data)
    return request_id, brief, warnings
```

**你要理解的**：持久化在校验之前（保证即使校验失败，原始输入也不丢）。

---

### 第 4 站：orchestrator/orchestrator.py → orchestrate()

**作用**：核心编排 — 生成 + 并行流水线 + 补充生成 + 排序

**关键逻辑**：
```python
async def orchestrate(brief, request_id, warnings):
    platform_spec = load_platform_spec(brief.target_platform)
    tool_failure_counter = [0]  # 共享计数器

    for refill_round in range(3):  # 最多 3 轮（初始 + 2 次补充）
        # 1. 生成候选
        gen_output = await creative_generator.generate(brief, platform_spec)

        # 2. 并行处理每个候选
        processed = await asyncio.gather(*[
            process_candidate(c, brief, platform_spec, deps)
            for c in gen_output.candidates
        ])

        # 3. 检查熔断（累计失败 > 30 次）
        if tool_failure_counter[0] > 30: raise CascadeFailureError

        # 4. 够 3 个合规候选就退出
        if count_compliant(all_processed) >= 3: break

    # 5. 排序输出
    return rank_candidates(all_processed, request_id, ...)
```

**你要理解的**：
- `asyncio.gather` = 并行跑所有候选的流水线
- 补充生成回路 = 如果合规过滤后不够 3 个，再生成一批
- 熔断 = 防止工具连续失败导致无限重试

---

### 第 5 站：orchestrator/pipeline.py → process_candidate()

**作用**：单个候选的 5 步处理流水线

**关键逻辑**：
```python
async def process_candidate(candidate, brief, platform_spec, deps):
    # Step 1: 合规首检
    report = await deps.compliance_checker.check(candidate.source_copy)
    candidate.compliance_report = report

    # Step 2: 翻译
    result = await deps.localization_tool.translate(candidate.source_copy, ...)
    candidate.localized_versions = result.localized_versions

    # Step 3: 关键词覆盖检查（不重写文案，只报告哪些关键词命中/缺失）
    result = await deps.keyword_embedder.embed(candidate.source_copy, keywords, ...)
    # source_copy 不变；keyword_coverage / hit_keywords / skipped_keywords 被填充

    # Step 4: 合规复检（因为嵌入可能引入新违规）
    report = await deps.compliance_checker.check(candidate.source_copy)
    candidate.compliance_report = report  # ← 用复检结果覆盖首检

    # Step 5: CTA 评分
    result = await deps.cta_optimizer.optimize(candidate, ...)
    candidate.cta_strength_score = result.cta_strength_score

    return candidate
```

**你要理解的**：
- Step 3 只**检查**关键词覆盖率，不重写文案（宁可标记缺失也不强行塞词导致文案变形）
- 每步失败都被 try/except 捕获，设默认值 + 加 warning，不会中断流水线

---

### 第 6 站：orchestrator/composite_scorer.py → rank_candidates()

**作用**：评分 + 过滤 BLOCK + 排序

**关键逻辑**：
```python
def rank_candidates(candidates, request_id, ...):
    # 1. 过滤掉含 BLOCK 违规的候选
    survivors = [c for c in candidates if not has_block(c)]

    # 2. 计算综合评分
    for c in survivors:
        c.composite_score = (
            0.5 * c.compliance_report.compliance_score
            + 0.25 * c.keyword_coverage
            + 0.25 * c.cta_strength_score
        )

    # 3. 排序（三级 tie-break）
    survivors.sort(key=lambda c: (
        -c.composite_score,      # 综合分高的在前
        -c.compliance_score,     # 合规分高的在前
        -c.cta_strength_score,   # CTA 强的在前
        c.generation_index       # 生成顺序早的在前
    ))

    return AB_Ranking(ranked_candidates=survivors, ...)
```

**你要理解的**：这是纯计算，没有 LLM 调用，没有 IO。

---

## 5 个工具速查表（只看输入/输出）

| 工具 | 输入 | 输出 | 调 LLM 吗 |
|------|------|------|-----------|
| Creative_Generator | brief + platform_spec | 15条标题/10条描述/5条CTA候选 | ✅ 多次(按角度并发) |
| Compliance_Checker | 文案 + 语言 | 合规报告（分数+违规列表） | ❌ 本地词典 |
| Localization_Tool | 文案 + 目标市场 | 多语言译文 | ✅ 每语言 1 次 |
| Keyword_Embedder | 文案 + 关键词列表 | 覆盖率报告（不重写文案）| ❌ 只检查匹配 |
| CTA_Optimizer | 候选 + 市场 + 类型 | CTA 强度评分 | ✅ 1 次 |

---

## 自测：能回答这 5 个问题就算理解了

1. **一条 brief 进来，第一步做什么？**
   → `api/gateway.py` 校验 + 分配 request_id + 持久化原始 Brief

2. **Creative_Generator 生成的候选怎么保证不重复？**
   → strip + lower + 去标点后比较（`_normalise` 函数）

3. **为什么 Keyword_Embedder 之后要再跑一次 Compliance？**
   → Keyword_Embedder 现在只做覆盖率检查，不重写文案，所以复检和首检结果相同；复检步骤保留是为了兼容未来如果重新开启嵌入改写时的安全保障

4. **如果 CTA_Optimizer 失败了会怎样？**
   → `cta_strength_score = 0.0`，candidate 不被剔除，warning 记录原因

5. **综合评分怎么算的？**
   → `0.5 × compliance_score + 0.25 × keyword_coverage + 0.25 × cta_strength_score`

---

## 文件清单（按重要性排序）

**必看（核心链路）**：
1. `api/handler.py` — 顶层入口
2. `api/gateway.py` — 校验
3. `orchestrator/orchestrator.py` — 编排
4. `orchestrator/pipeline.py` — 单候选流水线
5. `orchestrator/composite_scorer.py` — 评分排序

**了解即可（工具实现）**：
6. `tools/creative_generator.py` — 文案生成
7. `tools/compliance_checker.py` — 合规检查
8. `tools/localization_tool.py` — 翻译
9. `tools/keyword_embedder.py` — 关键词嵌入
10. `tools/cta_optimizer.py` — CTA 评分

**不用看（基础设施）**：
- `models/` — 数据结构定义，用到时再查
- `config/` — 配置加载，不影响理解主流程
- `llm/` — LLM 客户端，知道它"调 API 返回文本"就够了
- `errors/` — 异常类定义
- `observability/` — 日志和 trace
