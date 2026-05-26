# 代码逐行注释版 — 只看核心逻辑

> 每个文件只展示最关键的 20-30 行，每行都有中文解释。
> 看完这个文件你就能理解整个项目在干什么。

---

## 1. api/handler.py — 顶层入口（整个系统的"大门"）

```python
# 这个函数是整个系统的入口。用户发一个 JSON 请求进来，最终拿到排好序的广告文案列表。
async def handle_request(raw_input, orchestrator):
    """
    raw_input: 用户发来的 JSON（可以是字符串、字节、或已解析的字典）
    orchestrator: 编排器（里面装着 5 个工具）
    返回: {"status_code": 200, "body": {...排好序的候选列表...}}
    """
    request_id = ""  # 请求 ID，用来追踪这次请求的所有日志
    try:
        # 第一步：校验用户输入 + 分配请求 ID
        # 如果 JSON 格式错、字段缺失、枚举值非法，这里就会抛 ValidationError
        request_id, brief, warnings = await parse_and_validate(raw_input)

        # 第二步：把请求 ID 绑定到日志上下文（后续所有日志自动带上这个 ID）
        bind_request_context(request_id=request_id)

        # 第三步：调用编排器，执行完整的"生成→合规→翻译→嵌入→评分→排序"流程
        ranking = await orchestrator.orchestrate(
            brief=brief,           # 校验通过的 Brief 对象
            request_id=request_id, # 请求 ID
            warnings=warnings,     # 校验时产生的警告（比如关键词被截断）
        )

        # 第四步：把结果转成 JSON 字典返回
        body = ranking.model_dump(mode="json")  # Pydantic 模型 → 字典
        return {"status_code": 200, "body": body}  # 成功！

    # ===== 以下是错误处理 =====
    # 每种错误对应一个 HTTP 状态码，用户看到的是结构化的错误信息，不是堆栈跟踪

    except ValidationError as exc:          # 输入校验失败 → 400
        return _handle_agent_error(exc, request_id, "validation_error")

    except GenerationFailureError as exc:   # 文案生成失败（重试 2 次仍失败）→ 502
        return _handle_agent_error(exc, request_id, "generation_failure")

    except DegradedFailureError as exc:     # 补充生成后仍不够 3 个合规候选 → 503
        return _handle_agent_error(exc, request_id, "degraded_failure")

    except CascadeFailureError as exc:      # 累计工具失败超过阈值 → 503
        return _handle_agent_error(exc, request_id, "cascade_failure")

    except Exception as exc:                # 兜底：任何意外错误 → 500
        # 绝不暴露堆栈跟踪给用户，只返回"内部错误"
        return {"status_code": 500, "body": {"error": "Internal server error"}}

    finally:
        clear_request_context()  # 清理日志上下文，防止污染下一个请求
```

---

## 2. api/gateway.py — 输入校验（"保安"）

```python
# 这个函数负责：检查用户发来的 JSON 是否合法，分配请求 ID，保存原始输入
async def parse_and_validate(raw_input):
    """
    返回: (request_id, brief对象, warnings列表)
    如果输入有问题，直接抛 ValidationError，不会返回
    """
    # 1. 分配一个全局唯一的请求 ID（格式：req_时间戳_序号）
    request_id = f"req_{当前时间毫秒}_{递增序号}"

    # 2. 立刻把原始输入存到磁盘（即使后面校验失败，原始数据也不会丢）
    #    存到 traces/<request_id>/brief.json
    _persist_raw_input(request_id, raw_input)

    # 3. 解析 JSON
    data = json.loads(raw_input)  # 如果不是合法 JSON → 抛 MALFORMED_JSON 错误

    # 4. 检查必填字段（campaign_topic, target_platform, target_market, creative_type）
    #    缺失/空/纯空格 → 抛 MISSING_FIELD 错误
    _check_required_fields(data)

    # 5. 检查枚举值是否合法
    #    比如 target_platform 必须是 GOOGLE_ADS / FACEBOOK_ADS / TIKTOK_ADS 之一
    _check_enum_fields(data)

    # 6. 检查 campaign_topic 长度（1-200 字符）
    _check_topic_length(data)

    # 7. 如果关键词超过 20 个，截断到前 20 个，并记录一条 warning
    warnings = []
    data = _truncate_keywords(data, warnings)

    # 8. 用 Pydantic 构造 Brief 对象（会做最终的类型校验）
    brief = Creative_Brief.model_validate(data)

    return request_id, brief, warnings
```

---

## 3. orchestrator/orchestrator.py — 编排器（"指挥官"）

```python
# 这是整个系统的"大脑"，决定调用哪些工具、什么顺序、失败了怎么办
class Orchestrator:
    def __init__(self, creative_generator, compliance_checker,
                 localization_tool, keyword_embedder, cta_optimizer):
        # 把 5 个工具存起来，后面要用
        self._creative_generator = creative_generator
        self._compliance_checker = compliance_checker
        # ... 其他工具

    async def orchestrate(self, brief, request_id, warnings=None):
        """
        brief: 校验通过的创意请求
        返回: AB_Ranking（排好序的候选列表）
        """
        # 1. 加载平台规格（比如 Google Ads 标题限 30 字符）
        platform_spec = load_platform_spec(brief.target_platform)

        # 2. 初始化"工具失败计数器"（用来判断是否触发熔断）
        tool_failure_counter = [0]  # 用列表是因为要在多个函数间共享修改

        all_processed = []  # 存放所有处理完的候选
        refill_count = 0    # 补充生成了几次

        # 3. 生成 + 处理循环（最多 3 轮：1 次初始 + 2 次补充）
        for refill_round in range(3):

            # 3a. 调用 Creative_Generator 生成 ≥5 个候选文案
            gen_output = await self._creative_generator.generate(
                brief=brief,
                platform_spec=platform_spec,
                exclude_copies=[c.source_copy for c in all_processed],  # 避免重复
            )

            # 3b. 检查熔断（如果工具累计失败太多次，直接终止）
            if tool_failure_counter[0] > 30:
                raise CascadeFailureError(failure_count=tool_failure_counter[0])

            # 3c. 并行处理每个候选（这是性能关键！7 个候选同时跑流水线）
            processed = await asyncio.gather(*[
                process_candidate(c, brief, platform_spec, pipeline_deps)
                for c in gen_output.candidates
            ])
            all_processed.extend(processed)

            # 3d. 数一下有多少个候选是合规的（没有 BLOCK 违规）
            compliant_count = sum(1 for c in all_processed if not _has_block(c))

            # 3e. 够 3 个就退出循环
            if compliant_count >= 3:
                break

            refill_count = refill_round + 1  # 记录补充了几次

        # 4. 如果 3 轮之后还不够 3 个合规候选 → 报错
        if compliant_count < 3:
            raise DegradedFailureError(
                candidates_after_filter=compliant_count,
                refill_attempts=refill_count,
            )

        # 5. 调用 Composite Scorer 排序，输出最终结果
        return rank_candidates(all_processed, request_id, ...)
```

---

## 4. orchestrator/pipeline.py — 单候选流水线（"生产线"）

```python
# 每个候选文案都要经过这 5 步处理，就像工厂流水线
async def process_candidate(candidate, brief, platform_spec, deps):
    """
    candidate: 一条原始文案（刚从 Generator 出来的）
    返回: 处理完的 candidate（带上了合规分、翻译、关键词、CTA 评分）
    """

    # ===== Step 1: 合规首检 =====
    # 检查原始文案有没有违禁词（赌博、医疗承诺、歧视等）
    report = await deps.compliance_checker.check(candidate.source_copy, "en")
    candidate.compliance_report = report
    # 即使有 BLOCK 也不在这里剔除，留给最后排序时统一过滤

    # ===== Step 2: 翻译 =====
    # 把英文文案翻译成目标市场的语言（比如 PH 市场 → 菲律宾语 + 英语）
    result = await deps.localization_tool.translate(
        candidate.source_copy,
        target_market=brief.target_market,  # PH / TH / RU / EN_GLOBAL
    )
    candidate.localized_versions = result.localized_versions  # {"fil": "...", "en": "..."}
    candidate.failed_languages = result.failed_languages      # 翻译失败的语言

    # ===== Step 3: 关键词嵌入 =====
    # 把用户给的 SEO 关键词自然地嵌入文案（比如把 "topup" 塞进去）
    result = await deps.keyword_embedder.embed(
        candidate.source_copy,       # 原始文案
        brief.keywords,              # ["topup", "bonus"]
        platform_spec,               # 字符上限
        brief.creative_type,         # HEADLINE / DESCRIPTION / CTA
    )
    candidate.source_copy = result.embedded_copy  # ⚠️ 注意：这里替换了原文案！
    candidate.keyword_coverage = result.keyword_coverage  # 0.0-1.0
    candidate.hit_keywords = result.hit_keywords          # 命中了哪些
    candidate.skipped_keywords = result.skipped_keywords  # 没塞进去的

    # ===== Step 4: 合规复检 =====
    # 为什么要再检查一次？因为 Step 3 嵌入关键词可能引入了新的违禁词！
    # 比如嵌入 "bet" 这个词可能触发赌博类违规
    report = await deps.compliance_checker.check(candidate.source_copy, "en")
    candidate.compliance_report = report  # 用复检结果覆盖首检结果

    # ===== Step 5: CTA 评分 =====
    # 评估文案的"行动号召"有多强（动词力度、紧迫感、收益明确性、文化适配）
    result = await deps.cta_optimizer.optimize(
        candidate,
        brief.target_market,
        "en",
        brief.creative_type,
    )
    candidate.cta_strength_score = result.cta_strength_score  # 0.0-1.0

    return candidate  # 处理完毕，返回给 Orchestrator
```

**关键理解**：
- 每一步失败都被 try/except 包住（上面省略了），失败时设默认值 + 加 warning
- Step 3 会**修改 source_copy**，所以 Step 4 必须在 Step 3 之后
- 这 5 步是**串行**的（一个候选内部），但**多个候选之间是并行**的

---

## 5. orchestrator/composite_scorer.py — 评分排序（"裁判"）

```python
# 这个函数做三件事：过滤违规的、算综合分、排序
def rank_candidates(candidates, request_id, ...):
    """
    candidates: 所有处理完的候选（可能有些带 BLOCK 违规）
    返回: AB_Ranking（只包含合规的候选，按综合分从高到低排）
    """

    # 1. 过滤：把含有 BLOCK 违规的候选踢掉
    survivors = [c for c in candidates if not _has_block(c)]

    # 2. 算分：对每个存活的候选计算综合评分
    for c in survivors:
        compliance = c.compliance_report.compliance_score  # 合规分 0-1
        keyword = c.keyword_coverage                       # 关键词覆盖率 0-1
        cta = c.cta_strength_score                         # CTA 强度 0-1

        # 公式：合规占一半权重（因为违规的代价最大），关键词和 CTA 各占 1/4
        c.composite_score = 0.5 * compliance + 0.25 * keyword + 0.25 * cta

    # 3. 排序：综合分高的排前面
    #    如果综合分一样，合规分高的排前面
    #    如果还一样，CTA 强的排前面
    #    如果还一样，先生成的排前面
    survivors.sort(key=lambda c: (
        -c.composite_score,                          # 综合分 降序
        -c.compliance_report.compliance_score,       # 合规分 降序
        -c.cta_strength_score,                       # CTA   降序
        c.generation_index,                          # 生成顺序 升序
    ))

    # 4. 打包成 AB_Ranking 对象返回
    return AB_Ranking(
        ranked_candidates=survivors,
        total_candidates_generated=总共生成了多少,
        total_candidates_filtered_out=被踢掉了多少,
        ...
    )
```

---

## 6. 五个工具的"一句话总结"

| 工具 | 一句话 | 输入 | 输出 |
|------|--------|------|------|
| **Creative_Generator** | 调 LLM 生成 5-7 条广告文案 | Brief + 平台规格 | 候选列表 |
| **Compliance_Checker** | 用本地词典扫描违禁词 | 一条文案 | 合规报告（分数+违规列表） |
| **Localization_Tool** | 调 LLM 翻译成多语言 | 一条文案 + 目标市场 | 各语言译文 |
| **Keyword_Embedder** | 调 LLM 把关键词自然嵌入文案 | 文案 + 关键词 | 嵌入后的文案 + 覆盖率 |
| **CTA_Optimizer** | 调 LLM 评估 CTA 强度 | 一条文案 | 4 维度评分 |

---

## 7. 数据怎么流的（一图看懂）

```
用户 JSON
  │
  ├─ campaign_topic: "Game topup bonus"
  ├─ target_platform: "GOOGLE_ADS"
  ├─ target_market: "PH"
  ├─ creative_type: "HEADLINE"
  └─ keywords: ["topup"]
       │
       ▼ parse_and_validate()
       │
  Creative_Brief 对象
       │
       ▼ orchestrate()
       │
       ▼ creative_generator.generate()
       │
  7 个 Creative_Candidate（每个只有 source_copy）
       │
       ▼ process_candidate() × 7（并行）
       │
  7 个 Creative_Candidate（每个带上了：
       ├─ compliance_report（合规报告）
       ├─ localized_versions（多语言译文）
       ├─ keyword_coverage（关键词覆盖率）
       └─ cta_strength_score（CTA 评分）
       │
       ▼ rank_candidates()
       │
  AB_Ranking（过滤 BLOCK → 算综合分 → 排序）
       │
       ▼ model_dump(mode="json")
       │
  最终 JSON 响应 → 返回给用户
```

---

## 8. 错误处理怎么工作的

```
任何一步出错
    │
    ├─ 工具级别的错误（比如 LLM 超时）
    │   → 被 pipeline.py 的 try/except 捕获
    │   → 设默认值（score=0.0）+ 加 warning
    │   → 候选不被剔除，继续走下一步
    │   → tool_failure_counter += 1
    │
    ├─ 累计失败太多（counter > 30）
    │   → orchestrator 抛 CascadeFailureError
    │   → handler.py 捕获 → 返回 HTTP 503
    │
    ├─ 生成器彻底失败（重试 2 次仍失败）
    │   → 抛 GenerationFailureError
    │   → handler.py 捕获 → 返回 HTTP 502
    │
    ├─ 补充生成后仍不够 3 个合规候选
    │   → 抛 DegradedFailureError
    │   → handler.py 捕获 → 返回 HTTP 503
    │
    └─ 输入校验失败
        → 抛 ValidationError
        → handler.py 捕获 → 返回 HTTP 400
```

**核心设计思想**：单个工具失败不影响整体，只有累计失败太多才终止。
