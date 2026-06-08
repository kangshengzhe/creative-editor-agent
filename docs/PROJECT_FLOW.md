# Creative Editor Agent — 项目流程思维导图

> Coco AI 创意编辑工具 Agent 的端到端流程梳理

---

## 一、整体架构总览

```mermaid
mindmap
  root((Creative Editor Agent))
    输入层
      运营人员 Brief
        活动主题
        目标平台 GOOGLE_ADS / FB / TT
        目标市场 PH / TH / RU / EN
        文案类型 HEADLINE / DESCRIPTION / CTA / LONG_COPY
        SEO 关键词
        卖点
    校验层
      API Gateway
        JSON 解析
        必填字段校验
        枚举校验
        长度校验 1-200 字符
        关键词截断 ≤20
        分配 request_id
        持久化原始 Brief
    编排层
      Orchestrator
        加载 Platform_Spec
        生成 + 流水线 + 补充
        熔断检测 失败>5 次
        降级失败检测 候选<3
      Pipeline 单候选并行
      Composite Scorer 综合评分
    工具层
      Creative_Generator 文案生成
      Compliance_Checker 合规检查
      Localization_Tool 多语言翻译
      Keyword_Embedder 关键词嵌入
      CTA_Optimizer CTA 优化
    基础设施
      LLM 客户端
        RealLLMClient 阿里云百炼 DashScope
        MockLLMClient 测试用
      数据模型 Pydantic v2
      违禁词词典 4 语言
      Trace 可观测性
      错误处理 4 类错误码
    输出层
      AB_Ranking
        ranked_candidates 排序候选
        composite_score 综合评分
        compliance_report 合规报告
        localized_versions 多语言版本
        warnings 警告
```

---

## 二、端到端处理流程

```mermaid
flowchart TD
    Start([运营提交 Creative_Brief]) --> Gateway[API Gateway 校验 + 分配 request_id]

    Gateway -->|校验失败| ErrV[VALIDATION_ERROR 400]
    Gateway -->|校验通过| Persist[持久化原始 Brief]

    Persist --> LoadSpec[加载 Platform_Spec]
    LoadSpec --> Generate[Creative_Generator 生成 ≥5 候选]

    Generate -->|失败重试 2 次仍失败| ErrG[GENERATION_FAILURE 502]
    Generate -->|成功| ParPipeline{并行流水线 asyncio.gather}

    ParPipeline --> P1[候选 1 流水线]
    ParPipeline --> P2[候选 2 流水线]
    ParPipeline --> Pn[候选 N 流水线]

    P1 --> Compliance1[Compliance 首检]
    Compliance1 --> Localize[Localization 多语言翻译]
    Localize --> Embed[Keyword_Embedder 嵌入关键词]
    Embed --> Compliance2[Compliance 复检 嵌入可能引入新违规]
    Compliance2 --> CTA[CTA_Optimizer CTA 优化]

    CTA --> Collect[汇总所有候选]

    Collect --> Filter{合规过滤后<br/>≥3 个候选?}
    Filter -->|否, 重试 < 2| Generate
    Filter -->|否, 重试 = 2| ErrD[DEGRADED_FAILURE 503]
    Filter -->|是| Score[Composite Scorer 综合评分]

    Score --> Rank[按综合评分排序]
    Rank --> Output([返回 AB_Ranking])

    Cascade{累计失败<br/>>5 次?} -->|是| ErrC[CASCADE_FAILURE 503]

    style ErrV fill:#fee
    style ErrG fill:#fee
    style ErrD fill:#fee
    style ErrC fill:#fee
    style Output fill:#efe
```

---

## 三、五大核心工具职责拆解

```mermaid
mindmap
  root((五大核心工具))
    Creative_Generator
      输入 Creative_Brief + Platform_Spec
      输出 ≥5 个互不重复候选
      约束
        长度 ≤ char_limit
        差异化 角度/语气/卖点
        重试 ≤2 次
      调用 LLM 1 次
    Compliance_Checker
      输入 文案 + 语言
      输出 Compliance_Report
        compliance_score 0.0-1.0
        violations 列表
      检测策略 混合
        本地词典 BLOCK 类违禁词
        LLM 语义 WARN 类夸大
      评分公式
        含 BLOCK → 0.0
        N 个 WARN → max 0.1, 1 - 0.2N
        无违规 → 1.0
    Localization_Tool
      输入 源文案 + 目标市场
      输出 多语言版本
      市场映射
        PH → fil + en
        TH → th + en
        RU → ru + en
        EN_GLOBAL → en
      占位符保留 {name}
      货币符号 ₱ ฿ ₽ $
      日期格式 各市场专属
    Keyword_Embedder
      输入 文案 + SEO 关键词
      输出 嵌入后文案 + 覆盖率
      约束
        词边界匹配 大小写不敏感
        防堆砌 连续 ≤2 次
        长度 ≤ char_limit
      已存在关键词不重嵌
      覆盖率 = 命中数 / 总数
    CTA_Optimizer
      创意类型为 CTA 时
        生成 ≥5 个 CTA 候选
        合规过滤 BLOCK 剔除
      其他类型时
        识别文案末尾 CTA
        评分
      四维度评分
        verb_strength 动词号召力
        urgency 紧迫感
        benefit_clarity 收益明确
        cultural_fit 文化适配
```

---

## 四、综合评分公式

```mermaid
flowchart LR
    A[compliance_score] -->|权重 0.5| Sum((加权求和))
    B[keyword_coverage] -->|权重 0.25| Sum
    C[cta_strength_score] -->|权重 0.25| Sum
    Sum --> Composite[composite_score]

    Composite --> Sort{排序键}
    Sort --> S1[1. composite_score 降]
    S1 --> S2[2. compliance_score 降]
    S2 --> S3[3. cta_strength_score 降]
    S3 --> S4[4. generation_index 升]
    S4 --> Result[AB_Ranking]
```

**公式**：

```
composite_score = 0.5 × compliance_score
                + 0.25 × keyword_coverage
                + 0.25 × cta_strength_score
```

---

## 五、错误处理分层

```mermaid
mindmap
  root((错误处理 4 类))
    VALIDATION_ERROR 400
      触发 字段缺失/枚举非法/JSON 不合法
      子类型
        MISSING_FIELD
        INVALID_ENUM
        INVALID_LENGTH
        MALFORMED_JSON
      处理 立即返回错误
    GENERATION_FAILURE 502
      触发 Creative_Generator 重试 2 次仍失败
      处理 立即返回, 不返回部分结果
    DEGRADED_FAILURE 503
      触发 补充生成 2 次后合规候选仍 <3
      处理 返回部分结果信息
        candidates_after_filter
        refill_attempts
    CASCADE_FAILURE 503
      触发 累计工具失败 >5 次
      处理 立即终止请求 防止雪崩
    工具级降级 不抛出
      Compliance 失败 → 标 WARN, 不剔除
      Localization 单语言失败 → 跳过, 其他继续
      Keyword 失败 → coverage=0.0
      CTA 失败 → score=0.0
```

---

## 六、项目目录结构

```mermaid
graph TD
    Root[creative_editor_agent/]

    Root --> Spec[.kiro/specs/creative-editor-agent/]
    Spec --> SR[requirements.md]
    Spec --> SD[design.md]
    Spec --> ST[tasks.md]

    Root --> Src[src/creative_agent/]
    Src --> Api[api/]
    Api --> AG[gateway.py 校验]
    Api --> AH[handler.py 顶层入口]

    Src --> Orch[orchestrator/]
    Orch --> OO[orchestrator.py 主流程]
    Orch --> OP[pipeline.py 单候选流水线]
    Orch --> OS[composite_scorer.py 评分排序]

    Src --> Tools[tools/]
    Tools --> T1[creative_generator.py]
    Tools --> T2[compliance_checker.py]
    Tools --> T3[localization_tool.py]
    Tools --> T4[keyword_embedder.py]
    Tools --> T5[cta_optimizer.py]

    Src --> Models[models/]
    Models --> M1[brief.py]
    Models --> M2[candidate.py]
    Models --> M3[compliance.py]
    Models --> M4[ranking.py]
    Models --> M5[platform_spec.py]
    Models --> M6[enums.py]

    Src --> Config[config/]
    Config --> C1[platform_specs/ 3 个 JSON]
    Config --> C2[forbidden_terms/ 4 语言 JSON]
    Config --> C3[platform_loader.py]
    Config --> C4[forbidden_loader.py]

    Src --> Llm[llm/]
    Llm --> L1[client.py 抽象基类]
    Llm --> L2[real_client.py 阿里云百炼 DashScope]
    Llm --> L3[mock_client.py 测试用]

    Src --> Obs[observability/]
    Obs --> O1[logging.py 结构化日志]
    Obs --> O2[trace.py 链路追踪]

    Src --> Err[errors/]
    Err --> E1[codes.py 异常类]
    Err --> E2[responses.py ErrorResponse]

    Root --> Demo[demo_minimal.py 端到端验证]
    Root --> SetupEnv[setup_env.py token 安全配置]
    Root --> Sec[SECURITY.md]
    Root --> Env[.env 不进 git]
    Root --> Git[.gitignore 已加固]
```

---

## 七、开发进度时间线

```mermaid
timeline
    title 项目开发关键里程碑
    阶段 1 需求分析 : Requirements 9 个功能需求 : 10 个正确性属性
    阶段 2 技术设计 : Design 系统架构 : 工具接口 : 数据模型
    阶段 3 任务规划 : Tasks 61 个任务 : 17 核心 + 44 可选测试
    阶段 4 代码实现 : 项目骨架 : 数据模型 : 5 大工具 : 编排层 : API 层
    阶段 5 安全配置 : git + .env : setup_env.py : SECURITY.md
    阶段 6 端到端验证 : demo_minimal 跑通 : qwen3.7-max 调用 35-45s : compliance_score 1.0
    阶段 7 交付验收 : 前端界面 : 多语言 20 市场 : 213 测试全过
```

---

## 八、关键技术决策

```mermaid
mindmap
  root((技术决策))
    架构模式
      Orchestrator-Driven 而非 ReAct 自由调度
      原因 时延和合规是硬约束需要显式控制
    并行策略
      候选维度并行 asyncio.gather
      语言维度并行
      关键串行点 Embedder → Compliance 复检
    Compliance 实现
      混合策略
      本地词典 保证 BLOCK 单调性
      LLM 语义 识别夸大表述
    Platform_Spec
      配置化 JSON
      原因 平台规格频繁变更避免硬编码
    降级策略
      工具级降级 不抛出 标记 warning
      全局熔断 累计 >5 次终止
    LLM 解耦
      抽象基类 LLMClient
      RealLLMClient + MockLLMClient
      原因 工具与 LLM 提供商松耦合
    评分公式
      固定权重 0.5 / 0.25 / 0.25
      原因 简单可解释易于 PBT 验证
```

---

## 九、本次实习的交付物清单

| 类型 | 文件 | 说明 |
|------|------|------|
| 规范文档 | `requirements.md` | 9 个功能需求 + 10 个正确性属性，EARS 模式 |
| 规范文档 | `design.md` | 系统架构 + 工具接口 + Mermaid 流程图 |
| 规范文档 | `tasks.md` | 61 个任务，分 9 波依赖关系 |
| 核心代码 | `src/creative_agent/` | 完整 Python 实现 |
| 配置数据 | `platform_specs/*.json` | 3 个广告平台规格 |
| 配置数据 | `forbidden_terms/*.json` | 4 语言违禁词词典 |
| 端到端 demo | `demo_minimal.py` | 已验证跑通 |
| 安全工具 | `setup_env.py` | 交互式 token 配置 |
| 安全文档 | `SECURITY.md` | API key 保护规范 |
| 项目说明 | `README.md` | 快速上手 |

---

## 十、实习汇报可以怎么讲

> 我用 6 个阶段把"广告创意自动生成"这个业务需求落地成了一个可运行的 Agent。
>
> **业务价值**：原本运营人员一小时手写 5-10 条广告文案，现在 AI 一次能产出 5+ 条互不重复的候选，并且自动做合规检查（避免违反 Google Ads 政策被拒审）、多语言翻译（覆盖菲律宾/泰国/俄罗斯市场）、SEO 关键词嵌入和 CTA 优化。
>
> **技术亮点**：
> 1. **工程化的 Spec 驱动**：先写需求 → 设计 → 任务，再实现，每一步都有可追溯文档
> 2. **Orchestrator-Driven 架构**：5 个工具用显式编排，时延和合规可控
> 3. **正确性属性先行**：10 个 PBT 属性在需求阶段就锁定，比如"合规过滤后无 BLOCK"、"评分值域 0-1"、"翻译占位符保留"
> 4. **优雅降级**：单工具失败不影响整体，累计失败超阈值才熔断
> 5. **真实 LLM 验证**：接入阿里云百炼 qwen3.7-max，关闭深度思考模式后批量生成 35-45 秒，213 个测试全部通过
