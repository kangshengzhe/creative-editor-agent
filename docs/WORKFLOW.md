# 程序工作流说明 (Creative Editor Agent)

这份文档帮你从零理解：**一个请求从浏览器点「生成」开始，到拿到 20 条标题，中间到底发生了什么。**

> 图是用 Mermaid 画的。在 Kiro 里打开本文件，点右上角的预览（或 `Ctrl+Shift+V`）就能看到渲染后的图。

---

## 一、整体思维导图（一张图看懂全貌）

```mermaid
mindmap
  root((创意生成<br/>Agent))
    入口层
      frontend.html 前端页面
      server.py FastAPI 服务
      /api/creative 接口
    校验层 Gateway
      解析 JSON
      必填字段检查
      枚举值检查
      关键词截断到20
      存档 brief.json
    编排层 Orchestrator
      算出目标数量
        标题20 描述15 CTA10
      生成+补充循环 最多3轮
      调用各工具
      最后排序打包
    生成层 Generator
      角度拆分 AngleSplitter
        4到8个卖点角度
      语言选择 LanguagePrompt
        非英语母语生成
      每角度生成3条
      文本去重
    流水线 Pipeline 每条候选
      合规初检 Compliance
      本地化翻译 Localization
      关键词嵌入 Keyword
      合规复检
      CTA优化
    差异化三道防线
      文本去重 逐字重复
      语义去重 近义重复
      角度生成 结构差异
    收尾 Ranking
      综合打分
      多样性加成
      截断到目标数量
      输出 AB_Ranking
```

---

## 二、主流程图（请求从头到尾怎么走）

```mermaid
flowchart TD
    U[用户在前端填 Brief<br/>点 生成广告创意] --> S[server.py 收到 POST /api/creative]
    S --> G{API Gateway 校验}
    G -->|校验失败| ERR[返回错误 JSON<br/>400 等]
    G -->|校验通过| O[Orchestrator 开始编排]

    O --> TC[算出目标数量 target_count<br/>标题=20 描述=15 CTA=10]
    TC --> LOOP{生成+补充循环<br/>最多 3 轮}

    LOOP --> ANG[AngleSplitter 拆出 4-8 个角度]
    ANG --> GEN[CreativeGenerator 按角度轮转生成<br/>每角度每次 3 条]
    GEN --> TDEDUP[文本去重<br/>去掉逐字重复]
    TDEDUP --> SEM[语义去重 SemanticDiversityChecker<br/>轻量词法相似度 > 0.50 判为近义重复并丢弃]
    SEM --> PIPE[逐条过流水线]

    subgraph PIPE_DETAIL[每条候选的流水线]
        direction TB
        C1[合规初检] --> L1[本地化翻译<br/>源语言=主语言则跳过]
        L1 --> K1[关键词嵌入]
        K1 --> C2[合规复检]
        C2 --> CTA1[CTA 强度优化]
    end

    PIPE --> PIPE_DETAIL
    PIPE_DETAIL --> COUNT{合规候选数 >= 目标?}
    COUNT -->|否, 还有轮次| LOOP
    COUNT -->|是| RANK[Composite Scorer 排序]
    COUNT -->|3轮跑完仍不够| CHECK{够最低 3 条?}

    CHECK -->|否| FAIL[报降级失败<br/>DegradedFailureError]
    CHECK -->|是但不足目标| WARN[加 under-filled 警告<br/>继续排序]
    WARN --> RANK

    RANK --> MULT[乘多样性加成<br/>角度越多分越高]
    MULT --> TRUNC[截断到目标数量<br/>超量择优取前 N]
    TRUNC --> OUT[输出 AB_Ranking<br/>含 20 条标题+评分+翻译]
    OUT --> FE[前端展示结果]
```

---

## 三、五个核心工具各干什么

```mermaid
flowchart LR
    subgraph 工具箱
        T1[CreativeGenerator<br/>生成文案候选]
        T2[ComplianceChecker<br/>合规检查<br/>拦违禁词/承诺]
        T3[LocalizationTool<br/>翻译成目标市场语言]
        T4[KeywordEmbedder<br/>把SEO关键词嵌进文案]
        T5[CTAOptimizer<br/>给行动号召打分]
    end

    subgraph 新增组件
        N1[AngleSplitter<br/>卖点拆成4-8角度]
        N2[LanguagePromptSelector<br/>选母语生成提示词]
        N3[SemanticDiversityChecker<br/>嵌入算近义重复]
        N4[DisplayWidthCalculator<br/>算CJK显示宽度]
    end

    N1 --> T1
    N2 --> T1
    T1 --> N3
    N4 --> T1
```

---

## 四、关键概念（小白也能懂）

| 名词 | 大白话解释 |
|------|-----------|
| **Brief（创意请求）** | 你在前端填的那张表：主题、平台、市场、关键词、卖点 |
| **Orchestrator（编排器）** | 总指挥。决定生成多少条、循环几轮、调哪个工具、最后怎么排序 |
| **Angle（角度）** | 一个卖点切入点。比如"省钱""快速""安全"各是一个角度，保证标题不雷同 |
| **目标数量 target_count** | 这次要凑够多少条：标题20、描述15、CTA10（业务方定的，给运营留挑选余地） |
| **文本去重** | 拦"一模一样"的重复 |
| **语义去重** | 拦"换了说法但意思一样"的重复（用轻量词法相似度算分，>0.50 就丢；零额外依赖） |
| **本地化** | 把英文文案翻成目标市场语言（菲律宾语、泰语等），或直接用母语生成 |
| **合规检查** | 拦违禁词：赌博保证、医疗承诺、虚假紧迫感等 |
| **综合评分** | 合规分 + 关键词覆盖 + CTA强度 加权算出的总分，用来排序 |
| **截断** | 生成可能超量（如标题超过 20 条），最后只保留分数最高的目标数量交付 |
| **AB_Ranking** | 最终输出：排好序的候选列表 + 各项评分 + 翻译版本 |

---

## 五、差异化是怎么保证的（业务方最关心）

标题之间不雷同，靠**三道防线**叠加：

```mermaid
flowchart TD
    A[LLM 生成的一批候选] --> D1[第1道 文本去重]
    D1 -->|去掉逐字重复| D2[第2道 语义去重]
    D2 -->|词法相似度>0.50<br/>去掉近义重复| D3[第3道 角度生成]
    D3 -->|每条来自不同卖点角度<br/>保证结构性差异| R[剩下的就是互相差异化的候选]
```

- **第1道·文本去重**：`你好` 和 `你好` → 拦掉
- **第2道·语义去重**：`Quick topup bonus access` 和 `Easy topup bonus access`（相似度0.75）→ 拦掉
- **第3道·角度生成**：强制从"省钱/快速/安全/信任…"等不同角度各生成，从源头上拉开差异

> 注意：语义去重默认使用**轻量词法嵌入**（纯 Python 标准库，零额外依赖），server 启动时瞬时就绪（日志 `server.semantic_diversity_ready`）。如需更强的"完全不同措辞但同义"识别，可在配置里设 `lightweight=False` 切回 sentence-transformers 神经模型（需额外安装，约 460MB）。

---

## 六、一句话总结运行链路

```
浏览器填表 → server 收请求 → 校验 → 编排器算目标数量(20/15/10)
→ 拆角度 → 按角度生成 → 文本去重 → 语义去重 → 逐条(合规→翻译→关键词→复检→CTA)
→ 数量够了就排序 → 多样性加成 → 截断到目标数 → 返回结果 → 前端展示
```
