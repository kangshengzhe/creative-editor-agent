# Creative Editor Agent

Coco AI 创意编辑工具 Agent。基于 LLM Tool-Calling 模式，编排五个核心工具
（`Creative_Generator`、`Compliance_Checker`、`Localization_Tool`、`Keyword_Embedder`、
`CTA_Optimizer`），将运营人员提交的 `Creative_Brief` 转化为带有合规评分、关键词覆盖率、
CTA 强度评分以及多语言版本的 `AB_Ranking` 推荐列表。

**核心能力**：
- 精确交付 **15 条标题 / 10 条描述 / 5 条 CTA**（按业务方配额）
- **46 个目标市场**的母语生成 + 多语言本地化（20 种语言）
- **三层差异化**：文本去重 + 语义向量去重（阈值 0.60）+ 角度拆分
- **关键词覆盖优先**：语义相似度中被淘汰的近义候选作为储备池，当正选缺关键词时自动替补
- 合规检查（本地违禁词词典）、SEO 关键词本地化、香港运营审核翻译（简/繁中+英语）
- **Web 前端**：前端界面 + 批量生成 + 中断按钮 + 常换常新
- **推荐模型**：qwen3.7-max（阿里云百炼，需关闭思考模式，批量约 35-45 秒）

详见 `.kiro/specs/creative-editor-agent/` 下的 requirements / design / tasks 文档。

## 快速开始

```powershell
# 1. 创建虚拟环境（Python 3.10+）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install -r requirements.txt
# 或可编辑安装
pip install -e .

# 3. 配置 LLM 凭据
copy .env.example .env
# 编辑 .env：填入 TOKENPONY_API_KEY，确认 BASE_URL 和 MODEL 正确
# 推荐模型：qwen3.7-max（ENABLE_THINKING=false 已默认设好）

# 4. 启动 Web 服务（会自动弹出浏览器）
python server.py
# 前端界面: http://localhost:8001
# API 文档: http://localhost:8001/docs

# 5. 运行测试（213 个，全部通过）
pytest tests/
```

详细运行说明见 `START.md`。

## 项目结构

```
src/creative_agent/
├── api/             # API Gateway、请求校验、顶层入口
├── orchestrator/    # 编排逻辑、Composite Scorer、流水线
├── tools/           # 五个核心工具
├── models/          # Pydantic 数据模型与枚举
├── config/          # Platform_Spec 与 Forbidden_Term 词典加载
├── observability/   # 结构化日志、Trace 记录
├── errors/          # 错误码与响应模型
└── llm/             # LLM 客户端抽象（Real / Mock）

tests/
├── unit/            # 单元测试
├── property/        # 属性测试（Hypothesis）
└── integration/     # 集成测试（mock LLM / 真实 LLM 冒烟）
```
