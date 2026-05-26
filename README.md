# Creative Editor Agent

Coco AI 创意编辑工具 Agent。基于 LLM Tool-Calling 模式，编排五个核心工具
（`Creative_Generator`、`Compliance_Checker`、`Localization_Tool`、`Keyword_Embedder`、
`CTA_Optimizer`），将运营人员提交的 `Creative_Brief` 转化为带有合规评分、关键词覆盖率、
CTA 强度评分以及多语言版本的 `AB_Ranking` 推荐列表。

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
# 编辑 .env 填入 TOKENPONY_API_KEY / TOKENPONY_BASE_URL / TOKENPONY_MODEL

# 4. 运行测试
pytest tests/
```

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
