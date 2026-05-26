"""FastAPI HTTP server for the Creative Editor Agent.

启动方式:
    & "C:\\Users\\kangs\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" server.py

或者用 uvicorn 直接启动:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

启动后访问:
    http://localhost:8000        → 前端界面（填表单提交 Brief）
    http://localhost:8000/docs   → Swagger API 文档（自动生成）
    POST http://localhost:8000/api/creative  → API 端点
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保 src/ 在 Python 路径上
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from creative_agent import configure_logging, get_logger
from creative_agent.llm import RealLLMClient
from creative_agent.orchestrator import Orchestrator
from creative_agent.tools.compliance_checker import ComplianceChecker
from creative_agent.tools.creative_generator import CreativeGenerator
from creative_agent.tools.cta_optimizer import CTAOptimizer
from creative_agent.tools.keyword_embedder import KeywordEmbedder
from creative_agent.tools.localization_tool import LocalizationTool

# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

configure_logging(level="INFO")
log = get_logger("server")

# 组装工具链（进程启动时只做一次）
try:
    llm = RealLLMClient()
except ValueError as exc:
    print(f"ERROR: {exc}")
    print("请先运行 python setup_env.py 配置 .env")
    sys.exit(1)

compliance_checker = ComplianceChecker(llm=None)  # 本地词典模式
orchestrator = Orchestrator(
    creative_generator=CreativeGenerator(llm),
    compliance_checker=compliance_checker,
    localization_tool=LocalizationTool(llm),
    keyword_embedder=KeywordEmbedder(llm),
    cta_optimizer=CTAOptimizer(llm, compliance_checker=compliance_checker),
)

# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Creative Editor Agent",
    description="Coco AI 创意编辑工具 — 自动生成合规广告文案 + 多语言 + 排序",
    version="1.0.0",
)


@app.post("/api/creative")
async def create_creative(request: Request):
    """接收 Creative_Brief JSON，返回 AB_Ranking（单类型）。

    请求体示例:
    {
        "campaign_topic": "Game topup bonus",
        "target_platform": "GOOGLE_ADS",
        "target_market": "EN_GLOBAL",
        "creative_type": "HEADLINE",
        "keywords": ["topup", "bonus"]
    }
    """
    from creative_agent.api import handle_request

    body = await request.json()
    result = await handle_request(body, orchestrator)
    return JSONResponse(content=result["body"], status_code=result["status_code"])


@app.post("/api/creative/batch")
async def create_creative_batch(request: Request):
    """批量模式：一次生成 HEADLINE + DESCRIPTION + CTA 三种类型的完整广告组合。

    请求体（不需要 creative_type 字段，系统自动跑三种）:
    {
        "campaign_topic": "Game topup bonus",
        "target_platform": "GOOGLE_ADS",
        "target_market": "EN_GLOBAL",
        "keywords": ["topup", "bonus"],
        "selling_points": ["20% bonus", "instant credit"]
    }

    返回:
    {
        "headlines": { ...AB_Ranking... },
        "descriptions": { ...AB_Ranking... },
        "ctas": { ...AB_Ranking... },
        "total_time_ms": 12345,
        "errors": {}
    }
    """
    import time
    from creative_agent.api import handle_request

    body = await request.json()
    start = time.perf_counter()

    types = ["HEADLINE", "DESCRIPTION", "CTA"]
    results = {}
    errors = {}

    for creative_type in types:
        brief = {**body, "creative_type": creative_type}
        result = await handle_request(brief, orchestrator)
        if result["status_code"] == 200:
            results[creative_type.lower() + "s"] = result["body"]
        else:
            errors[creative_type.lower() + "s"] = result["body"]

    total_time_ms = int((time.perf_counter() - start) * 1000)

    response = {
        **results,
        "total_time_ms": total_time_ms,
        "errors": errors if errors else None,
    }
    # 如果全部失败返回 502，部分失败返回 207，全部成功返回 200
    if len(errors) == 3:
        status = 502
    elif errors:
        status = 207
    else:
        status = 200

    return JSONResponse(content=response, status_code=status)


@app.get("/", response_class=HTMLResponse)
async def frontend():
    """返回前端 HTML 页面。"""
    html_path = ROOT / "frontend.html"
    if html_path.is_file():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>前端文件 frontend.html 不存在</h1>", status_code=404)


@app.get("/health")
async def health():
    """健康检查端点。"""
    return {"status": "ok", "model": "creative-editor-agent", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# 直接运行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Creative Editor Agent — HTTP Server")
    print("=" * 60)
    print()
    print("  前端界面:  http://localhost:8000")
    print("  API 文档:  http://localhost:8000/docs")
    print("  API 端点:  POST http://localhost:8000/api/creative")
    print()
    print("按 Ctrl+C 停止服务器")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
