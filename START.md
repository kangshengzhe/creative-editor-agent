# 启动指南 (Creative Editor Agent)

每次开机后，照这个跑起来就行。

> **本项目用 8001 端口**（8000 被 Docker 里的 `auto_posting_platform` 占着）。
> 启动后浏览器会**自动弹出**，不用手动开。

---

## TL;DR（复制即用）

在 Kiro 的终端里依次运行：

```powershell
cd C:\Users\kangs\Desktop\intership_data\creative_editor_agent
.\.venv\Scripts\Activate.ps1
python server.py
```

启动后**浏览器自动弹出 http://localhost:8001**（就是前端页面）。
停止：在终端按 `Ctrl + C`。

> 启动会慢几秒（在预加载语义去重模型）。看到日志出现
> `server.semantic_diversity_ready` 就表示模型就绪、语义差异化已上线。

---

## 详细步骤

### 1. 进入项目目录

在 Kiro 里打开终端（Terminal → New Terminal），运行：

```powershell
cd C:\Users\kangs\Desktop\intership_data\creative_editor_agent
```

### 2. 激活虚拟环境

```powershell
.\.venv\Scripts\Activate.ps1
```

成功后命令行前面会出现 `(.venv)`。

> 如果提示「禁止运行脚本 / running scripts is disabled」，先运行一次：
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
> 用 CMD 而非 PowerShell 时，激活命令换成 `.\.venv\Scripts\activate.bat`。

### 3. 启动

```powershell
python server.py
```

`python server.py` 会：
- 在 **8001** 端口启动服务
- 预加载语义去重模型（启动慢几秒属正常）
- **自动打开浏览器**到 http://localhost:8001

看到下面这类输出就说明起来了：

```
前端界面:  http://localhost:8001
Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
Application startup complete.
```

> 想换端口：`$env:PORT=8002; python server.py`（浏览器会自动开对应地址）。

### 4. 使用

浏览器已自动弹出 http://localhost:8001，填 Brief → 点「🚀 生成广告创意」即可。

其他地址：
- API 文档（Swagger）： http://localhost:8001/docs
- 健康检查： http://localhost:8001/health

### 5. 停止

在运行服务的终端按 `Ctrl + C`。

---

## 关于 Docker

**本项目自身不需要 Docker。** 它是纯 Python 服务，venv + `python server.py` 就能跑。
之所以要留意 Docker，只是因为 `auto_posting_platform`（另一个项目）用 Docker 跑、
占了 8000 端口，所以本项目用 8001 避开它。

---

## 首次 / 换电脑才需要做

`.venv` 和 `.env` 都已配好，日常启动不用重复。只有全新环境或依赖/密钥丢失时才需要：

```powershell
python -m venv .venv                  # 建虚拟环境（仅首次）
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt       # 装依赖（仅首次或依赖更新后）
pip install sentence-transformers     # 语义差异化所需（仅首次）
python setup_env.py                   # 配 LLM 密钥（仅首次或轮换密钥时）
```

---

## 跑测试（可选）

```powershell
.\.venv\Scripts\Activate.ps1
pytest tests/
```

---

## 常见问题

| 现象 | 解决办法 |
|------|----------|
| 浏览器没自动弹 | 手动开 http://localhost:8001 即可 |
| 打开 localhost:8000 看到 `{"error":{"code":"HTTP_404"}}` | 那是 Docker 里的 auto_posting。本项目是 **8001** |
| 「生成」按钮报网络错误 | 别直接双击 `frontend.html`，要走 http://localhost:8001 |
| 激活脚本被禁止运行 | 运行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| 8001 被占用 | `$env:PORT=8002; python server.py`，浏览器会自动开新端口 |
| 启动报缺少模块 | 重新 `pip install -r requirements.txt` |
| 语义差异化日志显示 degraded | `pip install sentence-transformers`，再重启 |

---

## 附录：auto_posting_platform（这个才用 Docker，占 8000）

```powershell
cd C:\Users\kangs\Desktop\intership_data\auto_posting_platform
docker compose up -d        # 启动
docker compose ps           # 查看状态
docker compose down         # 停止
```
