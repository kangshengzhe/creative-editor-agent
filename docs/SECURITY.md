# API Key 与 Secrets 保护指南

本项目调用阿里云百炼 DashScope（Alibaba Cloud Model Studio）的 LLM API，需要携带 API Key。本文档列出
**必须遵守**的密钥保护规则、推荐工作流，以及泄露后的处置步骤。

---

## TL;DR

```powershell
# 第一次配置或轮换 key 时，运行：
python setup_env.py
```

脚本会用隐藏输入读取 token，写入 `.env`（已被 `.gitignore` 忽略），并把文件
权限锁定到只有你自己可读。**永远不要**把 token 直接贴进聊天、截图、日志或代码里。

---

## 1. 项目里 secrets 是怎么流动的

```
.env  (本地，gitignore)
  │
  ▼
python-dotenv 在 RealLLMClient import 时 load_dotenv()
  │
  ▼
os.getenv("TOKENPONY_API_KEY")
  │
  ▼
HTTPS Authorization: Bearer <token>  →  DashScope (Alibaba Cloud)
```

代码里**没有任何位置**硬编码 token。所有读取点都只有一个：
`creative_agent/llm/real_client.py` → `os.getenv("TOKENPONY_API_KEY")`。

---

## 2. 必须遵守的规则

### 永远不要做的事

- ❌ 把 token 贴进 IM、邮件、Issue、PR 描述、聊天对话窗口（包括与 AI 助手的对话）
- ❌ 把 token 写进任何 `.py`、`.json`、`.md`、`.html` 文件并提交到 git
- ❌ 把 `.env` 加入 git 跟踪（`.gitignore` 已经禁止，不要绕过）
- ❌ 把 token 截图发出去
- ❌ 把 token 输出到 stdout / 日志（`logging.info(token)` 之类的）
- ❌ 复用同一个 token 给多人/多机器使用

### 应该做的事

- ✅ 用 `python setup_env.py` 配置 token，输入是隐藏的
- ✅ 每个开发机器上都有自己独立的 `.env`，互不复用
- ✅ token 轮换时，再跑一次 `setup_env.py` 覆盖旧值
- ✅ 提交代码前用 `git status` 和 `git diff` 二次确认没把 secrets 加进去
- ✅ 在仓库根目录跑 `git check-ignore .env`，预期输出是 `.env`

---

## 3. .env 文件管理

### 配置（推荐方式）

```powershell
python setup_env.py
```

这个脚本会：

1. 用 `getpass.getpass()` 隐藏输入读取 token
2. 要求你重输 token 末尾 4 位做二次确认（防剪贴板事故）
3. 检查 `.gitignore` 已经覆盖 `.env`
4. 原子性写入 `.env`（先写 `.env.tmp` 再 `os.replace`，不会有半新半旧的中间状态）
5. 在 Windows 上用 `icacls` 把 ACL 改成 "只有当前用户可读写"
6. 在 macOS / Linux 上用 `chmod 600` 锁权限

### 手动方式（如果你不想用脚本）

```powershell
copy .env.example .env
notepad .env
```

填入：

```ini
TOKENPONY_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TOKENPONY_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
TOKENPONY_MODEL=qwen3.7-max
GENERATION_TEMPERATURE=0.8
ENABLE_THINKING=false
```

然后手动锁权限（Windows PowerShell 管理员）：

```powershell
icacls .env /inheritance:r
icacls .env /grant:r "$($env:USERNAME):F"
```

### 验证 .env 不会被提交

```powershell
# 这条命令会输出 ".env"，确认它在 ignore 列表里
git check-ignore -v .env
```

---

## 4. 多环境隔离

如果你有多套 token（开发 / 生产 / 测试），不要互相覆盖。用不同文件名：

- `.env` — 默认（开发用）
- `.env.production` — 生产 token
- `.env.staging` — 预发 token

`.gitignore` 已经覆盖 `.env.local`、`.env.*.local`、`.env.production`、
`.env.staging`、`.env.development`、`.env.backup`、`.env.bak`。

切换环境时，临时重命名或在启动命令里指定：

```powershell
$env:CREATIVE_AGENT_DOTENV = ".env.production"; python demo.py
```

（如果你需要这个特性，告诉我，我可以扩展 `RealLLMClient` 来读自定义路径。）

---

## 5. token 泄露后必做的 4 步

如果 token **可能**已经泄露（贴进聊天、误提交到 git、出现在截图、被陌生人看到屏幕……）：

1. **立刻吊销 / 重置** — 登录 TokenPony 后台，作废旧 token，重新生成一个新的
2. **审计调用记录** — 看 TokenPony 后台是否有异常 IP 或异常用量
3. **从 git 历史里清除**（如果误提交了）：

   ```powershell
   # 简单情况：尚未 push
   git reset --soft HEAD~1
   # 已 push：需要重写历史，建议用 git filter-repo
   pip install git-filter-repo
   git filter-repo --replace-text <(echo "tp-旧token==>REDACTED")
   git push --force-with-lease
   ```

   ⚠️ 重写已 push 的历史会影响协作者，先和团队商量

4. **跑 `setup_env.py`** 写入新 token，覆盖本地 `.env`

---

## 6. 自检清单（提交代码前过一遍）

```powershell
# 1. 确认 .env 没在跟踪里
git ls-files | findstr /R "\.env$"        # 应该没输出
git ls-files | findstr secrets             # 应该没输出

# 2. 搜整个仓库有没有 hardcoded token
findstr /S /R /C:"tp-[a-zA-Z0-9]" *.py *.md *.json *.html
# 应该只有 SECURITY.md 里这一行示例（tp-xxxx...），没有真实值

# 3. 确认 .gitignore 生效
git check-ignore -v .env
git check-ignore -v secrets.json
```

---

## 7. CI / 部署环境怎么处理

**永远不要**把 `.env` 打进容器镜像或部署包。

推荐方式：

| 环境 | 方案 |
|------|------|
| GitHub Actions | Repository Secrets → 注入为环境变量 |
| Docker | `--env-file` 或 secret manager |
| Kubernetes | Secret 资源（不要用 ConfigMap）|
| Azure / AWS / GCP | 各家的 Key Vault / Secrets Manager |

代码层面零变更 —— `os.getenv("TOKENPONY_API_KEY")` 在所有这些场景都自然工作。

---

## 8. 进一步加固（可选）

- **预提交钩子**：用 [`detect-secrets`](https://github.com/Yelp/detect-secrets)
  或 [`gitleaks`](https://github.com/gitleaks/gitleaks) 在 commit 时扫描 secrets

  ```powershell
  pip install detect-secrets
  detect-secrets scan > .secrets.baseline
  # 配置 pre-commit 钩子调用 detect-secrets-hook
  ```

- **环境变量管理工具**：[`direnv`](https://direnv.net/)（macOS / Linux）或
  [`dotenv-vault`](https://www.dotenv.org/)（跨平台）

- **硬件密钥**：YubiKey 这类硬件 token，在团队规模化后值得引入

---

## 9. 出问题怎么办

如果你不确定某个操作会不会泄露 token，**先停下来问**。轮换一个 token 比解释
一次泄露便宜得多。
