# OfficeGuider

**English:** [README.en.md](README.en.md)

> **想参与贡献？** 请阅读下文 **「贡献与协作（Contributing）」** 整节；我们欢迎 Issue 与 PR。

将**办公动作序列**建模为**排序问题**：在 2D 嵌入空间上，用预训练的 [VON（Versatile Ordering Network）](https://github.com/sysuvis/VON) 模型做贪心解码，得到推荐执行顺序；配套 **FastAPI** 后端与 **React + ECharts** 控制台，便于勾选动作、查看路径与过程信息。

> Python 包目录仍为 `backend/smartflow/`，与历史代码兼容；环境变量前缀多为 `SMARTFLOW_*`。

---

## 功能概览

| 能力 | 说明 |
|------|------|
| 动作目录 | 内置办公原子动作 + 用户自定义动作（JSON 持久化） |
| VON 排序 | 勾选 ≥2 个动作，调用 `POST /von/order` 得到推荐顺序与 2D 预览 |
| 预训练目录 | 支持环境变量 / `active_model.json` / 单次请求覆盖；前端可选「服务端本机目录选择」 |
| 训练任务 | 提交自定义 `user_pkl` 风格训练（后台子进程），轮询任务状态与日志 |
| Agent 面板 | 配置 OpenAI 兼容 API，经后端转发对话（密钥仅存浏览器） |
| 过程信息 | 流程、编码表、loc、排列、几何量、模型参数等分标签展示 |

---

## 贡献与协作（Contributing）

**我们欢迎并依赖社区贡献。** 无论你是修 bug、补文档、提需求，还是参与路线图中的能力（如实时训练、LLM 深度接入），都可以通过下面方式参与。

### 参与方式

1. **Issues（优先）**  
   - 报告 bug、讨论设计、认领功能：请先在仓库 **Issues** 列表中搜索是否已有类似主题，避免重复。  
   - **较大改动**（新 API、数据库结构、依赖大升级、与 VON 上游行为不一致的修改）请先开 Issue 简述方案，再提 PR。

2. **Pull Request**  
   - **Fork** 本仓库 → 从 `main`（或默认分支）拉 **topic 分支** → 小步提交 → 发起 PR。  
   - PR 标题与描述请写清**动机、改动范围、如何验证**（例如「本地 `npm run build` 通过」「手动点了排序按钮」）。  
   - 若 PR 关联某 Issue，正文写上 `Fixes #123` 等以便自动关联。

3. **提交前自检（强烈建议）**

   ```bash
   # 前端（在 frontend/）
   npm install && npm run build && npm run lint

   # 后端（在 backend/，已激活 venv）
   python -m py_compile main.py
   ```

4. **代码与协作约定**  
   - 改动尽量**聚焦单一主题**，避免无关格式化或大范围重命名。  
   - 与现有代码保持**命名、类型与注释风格**一致；新增依赖需在 PR 中说明理由。  
   - 涉及 **`third_party/VON`** 的修改请单独说明，并遵守其 **LICENSE**；优先通过上游 issue/PR 反馈通用问题。

5. **行为准则**  
   - 在 Issue/PR 讨论中保持尊重与建设性；**不接受**骚扰与歧视性内容。

6. **许可证**  
   - 向本仓库提交代码即表示你同意在 **MIT** 许可下授权你的贡献（与仓库根目录许可一致）。

---

## 技术栈

- **后端**：Python 3.10+、FastAPI、PyTorch（VON 推理）、Pydantic  
- **前端**：Vite 6、React 19、TypeScript、ECharts  
- **第三方**：`third_party/VON`（vendor，与上游许可证一致）

---

## 仓库结构

| 路径 | 说明 |
|------|------|
| `backend/smartflow/` | 原子任务与特征、指针排序、校验与办公目录等核心库 |
| `backend/main.py` | FastAPI 入口（REST API） |
| `backend/train_reinforce.py` | 可选轻量 REINFORCE；完整 VON 训练见 `third_party/VON` |
| `third_party/VON` | 官方 VON 源码（已内置），可按论文扩展 `mission` / 指标 |
| `frontend/` | Vite + React + ECharts 控制台 |
| `data/` | 示例日志、排序数据、2D 嵌入、Schema 等 |
| `.env.example` | 环境变量模板（复制为 `.env` 后按需填写） |

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/<你的组织或用户名>/<仓库名>.git
cd <仓库名>
```

将上述 URL 与目录名换成你在 GitHub 上创建的实际仓库。

### 2. 后端

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 3. 前端

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

浏览器打开终端提示的本地地址。开发环境下，API 通过 `frontend/vite.config.ts` 将 `/api` 代理到 `http://127.0.0.1:8000`。

### 4. VON 权重（排序必需）

将含 `args.json` 与 `epoch-*.pt` 的目录放到例如 `third_party/VON/pretrained/<子目录>/`，设置环境变量后重启后端：

```bash
export SMARTFLOW_VON_PRETRAINED=third_party/VON/pretrained/TSP
```

（路径相对仓库根；也可用绝对路径。）详见 `.env.example`。

---

## 环境变量

复制根目录 `.env.example` 为 `.env`，按需填写。常见项：

- **`SMARTFLOW_VON_PRETRAINED`**：VON 预训练目录（与官方 `eval.py` 加载方式一致）  
- **`SMARTFLOW_ATOMS_2D`**：原子 2D 坐标 JSON（默认有示例路径）  
- **`DEEPSEEK_*` / 类似**：若使用 `POST /dataset/generate` 等需 LLM 的接口  

完整说明见 `.env.example` 内注释。

---

## API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/atoms` | 原子任务列表 |
| GET | `/von/status` | 预训练目录是否就绪 |
| POST | `/von/order` | `{ "atom_ids": [...], "pretrained_dir?": "..." }`，VON 贪心排序 |
| POST | `/von/pick-pretrained-dir` | 服务端本机弹出系统目录（开发机常用） |
| POST | `/training/jobs` | 创建训练任务 |
| GET | `/training/jobs/{job_id}` | 查询任务与日志 |
| POST | `/training/set-active` | 将某产出目录设为当前排序模型 |
| POST | `/agent/chat` | OpenAI 兼容对话（需配置 base/key/model） |
| GET/POST/DELETE | `/custom/actions` | 自定义动作 CRUD |
| POST | `/recommend` | 指针引擎 Top-K 推荐（与 VON 路径不同） |
| POST | `/validate` | 序列与 Schema 校验 |
| POST | `/dataset/generate` | 用 LLM 生成日志（需 API Key） |
| … | `/mock/logs`、`/data/*` | 示例与数据校验 |

OpenAPI：启动后端后访问 `http://127.0.0.1:8000/docs`。

---

## 数据与脚本（进阶）

长链数据、2D 编码、序列张量等生成命令见原仓库脚本说明，例如：

```bash
PYTHONPATH=backend python scripts/generate_long_ordering_dataset.py --total 10000 --chain-len 50
pip install -r backend/requirements-encode.txt
PYTHONPATH=backend python scripts/encode_atoms_2d.py
```

更多细节仍适用本仓库内 `scripts/` 与 `data/` 目录布局。

---

## 与官方 VON 的关系

内置推理与 `third_party/VON` 中 `load_model` 路径一致（`args.json` + `epoch-*.pt`）。若要对齐论文中的完整训练与 `mission` 指标，请在 `third_party/VON` 内按官方流程扩展，并将产出目录指给 `SMARTFLOW_VON_PRETRAINED`。

---

## 路线图 / TODO

以下能力**已有雏形或占位**，但尚未作为「生产级、可对外承诺」的能力发布，仍在持续迭代中：

| 方向 | 说明 |
|------|------|
| **实时训练** | 训练日志与进度目前以**轮询**为主；**流式推送**（如 WebSocket/SSE）、更细粒度进度与资源监控、与前端 UI 的**真正实时**联动仍在 TODO。 |
| **LLM 接入** | 已支持 OpenAI 兼容接口转发与数据集生成等；**统一模型网关**、与办公场景深度结合的**工具调用/编排**、审计与配额等**完整 LLM 产品化接入**仍在 TODO。 |

欢迎在本仓库的 **Issues** 中讨论优先级与实现方案。

---

## 开源许可

OfficeGuider 示例代码以 **MIT** 许可发布；`third_party/VON` 遵循其原有 **LICENSE**。
