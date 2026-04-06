# OfficeGuider

**中文：** [README.md](README.md)

OfficeGuider models **office action sequences** as a **ordering problem**: it runs greedy decoding with a pretrained [VON (Versatile Ordering Network)](https://github.com/sysuvis/VON)-style model on 2D embeddings to produce a recommended execution order. It ships with a **FastAPI** backend and a **React + ECharts** console for selecting actions, inspecting paths, and viewing process details.

> The Python package still lives under `backend/smartflow/` for compatibility. Environment variables often use the `SMARTFLOW_*` prefix.

---

## Contributing

**Contributions are welcome and appreciated.** Whether you fix bugs, improve docs, propose features, or help with roadmap items (e.g. streaming training, deeper LLM integration), here is how to get involved.

### How to contribute

1. **Issues first**  
   - For bugs, design discussion, or claiming work: **search existing Issues** in this repo to avoid duplicates.  
   - For **large changes** (new APIs, schema/storage changes, major dependency bumps, behavior that diverges from upstream VON), open an **Issue** with a short proposal **before** sending a large PR.

2. **Pull requests**  
   - **Fork** → create a **topic branch** from `main` (or the default branch) → small commits → open a **PR**.  
   - Use a clear **title** and **description** (motivation, scope, how you tested—e.g. “`npm run build` passes”, “tested order button manually”).  
   - Link issues with `Fixes #123` when applicable.

3. **Checks before you push (recommended)**

   ```bash
   # Frontend (from frontend/)
   npm install && npm run build && npm run lint

   # Backend (from backend/, venv activated)
   python -m py_compile main.py
   ```

4. **Style & scope**  
   - Keep PRs **focused**; avoid unrelated formatting-only or wide renames.  
   - Match existing **naming, types, and comment style**; justify new dependencies in the PR.  
   - Changes under **`third_party/VON`** should be called out explicitly and respect its **LICENSE**; prefer upstream issues/PRs for generic fixes.

5. **Code of conduct**  
   - Be respectful and constructive in Issues and PRs. Harassment and discrimination are not tolerated.

6. **License**  
   - By contributing, you agree that your contributions are licensed under the same **MIT** license as this repository.

---

## Features

| Area | Description |
|------|-------------|
| Action catalog | Built-in office atoms + user-defined actions (JSON persistence) |
| VON ordering | Select ≥2 actions; `POST /von/order` returns order and 2D preview |
| Pretrained dir | Env / `active_model.json` / per-request override; optional server-side folder picker |
| Training jobs | Submit `user_pkl`-style training (subprocess); poll status and logs |
| Agent panel | OpenAI-compatible API via backend proxy (keys stored in browser only) |
| Process tabs | Pipeline, encoding, loc, permutation, geometry, model args, etc. |

---

## Stack

- **Backend:** Python 3.10+, FastAPI, PyTorch (VON inference), Pydantic  
- **Frontend:** Vite 6, React 19, TypeScript, ECharts  
- **Third party:** `third_party/VON` (vendored; follow upstream license)

---

## Repository layout

| Path | Role |
|------|------|
| `backend/smartflow/` | Core library: atoms, features, pointer orderer, validation, office catalog |
| `backend/main.py` | FastAPI app |
| `backend/train_reinforce.py` | Optional light REINFORCE; full VON training in `third_party/VON` |
| `third_party/VON` | Upstream VON (vendored) |
| `frontend/` | Vite + React + ECharts UI |
| `data/` | Sample logs, ordering data, 2D embeddings, schemas |
| `.env.example` | Environment template (copy to `.env`) |

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/<org-or-user>/<repo>.git
cd <repo>
```

### 2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open the URL printed by Vite. In dev, `/api` is proxied to `http://127.0.0.1:8000` (see `frontend/vite.config.ts`).

### 4. VON weights (required for ordering)

Place `args.json` and `epoch-*.pt` under e.g. `third_party/VON/pretrained/<subdir>/`, then:

```bash
export SMARTFLOW_VON_PRETRAINED=third_party/VON/pretrained/TSP
```

Paths can be relative to the repo root or absolute. See `.env.example`.

---

## Environment variables

Copy `.env.example` to `.env` and fill as needed. Common variables:

- **`SMARTFLOW_VON_PRETRAINED`** — VON checkpoint directory (same loading style as upstream `eval.py`)  
- **`SMARTFLOW_ATOMS_2D`** — Atom 2D coordinate JSON (defaults exist)  
- **`DEEPSEEK_*`** (or similar) — For `POST /dataset/generate` and other LLM-backed routes  

Details are in `.env.example`.

---

## API overview

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/atoms` | Atom metadata |
| GET | `/von/status` | Pretrained dir readiness |
| POST | `/von/order` | `{ "atom_ids": [...], "pretrained_dir?": "..." }` — greedy VON order |
| POST | `/von/pick-pretrained-dir` | Native folder dialog on server machine (dev) |
| POST | `/training/jobs` | Create training job |
| GET | `/training/jobs/{job_id}` | Job status and logs |
| POST | `/training/set-active` | Set active model directory |
| POST | `/agent/chat` | OpenAI-compatible chat |
| GET/POST/DELETE | `/custom/actions` | Custom actions CRUD |
| POST | `/recommend` | Pointer-engine Top-K (separate from VON path) |
| POST | `/validate` | Sequence + schema validation |
| POST | `/dataset/generate` | LLM-generated logs (API key) |
| … | `/mock/logs`, `/data/*` | Samples and validation |

Interactive docs: `http://127.0.0.1:8000/docs` when the backend is running.

---

## Data & scripts (advanced)

Example commands for long chains and 2D encoding:

```bash
PYTHONPATH=backend python scripts/generate_long_ordering_dataset.py --total 10000 --chain-len 50
pip install -r backend/requirements-encode.txt
PYTHONPATH=backend python scripts/encode_atoms_2d.py
```

See `scripts/` and `data/` for layout details.

---

## Relation to upstream VON

Inference matches `load_model` in `third_party/VON` (`args.json` + `epoch-*.pt`). For full paper-aligned training and `mission` metrics, extend inside `third_party/VON` and point `SMARTFLOW_VON_PRETRAINED` to your output directory.

---

## Roadmap / TODO

These areas have **early or partial** support but are **not** yet committed as production-grade:

| Area | Notes |
|------|--------|
| **Real-time training** | Logs/progress are mainly **polled**; **streaming** (WebSocket/SSE), finer progress, and tight real-time UI sync are **TODO**. |
| **LLM integration** | OpenAI-compatible proxy and dataset generation exist; a **unified gateway**, **tool use / orchestration** for office workflows, auditing, and quotas are **TODO**. |

Discuss priorities in **Issues**.

---

## License

OfficeGuider sample code is under the **MIT** License. `third_party/VON` follows its upstream **LICENSE**.
