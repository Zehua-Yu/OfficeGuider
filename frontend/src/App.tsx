import { useCallback, useEffect, useMemo, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

const LS_AGENT_BASE = 'officeguider_agent_api_base'
const LS_AGENT_KEY = 'officeguider_agent_api_key'
const LS_AGENT_MODEL = 'officeguider_agent_model'
const LS_ORDER_PRETRAINED = 'officeguider_order_pretrained_dir'

/** 旧版 SmartFlow 本地存储键，首次启动时迁移到 OfficeGuider 键 */
const LS_LEGACY = {
  agentBase: 'smartflow_agent_api_base',
  agentKey: 'smartflow_agent_api_key',
  agentModel: 'smartflow_agent_model',
  orderPretrained: 'smartflow_order_pretrained_dir',
} as const

type ProcessTabId =
  | 'pipeline'
  | 'encoding'
  | 'loc'
  | 'permutation'
  | 'geometry'
  | 'pairwise'
  | 'model'

const PROCESS_TABS: { id: ProcessTabId; label: string }[] = [
  { id: 'pipeline', label: '流程说明' },
  { id: 'encoding', label: '动作编码' },
  { id: 'loc', label: 'loc 张量' },
  { id: 'permutation', label: '排列 π' },
  { id: 'geometry', label: '几何量' },
  { id: 'pairwise', label: '成对矩阵' },
  { id: 'model', label: '模型与权重' },
]

const DEFAULT_TRAIN_JSON = `{
  "instances": [
    [[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8],[0.2,0.9],[0.4,0.1],[0.6,0.3],[0.8,0.5]],
    [[0.15,0.25],[0.35,0.45],[0.55,0.65],[0.75,0.85],[0.25,0.95],[0.45,0.15],[0.65,0.35],[0.85,0.55]],
    [[0.12,0.22],[0.32,0.42],[0.52,0.62],[0.72,0.82],[0.22,0.92],[0.42,0.12],[0.62,0.32],[0.82,0.52]]
  ],
  "metric": "tsp",
  "sample_size": 8,
  "n_epochs": 2,
  "epoch_size": 16,
  "batch_size": 8,
  "val_size": 4,
  "run_name": "officeguider_demo",
  "set_active_on_finish": true
}`

type Atom = {
  id: string
  display_name: string
  complexity: number
  est_duration_sec: number
  source?: 'catalog' | 'custom'
}

type VonProcessPayload = {
  catalog_path: string
  custom_actions_path?: string
  pipeline: { step: number; name: string; detail: string }[]
  encoding_input_order: {
    input_index: number
    atom_id: string
    display_name: string
    x: number
    y: number
  }[]
  loc_tensor: { shape: number[]; dtype: string; values_row_major: number[][] }
  permutation_pi: number[]
  permutation_note: string
  euclidean: {
    edge_lengths_along_output_order: number[]
    tour_length_open: number
    tour_length_closed: number
  }
  point_stats: {
    centroid: { x: number; y: number } | null
    bbox: Record<string, number> | null
  }
  pairwise_euclidean: number[][] | null
  pairwise_note: string
  model_cost_reported: number
  device: string
  model: { pretrained_dir: string; args: Record<string, unknown> }
}

export default function App() {
  const [atoms, setAtoms] = useState<Atom[]>([])
  const [filter, setFilter] = useState('')
  const [scope, setScope] = useState<'all' | 'catalog' | 'custom'>('all')
  const [selected, setSelected] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [vonResult, setVonResult] = useState<{
    ordered_atom_ids: string[]
    coords_ordered: { x: number; y: number }[]
    cost: number
    input_atom_ids: string[]
    index_order?: number[]
    process?: VonProcessPayload
  } | null>(null)
  const [vonLoading, setVonLoading] = useState(false)
  const [vonStatus, setVonStatus] = useState<{
    ready: boolean
    pretrained_dir: string | null
    using_default_pretrained?: boolean
    pretrained_source?: string
    active_model_file?: string
  } | null>(null)

  const [trainJson, setTrainJson] = useState(DEFAULT_TRAIN_JSON)
  const [trainJobId, setTrainJobId] = useState<string | null>(null)
  const [trainJob, setTrainJob] = useState<{
    status?: string
    log?: string
    last_line?: string
    pretrained_dir?: string | null
    error?: string | null
    exit_code?: number | null
  } | null>(null)
  const [trainSubmitting, setTrainSubmitting] = useState(false)

  const [agentApiBase, setAgentApiBase] = useState('')
  const [agentApiKey, setAgentApiKey] = useState('')
  const [agentModel, setAgentModel] = useState('deepseek-chat')
  const [chatMessages, setChatMessages] = useState<{ role: string; content: string }[]>([])
  const [chatInput, setChatInput] = useState('')
  const [agentLoading, setAgentLoading] = useState(false)
  const [pendingSuggested, setPendingSuggested] = useState<{ display_name: string }[]>([])
  const [manualCustomName, setManualCustomName] = useState('')

  useEffect(() => {
    try {
      const pairs: [string, string][] = [
        [LS_AGENT_BASE, LS_LEGACY.agentBase],
        [LS_AGENT_KEY, LS_LEGACY.agentKey],
        [LS_AGENT_MODEL, LS_LEGACY.agentModel],
        [LS_ORDER_PRETRAINED, LS_LEGACY.orderPretrained],
      ]
      for (const [k, old] of pairs) {
        if (!localStorage.getItem(k) && localStorage.getItem(old)) {
          localStorage.setItem(k, localStorage.getItem(old)!)
        }
      }
    } catch {
      /* ignore */
    }
  }, [])

  useEffect(() => {
    try {
      setAgentApiBase(
        localStorage.getItem(LS_AGENT_BASE) ?? localStorage.getItem(LS_LEGACY.agentBase) ?? 'https://api.deepseek.com/v1',
      )
      setAgentApiKey(localStorage.getItem(LS_AGENT_KEY) ?? localStorage.getItem(LS_LEGACY.agentKey) ?? '')
      setAgentModel(localStorage.getItem(LS_AGENT_MODEL) ?? localStorage.getItem(LS_LEGACY.agentModel) ?? 'deepseek-chat')
    } catch {
      /* ignore */
    }
  }, [])

  const persistAgentSettings = useCallback(() => {
    try {
      localStorage.setItem(LS_AGENT_BASE, agentApiBase)
      localStorage.setItem(LS_AGENT_MODEL, agentModel)
      localStorage.setItem(LS_AGENT_KEY, agentApiKey)
    } catch {
      /* ignore */
    }
  }, [agentApiBase, agentApiKey, agentModel])

  const nameById = useMemo(() => {
    const m = new Map<string, string>()
    for (const a of atoms) m.set(a.id, a.display_name)
    return m
  }, [atoms])

  const [rightListTab, setRightListTab] = useState<'recommend' | 'todo'>('recommend')
  const [todoCopied, setTodoCopied] = useState(false)
  const [processTab, setProcessTab] = useState<ProcessTabId>('pipeline')
  const [orderPretrainedDir, setOrderPretrainedDir] = useState(() => {
    try {
      return localStorage.getItem(LS_ORDER_PRETRAINED) ?? localStorage.getItem(LS_LEGACY.orderPretrained) ?? ''
    } catch {
      return ''
    }
  })
  const [pickDirBusy, setPickDirBusy] = useState(false)

  const todoTemplateText = useMemo(() => {
    if (!vonResult) return ''
    const lines = [
      '【OfficeGuider 待办清单】',
      `生成时间：${new Date().toLocaleString('zh-CN')}`,
      '',
      ...vonResult.ordered_atom_ids.map(
        (id, i) => `- [ ] ${i + 1}. ${nameById.get(id) ?? id}`,
      ),
      '',
      `VON cost: ${vonResult.cost?.toFixed?.(4) ?? vonResult.cost}`,
    ]
    return lines.join('\n')
  }, [vonResult, nameById])

  const copyTodoTemplate = useCallback(() => {
    if (!todoTemplateText) return
    navigator.clipboard.writeText(todoTemplateText).then(
      () => {
        setTodoCopied(true)
        window.setTimeout(() => setTodoCopied(false), 2000)
      },
      () => setError('复制失败（浏览器可能未授权剪贴板）'),
    )
  }, [todoTemplateText])

  const fetchAtoms = useCallback(async () => {
    const r = await fetch(`${API_BASE}/atoms`)
    if (!r.ok) throw new Error('加载动作列表失败')
    const j = await r.json()
    setAtoms(j.atoms ?? [])
  }, [])

  useEffect(() => {
    fetchAtoms().catch((e) => setError(String(e)))
  }, [fetchAtoms])

  useEffect(() => {
    try {
      localStorage.setItem(LS_ORDER_PRETRAINED, orderPretrainedDir)
    } catch {
      /* ignore */
    }
  }, [orderPretrainedDir])

  const refreshVonStatus = useCallback(() => {
    fetch(`${API_BASE}/von/status`)
      .then((r) => r.json())
      .then((j) =>
        setVonStatus({
          ready: j.ready,
          pretrained_dir: j.pretrained_dir,
          using_default_pretrained: j.using_default_pretrained,
          pretrained_source: j.pretrained_source,
          active_model_file: j.active_model_file,
        }),
      )
      .catch(() => setVonStatus(null))
  }, [])

  useEffect(() => {
    refreshVonStatus()
  }, [refreshVonStatus])

  useEffect(() => {
    if (!trainJobId) return
    const t = window.setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/training/jobs/${trainJobId}`)
        if (!r.ok) return
        const j = await r.json()
        setTrainJob(j)
        if (j.status === 'done' || j.status === 'failed') {
          window.clearInterval(t)
          refreshVonStatus()
        }
      } catch {
        /* ignore */
      }
    }, 1000)
    return () => window.clearInterval(t)
  }, [trainJobId, refreshVonStatus])

  const customAtoms = useMemo(
    () => atoms.filter((a) => a.source === 'custom'),
    [atoms],
  )

  const filteredAtoms = useMemo(() => {
    let list = atoms
    if (scope === 'catalog') list = list.filter((a) => a.source !== 'custom')
    if (scope === 'custom') list = list.filter((a) => a.source === 'custom')
    const q = filter.trim().toLowerCase()
    if (!q) return list
    return list.filter(
      (a) =>
        a.id.toLowerCase().includes(q) || a.display_name.toLowerCase().includes(q),
    )
  }, [atoms, filter, scope])

  const sendAgentChat = async () => {
    const text = chatInput.trim()
    if (!text) {
      setError('请输入对话内容')
      return
    }
    if (!agentApiBase.trim()) {
      setError('请填写 LLM API 地址')
      return
    }
    setAgentLoading(true)
    setError(null)
    persistAgentSettings()
    const nextMsgs = [...chatMessages, { role: 'user', content: text }]
    try {
      const r = await fetch(`${API_BASE}/agent/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: nextMsgs,
          api_base: agentApiBase.trim(),
          api_key: agentApiKey,
          model: agentModel.trim() || 'deepseek-chat',
        }),
      })
      if (!r.ok) throw new Error(await r.text())
      const j = await r.json()
      setChatMessages([...nextMsgs, { role: 'assistant', content: j.reply ?? '' }])
      setPendingSuggested(Array.isArray(j.suggested_actions) ? j.suggested_actions : [])
      setChatInput('')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setAgentLoading(false)
    }
  }

  const addSuggestedToCustom = async () => {
    if (!pendingSuggested.length) return
    setError(null)
    try {
      const r = await fetch(`${API_BASE}/custom/actions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actions: pendingSuggested }),
      })
      if (!r.ok) throw new Error(await r.text())
      setPendingSuggested([])
      await fetchAtoms()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const addManualCustom = async () => {
    const dn = manualCustomName.trim()
    if (!dn) return
    setError(null)
    try {
      const r = await fetch(`${API_BASE}/custom/actions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actions: [{ display_name: dn }] }),
      })
      if (!r.ok) throw new Error(await r.text())
      setManualCustomName('')
      await fetchAtoms()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const deleteCustom = async (id: string) => {
    setError(null)
    try {
      const r = await fetch(`${API_BASE}/custom/actions/${encodeURIComponent(id)}`, {
        method: 'DELETE',
      })
      if (!r.ok) throw new Error(await r.text())
      setSelected((s) => s.filter((x) => x !== id))
      await fetchAtoms()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const startVonTraining = async () => {
    setError(null)
    let body: Record<string, unknown>
    try {
      body = JSON.parse(trainJson) as Record<string, unknown>
    } catch {
      setError('训练配置 JSON 无法解析')
      return
    }
    setTrainSubmitting(true)
    setTrainJob(null)
    try {
      const r = await fetch(`${API_BASE}/training/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!r.ok) throw new Error(await r.text())
      const j = await r.json()
      const jid = j.job_id as string
      setTrainJobId(jid)
      const r2 = await fetch(`${API_BASE}/training/jobs/${jid}`)
      if (r2.ok) setTrainJob(await r2.json())
      else setTrainJob({ status: 'running' })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setTrainJobId(null)
    } finally {
      setTrainSubmitting(false)
    }
  }

  const setActiveFromPath = async (dir: string) => {
    setError(null)
    try {
      const r = await fetch(`${API_BASE}/training/set-active`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pretrained_dir: dir }),
      })
      if (!r.ok) throw new Error(await r.text())
      refreshVonStatus()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const toggle = (id: string) => {
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]))
  }

  const pickPretrainedDir = useCallback(async () => {
    setPickDirBusy(true)
    setError(null)
    try {
      const r = await fetch(`${API_BASE}/von/pick-pretrained-dir`, { method: 'POST' })
      const text = await r.text()
      if (!r.ok) throw new Error(text || `HTTP ${r.status}`)
      let j: { cancelled?: boolean; path?: string }
      try {
        j = JSON.parse(text) as { cancelled?: boolean; path?: string }
      } catch {
        throw new Error(text.slice(0, 200) || '服务端返回非 JSON')
      }
      if (j.cancelled) return
      if (j.path) setOrderPretrainedDir(j.path)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setPickDirBusy(false)
    }
  }, [])

  const runVonOrder = async () => {
    if (selected.length < 2) {
      setError('请至少勾选 2 个动作')
      return
    }
    setVonLoading(true)
    setError(null)
    try {
      const body: { atom_ids: string[]; pretrained_dir?: string } = { atom_ids: selected }
      const d = orderPretrainedDir.trim()
      if (d) body.pretrained_dir = d

      const r = await fetch(`${API_BASE}/von/order`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!r.ok) throw new Error(await r.text())
      setVonResult(await r.json())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setVonResult(null)
    } finally {
      setVonLoading(false)
    }
  }

  const chartOption = useMemo(() => {
    if (!vonResult?.coords_ordered?.length) return null
    const line = vonResult.coords_ordered.map((c) => [c.x, c.y] as [number, number])
    const labels = vonResult.ordered_atom_ids.map(
      (id) => nameById.get(id) ?? id,
    )
    return {
      title: {
        text: '推荐顺序在 2D 嵌入空间中的路径',
        left: 'center',
        textStyle: { fontSize: 14, color: '#18181b', fontWeight: 500 },
      },
      tooltip: { trigger: 'item' },
      grid: { left: 48, right: 24, top: 44, bottom: 32 },
      xAxis: { type: 'value', scale: true },
      yAxis: { type: 'value', scale: true },
      series: [
        {
          type: 'line',
          data: line,
          lineStyle: { color: '#2563eb', width: 2 },
          symbol: 'none',
        },
        {
          type: 'scatter',
          data: line.map((pt, i) => ({
            value: pt,
            label: {
              show: true,
              formatter: `${i + 1}. ${labels[i]}`.slice(0, 80),
              position: 'top',
              fontSize: 8,
              color: '#52525b',
            },
          })),
          symbolSize: 10,
          itemStyle: { color: '#1d4ed8' },
        },
      ],
    }
  }, [vonResult, nameById])

  const emptyChartOption = useMemo(
    () => ({
      title: {
        text: '2D 路径预览（排序后生成）',
        left: 'center',
        top: 'center',
        textStyle: { fontSize: 13, color: '#a1a1aa', fontWeight: 400 },
      },
      grid: { left: 48, right: 24, top: 40, bottom: 32 },
      xAxis: { type: 'value', scale: true, axisLabel: { color: '#71717a' } },
      yAxis: { type: 'value', scale: true, axisLabel: { color: '#71717a' } },
      series: [],
    }),
    [],
  )

  return (
    <div className="app app-one-screen">
      {error && <div className="error error-bar">{error}</div>}

      <div className="main-grid">
        <header className="header header-compact main-grid-header">
          <div className="header-row">
            <h1>OfficeGuider · VON</h1>
            {vonStatus && (
              <span className={`status-inline ${vonStatus.ready ? 'ok' : 'warn'}`}>
                {vonStatus.ready
                  ? `就绪 · ${vonStatus.pretrained_source === 'env' ? '环境变量' : vonStatus.pretrained_source === 'active_file' ? '用户训练' : '默认'}`
                  : '模型未就绪'}
              </span>
            )}
          </div>
        </header>

        <aside className="col-sidebar" aria-label="工具栏">
          <div className="col-sidebar-split">
      <section className="panel panel-tight agent-panel">
        <h2>Agent · 配置 LLM 并生成自定义动作</h2>
        <p className="hint agent-hint">
          填写 OpenAI 兼容接口（如 DeepSeek、OpenAI）。密钥仅保存在本机浏览器 localStorage；对话由后端转发至该地址。
        </p>
        <div className="agent-settings">
          <label className="agent-field">
            <span>API Base</span>
            <input
              type="url"
              value={agentApiBase}
              onChange={(e) => setAgentApiBase(e.target.value)}
              placeholder="https://api.deepseek.com/v1"
              autoComplete="off"
            />
          </label>
          <label className="agent-field">
            <span>API Key</span>
            <input
              type="password"
              value={agentApiKey}
              onChange={(e) => setAgentApiKey(e.target.value)}
              placeholder="sk-…"
              autoComplete="off"
            />
          </label>
          <label className="agent-field">
            <span>Model</span>
            <input
              type="text"
              value={agentModel}
              onChange={(e) => setAgentModel(e.target.value)}
              placeholder="deepseek-chat"
            />
          </label>
          <button type="button" className="btn-secondary" onClick={persistAgentSettings}>
            保存配置到浏览器
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => {
              setChatMessages([])
              setPendingSuggested([])
            }}
          >
            清空对话
          </button>
        </div>
        {chatMessages.length > 0 && (
          <div className="chat-log" role="log" aria-live="polite">
            {chatMessages.map((m, i) => (
              <div key={i} className={`chat-msg chat-msg-${m.role}`}>
                <span className="chat-role">{m.role === 'user' ? '你' : '助手'}</span>
                <pre className="chat-text">{m.content}</pre>
              </div>
            ))}
          </div>
        )}
        <div className="chat-compose">
          <textarea
            className="chat-input"
            rows={3}
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            placeholder="输入消息…（Enter 发送可用下方按钮）"
          />
          <button type="button" className="primary" onClick={sendAgentChat} disabled={agentLoading}>
            {agentLoading ? '请求中…' : '发送'}
          </button>
        </div>
        {pendingSuggested.length > 0 && (
          <div className="suggested-box">
            <p className="suggested-title">本轮解析到的动作（来自 JSON）</p>
            <ul className="suggested-list">
              {pendingSuggested.map((s, i) => (
                <li key={i}>{s.display_name}</li>
              ))}
            </ul>
            <button type="button" className="primary" onClick={addSuggestedToCustom}>
              全部加入自定义动作列表
            </button>
          </div>
        )}
      </section>

      <section className="panel panel-tight train-panel">
        <h2>VON 模型训练（自定义数据 + 度量）</h2>
        <p className="hint train-hint-short">
          <code>mission=user_pkl</code>，<strong>metric</strong>：<code>tsp</code> 或 <code>moransI</code>。完成后可将产出目录设为当前排序模型。
        </p>
        <textarea
          className="train-json"
          value={trainJson}
          onChange={(e) => setTrainJson(e.target.value)}
          spellCheck={false}
          aria-label="训练任务 JSON"
        />
        <div className="toolbar-inline" style={{ marginTop: '0.5rem' }}>
          <button type="button" className="primary" onClick={startVonTraining} disabled={trainSubmitting}>
            {trainSubmitting ? '提交中…' : '一键开始训练'}
          </button>
          {trainJobId && (
            <span className="train-job-id">
              任务 ID: <code>{trainJobId}</code>
            </span>
          )}
        </div>
        <div className="train-status">
          <div className="train-status-head">
            {trainJob ? (
              <>
                {trainJob.pretrained_dir && trainJob.status === 'done' && (
                  <p>
                    产出目录：<code>{trainJob.pretrained_dir}</code>{' '}
                    <button type="button" onClick={() => setActiveFromPath(trainJob.pretrained_dir!)}>
                      设为当前排序模型
                    </button>
                  </p>
                )}
              </>
            ) : (
              <p className="muted">尚未提交训练任务。</p>
            )}
          </div>
          <div className="train-log-region" aria-label="训练日志输出">
            <pre className="train-log">{trainJob?.log || '（尚无日志）'}</pre>
          </div>
        </div>
      </section>
          </div>
        </aside>

        <div className="col-center">
      <section className="panel panel-tight custom-panel">
        <h2>自定义办公动作（{customAtoms.length}）</h2>
        <p className="hint">可与内置动作一起勾选参与 VON 排序。</p>
        <div className="manual-add">
          <input
            type="text"
            value={manualCustomName}
            onChange={(e) => setManualCustomName(e.target.value)}
            placeholder="手动添加一条动作描述…"
          />
          <button type="button" onClick={addManualCustom}>
            添加
          </button>
        </div>
        {customAtoms.length === 0 ? (
          <p className="muted">暂无自定义动作。</p>
        ) : (
          <ul className="custom-list">
            {customAtoms.map((a) => (
              <li key={a.id}>
                <span className="custom-name">{a.display_name}</span>
                <code>{a.id}</code>
                <button type="button" className="btn-danger" onClick={() => deleteCustom(a.id)}>
                  删除
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel panel-tight action-select-panel">
        <h2>选择动作（勾选 {selected.length} / {atoms.length}）</h2>
        <div className="search-row">
          <input
            type="search"
            className="search-input"
            placeholder="按名称或 ID 筛选…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            aria-label="筛选动作"
          />
          <div className="scope-toggle" role="group" aria-label="动作范围">
            <button type="button" className={scope === 'all' ? 'active' : ''} onClick={() => setScope('all')}>
              全部
            </button>
            <button
              type="button"
              className={scope === 'catalog' ? 'active' : ''}
              onClick={() => setScope('catalog')}
            >
              仅内置
            </button>
            <button
              type="button"
              className={scope === 'custom' ? 'active' : ''}
              onClick={() => setScope('custom')}
            >
              仅自定义
            </button>
          </div>
          <div className="toolbar-inline">
            <button type="button" onClick={() => setSelected(filteredAtoms.map((x) => x.id))}>
              勾选当前筛选
            </button>
            <button type="button" onClick={() => setSelected(atoms.map((x) => x.id))}>
              全选
            </button>
            <button type="button" onClick={() => setSelected([])}>
              清空勾选
            </button>
            <button
              type="button"
              className={`order-pretrained-input${!orderPretrainedDir && !pickDirBusy ? ' order-pretrained-placeholder' : ''}`}
              onClick={pickPretrainedDir}
              disabled={pickDirBusy || vonLoading}
              aria-label="选择 VON 预训练模型目录"
              title={orderPretrainedDir || '在运行后端的电脑上打开系统文件夹选择'}
            >
              {pickDirBusy ? '正在打开…' : orderPretrainedDir || '点击选择模型目录…'}
            </button>
            <button
              type="button"
              className="primary"
              onClick={runVonOrder}
              disabled={vonLoading || selected.length < 2}
            >
              {vonLoading ? '排序中…' : '生成推荐顺序'}
            </button>
          </div>
        </div>
        <div className="action-grid" role="list">
          {filteredAtoms.map((a) => (
            <label key={a.id} className="action-chip" role="listitem">
              <input
                type="checkbox"
                checked={selected.includes(a.id)}
                onChange={() => toggle(a.id)}
              />
              <span className="action-meta">
                <span className="action-title">
                  <span className="action-title-text" title={a.display_name}>
                    {a.display_name}
                  </span>
                  {a.source === 'custom' && (
                    <span className="badge-custom" title="自定义动作">
                      自定义
                    </span>
                  )}
                </span>
                <span className="action-id" title={a.id}>
                  {a.id}
                </span>
              </span>
            </label>
          ))}
        </div>
        {filter && (
          <p className="hint">当前筛选 {filteredAtoms.length} 条；仍可对全部已勾选项排序。</p>
        )}
      </section>
        </div>

        <div className="col-right">
      <section
        className="panel recommend recommend-list-panel panel-tight"
        aria-label="推荐顺序与待办清单"
      >
        <div className="recommend-list-head">
          <div className="right-top-tabs" role="tablist" aria-label="右上列表切换">
            <button
              type="button"
              role="tab"
              aria-selected={rightListTab === 'recommend'}
              className={rightListTab === 'recommend' ? 'active' : ''}
              onClick={() => setRightListTab('recommend')}
            >
              推荐顺序
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={rightListTab === 'todo'}
              className={rightListTab === 'todo' ? 'active' : ''}
              onClick={() => setRightListTab('todo')}
            >
              待办清单
            </button>
          </div>
        </div>
        <div className="recommend-list-scroll">
          {rightListTab === 'recommend' ? (
            vonResult ? (
              <ol className="recommend-list">
                {vonResult.ordered_atom_ids.map((id, i) => (
                  <li key={`${i}-${id}`}>
                    <span className="step-num">{i + 1}</span>
                    <span className="step-name">{nameById.get(id) ?? id}</span>
                    <code className="step-id">{id}</code>
                  </li>
                ))}
              </ol>
            ) : (
              <div className="recommend-list-empty" aria-hidden="true" />
            )
          ) : vonResult ? (
            <div className="todo-template">
              <p className="todo-template-tip">左侧勾选跟踪，右侧为 Markdown 原文；可复制到备忘录 / Notion。</p>
              <div className="todo-template-body">
                <div className="todo-template-col todo-template-col-left">
                  <ul className="todo-checklist">
                    {vonResult.ordered_atom_ids.map((id, i) => (
                      <li key={`todo-${i}-${id}`}>
                        <label className="todo-checklist-row">
                          <input type="checkbox" />
                          <span className="todo-checklist-text">
                            {i + 1}. {nameById.get(id) ?? id}
                          </span>
                        </label>
                      </li>
                    ))}
                  </ul>
                  <button type="button" className="btn-secondary todo-copy-btn" onClick={copyTodoTemplate}>
                    {todoCopied ? '已复制' : '复制 Markdown'}
                  </button>
                </div>
                <div className="todo-template-col todo-template-col-right">
                  <div className="todo-template-md-label">Markdown</div>
                  <pre className="todo-template-pre">{todoTemplateText}</pre>
                </div>
              </div>
            </div>
          ) : (
            <p className="muted todo-tab-empty">暂无，请先在中间栏生成推荐顺序。</p>
          )}
        </div>
        {vonResult && (
          <p className="hint cost recommend-cost">
            模型报告代价 cost: {vonResult.cost?.toFixed?.(4) ?? vonResult.cost}
            {vonResult.process && (
              <>
                {' '}
                · 设备 {vonResult.process.device}
              </>
            )}
          </p>
        )}
      </section>

      <section className="panel recommend recommend-detail-panel panel-tight">
        <div className="chart-box chart-box-fixed">
          <ReactECharts
            option={chartOption ?? emptyChartOption}
            style={{ height: '100%', minHeight: 200 }}
            notMerge
            lazyUpdate
          />
        </div>

            <section className="process-panel" aria-label="排序过程详情">
              <div className="process-tabs" role="tablist" aria-label="过程信息">
                {PROCESS_TABS.map((t) => (
                  <button
                    key={t.id}
                    type="button"
                    role="tab"
                    id={`process-tab-${t.id}`}
                    aria-selected={processTab === t.id}
                    aria-controls={`process-panel-${t.id}`}
                    className={processTab === t.id ? 'active' : ''}
                    onClick={() => setProcessTab(t.id)}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
              <div
                className="process-tab-panel"
                role="tabpanel"
                id={`process-panel-${processTab}`}
                aria-labelledby={`process-tab-${processTab}`}
              >
                {(() => {
                  const p = vonResult?.process
                  if (!p) {
                    return <p className="muted process-tab-empty">暂无内容，完成排序后在此展示。</p>
                  }
                  switch (processTab) {
                    case 'pipeline':
                      return (
                        <ol className="process-pipeline">
                          {p.pipeline.map((step) => (
                            <li key={step.step}>
                              <strong>
                                {step.step}. {step.name}
                              </strong>
                              <p>{step.detail}</p>
                            </li>
                          ))}
                        </ol>
                      )
                    case 'encoding':
                      return (
                        <>
                          <p className="process-mini">
                            内置 catalog：<code>{p.catalog_path}</code>
                            {p.custom_actions_path && (
                              <>
                                <br />
                                自定义动作：<code>{p.custom_actions_path}</code>
                              </>
                            )}
                          </p>
                          <div className="table-wrap">
                            <table className="data-table">
                              <thead>
                                <tr>
                                  <th>#</th>
                                  <th>动作 ID</th>
                                  <th>名称</th>
                                  <th>x</th>
                                  <th>y</th>
                                </tr>
                              </thead>
                              <tbody>
                                {p.encoding_input_order.map((row) => (
                                  <tr key={row.input_index}>
                                    <td>{row.input_index}</td>
                                    <td>
                                      <code>{row.atom_id}</code>
                                    </td>
                                    <td>{row.display_name}</td>
                                    <td>{row.x}</td>
                                    <td>{row.y}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </>
                      )
                    case 'loc':
                      return (
                        <>
                          <p className="process-mini">
                            shape = [{p.loc_tensor.shape.join(', ')}]，{p.loc_tensor.dtype}
                          </p>
                          <pre className="json-pre">{JSON.stringify(p.loc_tensor.values_row_major, null, 2)}</pre>
                        </>
                      )
                    case 'permutation':
                      return (
                        <>
                          <p className="process-mini">{p.permutation_note}</p>
                          <pre className="json-pre">{JSON.stringify(p.permutation_pi)}</pre>
                          {vonResult.index_order && (
                            <p className="process-mini">
                              与 index_order 字段一致：{JSON.stringify(vonResult.index_order)}
                            </p>
                          )}
                        </>
                      )
                    case 'geometry':
                      return (
                        <>
                          <ul className="process-list">
                            <li>
                              沿输出顺序相邻边长：[{p.euclidean.edge_lengths_along_output_order.join(', ')}]
                            </li>
                            <li>开放路径欧氏总长：{p.euclidean.tour_length_open}</li>
                            <li>闭合回路欧氏总长（若首尾相连）：{p.euclidean.tour_length_closed}</li>
                            <li>
                              模型内部 cost（训练 metric）：{p.model_cost_reported.toFixed(6)}
                              （与欧氏长不一定相同）
                            </li>
                          </ul>
                          {p.point_stats.centroid && (
                            <p className="process-mini">
                              质心：({p.point_stats.centroid.x}, {p.point_stats.centroid.y})
                              {p.point_stats.bbox && (
                                <>
                                  {' '}
                                  · 包围盒 x∈[{p.point_stats.bbox.x_min},{p.point_stats.bbox.x_max}] y∈[
                                  {p.point_stats.bbox.y_min},{p.point_stats.bbox.y_max}]
                                </>
                              )}
                            </p>
                          )}
                        </>
                      )
                    case 'pairwise':
                      if (!p.pairwise_euclidean) {
                        return (
                          <p className="muted process-tab-empty">
                            暂无成对欧氏距离矩阵（例如 N&gt;28 或未返回）。
                          </p>
                        )
                      }
                      return (
                        <>
                          <p className="process-mini">{p.pairwise_note}</p>
                          <div className="table-wrap table-scroll">
                            <table className="data-table dense">
                              <thead>
                                <tr>
                                  <th />
                                  {p.encoding_input_order.map((_, i) => (
                                    <th key={i}>{i}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {p.pairwise_euclidean.map((row, i) => (
                                  <tr key={i}>
                                    <th>{i}</th>
                                    {row.map((v, j) => (
                                      <td key={j}>{v}</td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </>
                      )
                    case 'model':
                      return (
                        <div className="process-block-light">
                          <p className="process-mini">
                            <code>{p.model.pretrained_dir}</code>
                          </p>
                          <pre className="json-pre">{JSON.stringify(p.model.args, null, 2)}</pre>
                        </div>
                      )
                    default:
                      return null
                  }
                })()}
              </div>
            </section>
      </section>

        </div>
      </div>

      <footer className="footer">
        后端 API: <code>{API_BASE}</code> · VON 项目{' '}
        <a href="https://github.com/sysuvis/VON" target="_blank" rel="noreferrer">
          sysuvis/VON
        </a>
      </footer>
    </div>
  )
}
