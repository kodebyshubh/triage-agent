import { useState } from 'react'

// Set at build time (Vite env). Falls back to the local backend for `npm run dev`.
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
// Optional: only needed when the backend has TRIAGEFORGE_API_KEYS set. Anything in
// a VITE_ variable ships inside the JS bundle, so treat this as a demo key only.
const API_KEY = import.meta.env.VITE_API_KEY || ''

const ACTION_STYLES = {
  auto_resolved: { label: 'Auto-resolved', color: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' },
  escalated: { label: 'Escalated', color: 'bg-amber-500/15 text-amber-400 border-amber-500/30' },
  escalated_due_to_error: { label: 'Escalated (transport error)', color: 'bg-red-500/15 text-red-400 border-red-500/30' },
  escalated_output_blocked: { label: 'Escalated (output blocked)', color: 'bg-red-500/15 text-red-400 border-red-500/30' },
  escalated_incomplete_lookup: { label: 'Escalated (incomplete lookup)', color: 'bg-red-500/15 text-red-400 border-red-500/30' },
}

// Reads a Server-Sent Events stream from a fetch() Response (EventSource only
// supports GET, and the ticket endpoint is a POST). Calls onEvent(name, data).
async function readEventStream(resp, onEvent) {
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      let event = 'message'
      let data = ''
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7)
        else if (line.startsWith('data: ')) data += line.slice(6)
      }
      if (data) onEvent(event, JSON.parse(data))
    }
  }
}

function TraceStep({ step }) {
  return (
    <div className="relative pl-6 pb-6 last:pb-0 border-l border-white/10 last:border-transparent">
      <div className="absolute -left-[5px] top-1 h-[9px] w-[9px] rounded-full bg-violet-400" />
      <div className="text-sm font-medium text-white/90">{step.agent_name}</div>
      <div className="text-xs text-white/40 mt-0.5">
        {step.timestamp}
        {step.latency_ms != null && <span className="ml-2">{Math.round(step.latency_ms)} ms</span>}
      </div>
      <pre className="mt-2 text-xs bg-white/5 border border-white/10 rounded-md p-3 overflow-x-auto whitespace-pre-wrap break-words">
        {JSON.stringify(step.output, null, 2)}
      </pre>
    </div>
  )
}

function ResultCard({ result }) {
  const style = ACTION_STYLES[result.final_action] ?? { label: result.final_action, color: 'bg-white/10 text-white/70 border-white/20' }
  const r = result.response
  const isDraft = result.final_action === 'auto_resolved'

  return (
    <div className="border border-white/10 rounded-lg p-4 bg-white/[0.03]">
      <span className={`inline-block text-xs font-medium px-2 py-1 rounded-full border ${style.color}`}>
        {style.label}
      </span>

      {isDraft ? (
        <div className="mt-3 space-y-2">
          <p className="text-sm leading-relaxed">{r.draft_text}</p>
          <p className="text-xs text-white/40">Confidence: {r.confidence.toFixed(2)}</p>
          {r.citations?.length > 0 && (
            <p className="text-xs text-white/40">Sources: {[...new Set(r.citations)].join(', ')}</p>
          )}
        </div>
      ) : (
        <div className="mt-3 space-y-2">
          <p className="text-sm leading-relaxed">{r.summary}</p>
          <p className="text-xs text-white/40">Suggested priority: {r.suggested_priority}</p>
          <p className="text-xs text-white/40">Reason: {r.flagged_reason}</p>
        </div>
      )}
    </div>
  )
}

function App() {
  const [message, setMessage] = useState('')
  const [sessionId, setSessionId] = useState('demo-session')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  // Filled while the request streams; replaced by result.trace once it finishes.
  const [liveSteps, setLiveSteps] = useState([])
  const [liveStatus, setLiveStatus] = useState([])

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    setLiveSteps([])
    setLiveStatus([])

    try {
      const resp = await fetch(`${API_URL}/ticket/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
        },
        body: JSON.stringify({
          ticket_id: `ticket-${Date.now()}`,
          user_message: message,
          session_id: sessionId,
        }),
      })
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}))
        setError(data.detail ? JSON.stringify(data.detail) : `Request failed (${resp.status})`)
        return
      }
      await readEventStream(resp, (event, data) => {
        if (event === 'input_guardrail') {
          setLiveStatus((s) => [...s, data.is_safe ? 'Input guardrail passed' : `Input guardrail rejected (${data.flags.join(', ')})`])
        } else if (event === 'step') {
          setLiveSteps((s) => [...s, data])
        } else if (event === 'output_guardrail') {
          setLiveStatus((s) => [...s, data.passed ? 'Output guardrail passed' : 'Output guardrail blocked the reply'])
        } else if (event === 'final') {
          setResult(data)
        } else if (event === 'error') {
          setError(data.detail)
        }
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen max-w-2xl mx-auto px-4 py-10">
      <h1 className="text-2xl font-semibold text-white/90">TriageForge</h1>
      <p className="text-sm text-white/40 mt-1">Support ticket triage with agent trace</p>

      <form onSubmit={handleSubmit} className="mt-8 space-y-3">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Describe the issue..."
          rows={4}
          required
          className="w-full bg-white/5 border border-white/10 rounded-md p-3 text-sm placeholder:text-white/30 focus:outline-none focus:border-violet-400/50"
        />
        <input
          value={sessionId}
          onChange={(e) => setSessionId(e.target.value)}
          placeholder="Session ID"
          required
          className="w-full bg-white/5 border border-white/10 rounded-md p-3 text-sm placeholder:text-white/30 focus:outline-none focus:border-violet-400/50"
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-violet-500 hover:bg-violet-400 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium px-4 py-2 rounded-md transition-colors"
        >
          {loading ? 'Submitting...' : 'Submit ticket'}
        </button>
      </form>

      {error && (
        <p className="mt-6 text-sm text-red-400 bg-red-500/10 border border-red-500/30 rounded-md p-3">
          {error}
        </p>
      )}

      {loading && (liveStatus.length > 0 || liveSteps.length > 0) && (
        <div className="mt-10">
          <h2 className="text-sm font-medium text-white/60 mb-3">Live trace</h2>
          {liveStatus.map((s, i) => (
            <p key={i} className="text-xs text-white/50 mb-2">{s}</p>
          ))}
          <div>
            {liveSteps.map((step, i) => (
              <TraceStep key={i} step={step} />
            ))}
          </div>
          <p className="text-xs text-white/30 mt-3 animate-pulse">Waiting for the next step...</p>
        </div>
      )}

      {result && (
        <div className="mt-10 space-y-8">
          <div>
            <h2 className="text-sm font-medium text-white/60 mb-3">Raw response</h2>
            <pre className="text-xs bg-white/5 border border-white/10 rounded-md p-3 overflow-x-auto whitespace-pre-wrap break-words">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>

          <div>
            <h2 className="text-sm font-medium text-white/60 mb-3">
              Agent trace
              {result.latency_ms != null && (
                <span className="ml-2 text-white/30 font-normal">total {(result.latency_ms / 1000).toFixed(1)} s</span>
              )}
            </h2>
            {result.trace.length > 0 ? (
              <div>
                {result.trace.map((step, i) => (
                  <TraceStep key={i} step={step} />
                ))}
              </div>
            ) : (
              <p className="text-xs text-white/40">No trace - rejected before reaching the orchestrator.</p>
            )}
          </div>

          <div>
            <h2 className="text-sm font-medium text-white/60 mb-3">Result</h2>
            <ResultCard result={result} />
          </div>
        </div>
      )}
    </div>
  )
}

export default App
