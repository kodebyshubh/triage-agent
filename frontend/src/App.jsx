import { useState } from 'react'

const API_URL = 'http://127.0.0.1:8000'

const ACTION_STYLES = {
  auto_resolved: { label: 'Auto-resolved', color: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' },
  escalated: { label: 'Escalated', color: 'bg-amber-500/15 text-amber-400 border-amber-500/30' },
  escalated_due_to_error: { label: 'Escalated (transport error)', color: 'bg-red-500/15 text-red-400 border-red-500/30' },
  escalated_output_blocked: { label: 'Escalated (output blocked)', color: 'bg-red-500/15 text-red-400 border-red-500/30' },
}

function TraceStep({ step }) {
  return (
    <div className="relative pl-6 pb-6 last:pb-0 border-l border-white/10 last:border-transparent">
      <div className="absolute -left-[5px] top-1 h-[9px] w-[9px] rounded-full bg-violet-400" />
      <div className="text-sm font-medium text-white/90">{step.agent_name}</div>
      <div className="text-xs text-white/40 mt-0.5">{step.timestamp}</div>
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

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const resp = await fetch(`${API_URL}/ticket`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticket_id: `ticket-${Date.now()}`,
          user_message: message,
          session_id: sessionId,
        }),
      })
      const data = await resp.json()
      if (!resp.ok) {
        setError(data.detail ? JSON.stringify(data.detail) : 'Request failed')
      } else {
        setResult(data)
      }
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

      {result && (
        <div className="mt-10 space-y-8">
          <div>
            <h2 className="text-sm font-medium text-white/60 mb-3">Raw response</h2>
            <pre className="text-xs bg-white/5 border border-white/10 rounded-md p-3 overflow-x-auto whitespace-pre-wrap break-words">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>

          <div>
            <h2 className="text-sm font-medium text-white/60 mb-3">Agent trace</h2>
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
