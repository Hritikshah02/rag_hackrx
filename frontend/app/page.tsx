'use client'

import { useState, useCallback } from 'react'

interface QAPair {
  question: string
  answer: string
}

export default function Home() {
  const [documentUrl, setDocumentUrl] = useState('')
  const [questions, setQuestions] = useState<string[]>([''])
  const [results, setResults] = useState<QAPair[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null)
  const [processingStep, setProcessingStep] = useState(0)

  const addQuestion = useCallback(() => {
    setQuestions(q => [...q, ''])
  }, [])

  const removeQuestion = useCallback((i: number) => {
    setQuestions(q => q.filter((_, idx) => idx !== i))
  }, [])

  const updateQuestion = useCallback((i: number, val: string) => {
    setQuestions(q => {
      const next = [...q]
      next[i] = val
      return next
    })
  }, [])

  const copyAnswer = useCallback((i: number, text: string) => {
    navigator.clipboard.writeText(text)
    setCopiedIdx(i)
    setTimeout(() => setCopiedIdx(null), 2000)
  }, [])

  const handleSubmit = async () => {
    const filtered = questions.filter(q => q.trim())
    if (!documentUrl.trim()) {
      setError('Document URL is required.')
      return
    }
    if (filtered.length === 0) {
      setError('At least one query is required.')
      return
    }

    setLoading(true)
    setError('')
    setResults([])
    setProcessingStep(0)

    // Simulate step progression for UX
    const stepTimer = setInterval(() => {
      setProcessingStep(s => Math.min(s + 1, 2))
    }, 3000)

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ documents: documentUrl.trim(), questions: filtered }),
      })

      clearInterval(stepTimer)
      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.detail || `Server error: ${res.status}`)
      }

      setResults(
        filtered.map((q, i) => ({
          question: q,
          answer: (data.answers?.[i] as string) || 'No answer returned.',
        }))
      )
    } catch (err) {
      clearInterval(stepTimer)
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setLoading(false)
      setProcessingStep(0)
    }
  }

  const steps = [
    'Fetching & parsing document...',
    'Running semantic search across chunks...',
    'Generating answers via LLM...',
  ]

  return (
    <div className="min-h-screen bg-[#050508]">
      {/* Background grid */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage:
            'linear-gradient(rgba(6,182,212,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(6,182,212,0.03) 1px, transparent 1px)',
          backgroundSize: '48px 48px',
        }}
      />

      {/* Top accent line */}
      <div className="fixed top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-60" />

      <div className="relative max-w-2xl mx-auto px-5 pt-12 pb-20">

        {/* ── Header ── */}
        <header className="mb-10">
          <h1 className="text-3xl font-bold tracking-tight leading-tight">
            <span className="text-cyan-400">Agentic</span>
            <span className="text-slate-300"> Document QA System</span>
          </h1>
        </header>

        {/* ── Input panel ── */}
        <section className="rounded-2xl border border-slate-800 bg-[#090b14] p-6 mb-5 shadow-xl">

          {/* Document URL */}
          <div className="mb-6">
            <label className="flex items-center gap-2 mb-2">
              <span className="text-cyan-500 text-xs">01</span>
              <span className="text-[10px] uppercase tracking-widest text-slate-500">
                Document Source
              </span>
            </label>
            <div className="relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-600 text-sm select-none">
                →
              </span>
              <input
                type="url"
                value={documentUrl}
                onChange={e => setDocumentUrl(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSubmit()}
                placeholder="https://example.com/document.pdf"
                spellCheck={false}
                className="w-full bg-[#050508] border border-slate-800 rounded-xl pl-8 pr-4 py-3 text-sm text-slate-200 placeholder-slate-700 focus:border-cyan-700 focus:outline-none transition-colors"
              />
            </div>
          </div>

          {/* Questions */}
          <div>
            <label className="flex items-center gap-2 mb-3">
              <span className="text-cyan-500 text-xs">02</span>
              <span className="text-[10px] uppercase tracking-widest text-slate-500">
                Queries
              </span>
              <span className="ml-auto text-[10px] text-slate-700">
                {questions.filter(q => q.trim()).length} active
              </span>
            </label>

            <div className="space-y-2">
              {questions.map((q, i) => (
                <div key={i} className="flex items-center gap-2 group">
                  <span className="text-[10px] text-slate-800 group-hover:text-slate-600 transition-colors w-5 text-right shrink-0 select-none">
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <input
                    type="text"
                    value={q}
                    onChange={e => updateQuestion(i, e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') {
                        e.preventDefault()
                        if (i === questions.length - 1) addQuestion()
                      }
                    }}
                    placeholder={`Query ${i + 1}…`}
                    spellCheck={false}
                    className="flex-1 bg-[#050508] border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-800 focus:border-cyan-800 focus:outline-none transition-colors"
                  />
                  {questions.length > 1 && (
                    <button
                      onClick={() => removeQuestion(i)}
                      className="opacity-0 group-hover:opacity-100 text-slate-700 hover:text-red-500 transition-all text-xs w-5 shrink-0"
                      title="Remove query"
                    >
                      ✕
                    </button>
                  )}
                </div>
              ))}
            </div>

            <div className="flex items-center justify-between mt-4">
              <button
                onClick={addQuestion}
                className="text-[11px] text-slate-600 hover:text-cyan-400 transition-colors flex items-center gap-1.5"
              >
                <span className="text-base leading-none">+</span>
                <span>add query</span>
              </button>

              <button
                onClick={handleSubmit}
                disabled={loading}
                className="
                  flex items-center gap-2 px-6 py-2 rounded-lg text-xs font-bold
                  bg-cyan-500 text-black
                  hover:bg-cyan-400
                  disabled:bg-slate-800 disabled:text-slate-600 disabled:cursor-not-allowed
                  transition-all shadow-lg shadow-cyan-900/20
                "
              >
                {loading ? (
                  <>
                    <span className="w-3 h-3 border-2 border-slate-500 border-t-transparent rounded-full animate-spin" />
                    running
                  </>
                ) : (
                  <>▶ run</>
                )}
              </button>
            </div>
          </div>
        </section>

        {/* ── Error ── */}
        {error && (
          <div className="fade-in-up rounded-xl border border-red-900/60 bg-red-950/20 px-4 py-3 mb-5 flex items-start gap-3">
            <span className="text-red-500 text-sm shrink-0 mt-0.5">✗</span>
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        {/* ── Loading state ── */}
        {loading && (
          <div className="fade-in-up rounded-2xl border border-slate-800 bg-[#090b14] p-6 mb-5">
            <div className="flex items-center gap-3 mb-4">
              <div className="flex gap-1">
                {[0, 1, 2].map(i => (
                  <span
                    key={i}
                    className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-bounce"
                    style={{ animationDelay: `${i * 120}ms` }}
                  />
                ))}
              </div>
              <span className="text-sm text-slate-400">Processing</span>
            </div>

            <div className="space-y-2">
              {steps.map((step, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span
                    className={`text-xs transition-colors ${
                      i < processingStep
                        ? 'text-green-500'
                        : i === processingStep
                        ? 'text-cyan-400'
                        : 'text-slate-800'
                    }`}
                  >
                    {i < processingStep ? '✓' : i === processingStep ? '›' : '·'}
                  </span>
                  <span
                    className={`text-xs transition-colors ${
                      i < processingStep
                        ? 'text-slate-500 line-through'
                        : i === processingStep
                        ? 'text-slate-300'
                        : 'text-slate-800'
                    }`}
                  >
                    {step}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Results ── */}
        {results.length > 0 && (
          <div className="fade-in-up">
            <div className="flex items-center gap-3 mb-4 px-1">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
              <span className="text-xs text-slate-500">
                {results.length} answer{results.length !== 1 ? 's' : ''} returned
              </span>
              <span className="ml-auto text-[10px] text-slate-700">
                {new Date().toLocaleTimeString()}
              </span>
            </div>

            <div className="space-y-3">
              {results.map((r, i) => (
                <div
                  key={i}
                  className="result-card fade-in-up rounded-2xl border border-slate-800 bg-[#090b14] overflow-hidden shadow-lg"
                  style={{ animationDelay: `${i * 60}ms` }}
                >
                  {/* Question header */}
                  <div className="flex items-start gap-3 px-5 py-3.5 border-b border-slate-800/80 bg-[#0c0f1c]">
                    <span className="text-[10px] text-cyan-700 pt-0.5 shrink-0 w-6">
                      Q{String(i + 1).padStart(2, '0')}
                    </span>
                    <p className="text-sm text-slate-300 leading-snug">{r.question}</p>
                  </div>

                  {/* Answer body */}
                  <div className="px-5 py-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <span className="w-1 h-1 rounded-full bg-cyan-500" />
                        <span className="text-[9px] uppercase tracking-widest text-cyan-700">
                          Answer
                        </span>
                      </div>
                      <button
                        onClick={() => copyAnswer(i, r.answer)}
                        className="text-[10px] text-slate-700 hover:text-cyan-400 transition-colors flex items-center gap-1"
                      >
                        {copiedIdx === i ? (
                          <><span className="text-green-400">✓</span> copied</>
                        ) : (
                          <>copy</>
                        )}
                      </button>
                    </div>

                    <p className="answer-text text-sm text-slate-200 whitespace-pre-wrap leading-relaxed">
                      {r.answer}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            {/* Footer action */}
            <div className="mt-5 flex justify-center">
              <button
                onClick={() => {
                  setResults([])
                  setQuestions([''])
                  setDocumentUrl('')
                }}
                className="text-xs text-slate-700 hover:text-slate-500 transition-colors"
              >
                ↺ reset
              </button>
            </div>
          </div>
        )}

      </div>
    </div>
  )
}
