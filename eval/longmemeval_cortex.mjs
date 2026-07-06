#!/usr/bin/env node
/**
 * CORTEX × LongMemEval Adapter v1.1
 * ====================================
 * Conecta el benchmark oficial LongMemEval (ICLR 2025) a CORTEX via MCP stdio.
 * No modifica ningún archivo de src/ — solo llama las herramientas MCP existentes.
 *
 * Dataset oficial: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
 *
 * FORMATO REAL DEL DATASET (confirmado):
 *   {
 *     question_id:          string
 *     question_type:        string  (single-session-user, temporal-reasoning, ...)
 *     question:             string
 *     answer:               string
 *     question_date:        string
 *     haystack_session_ids: string[]
 *     haystack_dates:       string[]
 *     haystack_sessions:    Array<Array<{role, content, has_answer?}>>
 *     answer_session_ids:   string[]
 *   }
 *
 * FLUJO:
 *   1. Lee dataset JSON
 *   2. Por cada caso: ingesta haystack_sessions via batch_observe → index_temp
 *   3. Lanza recall + get_context_for con la question
 *   4. Evaluación local por keyword match (sin LLM)
 *   5. Guarda outputs en formato JSONL para evaluate_qa.py oficial
 *
 * USO:
 *   node eval/longmemeval_cortex.mjs
 *   node eval/longmemeval_cortex.mjs --dataset /tmp/LongMemEval/data/longmemeval_oracle.json
 *   node eval/longmemeval_cortex.mjs --dataset longmemeval_oracle.json --limit 50
 *   node eval/longmemeval_cortex.mjs --limit 50 --k 3 --no-cleanup
 *
 * PARA EL SCORER OFICIAL (requiere OpenAI API):
 *   cd /tmp/LongMemEval/src/evaluation
 *   python3 evaluate_qa.py gpt-4o ../../<out>.jsonl ../../data/longmemeval_oracle.json
 */

import { spawn }       from 'child_process'
import { readFileSync, writeFileSync, mkdirSync, existsSync, appendFileSync } from 'fs'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dir = dirname(fileURLToPath(import.meta.url))
const SERVER = resolve(__dir, '../dist/server.js')

// ─── CLI args ─────────────────────────────────────────────────────────────────

const argv    = process.argv.slice(2)
const getArg  = (f) => { const i = argv.indexOf(f); return i !== -1 ? argv[i + 1] : null }
const hasFlag = (f) => argv.includes(f)

const DATASET_PATH = getArg('--dataset')
  || resolve(__dir, '../data/longmemeval_oracle.json')
const LIMIT        = Number(getArg('--limit') || 0)      // 0 = sin límite
const K            = Number(getArg('--k') || 5)
const NO_CLEANUP   = hasFlag('--no-cleanup')
const TOOL_TIMEOUT = Number(getArg('--timeout') || 60_000)
const PROJECT_BASE = 'lme'

// ─── Outputs ──────────────────────────────────────────────────────────────────

const RESULTS_DIR = resolve(__dir, 'results')
if (!existsSync(RESULTS_DIR)) mkdirSync(RESULTS_DIR, { recursive: true })

const TS       = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
// JSONL para evaluate_qa.py oficial (campos: question_id + hypothesis)
const JSONL_OUT = resolve(RESULTS_DIR, `longmemeval_cortex_${TS}.jsonl`)
// JSON completo con métricas locales
const JSON_OUT  = resolve(RESULTS_DIR, `longmemeval_cortex_${TS}.full.json`)
const SUM_OUT   = resolve(RESULTS_DIR, `longmemeval_cortex_${TS}.summary.json`)

// ─── MCP runner ───────────────────────────────────────────────────────────────

let reqId = 0

async function callTool(name, toolArgs = {}) {
  return new Promise((res, rej) => {
    const proc = spawn('node', [SERVER], { stdio: ['pipe', 'pipe', 'inherit'] })
    const id   = ++reqId
    const req  = JSON.stringify({
      jsonrpc: '2.0', id,
      method: 'tools/call',
      params: { name, arguments: toolArgs },
    })
    let out = ''
    proc.stdout.on('data', d => { out += d })
    proc.on('close', () => {
      const lines = out.trim().split('\n')
      for (const line of lines.reverse()) {
        try {
          const p = JSON.parse(line)
          if (p.id === id || p.result !== undefined || p.error !== undefined) {
            res(p); return
          }
        } catch { /* ignorar líneas no-JSON */ }
      }
      rej(new Error(`Sin respuesta JSON para tool=${name}\nout: ${out.slice(0, 200)}`))
    })
    proc.stdin.write(req + '\n')
    proc.stdin.end()
    setTimeout(() => { proc.kill(); rej(new Error(`Timeout ${TOOL_TIMEOUT}ms en ${name}`)) }, TOOL_TIMEOUT)
  })
}

const getText = (r) => r?.result?.content?.[0]?.text ?? ''

// ─── Ingesta: haystack_sessions → CORTEX ────────────────────────────────────
//
// haystack_sessions es Array<Array<{role, content, has_answer?}>>
// Cada sesión (índice i) tiene fecha haystack_dates[i].
// Agrupamos turnos user+assistant en pares para más contexto semántico.

async function ingestCase(projectName, item) {
  const sessions = item.haystack_sessions
  const dates    = item.haystack_dates ?? []
  const memories = []

  for (let si = 0; si < sessions.length; si++) {
    const session  = sessions[si]
    const dateTag  = dates[si] ? `[${dates[si]}] ` : ''

    for (let ti = 0; ti < session.length; ti++) {
      const turn = session[ti]
      if (!turn?.content?.trim()) continue

      // Solo turnos de usuario (son los que contienen los hechos que el sistema debe recordar)
      // Añadimos el contexto del asistente si sigue inmediatamente
      if (turn.role === 'user') {
        const nextTurn = session[ti + 1]
        if (nextTurn?.role === 'assistant' && nextTurn.content?.trim()) {
          memories.push(`${dateTag}User: ${turn.content.trim()} | Assistant: ${nextTurn.content.trim()}`)
        } else {
          memories.push(`${dateTag}User: ${turn.content.trim()}`)
        }
      }
    }
  }

  if (memories.length === 0) return 0

  // batch_observe: rápido, sin LLM, sin embedding — va al buffer temporal
  await callTool('batch_observe', { projectName, memories })
  return memories.length
}

// ─── Evaluación local por keyword ────────────────────────────────────────────
//
// El scorer oficial (evaluate_qa.py) usa GPT-4o como juez.
// Aquí hacemos una evaluación por substring/keyword para métricas rápidas.

function localScore(groundTruth, systemOutput) {
  const gt = String(groundTruth).toLowerCase().trim()
  if (!gt) return 0

  const corpus = systemOutput.toLowerCase()

  // Match exacto
  if (corpus.includes(gt)) return 1.0

  // Match por palabras significativas (todas presentes)
  const words = gt.split(/\s+/).filter(w => w.length > 3)
  if (words.length > 0 && words.every(w => corpus.includes(w))) return 0.8

  // Match parcial (≥60% de palabras)
  if (words.length > 1) {
    const hits = words.filter(w => corpus.includes(w)).length
    if (hits / words.length >= 0.6) return 0.5
  }

  return 0.0
}

// ─── Main ─────────────────────────────────────────────────────────────────────

console.log('\n🧠 CORTEX × LongMemEval Adapter v1.1')
console.log('═'.repeat(62))

// 1. Cargar dataset
if (!existsSync(DATASET_PATH)) {
  console.error(`\n❌ Dataset no encontrado: ${DATASET_PATH}`)
  console.error(`\nDescarga el dataset primero:`)
  console.error(`  mkdir -p /tmp/LongMemEval/data`)
  console.error(`  cd /tmp/LongMemEval/data`)
  console.error(`  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json`)
  process.exit(1)
}

const allItems = JSON.parse(readFileSync(DATASET_PATH, 'utf8'))
const items    = LIMIT > 0 ? allItems.slice(0, LIMIT) : allItems

console.log(`✅ ${items.length} casos cargados${LIMIT ? ` (limitado a ${LIMIT})` : ''}`)
console.log(`📁 Dataset: ${DATASET_PATH}`)
console.log(`📁 Outputs: ${RESULTS_DIR}/`)
console.log(`⚙️  K=${K} | timeout=${TOOL_TIMEOUT}ms | cleanup=${!NO_CLEANUP}`)

// 2. Distribución de tipos
const typeCount = {}
for (const it of items) typeCount[it.question_type] = (typeCount[it.question_type] ?? 0) + 1
console.log('\n📊 Tipos de pregunta:')
for (const [t, n] of Object.entries(typeCount).sort((a, b) => b[1] - a[1]))
  console.log(`   ${t.padEnd(38)} ${n}`)
console.log('═'.repeat(62))

// 3. Evaluación caso a caso
const fullResults = []
const byType      = {}

for (let idx = 0; idx < items.length; idx++) {
  const item    = items[idx]
  const projId  = item.question_id.replace(/\W/g, '_').slice(0, 40)
  const project = `${PROJECT_BASE}_${projId}`
  const prefix  = `[${String(idx + 1).padStart(3)}/${items.length}]`

  process.stdout.write(`${prefix} ${item.question_id.padEnd(22)} `)

  let hypothesis  = ''
  let localSc     = 0
  let ingested    = 0
  let errorMsg    = null

  try {
    // a) Limpiar proyecto previo
    await callTool('delete_all_memories', { projectName: project, confirm: true }).catch(() => {})

    // b) Ingestar sesiones (batch_observe → buffer)
    ingested = await ingestCase(project, item)

    if (ingested === 0) {
      console.log('⚠️  sin contenido — skip')
      continue
    }

    // c) Indexar con ONNX local (sin LLM = rápido)
    await callTool('index_temp', { projectName: project, batchSize: 50, skipScoring: true })

    // d) Recall
    const recallRes  = await callTool('recall', { projectName: project, query: item.question, limit: K })
    const contextRes = await callTool('get_context_for', {
      projectName: project,
      message:     item.question,
      maxItems:    K,
    })

    const recallText  = getText(recallRes)
    const contextText = getText(contextRes)

    // El hypothesis para el scorer oficial = contexto más relevante
    hypothesis = contextText || recallText

    // e) Score local
    localSc = localScore(item.answer, recallText + '\n' + contextText)

    const sym = localSc >= 0.8 ? '✅' : localSc >= 0.5 ? '🟡' : '❌'
    console.log(`${sym} score=${localSc.toFixed(1)} mem=${ingested} "${item.question.slice(0, 38)}"`)

  } catch (err) {
    errorMsg = err.message
    console.log(`💥 ${err.message.slice(0, 60)}`)
  } finally {
    if (!NO_CLEANUP) {
      await callTool('delete_all_memories', { projectName: project, confirm: true }).catch(() => {})
    }
  }

  // f) Guardar JSONL para scorer oficial (formato: question_id + hypothesis)
  appendFileSync(JSONL_OUT, JSON.stringify({ question_id: item.question_id, hypothesis }) + '\n')

  // g) Acumular métricas locales
  const rec = {
    question_id:   item.question_id,
    question_type: item.question_type,
    question:      item.question,
    answer:        item.answer,
    hypothesis:    hypothesis.slice(0, 500),
    local_score:   localSc,
    memories_ingested: ingested,
    error:         errorMsg,
  }
  fullResults.push(rec)

  if (!byType[item.question_type]) byType[item.question_type] = []
  byType[item.question_type].push(localSc)

  // h) Flush JSON completo cada 10 casos
  if ((idx + 1) % 10 === 0) writeFileSync(JSON_OUT, JSON.stringify(fullResults, null, 2))
}

// ─── Métricas finales ─────────────────────────────────────────────────────────

writeFileSync(JSON_OUT, JSON.stringify(fullResults, null, 2))

const scores = fullResults.map(r => r.local_score)
const avg    = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0

const summary = {
  timestamp:   new Date().toISOString(),
  dataset:     DATASET_PATH,
  total_cases: fullResults.length,
  k:           K,
  local_scores: {
    full_match:    +(scores.filter(s => s >= 0.8).length / scores.length).toFixed(4),
    partial_match: +(scores.filter(s => s >= 0.5).length / scores.length).toFixed(4),
    no_match:      +(scores.filter(s => s === 0).length  / scores.length).toFixed(4),
    mean_score:    +avg(scores).toFixed(4),
  },
  by_question_type: Object.fromEntries(
    Object.entries(byType).map(([t, arr]) => [t, {
      n:             arr.length,
      full_match_pct: +((arr.filter(s => s >= 0.8).length / arr.length) * 100).toFixed(1),
      partial_pct:    +((arr.filter(s => s >= 0.5).length / arr.length) * 100).toFixed(1),
      mean_score:     +avg(arr).toFixed(3),
    }])
  ),
  headline_local: `CORTEX local Recall@${K}: ${(scores.filter(s => s >= 0.8).length / scores.length * 100).toFixed(1)}% full-match`,
  next_step: `cd /tmp/LongMemEval/src/evaluation && python3 evaluate_qa.py gpt-4o ${JSONL_OUT} ../../data/longmemeval_oracle.json`,
}

writeFileSync(SUM_OUT, JSON.stringify(summary, null, 2))

// ─── Reporte ──────────────────────────────────────────────────────────────────

const pct = (n) => `${(n * 100).toFixed(1)}%`
const bar = (label, v, w = 32) => {
  const f = Math.round(v * w)
  console.log(`  ${label.padEnd(22)} [${'█'.repeat(f)}${'░'.repeat(w - f)}] ${pct(v)}`)
}

console.log('\n' + '═'.repeat(62))
console.log('📊 RESULTADOS — Evaluación local (keyword match)\n')
bar(`Full match  (≥0.8)`, summary.local_scores.full_match)
bar(`Partial     (≥0.5)`, summary.local_scores.partial_match)
bar(`No match    (0.0)`,  summary.local_scores.no_match)
console.log(`\n  Mean score: ${summary.local_scores.mean_score}`)

console.log('\n📂 Por tipo de pregunta:\n')
for (const [type, m] of Object.entries(summary.by_question_type).sort((a, b) => b[1].full_match_pct - a[1].full_match_pct)) {
  console.log(`  ${type.padEnd(38)} full=${String(m.full_match_pct + '%').padStart(6)}  partial=${String(m.partial_pct + '%').padStart(6)}  n=${m.n}`)
}

console.log('\n' + '═'.repeat(62))
console.log(`🏆 ${summary.headline_local}`)
console.log('═'.repeat(62))

console.log(`
📄 Archivos generados:
   JSONL (scorer oficial): ${JSONL_OUT}
   JSON  (detalle):        ${JSON_OUT}
   JSON  (summary):        ${SUM_OUT}

🔬 Para el score oficial con LLM judge (GPT-4o):
   export OPENAI_API_KEY=sk-...
   cd /tmp/LongMemEval/src/evaluation
   python3 evaluate_qa.py gpt-4o \\
     ${JSONL_OUT} \\
     /tmp/LongMemEval/data/longmemeval_oracle.json
`)

process.exit(0)
