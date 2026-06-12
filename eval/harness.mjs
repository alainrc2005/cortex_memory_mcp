#!/usr/bin/env node
/**
 * CORTEX — Evaluation Harness v1.0
 * Inspirado en LoCoMo / LongMemEval
 *
 * Mide la calidad de recuperación del sistema de memoria con 3 métricas:
 *   • Recall@K  — ¿el engrama esperado aparece en los top-K resultados?
 *   • MRR       — Mean Reciprocal Rank (1/posición del primer acierto)
 *   • Hit@1     — ¿el engrama correcto es el #1?
 *
 * Uso:
 *   node eval/harness.mjs              # corre todos los casos
 *   node eval/harness.mjs --no-cleanup # no borra el proyecto de prueba al final
 *   node eval/harness.mjs --k 3        # Recall@3 (default: 5)
 */

import { spawn } from 'child_process'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dir = dirname(fileURLToPath(import.meta.url))
const SERVER = resolve(__dir, '../dist/server.js')

// ─── Config ───────────────────────────────────────────────────────────────────

const EVAL_PROJECT = 'eval_harness_tmp'
const args_cli     = process.argv.slice(2)
const K            = Number(args_cli[args_cli.indexOf('--k') + 1] || 5)
const NO_CLEANUP   = args_cli.includes('--no-cleanup')
const TOOL_TIMEOUT = 30_000  // ms por llamada (observe tarda por scoring)

// ─── Dataset de evaluación ────────────────────────────────────────────────────
//
// Cada caso tiene:
//   - id       : identificador legible del caso
//   - memory   : texto a insertar en CORTEX
//   - query    : consulta que debería recuperar esa memoria
//   - category : tipo de caso (semántico, keyword, tipo, recencia)
//
// Los IDs de engrama se asignan en tiempo de ejecución y se rastrean por índice.

const DATASET = [
  // ── Recuperación semántica básica ────────────────────────────────────────
  {
    id: 'SEM-01',
    category: 'semantic',
    memory: 'Usamos Zod para validación de esquemas en todos los endpoints REST.',
    query:  'validación de datos en la API',
  },
  {
    id: 'SEM-02',
    category: 'semantic',
    memory: 'El módulo de autenticación usa JWT con refresh tokens de 7 días.',
    query:  'cómo funciona el login y la sesión de usuario',
  },
  {
    id: 'SEM-03',
    category: 'semantic',
    memory: 'La base de datos principal es PostgreSQL 15 corriendo en Docker.',
    query:  'qué base de datos estamos usando',
  },
  // ── Recuperación por keyword exacta ──────────────────────────────────────
  {
    id: 'KEY-01',
    category: 'keyword',
    memory: 'El puerto del servidor de desarrollo es 3001, no el 3000 por defecto.',
    query:  'puerto 3001 servidor',
  },
  {
    id: 'KEY-02',
    category: 'keyword',
    memory: 'Variable de entorno DATABASE_URL apunta a postgres://localhost:5432/app_dev.',
    query:  'DATABASE_URL conexión postgres',
  },
  // ── Recuperación por tipo de engrama ─────────────────────────────────────
  {
    id: 'TYPE-01',
    category: 'type',
    memory: 'DECISIÓN: descartamos Redis para caché y usamos memoria en proceso (LRU).',
    query:  'decisión sobre caché y almacenamiento temporal',
  },
  {
    id: 'TYPE-02',
    category: 'type',
    memory: 'ERROR RESUELTO: import circular entre authService y userService — fix: extraer tipos a types/auth.ts.',
    query:  'error de import circular en servicios',
  },
  // ── Recuperación de contexto técnico ─────────────────────────────────────
  {
    id: 'CTX-01',
    category: 'context',
    memory: 'El pipeline de CI/CD usa GitHub Actions con deploy automático a Fly.io en cada merge a main.',
    query:  'cómo se hace el deployment',
  },
  {
    id: 'CTX-02',
    category: 'context',
    memory: 'La arquitectura del frontend usa React 18 con Vite y React Query para el estado del servidor.',
    query:  'stack tecnológico del frontend',
  },
  // ── Resistencia a ruido (distractores semánticos) ─────────────────────────
  {
    id: 'NOISE-01',
    category: 'noise',
    memory: 'El timeout de las peticiones HTTP al proveedor externo es de 15 segundos.',
    query:  'timeout en llamadas a servicios externos',
    // Este caso verifica que el sistema no se confunde con otros engramas de timeout/HTTP
  },
]

// ─── Runner MCP ───────────────────────────────────────────────────────────────

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
      try { res(JSON.parse(out)) }
      catch { rej(new Error(`JSON inválido: ${out.slice(0, 120)}`)) }
    })
    proc.stdin.write(req + '\n')
    proc.stdin.end()
    setTimeout(() => { proc.kill(); rej(new Error(`timeout (${TOOL_TIMEOUT}ms) en ${name}`)) }, TOOL_TIMEOUT)
  })
}

function getText(res) {
  return res?.result?.content?.[0]?.text ?? ''
}

// ─── Métricas ─────────────────────────────────────────────────────────────────

function computeMetrics(results) {
  const n = results.length
  if (n === 0) return { recallAtK: 0, mrr: 0, hitAt1: 0 }

  let hits   = 0
  let rrSum  = 0
  let hitAt1 = 0

  for (const r of results) {
    if (r.rank !== null) {
      hits++
      rrSum += 1 / r.rank
      if (r.rank === 1) hitAt1++
    }
  }

  return {
    recallAtK: hits / n,
    mrr:       rrSum / n,
    hitAt1:    hitAt1 / n,
  }
}

// ─── Utilidades ───────────────────────────────────────────────────────────────

function pad(str, len) { return String(str).padEnd(len) }
function pct(n)        { return `${(n * 100).toFixed(1)}%` }

function printBar(label, value, width = 30) {
  const filled = Math.round(value * width)
  const bar    = '█'.repeat(filled) + '░'.repeat(width - filled)
  console.log(`  ${pad(label, 12)} [${bar}] ${pct(value)}`)
}

// ─── Main ─────────────────────────────────────────────────────────────────────

console.log('\n🧪 CORTEX Evaluation Harness v1.0')
console.log(`   Proyecto: ${EVAL_PROJECT} | K=${K} | ${DATASET.length} casos`)
console.log('═'.repeat(56))

// 1. Limpiar proyecto de prueba anterior si existe
process.stdout.write('\n[1/4] Limpiando estado anterior... ')
await callTool('delete_all_memories', { projectName: EVAL_PROJECT, confirm: true })
  .catch(() => {})
console.log('OK')

// 2. Insertar memorias del dataset (sin scoring para no quemar CPU)
process.stdout.write(`[2/4] Insertando ${DATASET.length} memorias en buffer... `)
const engramaIds = []

for (const item of DATASET) {
  const res  = await callTool('quick_observe', { projectName: EVAL_PROJECT, content: item.memory })
  const text = getText(res)
  // Extraer el ID del mensaje "• ID: <uuid>"
  const match = text.match(/ID:\s*([0-9a-f-]{36})/i)
  engramaIds.push(match ? match[1] : null)
}
console.log('OK')

// 3. Indexar con skipScoring=true (solo embedding ONNX, sin qwen3)
process.stdout.write('[3/4] Indexando con fastembed ONNX (sin LLM)... ')
await callTool('index_temp', { projectName: EVAL_PROJECT, batchSize: 20, skipScoring: true })
console.log('OK')

// 4. Evaluar: por cada caso, hacer recall y buscar el ID esperado en los resultados
console.log(`[4/4] Evaluando ${DATASET.length} queries (Recall@${K}, MRR, Hit@1)...\n`)

const evalResults = []
const byCategory  = new Map()

for (let i = 0; i < DATASET.length; i++) {
  const item      = DATASET[i]
  const targetId  = engramaIds[i]
  const category  = item.category

  process.stdout.write(`  ${pad(item.id, 10)} `)

  const res  = await callTool('recall', { projectName: EVAL_PROJECT, query: item.query, limit: K })
  const text = getText(res)

  // Extraer IDs de la respuesta (aparecen como "ID: <uuid>" o en el contexto)
  // recall no devuelve IDs directamente — buscamos el contenido de la memoria
  // como proxy (match parcial de los primeros 40 chars)
  const memorySnippet = item.memory.slice(0, 40).toLowerCase()
  const lines         = text.toLowerCase().split('\n')

  let rank = null
  let lineIdx = 0
  for (const line of lines) {
    if (line.match(/^\s*\d+\./)) lineIdx++   // nueva entrada numerada
    if (line.includes(memorySnippet)) { rank = lineIdx || 1; break }
  }

  // Fallback: si no encontramos por snippet, buscar por palabras clave únicas
  if (rank === null) {
    const keywords = item.memory.split(' ').filter(w => w.length > 6).slice(0, 3)
    lineIdx = 0
    for (const line of lines) {
      if (line.match(/^\s*\d+\./)) lineIdx++
      if (keywords.every(kw => line.includes(kw.toLowerCase()))) { rank = lineIdx || 1; break }
    }
  }

  // Limitar rank a K (si está pero después de K, no cuenta)
  if (rank !== null && rank > K) rank = null

  const symbol = rank === null ? '❌' : rank === 1 ? '🎯' : '✅'
  console.log(`${symbol}  rank=${rank ?? `>${K}`}  "${item.query.slice(0, 35)}"`)

  evalResults.push({ ...item, rank, targetId })

  if (!byCategory.has(category)) byCategory.set(category, [])
  byCategory.get(category).push({ ...item, rank })
}

// ─── Resultados ───────────────────────────────────────────────────────────────

const global_ = computeMetrics(evalResults)

console.log('\n' + '═'.repeat(56))
console.log('📊 Métricas globales\n')
printBar(`Recall@${K}`, global_.recallAtK)
printBar('MRR',         global_.mrr)
printBar('Hit@1',       global_.hitAt1)

console.log('\n📂 Por categoría\n')
for (const [cat, items] of byCategory.entries()) {
  const m = computeMetrics(items)
  console.log(`  ${pad(cat, 10)}  Recall@${K}=${pct(m.recallAtK)}  MRR=${pct(m.mrr)}  Hit@1=${pct(m.hitAt1)}  (n=${items.length})`)
}

// ─── Casos fallidos ───────────────────────────────────────────────────────────

const failed = evalResults.filter(r => r.rank === null)
if (failed.length > 0) {
  console.log('\n⚠️  Casos no recuperados:\n')
  for (const f of failed) {
    console.log(`  [${f.id}] "${f.query}"`)
    console.log(`         → "${f.memory.slice(0, 60)}..."`)
  }
}

// ─── Cleanup ──────────────────────────────────────────────────────────────────

if (!NO_CLEANUP) {
  process.stdout.write('\n🧹 Limpiando proyecto de prueba... ')
  await callTool('delete_all_memories', { projectName: EVAL_PROJECT, confirm: true })
    .catch(() => {})
  console.log('OK')
} else {
  console.log(`\n⚠️  --no-cleanup activo. Proyecto "${EVAL_PROJECT}" conservado en Qdrant.`)
}

// ─── Score final ──────────────────────────────────────────────────────────────

const score = (global_.recallAtK + global_.mrr + global_.hitAt1) / 3
console.log('\n' + '═'.repeat(56))
console.log(`✨ Score compuesto: ${pct(score)}  (Recall@${K} + MRR + Hit@1) / 3`)

if (score >= 0.8)       console.log('   🟢 Excelente — sistema bien calibrado')
else if (score >= 0.6)  console.log('   🟡 Aceptable — hay margen de mejora')
else                    console.log('   🔴 Bajo — revisar embeddings o scoring')

console.log('═'.repeat(56) + '\n')

// Exit code para CI
process.exit(failed.length > 0 ? 1 : 0)
