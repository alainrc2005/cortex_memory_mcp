/**
 * CORTEX — Migration Wizard
 * Clasificación automática: heurísticas locales + OpenRouter batch.
 * Las aceptadas van a temp_memories → listas para index_temp.
 */

import http from 'http'
import fs   from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { randomUUID } from 'crypto'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// ── Config ───────────────────────────────────────────────────────────────────
const QDRANT_URL       = process.env.QDRANT_URL          || 'http://localhost:6333'
const QDRANT_KEY       = process.env.QDRANT_API_KEY      || 'aJmUm1IN0Nws4d8mX5ecw1AiABp6JVPp'
const OPENROUTER_KEY   = process.env.OPENROUTER_API_KEY  || ''
const OPENROUTER_MODEL = process.env.OPENROUTER_MODEL    || 'google/gemma-4-31b-it:free'
const OPENROUTER_URL   = process.env.OPENROUTER_BASE_URL || 'https://openrouter.ai/api/v1'
const PORT             = Number(process.env.MIGRATE_PORT || 7474)

// ── Heurísticas de ruido obvio (sin LLM, instantáneo) ────────────────────────
const NOISE_PATTERNS = [
  /^(The AI|AI |Zeus )/i,
  /^The user is (seeking|asking|trying|looking|requesting)/i,
  /^Detected a gap/i,
  /^(User|Operator) (asked|inquired|requested|mentioned|provided|confirmed)/i,
  /^(Discussion focused|Conversation (about|regarding)|Session (about|focused))/i,
  /displays? a (consolidated|report|summary)/i,
  /^AI (generated|displays|removed|added|created|updated|fixed|changed|made|responded|returned)/i,
  /^The (assistant|model|system) (generated|responded|created|updated|fixed|returned)/i,
  /^(I |We )(generated|created|added|removed|updated|fixed|changed|made|responded)/i,
  /^(Operator|User) (was|is) (asking|seeking|looking|trying)/i,
]

function isObviousNoise(content) {
  if (!content || content.trim().length < 35) return true
  const c = content.trim()
  return NOISE_PATTERNS.some(re => re.test(c))
}

// ── Colecciones legacy ────────────────────────────────────────────────────────
const LEGACY_COLLECTIONS = [
  'work_memories',
  'social_memories',
  'archive_memories',
  'directive_memories',
  'core_directives',
  'signal_memories',
]

const CORTEX_PROJECTS = [
  'cortex', 'pets', 'hotetec', 'tickets-venue-editor', 'school', 'global',
]

// ── Qdrant helpers ────────────────────────────────────────────────────────────
async function qdrantReq(method, urlPath, body) {
  const res = await fetch(`${QDRANT_URL}${urlPath}`, {
    method,
    headers: { 'api-key': QDRANT_KEY, 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`Qdrant ${method} ${urlPath} → ${res.status}`)
  return res.json()
}

async function fetchPage(collection, limit = 10, offset = null) {
  const body = { limit, with_payload: true, with_vector: false }
  if (offset) body.offset = offset
  const data = await qdrantReq('POST', `/collections/${collection}/points/scroll`, body)
  return { points: data.result?.points || [], nextOffset: data.result?.next_page_offset || null }
}

async function fetchBatch(collection, limit, offset) {
  return fetchPage(collection, limit, offset)
}

async function countPoints(collection) {
  try {
    const data = await qdrantReq('GET', `/collections/${collection}`)
    return data.result?.points_count ?? 0
  } catch { return 0 }
}

async function deletePoint(collection, id) {
  await qdrantReq('POST', `/collections/${collection}/points/delete`, { points: [id] })
}

async function deleteBatch(collection, ids) {
  if (!ids.length) return
  await qdrantReq('POST', `/collections/${collection}/points/delete`, { points: ids })
}

async function writeToTemp(content, projectName) {
  const id  = randomUUID()
  const now = Date.now()
  await qdrantReq('PUT', '/collections/temp_memories/points', {
    wait: true,
    points: [{ id, vector: new Array(384).fill(0), payload: { id, content, projectName, createdAt: now, status: 'pending' } }],
  })
  return id
}

// ── OpenRouter batch classifier ───────────────────────────────────────────────
async function classifyWithOpenRouter(items, attempt = 0) {
  if (!OPENROUTER_KEY) throw new Error('OPENROUTER_API_KEY no configurada')

  const numbered = items.map((it, i) => `${i + 1}. ${it.content.slice(0, 200)}`).join('\n')

  const prompt = `Eres un clasificador de memorias de un sistema de IA personal. 
Tu tarea: decidir cuáles de estas memorias VALE LA PENA conservar para contexto futuro y cuáles son RUIDO a eliminar.

RUIDO = descripciones de acciones triviales del AI, frases genéricas sin info técnica, logs de conversación sin valor, acciones de UI.
VALIOSA = decisiones técnicas, errores resueltos, preferencias del operador, configuraciones, hitos, contexto de proyectos reales.

Memorias:
${numbered}

Responde SOLO con un JSON array con un objeto por ítem:
[{"n":1,"keep":true/false,"reason":"<5 palabras max>"},...]
Sin markdown, sin explicaciones adicionales.`

  const res = await fetch(`${OPENROUTER_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENROUTER_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: OPENROUTER_MODEL,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 800,
    }),
  })

  // Retry con backoff si hay rate limit
  if (res.status === 429) {
    if (attempt >= 2) throw new Error('OpenRouter rate limit — 3 intentos fallidos')
    const wait = (attempt + 1) * 2000
    console.log(`[CLASSIFY] 429 rate limit — reintentando en ${wait}ms (intento ${attempt + 1}/3)`)
    await new Promise(r => setTimeout(r, wait))
    return classifyWithOpenRouter(items, attempt + 1)
  }

  if (!res.ok) throw new Error(`OpenRouter error ${res.status}`)
  const data = await res.json()
  const raw  = data.choices?.[0]?.message?.content?.trim() || '[]'

  // Parse robusto: buscar el JSON array aunque haya texto alrededor
  const match = raw.match(/\[[\s\S]*\]/)
  if (!match) throw new Error(`OpenRouter no devolvió JSON válido: ${raw.slice(0, 100)}`)
  return JSON.parse(match[0])
}

// ── Estado del servidor ───────────────────────────────────────────────────────
const cursors = {}
for (const col of LEGACY_COLLECTIONS) cursors[col] = { offset: null, done: false }

let stats = { accepted: 0, skipped: 0, deleted: 0, autoDeleted: 0 }

// ── HTTP helpers ──────────────────────────────────────────────────────────────
function jsonResp(res, status, data) {
  res.writeHead(status, { 'Content-Type': 'application/json' })
  res.end(JSON.stringify(data))
}

async function readBody(req) {
  let body = ''
  for await (const chunk of req) body += chunk
  return JSON.parse(body)
}

// ── API: /api/next ────────────────────────────────────────────────────────────
// Devuelve el siguiente punto que pasa el filtro heurístico.
// Auto-elimina y salta los que son ruido obvio (hasta 50 consecutivos).
async function apiNext(req, res) {
  const url    = new URL(req.url, `http://localhost:${PORT}`)
  const col    = url.searchParams.get('collection') || LEGACY_COLLECTIONS[0]
  const skip   = url.searchParams.get('skipNoise') !== 'false'  // default: true
  const cursor = cursors[col]

  if (cursor.done) return jsonResp(res, 200, { done: true, collection: col })

  let attempts = 0
  while (attempts < 200) {
    const { points, nextOffset } = await fetchPage(col, 1, cursor.offset)

    cursor.offset = nextOffset
    if (!nextOffset) cursor.done = true

    if (!points || points.length === 0) {
      cursor.done = true
      return jsonResp(res, 200, { done: true, collection: col })
    }

    const pt      = points[0]
    const content = pt.payload?.content || pt.payload?.raw_content_preview || ''

    if (skip && isObviousNoise(content)) {
      // Auto-eliminar ruido obvio silenciosamente
      try { await deletePoint(col, pt.id) } catch {}
      stats.autoDeleted++
      attempts++
      if (cursor.done) return jsonResp(res, 200, { done: true, collection: col })
      continue
    }

    return jsonResp(res, 200, { point: pt, collection: col, autoDeleted: stats.autoDeleted })
  }

  return jsonResp(res, 200, { done: true, collection: col })
}

// ── API: /api/batch-classify ──────────────────────────────────────────────────
// Lee N memorias, clasifica con OpenRouter, auto-elimina las noise, devuelve las keep.
async function apiBatchClassify(req, res) {
  const body       = await readBody(req)
  const collection = body.collection || LEGACY_COLLECTIONS[0]
  const batchSize  = Math.min(Number(body.batchSize || 20), 50)
  const cursor     = cursors[collection]

  if (cursor.done) return jsonResp(res, 200, { done: true, kept: [], deleted: 0 })

  // Leer lote del legacy
  const { points, nextOffset } = await fetchBatch(collection, batchSize, cursor.offset)
  cursor.offset = nextOffset
  if (!nextOffset) cursor.done = true

  if (!points.length) {
    cursor.done = true
    return jsonResp(res, 200, { done: true, kept: [], deleted: 0 })
  }

  // Paso 1: filtro heurístico local (gratis, instantáneo)
  const obviousNoise = []
  const candidates   = []

  for (const pt of points) {
    const content = pt.payload?.content || pt.payload?.raw_content_preview || ''
    if (isObviousNoise(content)) {
      obviousNoise.push({ id: pt.id, content, reason: 'heurística' })
    } else {
      candidates.push({ pt, content })
    }
  }

  // Borrar ruido obvio inmediatamente
  if (obviousNoise.length) {
    await deleteBatch(collection, obviousNoise.map(p => p.id))
    stats.autoDeleted += obviousNoise.length
  }

  if (!candidates.length) {
    return jsonResp(res, 200, {
      kept: [],
      deletedItems: obviousNoise,
      deleted: obviousNoise.length,
      heuristic: obviousNoise.length,
      llm: 0,
      stats,
    })
  }

  // Paso 2: clasificar candidatos con OpenRouter
  let llmDeleted  = 0
  let kept        = []
  const llmNoise  = []

  try {
    const items = candidates.map(c => ({ id: c.pt.id, content: c.content }))
    const results = await classifyWithOpenRouter(items)

    const keepIds   = new Set()
    const deleteMap = new Map()   // id → reason

    for (const r of results) {
      const idx = (r.n || 0) - 1
      if (idx < 0 || idx >= candidates.length) continue
      const c = candidates[idx]
      if (r.keep) {
        keepIds.add(c.pt.id)
      } else {
        deleteMap.set(c.pt.id, r.reason || 'LLM: descartada')
      }
    }

    // Candidatos no mencionados → keep (conservador)
    for (const c of candidates) {
      if (!keepIds.has(c.pt.id) && !deleteMap.has(c.pt.id)) keepIds.add(c.pt.id)
    }

    const deleteIds = [...deleteMap.keys()]
    for (const c of candidates) {
      if (deleteMap.has(c.pt.id)) {
        llmNoise.push({ id: c.pt.id, content: c.content, reason: deleteMap.get(c.pt.id) })
      }
    }

    if (deleteIds.length) {
      await deleteBatch(collection, deleteIds)
      stats.autoDeleted += deleteIds.length
      llmDeleted = deleteIds.length
    }

    kept = candidates
      .filter(c => keepIds.has(c.pt.id))
      .map(c => c.pt)

  } catch (err) {
    console.error('[CLASSIFY ERROR]', err.message)
    kept = candidates.map(c => c.pt)
  }

  return jsonResp(res, 200, {
    kept,
    deletedItems: [...obviousNoise, ...llmNoise],
    deleted: obviousNoise.length + llmDeleted,
    heuristic: obviousNoise.length,
    llm: llmDeleted,
    stats,
  })
}

// ── API: /api/action ──────────────────────────────────────────────────────────
async function apiAction(req, res) {
  const { action, collection, id, content, projectName } = await readBody(req)

  try {
    if (action === 'accept') {
      if (!content || !projectName) return jsonResp(res, 400, { error: 'content y projectName requeridos' })
      const tempId = await writeToTemp(content, projectName)
      await deletePoint(collection, id)
      stats.accepted++
      return jsonResp(res, 200, { ok: true, tempId, stats })

    } else if (action === 'skip') {
      stats.skipped++
      return jsonResp(res, 200, { ok: true, stats })

    } else if (action === 'delete') {
      await deletePoint(collection, id)
      stats.deleted++
      return jsonResp(res, 200, { ok: true, stats })

    } else {
      return jsonResp(res, 400, { error: `Acción desconocida: ${action}` })
    }
  } catch (err) {
    return jsonResp(res, 500, { error: err.message })
  }
}

// ── API: /api/status ──────────────────────────────────────────────────────────
async function apiStatus(req, res) {
  const counts = {}
  for (const col of LEGACY_COLLECTIONS) counts[col] = await countPoints(col)
  return jsonResp(res, 200, { counts, stats, cursors: Object.fromEntries(
    Object.entries(cursors).map(([k, v]) => [k, { done: v.done }])
  )})
}

// ── API: /api/projects ────────────────────────────────────────────────────────
async function apiProjects(req, res) {
  try {
    const data = await qdrantReq('GET', '/collections')
    const cortexCols = (data.result?.collections || [])
      .map(c => c.name)
      .filter(n => n.startsWith('cortex_'))
      .map(n => n.replace(/^cortex_/, '').replace(/_/g, '-'))
    const projects = [...new Set([...CORTEX_PROJECTS, ...cortexCols])].sort()
    return jsonResp(res, 200, { projects })
  } catch {
    return jsonResp(res, 200, { projects: CORTEX_PROJECTS })
  }
}

// ── Servidor HTTP ─────────────────────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`)

  res.setHeader('Access-Control-Allow-Origin', '*')
  if (req.method === 'OPTIONS') { res.writeHead(204); return res.end() }

  try {
    if (url.pathname === '/api/next'            && req.method === 'GET')  return await apiNext(req, res)
    if (url.pathname === '/api/action'          && req.method === 'POST') return await apiAction(req, res)
    if (url.pathname === '/api/batch-classify'  && req.method === 'POST') return await apiBatchClassify(req, res)
    if (url.pathname === '/api/status'          && req.method === 'GET')  return await apiStatus(req, res)
    if (url.pathname === '/api/projects'        && req.method === 'GET')  return await apiProjects(req, res)

    if (url.pathname === '/' || url.pathname === '/index.html') {
      const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8')
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
      return res.end(html)
    }

    res.writeHead(404); res.end('Not found')
  } catch (err) {
    console.error('[MIGRATE]', err)
    res.writeHead(500, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: err.message }))
  }
})

server.listen(PORT, () => {
  console.log(`\n🧠 CORTEX Migration Wizard — http://localhost:${PORT}`)
  console.log(`   OpenRouter: ${OPENROUTER_KEY ? '✅ activo' : '❌ no configurado'}`)
  console.log(`   Ctrl+C para detener\n`)
})
