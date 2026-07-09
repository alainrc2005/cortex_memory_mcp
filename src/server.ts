import './bootstrap.js'
import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js'
import * as dotenv from 'dotenv'
import { v4 as uuidv4 } from 'uuid'
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'fs'
import { createRequire } from 'module'
import * as path from 'path'
import * as os from 'os'

// Lee la versión una sola vez desde package.json — fuente de verdad única
const _require = createRequire(import.meta.url)
const PKG_VERSION: string = (_require('../package.json') as { version: string }).version

import { observeGraph } from './graph/observe/workflow.js'
import { runConsolidation } from './graph/consolidate/nodes.js'
import { warmupEmbedding, getEmbedding, getSparseEmbedding, getEmbeddingBatch, getSparseEmbeddingBatch } from './services/fastembed.js'
import { generateText, scoreAndTag, batchScoreAndTag, rerankWithLLM } from './services/llm.js'
import { qdrant, searchCrossProject, searchHybridCrossProject, patchPayload, upsertEngrama, scrollAll, getOperatorProfile, saveOperatorProfile, listCortexCollections, deleteEngramas, deleteAllInProject, exportProject, upsertTemp, scrollTemp, searchTempByKeyword, deleteTempIds, ensureProjectCollection, findEngramaById, TEMP_COLLECTION } from './services/qdrant.js'
import type { SearchHit, TempMemory } from './services/qdrant.js'
import { calcDecay, combinedScore } from './services/decay.js'
import { createEpisode, appendEvent, getOpenSession, closeEpisode, searchEpisodes, getRecentEpisodes, episodeCollectionFor } from './services/episode.js'
import { ensureSchema, queryNeighbors, queryTimeline, rawQuery, queryGraphContext, listEntities } from './services/kuzu.js'
import type { Engrama } from './types/engrama.js'

dotenv.config()

// ─── Feature flags ──────────────────────────────────────────────────
// false por defecto: en CPU el cross-encoder Ollama bloquea 90-130s por llamada
const RERANKER_ENABLED = process.env.CORTEX_RERANKER_ENABLED === 'true'

// ─── Logger ───────────────────────────────────────────────────────────────────

const LOG_DIR  = process.env.CORTEX_LOG_DIR ?? path.join(os.homedir(), '.cortex', 'logs')
const LOG_FILE = path.join(LOG_DIR, 'cortex-mcp.log')
try { mkdirSync(LOG_DIR, { recursive: true }) } catch {}

function log(event: string, data?: unknown) {
  const line = `[${new Date().toISOString()}] ${event}${data !== undefined ? ' ' + JSON.stringify(data) : ''}\n`
  try { appendFileSync(LOG_FILE, line) } catch {}
  // También a stderr para que el broker lo capture si está configurado
  process.stderr.write(line)
}
// ─── Helpers de validación ────────────────────────────────────────────────────

function requireString(args: Record<string, unknown> | undefined, key: string, label?: string): string {
  const val = args?.[key]
  if (!val || typeof val !== 'string' || !(val as string).trim()) {
    throw new Error(`Parámetro requerido: ${label ?? key}`)
  }
  return (val as string).trim()
}

function errorResponse(msg: string, detail?: string) {
  const text = detail ? `❌ ${msg}\n\nDetalle: ${detail}` : `❌ ${msg}`
  return { content: [{ type: 'text' as const, text }] }
}


// ─── MCP Server ───────────────────────────────────────────────────────────────

log('SERVER_START', { version: PKG_VERSION, pid: process.pid })

// Pre-carga el modelo ONNX en background al arrancar.
// Primera petición no paga el costo de carga (~1-3s).
warmupEmbedding()

// ─── Maintenance Scheduler ───────────────────────────────────────────────────
// Ejecuta consolidate + detect_patterns automáticamente en background.
// Nunca bloquea el servidor ni las peticiones del agente.

const MAINT_FILE                  = path.join(os.homedir(), '.cortex', 'maintenance.json')
const MAINT_CONSOLIDATE_INTERVAL  = 7  * 24 * 60 * 60 * 1000   // 7 días
const MAINT_DETECT_INTERVAL       = 14 * 24 * 60 * 60 * 1000   // 14 días
const MAINT_MIN_ENGRAMAS_CONSOL   = 25   // mínimo para que valga la pena consolidar
const MAINT_MIN_ENGRAMAS_DETECT   = 15   // mínimo para detectar patrones
const MAINT_INDEX_BATCH_SIZE      = 10   // memorias del buffer a indexar por proyecto por ciclo
const MAINT_CHECK_INTERVAL        = 60  * 60 * 1000             // revisar cada hora
const MAINT_STARTUP_DELAY         = 5   * 60 * 1000             // esperar 5 min al arrancar
const MAINT_INDEX_STARTUP_DELAY   = 60  * 1000                  // indexar buffer en 60s al arrancar

interface MaintenanceState {
  lastConsolidate:    Record<string, number>  // projectName → epoch ms
  lastDetectPatterns: Record<string, number>
  lastIndexTemp:      Record<string, number>  // projectName → epoch ms del último auto-index
}

function loadMaintState(): MaintenanceState {
  try { return JSON.parse(readFileSync(MAINT_FILE, 'utf-8')) as MaintenanceState }
  catch { return { lastConsolidate: {}, lastDetectPatterns: {}, lastIndexTemp: {} } }
}
function saveMaintState(s: MaintenanceState): void {
  try { writeFileSync(MAINT_FILE, JSON.stringify(s, null, 2)) } catch {}
}

/**
 * Lógica central de detect_patterns: compartida entre el tool MCP y el scheduler.
 * Recibe los engramas ya cargados para evitar doble fetch cuando viene del scheduler.
 */
async function autoDetectPatterns(projectName: string, allEngramas: Engrama[]): Promise<string[]> {
  if (allEngramas.length < 5) return []

  const nowMs   = Date.now()
  const maxAge  = 30 * 24 * 60 * 60 * 1000
  const sample  = [...allEngramas]
    .map(e => ({
      e,
      score: 0.7 * Math.max(0, 1 - (nowMs - (e.lastAccessed ?? e.createdAt ?? 0)) / maxAge)
           + 0.3 * ((e.importance ?? 5) / 10),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 50)
    .map(x => x.e)

  const contentList = sample
    .map((e, i) => `${i + 1}. [${e.type ?? 'FACT'}] ${e.content}`)
    .join('\n')

  const prompt =
    `Analiza estas memorias técnicas del proyecto "${projectName}" y detecta:\n` +
    `1. Preferencias recurrentes del operador (cómo prefiere hacer las cosas)\n` +
    `2. Patrones de trabajo habituales (flujo típico)\n` +
    `3. Errores o problemas recurrentes\n\n` +
    `Memorias:\n${contentList}\n\n` +
    `Responde con una lista. Cada patrón en una línea comenzando con "- ".\n` +
    `Sé específico y conciso. Máximo 8 patrones.`

  const response = await generateText(prompt)
  const patterns = response
    .split('\n')
    .map(l => l.trim())
    .filter(l => l.startsWith('-'))
    .map(l => l.replace(/^-\s*/, '').trim())
    .filter(l => l.length > 10)

  if (patterns.length === 0) return []

  // Guardar cada patrón como engrama PATTERN con importancia 10
  for (const pattern of patterns) {
    const engramaId = uuidv4()
    const scored    = await scoreAndTag(pattern)
    const embedding = await getEmbedding(pattern)
    const now       = Date.now()
    await upsertEngrama(engramaId, embedding, {
      content: pattern, projectName, createdAt: now,
      importance: 10, accessCount: 0, lastAccessed: now,
      type: 'PATTERN', tags: scored.tags, linkedTo: [],
    })
  }

  // Actualizar Operator Profile
  const existingProfile = await getOperatorProfile()
  const activeProjects  = existingProfile?.activeProjects ?? []
  if (!activeProjects.includes(projectName)) activeProjects.push(projectName)
  await saveOperatorProfile({
    codingPreferences: patterns.filter(p =>
      p.toLowerCase().includes('usa') ||
      p.toLowerCase().includes('prefer') ||
      p.toLowerCase().includes('siempre')
    ),
    activeProjects,
    detectedPatterns: patterns,
    lastUpdated: Date.now(),
  })

  return patterns
}

/**
 * Auto-indexa las memorias pendientes del buffer (temp_memories) de un proyecto.
 * Compartida entre el tool MCP index_temp y el MaintenanceScheduler.
 *
 * @param projectName  Proyecto a indexar
 * @param batchSize    Memorias a procesar en este ciclo (default: MAINT_INDEX_BATCH_SIZE)
 * @param skipScoring  true = sin LLM (más rápido); false = con scoring qwen3 (mejor calidad)
 * @returns número de memorias indexadas
 */
async function autoIndexTemp(projectName: string, batchSize = MAINT_INDEX_BATCH_SIZE, skipScoring = false): Promise<number> {
  const pending = await scrollTemp(projectName, 'pending', batchSize)
  if (pending.length === 0) return 0

  const targetCol = await ensureProjectCollection(projectName)
  const contents  = pending.map(m => m.content)

  // Paso 1: Embeddings en batch
  let denseVecs: number[][] = []
  let sparseVecs: Array<{ indices: number[], values: number[] }> = []
  try {
    ;[denseVecs, sparseVecs] = await Promise.all([
      getEmbeddingBatch(contents),
      getSparseEmbeddingBatch(contents).catch(() => contents.map(() => ({ indices: [], values: [] }))),
    ])
  } catch {
    denseVecs  = await Promise.all(contents.map(c => getEmbedding(c)))
    sparseVecs = contents.map(() => ({ indices: [], values: [] }))
  }

  // Paso 2: Scoring opcional
  let scores: Array<{ importance: number, type: string, tags: string[] }>
  if (!skipScoring) {
    try { scores = await batchScoreAndTag(contents) }
    catch { scores = pending.map(m => ({ importance: m.importance ?? 5, type: m.type ?? 'FACT', tags: m.tags ?? [] })) }
  } else {
    scores = pending.map(m => ({ importance: m.importance ?? 5, type: m.type ?? 'FACT', tags: m.tags ?? [] }))
  }

  // Paso 3: Upsert en Qdrant (batch)
  const now    = Date.now()
  const points = pending.map((mem, i) => ({
    id:      mem.id,
    vector:  sparseVecs[i]?.indices?.length
      ? ({ '': denseVecs[i], bm25: sparseVecs[i] } as unknown as number[])
      : denseVecs[i],
    payload: {
      content: mem.content, projectName,
      createdAt: mem.createdAt ?? now, importance: scores[i].importance,
      accessCount: 0, lastAccessed: now,
      type: scores[i].type, tags: scores[i].tags, linkedTo: [],
    } as Record<string, unknown>,
  }))

  const indexedIds: string[] = []
  try {
    await qdrant.upsert(targetCol, { wait: true, points })
    indexedIds.push(...pending.map(m => m.id))
  } catch {
    // Fallback individual
    for (let i = 0; i < pending.length; i++) {
      const sparseArg = sparseVecs[i]?.indices?.length ? sparseVecs[i] : undefined
      try {
        await upsertEngrama(pending[i].id, denseVecs[i], {
          content: pending[i].content, projectName,
          createdAt: pending[i].createdAt ?? now, importance: scores[i].importance,
          accessCount: 0, lastAccessed: now,
          type: scores[i].type as Engrama['type'], tags: scores[i].tags, linkedTo: [],
        }, targetCol, sparseArg)
        indexedIds.push(pending[i].id)
      } catch { /* log ya en el caller */ }
    }
  }

  // Limpiar del buffer los indexados
  if (indexedIds.length > 0) await deleteTempIds(indexedIds)

  return indexedIds.length
}

/**
 * Indexa TODO el buffer pendiente, recorriendo TODOS los proyectos.
 * Para el arranque del servidor (rescata memorias huérfanas de sesiones anteriores).
 */
async function autoIndexAllPending(): Promise<void> {
  log('MAINTENANCE_INDEX_ALL_START')
  try {
    // scrollTemp sin projectName trae todo
    const allPending = await scrollTemp(undefined, 'pending', 500)
    if (allPending.length === 0) {
      log('MAINTENANCE_INDEX_ALL_EMPTY')
      return
    }
    // Agrupar por proyecto
    const byProject = new Map<string, number>()
    for (const m of allPending) byProject.set(m.projectName, (byProject.get(m.projectName) ?? 0) + 1)
    log('MAINTENANCE_INDEX_ALL_FOUND', {
      total: allPending.length,
      projects: Object.fromEntries(byProject),
    })
    const state = loadMaintState()
    const now   = Date.now()
    for (const [projectName, count] of byProject.entries()) {
      log('MAINTENANCE_INDEX_PROJECT_START', { projectName, count })
      // Procesar en lotes de MAINT_INDEX_BATCH_SIZE hasta vaciar el buffer
      let remaining = count
      while (remaining > 0) {
        const indexed = await autoIndexTemp(projectName, MAINT_INDEX_BATCH_SIZE, false)
        if (indexed === 0) break   // sin progreso, evitar loop infinito
        remaining -= indexed
        log('MAINTENANCE_INDEX_PROJECT_BATCH', { projectName, indexed, remaining })
      }
      state.lastIndexTemp = state.lastIndexTemp ?? {}
      state.lastIndexTemp[projectName] = now
    }
    saveMaintState(state)
    log('MAINTENANCE_INDEX_ALL_DONE', { projects: byProject.size, total: allPending.length })
  } catch (e) {
    log('MAINTENANCE_INDEX_ALL_ERROR', { error: String(e) })
  }
}

let maintenanceBusy = false

async function runAutoMaintenance(): Promise<void> {
  if (maintenanceBusy) { log('MAINTENANCE_SKIPPED', { reason: 'already running' }); return }
  maintenanceBusy = true
  log('MAINTENANCE_CYCLE_START')
  try {
    const state = loadMaintState()
    const now   = Date.now()

    // ── Paso 0: Auto-Index del buffer (SIEMPRE, antes de consolidar) ───────────
    // Si hay memorias pendientes en temp_memories, las indexamos primero.
    // Así consolidate las verá en este mismo ciclo.
    const allPending = await scrollTemp(undefined, 'pending', 500)
    if (allPending.length > 0) {
      const byProject = new Map<string, number>()
      for (const m of allPending) byProject.set(m.projectName, (byProject.get(m.projectName) ?? 0) + 1)
      log('MAINTENANCE_INDEX_PENDING', { total: allPending.length, projects: byProject.size })
      state.lastIndexTemp = state.lastIndexTemp ?? {}
      for (const [projectName] of byProject.entries()) {
        // Indexar hasta MAINT_INDEX_BATCH_SIZE memorias por proyecto (sin bloquear demasiado)
        const indexed = await autoIndexTemp(projectName, MAINT_INDEX_BATCH_SIZE, false).catch(() => 0)
        if (indexed > 0) {
          state.lastIndexTemp[projectName] = now
          log('MAINTENANCE_INDEX_DONE', { projectName, indexed })
        }
      }
      saveMaintState(state)
    }

    const collections = await listCortexCollections()
    const projects    = collections
      .map(c => c.replace(/^cortex_/, '').replace(/_/g, '-'))
      .filter(p => p !== 'global' && !p.startsWith('episodes') && !p.startsWith('cortex-episodes'))

    for (const projectName of projects) {
      // ── Auto-Consolidate ──────────────────────────────────────────────────
      const lastConsol   = state.lastConsolidate[projectName] ?? 0
      const dueConsol    = (now - lastConsol) > MAINT_CONSOLIDATE_INTERVAL
      if (dueConsol) {
        const engramas = await scrollAll(projectName).catch(() => [] as Engrama[])
        if (engramas.length >= MAINT_MIN_ENGRAMAS_CONSOL) {
          log('MAINTENANCE_CONSOLIDATE_START', { projectName, engramas: engramas.length })
          const result = await runConsolidation(projectName)
          state.lastConsolidate[projectName] = now
          saveMaintState(state)
          log('MAINTENANCE_CONSOLIDATE_DONE', { projectName, report: result.report.slice(0, 120) })
        } else {
          log('MAINTENANCE_CONSOLIDATE_SKIP', { projectName, reason: 'not enough engramas', count: engramas.length })
        }
      }

      // ── Auto-Detect Patterns ──────────────────────────────────────────────
      const lastDetect  = state.lastDetectPatterns[projectName] ?? 0
      const dueDetect   = (now - lastDetect) > MAINT_DETECT_INTERVAL
      if (dueDetect) {
        const engramas = await scrollAll(projectName).catch(() => [] as Engrama[])
        if (engramas.length >= MAINT_MIN_ENGRAMAS_DETECT) {
          log('MAINTENANCE_DETECT_START', { projectName, engramas: engramas.length })
          const patterns = await autoDetectPatterns(projectName, engramas)
          state.lastDetectPatterns[projectName] = now
          saveMaintState(state)
          log('MAINTENANCE_DETECT_DONE', { projectName, patterns: patterns.length })
        } else {
          log('MAINTENANCE_DETECT_SKIP', { projectName, reason: 'not enough engramas', count: engramas.length })
        }
      }
    }
  } catch (e) {
    log('MAINTENANCE_ERROR', { error: String(e) })
  } finally {
    maintenanceBusy = false
    log('MAINTENANCE_CYCLE_END')
  }
}

// Bootstrap de mantenimiento — dos relojes independientes:
//
//   1. Auto-index del buffer: arranca en 60s (el servidor ya está caliente,
//      rescata memorias huérfanas de sesiones anteriores).
//
//   2. Ciclo completo (index + consolidate + detect_patterns):
//      arranca a los 5 min y luego cada hora. Usa un timer independiente
//      para no cancelarse si autoIndexAllPending() tarda.

setTimeout(() => {
  void autoIndexAllPending()    // rescate inmediato del buffer
}, MAINT_INDEX_STARTUP_DELAY)

setTimeout(() => {
  void runAutoMaintenance()
  setInterval(() => { void runAutoMaintenance() }, MAINT_CHECK_INTERVAL)
}, MAINT_STARTUP_DELAY)

log('MAINTENANCE_SCHEDULER_REGISTERED', {
  indexBufferDelaySeconds: 60,
  fullCycleStartupDelayMinutes: 5,
  fullCycleIntervalHours: 1,
})

const server = new Server(
  { name: 'cortex-memory', version: PKG_VERSION },
  { capabilities: { tools: {} } },
)

// ─── Tool Definitions ─────────────────────────────────────────────────────────

server.setRequestHandler(ListToolsRequestSchema, async () => {
  log('TOOLS_LIST_CALLED')
  return {  tools: [
    // ── FASE 1 ──────────────────────────────────────────────────────────────
    {
      name: 'observe',
      description: 'Guarda un hecho o decisión en la memoria CORTEX. Lo puntúa, categoriza, genera tags y crea vínculos automáticamente.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Nombre del proyecto (ej: my-api, ecommerce, backend)' },
          content: { type: 'string', description: 'El hecho, decisión o contexto a memorizar' },
        },
        required: ['projectName', 'content'],
      },
    },
    {
      name: 'recall',
      description: 'Recupera memorias relevantes de CORTEX ponderadas por relevancia semántica y decay temporal.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Nombre del proyecto' },
          query: { type: 'string', description: 'Texto o pregunta para buscar' },
          limit: { type: 'number', description: 'Máximo de resultados (default: 5)' },
        },
        required: ['projectName', 'query'],
      },
    },
    {
      name: 'consolidate',
      description: 'Comprime la memoria de un proyecto: detecta duplicados y los fusiona en engramas más ricos.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Nombre del proyecto' },
        },
        required: ['projectName'],
      },
    },
    // ── FASE 2 ──────────────────────────────────────────────────────────────
    {
      name: 'get_context_for',
      description: 'RAG automático: recupera el contexto más relevante de la memoria para inyectarlo en el system prompt al inicio de cada sesión. Devuelve un bloque de texto listo para usar.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Nombre del proyecto activo' },
          message: { type: 'string', description: 'El primer mensaje o tema de la sesión actual' },
          maxItems: { type: 'number', description: 'Máximo de memorias a incluir (default: 6)' },
        },
        required: ['projectName', 'message'],
      },
    },
    {
      name: 'detect_patterns',
      description: 'Analiza las memorias recientes de un proyecto y detecta patrones, preferencias recurrentes y errores habituales. Los guarda como engramas PATTERN de alta importancia.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Nombre del proyecto a analizar' },
          limit: { type: 'number', description: 'Número de engramas recientes a analizar (default: 50)' },
        },
        required: ['projectName'],
      },
    },
    // ── FASE 3 ──────────────────────────────────────────────────────────────
    {
      name: 'get_operator_profile',
      description: 'Recupera el perfil del operador: preferencias de código, proyectos activos y patrones detectados. Úsalo al inicio de sesión para personalizar respuestas.',
      inputSchema: { type: 'object', properties: {} },
    },
    {
      name: 'cortex_status',
      description: 'Estado del sistema CORTEX: colecciones activas, número de engramas por proyecto, y salud general.',
      inputSchema: { type: 'object', properties: {} },
    },
    // ── P1: CRUD completo ────────────────────────────────────────────
    {
      name: 'delete_memory',
      description: 'Elimina un engrama de la memoria por su ID.',
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'UUID del engrama a eliminar' },
        },
        required: ['id'],
      },
    },
    {
      name: 'update_memory',
      description: 'Actualiza el contenido de un engrama existente. Recalcula su embedding y lo re-persiste.',
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'UUID del engrama a actualizar' },
          content: { type: 'string', description: 'Nuevo contenido del engrama' },
        },
        required: ['id', 'content'],
      },
    },
    {
      name: 'get_all_memories',
      description: 'Retorna todos los engramas de un proyecto ordenados por decay score (los más relevantes primero).',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Nombre del proyecto' },
          limit: { type: 'number', description: 'Máximo de engramas a retornar (default: 20)' },
        },
        required: ['projectName'],
      },
    },
    // ── P3: Gestión avanzada ─────────────────────────────────────────
    {
      name: 'delete_all_memories',
      description: 'Elimina TODOS los engramas de un proyecto. Operación irreversible. Requiere confirmar con confirm=true.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Nombre del proyecto a limpiar' },
          confirm: { type: 'boolean', description: 'Debe ser true para confirmar la operación' },
        },
        required: ['projectName', 'confirm'],
      },
    },
    {
      name: 'batch_observe',
      description: 'Guarda múltiples memorias de golpe. Más eficiente que llamar observe N veces.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Proyecto donde guardar las memorias' },
          memories: {
            type: 'array',
            description: 'Lista de textos a memorizar (máx. 20)',
            items: { type: 'string' },
          },
        },
        required: ['projectName', 'memories'],
      },
    },
    {
      name: 'export_memories',
      description: 'Exporta todas las memorias de un proyecto como JSON. Útil para backup o migración.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Proyecto a exportar' },
        },
        required: ['projectName'],
      },
    },
    // ── TEMP_MEMORIES: buffer sin LLM ─────────────────────────────────────────────
    {
      name: 'quick_observe',
      description: 'Guarda una memoria en el buffer temporal (temp_memories) SIN usar LLM ni embedding. Instantáneo y sin costo de CPU. Usa list_pending para ver qué hay pendiente e index_temp para indexar cuando tengas recursos.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Nombre del proyecto' },
          content: { type: 'string', description: 'El hecho, decisión o contexto a guardar' },
        },
        required: ['projectName', 'content'],
      },
    },
    {
      name: 'list_pending',
      description: 'Lista las memorias pendientes de indexar en temp_memories. Muestra cuántas hay por proyecto y su contenido.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Proyecto (omitir para ver todos los proyectos)' },
          limit: { type: 'number', description: 'Máximo de resultados (default: 20)' },
        },
        required: [],
      },
    },
    {
      name: 'index_temp',
      description: 'Indexa memorias de temp_memories hacia la colección del proyecto (cortex_*) con embedding ONNX (fastembed) y scoring LLM. Llamar manualmente cuando haya CPU disponible. Procesa en lotes controlados.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Proyecto a indexar' },
          batchSize: { type: 'number', description: 'Número de memorias a procesar (default: 5, máx: 20)' },
          skipScoring: { type: 'boolean', description: 'Si true, omite el scoring LLM (qwen3) y usa valores por defecto. Más rápido pero sin etiquetas automáticas.' },
        },
        required: ['projectName'],
      },
    },
    {
      name: 'recall_hybrid',
      description: 'Busca memorias en ambas fuentes: keyword en temp_memories (buffer, sin LLM) + semántica en la colección del proyecto (fastembed). Ideal para encontrar lo que guardaste recientemente.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Nombre del proyecto' },
          query: { type: 'string', description: 'Texto o pregunta para buscar' },
          limit: { type: 'number', description: 'Máximo de resultados por fuente (default: 5)' },
        },
        required: ['projectName', 'query'],
      },
    },
    // ── EPISODIC MEMORY (v3.2) ────────────────────────────────────────────────
    {
      name: 'start_session',
      description: 'Abre una sesión episódica para el proyecto. Registra el contexto/objetivo de la conversación actual y cierra automáticamente cualquier sesión anterior abierta. Sin LLM — instantáneo.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Nombre del proyecto' },
          context: { type: 'string', description: 'Tema u objetivo de esta sesión (ej: "implementar refresh tokens JWT")' },
        },
        required: ['projectName', 'context'],
      },
    },
    {
      name: 'log_event',
      description: 'Registra un evento en la sesión episódica activa. Úsalo para marcar decisiones importantes, errores encontrados, soluciones aplicadas o insights relevantes durante la conversación.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Nombre del proyecto' },
          sessionId: { type: 'string', description: 'ID de la sesión activa (retornado por start_session)' },
          eventType: {
            type: 'string',
            description: 'Tipo de evento',
            enum: ['DECISION', 'ERROR', 'SOLUTION', 'INSIGHT', 'CONTEXT_CHANGE'],
          },
          description: { type: 'string', description: 'Descripción del evento (qué pasó, qué se decidió)' },
        },
        required: ['projectName', 'sessionId', 'eventType', 'description'],
      },
    },
    {
      name: 'recall_sessions',
      description: 'Busca sesiones pasadas relevantes para una consulta. Retorna sesiones ordenadas por similitud semántica con sus eventos y contexto. Sin LLM — usa fastembed ONNX.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Nombre del proyecto' },
          query: { type: 'string', description: 'Qué estás buscando (ej: "decisiones sobre autenticación")' },
          limit: { type: 'number', description: 'Máximo de sesiones a retornar (default: 3)' },
        },
        required: ['projectName', 'query'],
      },
    },
    // ── KNOWLEDGE GRAPH (v4.0) ────────────────────────────────────────────────
    {
      name: 'graph_neighbors',
      description: 'Muestra las entidades directamente conectadas a una entidad en el Knowledge Graph. Sin LLM — consulta Kuzu.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Nombre del proyecto' },
          entity:      { type: 'string', description: 'Nombre exacto de la entidad (ej: "PostgreSQL", "reactor-netty")' },
          limit:       { type: 'number', description: 'Máximo de vecinos (default: 10)' },
        },
        required: ['projectName', 'entity'],
      },
    },
    {
      name: 'graph_timeline',
      description: 'Muestra la evolución temporal de una entidad: qué relaciones se registraron y cuándo. Sin LLM.',
      inputSchema: {
        type: 'object',
        properties: {
          projectName: { type: 'string', description: 'Nombre del proyecto' },
          entity:      { type: 'string', description: 'Nombre de la entidad' },
          limit:       { type: 'number', description: 'Máximo de eventos (default: 20)' },
        },
        required: ['projectName', 'entity'],
      },
    },
    {
      name: 'graph_query',
      description: 'Ejecuta una consulta Cypher directa sobre el Knowledge Graph (Kuzu). Para usuarios avanzados.',
      inputSchema: {
        type: 'object',
        properties: {
          cypher: { type: 'string', description: 'Consulta Cypher (ej: MATCH (n:Entity) RETURN n.name LIMIT 10)' },
        },
        required: ['cypher'],
      },
    },
  ]}
})

// ─── Tool Handlers ────────────────────────────────────────────────────────────

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params
  log('TOOL_CALLED', { name, args: JSON.stringify(args).slice(0, 200) })

  try {
  // ── observe ──────────────────────────────────────────────────────────────
  if (name === 'observe') {
    const projectName = requireString(args, 'projectName', 'projectName (ej: "my-project")')
    const content = requireString(args, 'content', 'content (texto a memorizar)')

    const engramaId = uuidv4()
    const rawResult = await observeGraph.invoke({
      content: content.trim(),
      projectName,
      engramaId,
      importance: 5,
      type: 'FACT',
      tags: [],
      embedding: [],
      linkedTo: [],
      supersededIds: [],
      status: 'starting',
    })
    const result = rawResult as unknown as import('./graph/observe/state.js').ObserveState

    const linksText = result.linkedTo?.length > 0
      ? `\nVinculado con ${result.linkedTo.length} memorias relacionadas.`
      : ''
    const supersededText = result.supersededIds?.length > 0
      ? `\n⚠️ Invalidó ${result.supersededIds.length} memoria(s) obsoleta(s).`
      : ''

    return {
      content: [{
        type: 'text',
        text: `✅ Engrama guardado.\n• ID: ${engramaId}\n• Importancia: ${result.importance}/10\n• Tipo: ${result.type}\n• Tags: ${result.tags?.join(', ') || 'ninguno'}${linksText}${supersededText}`,
      }],
    }
  }

  // ── recall ───────────────────────────────────────────────────────────────
  if (name === 'recall') {
    const projectName = requireString(args, 'projectName', 'projectName (ej: "my-project")')
    const query = requireString(args, 'query', 'query (texto de búsqueda)')
    const limit = Math.min(Number(args?.limit ?? 5), 20)

    const queryEmbedding = await getEmbedding(query)
    // Generar sparse en paralelo con el dense para no añadir latencia
    const [sparseVec, hits] = await Promise.all([
      getSparseEmbedding(query).catch(() => ({ indices: [], values: [] })),
      // pre-fetch dense solo para tener el embedding listo; la búsqueda real es híbrida
      Promise.resolve(null),
    ])
    // Traer el doble de candidatos para que el reranker tenga más contexto
    const hybridHits = await searchHybridCrossProject(queryEmbedding, sparseVec, projectName, limit * 2)
    void hits  // satisfacer TypeScript

    if (hybridHits.length === 0) {
      return { content: [{ type: 'text', text: 'No se encontraron memorias relevantes.' }] }
    }

    const engramas: Engrama[] = hybridHits
      .map((h: SearchHit) => ({ id: h.id, ...h.payload } as Engrama))
      .filter(e => e.status !== 'superseded')  // excluir memorias invalidadas

    // ── Cross-encoder reranking (batch, una sola llamada Ollama) ────────────────
    // Controlado por CORTEX_RERANKER_ENABLED en .env (false = skip, usa solo fastembed)
    const rerankMap = new Map<string, number>()
    if (RERANKER_ENABLED && hybridHits.length >= 3) {
      const candidates = engramas.map(e => ({ id: e.id, content: e.content ?? '' }))
      const rerankResults = await rerankWithLLM(query, candidates)
      for (const r of rerankResults) rerankMap.set(r.id, r.rerankScore)
      log('RECALL_RERANK', { query: query.slice(0, 60), candidates: candidates.length })
    } else if (!RERANKER_ENABLED) {
      log('RECALL_RERANK_SKIPPED', { reason: 'CORTEX_RERANKER_ENABLED=false' })
    }

    const ranked = engramas
      .map(e => {
        const hit = hybridHits.find((h: SearchHit) => h.id === e.id)!
        const rerankScore = rerankMap.get(e.id)
        return {
          engrama: e,
          semantic: hit.score,
          decay: calcDecay(e),
          combined: combinedScore(hit.score, calcDecay(e), e.type, rerankScore),
          reranked: rerankScore !== undefined,
        }
      })
      .sort((a, b) => b.combined - a.combined)
      .slice(0, limit)

    const now = Date.now()
    for (const item of ranked) {
      await patchPayload(item.engrama.id, {
        accessCount: (item.engrama.accessCount ?? 0) + 1,
        lastAccessed: now,
      }).catch(() => {})
    }

    const lines = ranked.map((item, i) => {
      const e = item.engrama
      const tags = e.tags?.join(', ') || ''
      const links = e.linkedTo?.length ? ` [${e.linkedTo.length} vínculos]` : ''
      const rerankBadge = item.reranked ? ` ★${(item.combined * 10).toFixed(1)}` : ''
      return `${i + 1}. [${e.type ?? 'FACT'}] [imp:${e.importance ?? '?'}/10]${rerankBadge} ${e.content}\n   Tags: ${tags}${links}`
    })

    const rerankNote = rerankMap.size > 0 ? ` (hybrid BM25+dense, reranked)` : ` (hybrid BM25+dense)`
    return {
      content: [{
        type: 'text',
        text: `📚 ${ranked.length} memorias para "${query}" en ${projectName}${rerankNote}:\n\n${lines.join('\n\n')}`,
      }],
    }
  }

  // ── consolidate ──────────────────────────────────────────────────────────
  if (name === 'consolidate') {
    const projectName = args?.projectName as string
    // La colección del proyecto se crea automáticamente si no existe
    const result = await runConsolidation(projectName)
    return {
      content: [{
        type: 'text',
        text: `🧹 Consolidación completada para ${projectName}:\n${result.report}`,
      }],
    }
  }

  // ── get_context_for (FASE 2+3) ─────────────────────────────────────────────
  if (name === 'get_context_for') {
    const projectName = (args?.projectName || args?.project_name || 'global') as string
    const maxItems = Number(args?.maxItems ?? 6)

    // Fallback robusto si message viene vacío o undefined
    const rawMessage = (args?.message || args?.query || '') as string
    const searchQuery = rawMessage.trim() || `proyecto ${projectName} estado decisiones`

    log('GET_CONTEXT_FOR', { projectName, query: searchQuery.slice(0, 80) })

    // Las colecciones se crean automáticamente al observar
    // Dense + sparse en paralelo para no añadir latencia secuencial
    const [queryEmbedding, sparseVecCtx] = await Promise.all([
      getEmbedding(searchQuery),
      getSparseEmbedding(searchQuery).catch(() => ({ indices: [], values: [] })),
    ])
    // Búsqueda híbrida cross-project (BM25+dense, colección propia + global)
    const hits = await searchHybridCrossProject(queryEmbedding, sparseVecCtx, projectName, maxItems * 2)

    if (hits.length === 0) {
      return {
        content: [{
          type: 'text',
          text: `<!-- CORTEX: Sin memorias previas para el proyecto "${projectName}" -->`,
        }],
      }
    }

    // Rerank por combined score (semántico + decay)
    const engramas: Engrama[] = hits
      .map(h => ({ id: h.id, ...h.payload } as Engrama))
      .filter(e => e.status !== 'superseded')  // excluir memorias invalidadas

    // ── Cross-encoder reranking (batch, una sola llamada Ollama) ────────────────
    // Controlado por CORTEX_RERANKER_ENABLED en .env (false = skip, usa solo fastembed)
    const rerankMap = new Map<string, number>()
    if (RERANKER_ENABLED && hits.length >= 3) {
      const candidates = engramas.map(e => ({ id: e.id, content: e.content ?? '' }))
      const rerankResults = await rerankWithLLM(searchQuery, candidates)
      for (const r of rerankResults) rerankMap.set(r.id, r.rerankScore)
      log('GET_CONTEXT_RERANK', { query: searchQuery.slice(0, 60), candidates: candidates.length })
    } else if (!RERANKER_ENABLED) {
      log('GET_CONTEXT_RERANK_SKIPPED', { reason: 'CORTEX_RERANKER_ENABLED=false' })
    }

    const ranked = engramas
      .map(e => {
        const hit = hits.find(h => h.id === e.id)!
        const rerankScore = rerankMap.get(e.id)
        return {
          engrama: e,
          combined: combinedScore(hit.score, calcDecay(e), e.type, rerankScore),
        }
      })
      .sort((a, b) => b.combined - a.combined)
      .slice(0, maxItems)

    // Actualizar access stats
    const now = Date.now()
    for (const item of ranked) {
      await patchPayload(item.engrama.id, {
        accessCount: (item.engrama.accessCount ?? 0) + 1,
        lastAccessed: now,
      }).catch(() => {})
    }

    // Formatear como bloque de contexto inyectable en system prompt
    const contextLines = ranked.map(item => {
      const e = item.engrama
      const type = e.type ?? 'FACT'
      const imp = e.importance ?? 5
      return `• [${type}] ${e.content} (imp:${imp}/10)`
    })

    // ── Bloque episódico: sesiones recientes (sin LLM) ──────────────────────
    const recentEpisodes = await getRecentEpisodes(projectName, 2)
    const episodeLines: string[] = []
    for (const ep of recentEpisodes) {
      const date     = new Date(ep.startedAt).toLocaleDateString('es-MX')
      const duration = ep.endedAt
        ? `${Math.round((ep.endedAt - ep.startedAt) / 60000)} min`
        : 'en curso'
      const highlights = ep.events
        .filter(e => e.eventType === 'DECISION' || e.eventType === 'SOLUTION')
        .slice(0, 3)
        .map(e => `  → ${e.description}`)
        .join('\n')
      episodeLines.push(`• **${date}** (${duration}): ${ep.context}${highlights ? '\n' + highlights : ''}`)
    }

    const episodeBlock = episodeLines.length > 0
      ? ['\n## 📼 Sesiones recientes', ...episodeLines].join('\n')
      : ''

    // ── Bloque de Knowledge Graph: entidades relacionadas (sin LLM) ──────────
    const keywords = searchQuery
      .split(/\s+/)
      .filter(w => w.length > 4)
      .slice(0, 5)
    const graphBlock = await queryGraphContext(keywords, projectName, 5).catch(() => '')

    const contextBlock = [
      `## 🧠 Memoria CORTEX — Proyecto: ${projectName}`,
      `*(${ranked.length} engramas más relevantes${rerankMap.size > 0 ? ', reranked' : ''})*`,
      '',
      ...contextLines,
      episodeBlock,
      graphBlock,
      '',
      `> Usa este contexto como conocimiento previo. No lo repitas textualmente.`,
    ].join('\n')

    log('GET_CONTEXT_FOR_RESULT', { 
      projectName, 
      count: ranked.length,
      reranked: rerankMap.size > 0,
      episodes: recentEpisodes.length,
      items: ranked.map(r => r.engrama.content?.slice(0, 60))
    })

    return {
      content: [{ type: 'text', text: contextBlock }],
    }
  }



  // ── detect_patterns (FASE 2) ──────────────────────────────────────────────
  // La lógica central está en autoDetectPatterns() — compartida con el scheduler.
  if (name === 'detect_patterns') {
    const projectName = args?.projectName as string

    const allEngramas = await scrollAll(projectName)
    if (allEngramas.length < 5) {
      return {
        content: [{
          type: 'text',
          text: `⚠️ Necesitas al menos 5 engramas en "${projectName}" para detectar patrones. Actualmente: ${allEngramas.length}`,
        }],
      }
    }

    const patterns = await autoDetectPatterns(projectName, allEngramas)

    if (patterns.length === 0) {
      return {
        content: [{ type: 'text', text: '⚠️ No se pudieron detectar patrones claros con las memorias actuales.' }],
      }
    }

    // Actualizar timestamp de mantenimiento manual también
    const state = loadMaintState()
    state.lastDetectPatterns[projectName] = Date.now()
    saveMaintState(state)

    log('OPERATOR_PROFILE_SAVED', { projectName, patterns: patterns.length, trigger: 'manual' })

    return {
      content: [{
        type: 'text',
        text: `🔍 ${patterns.length} patrones detectados en "${projectName}" → guardados + Operator Profile actualizado:\n\n${patterns.map((p, i) => `${i + 1}. ${p}`).join('\n')}`,
      }],
    }
  }

  // ── get_operator_profile (FASE 3) ─────────────────────────────────────────
  if (name === 'get_operator_profile') {
    const profile = await getOperatorProfile()

    if (!profile) {
      return {
        content: [{
          type: 'text',
          text: `## 👤 Operator Profile — CORTEX\n\n*Sin perfil guardado todavía.*\n\nUsa \`detect_patterns\` en tus proyectos activos para que CORTEX genere tu perfil automáticamente.`,
        }],
      }
    }

    const lines = [
      `## 👤 Operator Profile — CORTEX`,
      `*Última actualización: ${new Date(profile.lastUpdated).toLocaleDateString('es-MX')}*`,
      '',
      `**Proyectos activos**: ${profile.activeProjects.join(', ') || 'ninguno'}`,
      '',
      `**Preferencias de código**:`,
      ...(profile.codingPreferences.map(p => `• ${p}`)),
      '',
      `**Patrones detectados**:`,
      ...(profile.detectedPatterns.map(p => `• ${p}`)),
    ]

    return {
      content: [{ type: 'text', text: lines.join('\n') }],
    }
  }

  // ── cortex_status (FASE 3) ────────────────────────────────────────────────
  if (name === 'cortex_status') {
    const collections = await listCortexCollections()
    const stats: string[] = []

    for (const col of collections) {
      try {
        const info = await qdrant.getCollection(col)
        const count = info.points_count ?? 0
        const project = col.replace(/^cortex_/, '').replace(/_/g, '-')
        stats.push(`• **${project}**: ${count} engramas`)
      } catch {
        stats.push(`• ${col}: error`)
      }
    }

    const profile = await getOperatorProfile()

    // ── Conteo de temp_memories pendientes ──
    const allPending = await scrollTemp(undefined, 'pending', 200)
    const tempLines: string[] = []
    if (allPending.length > 0) {
      const byProject = new Map<string, number>()
      for (const m of allPending) {
        const proj = m.projectName || 'sin-proyecto'
        byProject.set(proj, (byProject.get(proj) ?? 0) + 1)
      }
      for (const [proj, count] of byProject.entries()) {
        tempLines.push(`  · ${proj}: ${count} pendiente${count !== 1 ? 's' : ''}`)
      }
    }

    const tempBlock = allPending.length > 0
      ? [
          `**Buffer temporal (temp_memories)**: ⏳ ${allPending.length} pendiente${allPending.length !== 1 ? 's' : ''} de indexar`,
          `  _(el scheduler los indexará automáticamente — próximo ciclo: ~60s si acaba de arrancar, o en el siguiente ciclo horario)_`,
          ...tempLines,
        ]
      : [`**Buffer temporal (temp_memories)**: ✅ vacío`]

    const statusBlock = [
      `## 🧠 CORTEX Status`,
      `*v${PKG_VERSION} — LangGraph.js + Qdrant + fastembed*`,
      '',
      `**Colecciones activas**: ${collections.length}`,
      ...stats,
      '',
      ...tempBlock,
      '',
      `**Operator Profile**: ${profile ? `✅ (actualizado ${new Date(profile.lastUpdated).toLocaleDateString('es-MX')})` : '❌ sin generar'}`,
      '',
      `**🔧 Mantenimiento automático**:`,
      ...(() => {
        const maint = loadMaintState()
        const now   = Date.now()
        const lines: string[] = []
        const projects = collections
          .map(c => c.replace(/^cortex_/, '').replace(/_/g, '-'))
          .filter(p => p !== 'global' && !p.startsWith('episodes'))
        for (const p of projects) {
          const lc = maint.lastConsolidate[p]
          const ld = maint.lastDetectPatterns[p]
          const li = (maint.lastIndexTemp ?? {})[p]
          const consolAge = lc ? Math.round((now - lc) / 86400000) + 'd' : 'nunca'
          const detectAge = ld ? Math.round((now - ld) / 86400000) + 'd' : 'nunca'
          const indexAge  = li ? Math.round((now - li) / 3600000) + 'h' : 'nunca'
          lines.push(`  · ${p}: index_temp=${indexAge} · consolidate=${consolAge} · detect=${detectAge}`)
        }
        return lines.length > 0 ? lines : ['  · Sin proyectos indexados aún']
      })(),
    ].join('\n')


    return {
      content: [{ type: 'text', text: statusBlock }],
    }
  }

  // ── delete_memory (P1) ─────────────────────────────────────────────────
  if (name === 'delete_memory') {
    const id = args?.id as string
    if (!id) throw new Error('Parámetro requerido: id')
    const found = await findEngramaById(id)
    if (!found) return errorResponse(`No se encontró ningún engrama con ID ${id}`)
    await deleteEngramas([id], found.collection)
    log('DELETE_MEMORY', { id, collection: found.collection })
    return { content: [{ type: 'text', text: `✅ Engrama ${id} eliminado.` }] }
  }

  // ── update_memory (P1) ─────────────────────────────────────────────────
  if (name === 'update_memory') {
    const id = args?.id as string
    const content = args?.content as string
    if (!id || !content) throw new Error('Parámetros requeridos: id, content')

    const found = await findEngramaById(id)
    if (!found) return errorResponse(`No se encontró ningún engrama con ID ${id}`)

    const [newEmbedding, sparse, scored] = await Promise.all([
      getEmbedding(content),
      getSparseEmbedding(content).catch(() => undefined),
      scoreAndTag(content),
    ])

    // Merge sobre el payload existente para no perder projectName, createdAt,
    // accessCount, linkedTo, status, etc. al re-upsertear con el nuevo vector.
    const merged = {
      ...found.payload,
      content,
      importance: scored.importance,
      tags: scored.tags,
      type: scored.type,
      lastAccessed: Date.now(),
    }

    await upsertEngrama(id, newEmbedding, merged, found.collection, sparse)
    log('UPDATE_MEMORY', { id, collection: found.collection, content: content.slice(0, 60) })
    return { content: [{ type: 'text', text: `✅ Engrama ${id} actualizado. Importancia: ${scored.importance}/10` }] }
  }

  // ── get_all_memories (P1) ─────────────────────────────────────────────
  if (name === 'get_all_memories') {
    const projectName = (args?.projectName || args?.project_name) as string
    const limit = Number(args?.limit ?? 20)
    if (!projectName) throw new Error('Parámetro requerido: projectName')

    const all = await scrollAll(projectName)
    const ranked = all
      .map(e => ({ e, decay: calcDecay(e) }))
      .sort((a, b) => b.decay - a.decay)
      .slice(0, limit)

    const lines = ranked.map((item, i) => {
      const { e, decay } = item
      return `${i + 1}. [${e.type ?? 'FACT'}] [imp:${e.importance ?? '?'}/10] [decay:${decay.toFixed(2)}] ${e.content?.slice(0, 80)}\n   ID: ${e.id} | Tags: ${e.tags?.join(', ') || 'ninguno'}`
    })

    return {
      content: [{ type: 'text', text: `💡 ${ranked.length} engramas en "${projectName}" (de ${all.length} total):\n\n${lines.join('\n\n')}` }],
    }
  }

  // ── delete_all_memories (P3) ──────────────────────────────────────────────
  if (name === 'delete_all_memories') {
    const projectName = requireString(args, 'projectName', 'projectName')
    const confirm = args?.confirm

    if (confirm !== true) {
      return errorResponse(
        `Operación cancelada — debes pasar confirm: true para borrar TODOS los engramas de "${projectName}"`,
      )
    }

    const deleted = await deleteAllInProject(projectName)
    log('DELETE_ALL', { projectName, deleted })
    return {
      content: [{ type: 'text', text: `🗑️ ${deleted} engramas eliminados del proyecto "${projectName}".` }],
    }
  }

  // ── batch_observe (P3) ────────────────────────────────────────────────────
  // Sin LLM, sin embedding — escribe al buffer temporal igual que quick_observe.
  // Usar index_temp cuando haya CPU disponible para indexar con scoring.
  if (name === 'batch_observe') {
    const projectName = requireString(args, 'projectName', 'projectName')
    const memories = args?.memories

    if (!Array.isArray(memories) || memories.length === 0) {
      throw new Error('Parámetro requerido: memories (array de strings no vacío)')
    }

    const batch = (memories as string[]).slice(0, 20).filter(m => m?.trim())
    const ids: string[] = []

    for (const mem of batch) {
      const id = uuidv4()
      await upsertTemp(id, mem.trim(), projectName)
      ids.push(id)
    }

    log('BATCH_OBSERVE', { projectName, count: batch.length })
    return {
      content: [{
        type: 'text',
        text: `⚡ ${batch.length} memorias guardadas en buffer temporal de "${projectName}".\nSin LLM · Sin embedding · Instantáneo\n\nUsa \`index_temp\` cuando tengas CPU disponible para indexarlas con scoring.`,
      }],
    }
  }

  // ── export_memories (P3) ──────────────────────────────────────────────────
  if (name === 'export_memories') {
    const projectName = requireString(args, 'projectName', 'projectName')
    const exported = await exportProject(projectName)
    const json = JSON.stringify(exported, null, 2)

    log('EXPORT', { projectName, count: exported.count })
    return {
      content: [{
        type: 'text',
        text: `📦 Export de "${projectName}" — ${exported.count} engramas:\n\`\`\`json\n${json.slice(0, 4000)}${json.length > 4000 ? '\n... (truncado, usa el CLI para exportación completa)' : ''}\n\`\`\``,
      }],
    }
  }

  // ── quick_observe ──────────────────────────────────────────────────────────────
  if (name === 'quick_observe') {
    const projectName = requireString(args, 'projectName', 'projectName')
    const content = requireString(args, 'content', 'content')
    const id = uuidv4()
    await upsertTemp(id, content.trim(), projectName)
    log('QUICK_OBSERVE', { projectName, id, content: content.slice(0, 80) })
    return {
      content: [{
        type: 'text',
        text: `⚡ Guardado en buffer temporal.\n• ID: ${id}\n• Proyecto: ${projectName}\n• Sin LLM, sin embedding — usa index_temp para indexar cuando tengas CPU disponible.`,
      }],
    }
  }

  // ── list_pending ─────────────────────────────────────────────────────────────
  if (name === 'list_pending') {
    const projectName = args?.projectName as string | undefined
    const limit = Math.min(Number(args?.limit ?? 20), 100)
    const pending = await scrollTemp(projectName, 'pending', limit)

    if (pending.length === 0) {
      const msg = projectName
        ? `✅ No hay memorias pendientes en "${projectName}".`
        : '✅ No hay memorias pendientes en temp_memories.'
      return { content: [{ type: 'text', text: msg }] }
    }

    // Agrupar por proyecto para mejor legibilidad
    const byProject = new Map<string, TempMemory[]>()
    for (const m of pending) {
      const proj = m.projectName || 'sin-proyecto'
      if (!byProject.has(proj)) byProject.set(proj, [])
      byProject.get(proj)!.push(m)
    }

    const sections: string[] = []
    for (const [proj, mems] of byProject.entries()) {
      sections.push(`**${proj}** (${mems.length} pendientes):`)
      mems.forEach((m, i) => {
        const date = new Date(m.createdAt).toLocaleString('es-MX', { timeZone: 'America/Mexico_City' })
        sections.push(`  ${i + 1}. [${date}] ${m.content.slice(0, 100)}${m.content.length > 100 ? '...' : ''}\n     ID: ${m.id}`)
      })
    }

    return {
      content: [{
        type: 'text',
        text: `📥 ${pending.length} memorias pendientes en temp_memories:\n\n${sections.join('\n')}`,
      }],
    }
  }

  // ── index_temp ───────────────────────────────────────────────────────────────
  //
  // Pipeline optimizado para CPU-only (sin GPU):
  //   1. getEmbeddingBatch()       → 1 sola llamada ONNX para todos los textos
  //   2. getSparseEmbeddingBatch() → 1 sola llamada SPLADE para todos los textos
  //   3. batchScoreAndTag()        → 1 sola llamada qwen3 para clasificar todos
  //   4. upsert batch en Qdrant    → 1 sola operación con wait:false
  //
  // Speedup vs versión anterior (secuencial N×): ~3-8× dependiendo del batchSize.
  if (name === 'index_temp') {
    const projectName = requireString(args, 'projectName', 'projectName')
    const batchSize   = Math.min(Number(args?.batchSize ?? 5), 20)
    const skipScoring = args?.skipScoring === true
    const t0 = Date.now()

    // Verificar si hay pendientes antes de llamar autoIndexTemp
    const before = await scrollTemp(projectName, 'pending', batchSize)
    if (before.length === 0) {
      return {
        content: [{ type: 'text', text: `✅ No hay memorias pendientes en "${projectName}". Todo está indexado.` }],
      }
    }

    // Delegar en la lógica compartida con el scheduler
    const indexed = await autoIndexTemp(projectName, batchSize, skipScoring)

    // Actualizar timestamp en maintenance.json también para llamadas manuales
    const state = loadMaintState()
    state.lastIndexTemp = state.lastIndexTemp ?? {}
    state.lastIndexTemp[projectName] = Date.now()
    saveMaintState(state)

    const remaining    = await scrollTemp(projectName, 'pending', 1)
    const pendingNote  = remaining.length > 0 ? '⏳ Aún hay pendientes — vuelve a llamar index_temp o el scheduler los indexará automáticamente.' : '✅ Todo indexado.'
    const mode         = skipScoring ? 'sin scoring LLM' : 'con scoring qwen3 (batch)'
    const elapsed      = Date.now() - t0

    log('INDEX_TEMP', { projectName, indexed, batchSize, skipScoring, totalMs: elapsed })
    return {
      content: [{ type: 'text', text: `🔄 ${indexed}/${before.length} memorias indexadas en "${projectName}" ${mode} (${elapsed}ms)\n\n${pendingNote}` }],
    }
  }

  // ── recall_hybrid ─────────────────────────────────────────────────────────────
  if (name === 'recall_hybrid') {
    const projectName = requireString(args, 'projectName', 'projectName')
    const query = requireString(args, 'query', 'query')
    const limit = Math.min(Number(args?.limit ?? 5), 20)

    const sections: string[] = []
    let totalFound = 0

    // 1. Búsqueda en temp_memories por keyword (sin embedding)
    const tempHits = await searchTempByKeyword(query, projectName, limit)
    if (tempHits.length > 0) {
      totalFound += tempHits.length
      sections.push(`📥 **Buffer temporal** (${tempHits.length} resultados por keyword):`)
      tempHits.forEach((m, i) => {
        const date = new Date(m.createdAt).toLocaleString('es-MX', { timeZone: 'America/Mexico_City' })
        sections.push(`  ${i + 1}. [TEMP] ${m.content}\n     ID: ${m.id} | ${date}`)
      })
    }

    // 2. Búsqueda semántica en colección del proyecto (fastembed ONNX)
    try {
      const queryEmbedding = await getEmbedding(query)
      const semanticHits: SearchHit[] = await searchCrossProject(queryEmbedding, projectName, limit * 2)

      if (semanticHits.length > 0) {
        // Excluir IDs que ya aparecen en temp
        const tempIds = new Set(tempHits.map(m => m.id))
        const filtered = semanticHits
          .filter(h => !tempIds.has(h.id))
          .map(h => ({ engrama: { id: h.id, ...h.payload } as any, score: h.score }))
          .sort((a, b) => b.score - a.score)
          .slice(0, limit)

        if (filtered.length > 0) {
          totalFound += filtered.length
          sections.push(`\n🧠 **Memoria indexada** (${filtered.length} resultados semánticos):`)
          filtered.forEach((item, i) => {
            const e = item.engrama
            const tags = e.tags?.join(', ') || ''
            sections.push(`  ${i + 1}. [${e.type ?? 'FACT'}] [imp:${e.importance ?? '?'}/10] ${e.content}\n     Tags: ${tags}`)
          })

          // Actualizar access stats
          const now = Date.now()
          for (const item of filtered) {
            await patchPayload(item.engrama.id, {
              accessCount: (item.engrama.accessCount ?? 0) + 1,
              lastAccessed: now,
            }).catch(() => {})
          }
        }
      }
    } catch (err) {
      sections.push(`\n⚠️ Búsqueda semántica no disponible: ${err instanceof Error ? err.message : String(err)}`)
    }

    if (totalFound === 0) {
      return { content: [{ type: 'text', text: `No se encontraron memorias para "${query}" en ${projectName}.` }] }
    }

    log('RECALL_HYBRID', { projectName, query: query.slice(0, 60), tempHits: tempHits.length, totalFound })
    return {
      content: [{
        type: 'text',
        text: `🔍 Búsqueda híbrida para "${query}" en ${projectName}:\n\n${sections.join('\n')}`,
      }],
    }
  }

  // ── start_session (EPISODIC v3.2) ─────────────────────────────────────────
  if (name === 'start_session') {
    const projectName = requireString(args, 'projectName', 'projectName')
    const context     = requireString(args, 'context', 'context (tema de la sesión)')
    const sessionId   = uuidv4()

    // Auto-cerrar sesión anterior si quedó abierta (sin LLM)
    const openSession = await getOpenSession(projectName)
    let closedMsg = ''
    if (openSession) {
      await closeEpisode(projectName, openSession.id)
      const duration = Math.round((Date.now() - openSession.startedAt) / 60000)
      closedMsg = `\n⏹️ Sesión anterior cerrada (${duration} min, ${openSession.eventCount} eventos)`
      log('SESSION_AUTO_CLOSED', { projectName, sessionId: openSession.id, events: openSession.eventCount })
    }

    await createEpisode(projectName, context, sessionId)
    log('SESSION_START', { projectName, sessionId, context: context.slice(0, 80) })

    return {
      content: [{
        type: 'text',
        text: `▶️ Sesión episódica iniciada.${closedMsg}\n• ID: ${sessionId}\n• Proyecto: ${projectName}\n• Contexto: ${context}\n\nUsa \`log_event\` para registrar decisiones, errores o insights durante la sesión.`,
      }],
    }
  }

  // ── log_event (EPISODIC v3.2) ─────────────────────────────────────────────
  if (name === 'log_event') {
    const projectName = requireString(args, 'projectName', 'projectName')
    const sessionId   = requireString(args, 'sessionId', 'sessionId')
    const description = requireString(args, 'description', 'description')
    const eventType   = (args?.eventType as string) || 'INSIGHT'

    const validTypes = ['DECISION', 'ERROR', 'SOLUTION', 'INSIGHT', 'CONTEXT_CHANGE']
    if (!validTypes.includes(eventType)) {
      return errorResponse(`eventType inválido: "${eventType}". Válidos: ${validTypes.join(', ')}`)
    }

    const event = {
      timestamp: Date.now(),
      eventType: eventType as any,
      description,
    }

    const updated = await appendEvent(projectName, sessionId, event)
    if (!updated) {
      return errorResponse(`Sesión no encontrada: ${sessionId}`)
    }

    log('EVENT_LOGGED', { projectName, sessionId, eventType, description: description.slice(0, 80) })

    const icon = { DECISION: '🔵', ERROR: '🔴', SOLUTION: '🟢', INSIGHT: '💡', CONTEXT_CHANGE: '🔄' }[eventType] ?? '•'
    return {
      content: [{
        type: 'text',
        text: `${icon} Evento registrado en sesión [${sessionId.slice(0, 8)}...]\n• Tipo: ${eventType}\n• Descripción: ${description}\n• Total eventos en sesión: ${updated.eventCount}`,
      }],
    }
  }

  // ── recall_sessions (EPISODIC v3.2) ──────────────────────────────────────
  if (name === 'recall_sessions') {
    const projectName = requireString(args, 'projectName', 'projectName')
    const query       = requireString(args, 'query', 'query')
    const limit       = Math.min(Number(args?.limit ?? 3), 10)

    // Búsqueda semántica sobre summaries de sesiones cerradas (fastembed ONNX, sin LLM)
    const queryVector = await getEmbedding(query)
    const episodes    = await searchEpisodes(projectName, queryVector, limit)

    if (episodes.length === 0) {
      return {
        content: [{
          type: 'text',
          text: `No se encontraron sesiones pasadas para "${query}" en ${projectName}.\n\nUsa \`start_session\` para comenzar a registrar sesiones episódicas.`,
        }],
      }
    }

    const lines = episodes.map((ep, i) => {
      const date     = new Date(ep.startedAt).toLocaleDateString('es-MX')
      const duration = ep.endedAt
        ? `${Math.round((ep.endedAt - ep.startedAt) / 60000)} min`
        : 'abierta'
      const events   = ep.events
        .map(e => `    ${e.eventType === 'DECISION' ? '🔵' : e.eventType === 'ERROR' ? '🔴' : e.eventType === 'SOLUTION' ? '🟢' : '💡'} ${e.description}`)
        .join('\n')

      return [
        `${i + 1}. 📼 **${date}** (${duration}) — ${ep.context}`,
        events || '    *(sin eventos registrados)*',
      ].join('\n')
    })

    log('RECALL_SESSIONS', { projectName, query: query.slice(0, 60), found: episodes.length })
    return {
      content: [{
        type: 'text',
        text: `📼 ${episodes.length} sesiones relevantes para "${query}" en ${projectName}:\n\n${lines.join('\n\n')}`,
      }],
    }
  }

  // ── graph_neighbors ────────────────────────────────────────────────────
  if (name === 'graph_neighbors') {
    const projectName = requireString(args, 'projectName', 'projectName')
    const entity      = requireString(args, 'entity', 'entity')
    const limit       = Math.min(Number(args?.limit ?? 10), 30)

    await ensureSchema()
    const rows = await queryNeighbors(entity, projectName, limit)

    if (rows.length === 0) {
      return { content: [{ type: 'text', text: `❌ No se encontró la entidad "${entity}" en el grafo de "${projectName}". Asegúrate de que existen engramas que la mencionen.` }] }
    }

    const lines = rows.map(r => {
      const arrow = r.direction === 'out' ? `→ ${r.neighbor}` : `← ${r.neighbor}`
      return `• [${r.relType}] ${arrow}${r.label ? ` — ${r.label}` : ''}`
    })

    return {
      content: [{ type: 'text', text: `🕸️ Vecinos de "${entity}" en ${projectName}:\n\n${lines.join('\n')}` }],
    }
  }

  // ── graph_timeline ────────────────────────────────────────────────────
  if (name === 'graph_timeline') {
    const projectName = requireString(args, 'projectName', 'projectName')
    const entity      = requireString(args, 'entity', 'entity')
    const limit       = Math.min(Number(args?.limit ?? 20), 50)

    await ensureSchema()
    const rows = await queryTimeline(entity, projectName, limit)

    if (rows.length === 0) {
      return { content: [{ type: 'text', text: `❌ No hay eventos en el timeline de "${entity}" en "${projectName}".` }] }
    }

    const lines = rows.map(r => {
      const date = new Date(r.createdAt).toLocaleString('es-MX', { timeZone: 'America/Mexico_City' })
      return `• [${date}] [${r.relType}] → ${r.neighbor}${r.label ? ` — ${r.label}` : ''}`
    })

    return {
      content: [{ type: 'text', text: `📅 Timeline de "${entity}" en ${projectName}:\n\n${lines.join('\n')}` }],
    }
  }

  // ── graph_query ───────────────────────────────────────────────────────
  if (name === 'graph_query') {
    const cypher = requireString(args, 'cypher', 'cypher (consulta Cypher)')

    await ensureSchema()
    const results = await rawQuery(cypher)

    if (results.length === 0) {
      return { content: [{ type: 'text', text: '❌ La consulta no retornó resultados.' }] }
    }

    const json = JSON.stringify(results, null, 2)
    return {
      content: [{ type: 'text', text: `🔍 ${results.length} resultados:\n\`\`\`json\n${json.slice(0, 3000)}${json.length > 3000 ? '\n... (truncado)' : ''}\n\`\`\`` }],
    }
  }

    throw new Error(`Tool no encontrado: ${name}`)

  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err)
    log('TOOL_ERROR', { name, error: msg })

    // Si es un error de parámetros (validación), devuelve ayuda útil
    if (msg.startsWith('Parámetro requerido:')) {
      return errorResponse(msg)
    }

    // Errores de conectividad
    if (msg.includes('ECONNREFUSED') || msg.includes('connect')) {
      return errorResponse(
        'Servicio no disponible',
        msg.includes('6333')
          ? 'Qdrant no responde en localhost:6333. ¿Está corriendo?'
          : msg.includes('11434')
          ? 'Ollama no responde en localhost:11434. ¿Está corriendo?'
          : msg,
      )
    }

    // Error genérico — devuelve mensaje limpio
    return errorResponse(`Error en tool "${name}"`, msg)
  }
})

// ─── Start ────────────────────────────────────────────────────────────────────

const transport = new StdioServerTransport()
await server.connect(transport)
