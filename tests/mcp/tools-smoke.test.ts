/**
 * LAYER 3 — MCP SMOKE TESTS: Todos los tools del protocolo
 *
 * Verifica que:
 *   1. Los 22 tools están registrados (detecta tools nuevos no agregados)
 *   2. Los parámetros requeridos son validados
 *   3. Las respuestas tienen el formato MCP correcto
 *   4. Los safety gates funcionan (delete_all_memories, etc.)
 *   5. El flag CORTEX_RERANKER_ENABLED=false es respetado — BUG DE PRODUCCIÓN
 *
 * A diferencia del test_cortex.mjs original, este reutiliza una sola instancia
 * del servidor (mucho más rápido) y cubre los 22 tools actuales.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { spawn, type ChildProcess } from 'child_process'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dir  = dirname(fileURLToPath(import.meta.url))
const SERVER = resolve(__dir, '../../dist/server.js')

// ─── MCP Client mínimo (reutiliza proceso) ────────────────────────────────────

let serverProc: ChildProcess
let buffer = ''
let reqId  = 0
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>()

function startServer() {
  serverProc = spawn('node', [SERVER], {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: {
      ...process.env,
      CORTEX_LLM_BACKEND:      'none',
      CORTEX_RERANKER_ENABLED: 'false',
      QDRANT_URL:              'http://localhost:6333',
    },
  })

  serverProc.stdout!.on('data', (chunk: Buffer) => {
    buffer += chunk.toString()
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.trim()) continue
      try {
        const msg = JSON.parse(line) as { id?: number; result?: unknown; error?: unknown }
        if (msg.id !== undefined && pending.has(msg.id)) {
          const { resolve } = pending.get(msg.id)!
          pending.delete(msg.id)
          resolve(msg)
        }
      } catch {}
    }
  })

  serverProc.stderr!.on('data', () => {}) // silenciar logs del servidor
}

function send(method: string, params: unknown): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const id = ++reqId
    pending.set(id, { resolve, reject })
    const req = JSON.stringify({ jsonrpc: '2.0', id, method, params })
    serverProc.stdin!.write(req + '\n')
    setTimeout(() => {
      if (pending.has(id)) {
        pending.delete(id)
        reject(new Error(`timeout: ${method}`))
      }
    }, 45_000)
  })
}

function callTool(name: string, args: Record<string, unknown> = {}) {
  return send('tools/call', { name, arguments: args }) as Promise<{
    result?: { content?: Array<{ text?: string }> }
    error?:  { message: string }
  }>
}

function getText(res: Awaited<ReturnType<typeof callTool>>): string {
  return res?.result?.content?.[0]?.text ?? ''
}

beforeAll(async () => {
  startServer()
  // Esperar a que el servidor esté listo
  await new Promise(r => setTimeout(r, 2000))
})

afterAll(() => {
  serverProc?.kill()
})

// ─── 1. Registro de tools ─────────────────────────────────────────────────────

describe('tools/list — registro completo', () => {
  it('devuelve exactamente los 22 tools esperados', async () => {
    const res = await send('tools/list', {}) as { result?: { tools?: Array<{ name: string }> } }
    const tools = res.result?.tools?.map(t => t.name) ?? []

    const EXPECTED_TOOLS = [
      // Core
      'observe', 'recall', 'consolidate', 'get_context_for',
      'detect_patterns', 'get_operator_profile', 'cortex_status',
      // CRUD
      'delete_memory', 'update_memory', 'get_all_memories', 'delete_all_memories',
      'batch_observe', 'export_memories',
      // Buffer
      'quick_observe', 'list_pending', 'index_temp', 'recall_hybrid',
      // Episodic
      'start_session', 'log_event', 'recall_sessions',
      // Knowledge Graph
      'graph_neighbors', 'graph_timeline', 'graph_query',
    ]

    expect(tools.length).toBe(EXPECTED_TOOLS.length)
    for (const t of EXPECTED_TOOLS) {
      expect(tools, `Tool "${t}" debe estar registrado`).toContain(t)
    }
  }, 15_000)
})

// ─── 2. Validación de parámetros requeridos ───────────────────────────────────

describe('Validación de parámetros requeridos', () => {
  const validationCases: Array<[string, Record<string, unknown>, string]> = [
    ['observe',          {},                        'projectName'],
    ['observe',          { projectName: 'test' },   'content'],
    ['recall',           { projectName: 'test' },   'query'],
    ['recall',           {},                        'projectName'],
    ['delete_memory',    {},                        'id'],
    ['update_memory',    { id: 'x' },               'content'],
    ['get_all_memories', {},                        'projectName'],
    ['quick_observe',    {},                        'projectName'],
    ['quick_observe',    { projectName: 'test' },   'content'],
    ['recall_hybrid',    {},                        'projectName'],
    ['graph_neighbors',  { projectName: 'test' },   'entity'],
    ['graph_query',      {},                        'cypher'],
    ['start_session',    { projectName: 'test' },   'context'],
    ['log_event',        { projectName: 'test', sessionId: 'x', eventType: 'DECISION' }, 'description'],
  ]

  for (const [tool, args, missing] of validationCases) {
    it(`${tool}: sin "${missing}" → error de validación`, async () => {
      const res = await callTool(tool, args)
      const text = getText(res)
      expect(
        text.toLowerCase().includes('requerido') ||
        text.toLowerCase().includes('required') ||
        text.toLowerCase().includes('error') ||
        text.includes('❌'),
        `Esperaba error de validación, recibí: "${text.slice(0, 100)}"`
      ).toBe(true)
    }, 15_000)
  }
})

// ─── 3. Safety gates ──────────────────────────────────────────────────────────

describe('Safety gates', () => {
  it('delete_all_memories sin confirm → rechazado', async () => {
    const res = await callTool('delete_all_memories', { projectName: 'test' })
    const text = getText(res)
    expect(text.toLowerCase()).toMatch(/cancel|rechaz|confirm|operaci/i)
  }, 15_000)

  it('delete_all_memories con confirm=false → rechazado', async () => {
    const res = await callTool('delete_all_memories', { projectName: 'test', confirm: false })
    const text = getText(res)
    expect(text.toLowerCase()).toMatch(/cancel|rechaz|confirm|operaci/i)
  }, 15_000)

  it('delete_all_memories con confirm=true en proyecto vacío → OK o error limpio', async () => {
    const tmpProject = `test_safety_del_${Date.now()}`
    const res = await callTool('delete_all_memories', { projectName: tmpProject, confirm: true })
    const text = getText(res)
    // Debe responder algo (no crash), sea éxito o "no existe"
    expect(text.length).toBeGreaterThan(0)
  }, 15_000)
})

// ─── 4. BUG DE PRODUCCIÓN — Flag del reranker ────────────────────────────────

describe('CORTEX_RERANKER_ENABLED=false — BUG DE PRODUCCIÓN', () => {
  it('cortex_status responde sin llamar al reranker (no timeout)', async () => {
    // Con CORTEX_RERANKER_ENABLED=false el servidor no debe llamar al LLM
    // Si lo llamara, con LLM_BACKEND=none daría error o timeout
    const start = Date.now()
    const res = await callTool('cortex_status')
    const elapsed = Date.now() - start
    const text = getText(res)

    expect(text).toContain('CORTEX')
    // Con reranker desactivado, cortex_status debe ser rápido (<5s)
    expect(elapsed).toBeLessThan(5000)
  }, 15_000)

  it('recall con query devuelve respuesta sin reranker (no llama LLM)', async () => {
    // Primero guardamos algo para tener algo que recuperar
    await callTool('quick_observe', {
      projectName: 'test_reranker_flag',
      content: 'Test del flag de reranker: Redis para caché',
    })

    const start = Date.now()
    const res = await callTool('recall', {
      projectName: 'test_reranker_flag',
      query: 'Redis caché',
      limit: 3,
    })
    const elapsed = Date.now() - start
    const text = getText(res)

    // Con reranker=false y LLM=none, debe completar rápido
    expect(elapsed).toBeLessThan(10_000)
    // El texto debe indicar que no usó reranker
    expect(text).not.toContain('reranked')
  }, 20_000)
})

// ─── 5. Responses well-formed ─────────────────────────────────────────────────

describe('Formato de respuesta MCP correcto', () => {
  it('cortex_status tiene todos los campos esperados', async () => {
    const res = await callTool('cortex_status')
    const text = getText(res)
    expect(text).toContain('CORTEX')
    expect(text).toContain('engramas')
    expect(text).toContain('Operator Profile')
    expect(text).toContain('Mantenimiento')
  }, 15_000)

  it('get_operator_profile retorna header correcto', async () => {
    const res = await callTool('get_operator_profile')
    const text = getText(res)
    expect(text).toContain('Operator Profile')
  }, 15_000)

  it('get_context_for sin project retorna fallback válido (no crash)', async () => {
    const res = await callTool('get_context_for', {})
    const text = getText(res)
    expect(text).toBeTruthy()
    expect(text.length).toBeGreaterThan(0)
  }, 15_000)

  it('quick_observe guarda y retorna confirmación', async () => {
    const res = await callTool('quick_observe', {
      projectName: 'test_smoke',
      content: 'Test smoke: sistema de caché con Redis implementado',
    })
    const text = getText(res)
    expect(text).toContain('buffer')
    expect(text).toMatch(/ID:|id:|guardado|saved/i)
  }, 15_000)

  it('list_pending responde con estructura válida', async () => {
    const res = await callTool('list_pending', {})
    const text = getText(res)
    expect(text.length).toBeGreaterThan(0)
  }, 15_000)

  it('tool desconocido → error limpio (no crash del servidor)', async () => {
    const res = await callTool('tool_que_no_existe', {})
    const text = getText(res)
    expect(text.toLowerCase()).toMatch(/no encontrado|not found|error|desconocido|unknown/i)
  }, 15_000)
})

// ─── 6. episodic memory ───────────────────────────────────────────────────────

describe('Memoria episódica — ciclo completo', () => {
  it('start_session + log_event fluye sin errores', async () => {
    const sessionRes = await callTool('start_session', {
      projectName: 'test_smoke',
      context: 'Testing sesión episódica - implementar autenticación',
    })
    const sessionText = getText(sessionRes)
    expect(sessionText).toBeTruthy()

    // Extraer sessionId de la respuesta
    const idMatch = sessionText.match(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i)
    if (idMatch) {
      const sessionId = idMatch[0]
      const eventRes = await callTool('log_event', {
        projectName: 'test_smoke',
        sessionId,
        eventType:   'DECISION',
        description: 'Decidimos usar JWT con refresh tokens en Redis',
      })
      const eventText = getText(eventRes)
      expect(eventText).toBeTruthy()
    }
  }, 20_000)
})
