/**
 * LAYER 2 — INTEGRATION TESTS: Buffer pipeline (quick_observe → index_temp)
 *
 * Verifica el flujo de dos fases: guardado instantáneo en buffer,
 * luego indexado con embeddings cuando hay CPU disponible.
 */

import { describe, it, expect, afterAll } from 'vitest'
import { v4 as uuidv4 } from 'uuid'
import {
  upsertTemp,
  scrollTemp,
  deleteTempIds,
  searchTempByKeyword,
} from '../../src/services/qdrant.js'

const TS = Date.now()
const PROJECT_A = `test_buffer_a_${TS}`
const PROJECT_B = `test_buffer_b_${TS}`

// Track IDs to cleanup
const insertedIds: string[] = []

afterAll(async () => {
  if (insertedIds.length > 0) {
    await deleteTempIds(insertedIds).catch(() => {})
  }
})

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('Buffer temporal (temp_memories)', () => {
  it('upsertTemp guarda sin LLM ni embedding', async () => {
    const id = uuidv4()
    insertedIds.push(id)
    await upsertTemp(id, 'Test: decisión de usar TypeScript strict mode', PROJECT_A)

    const pending = await scrollTemp(PROJECT_A, 'pending', 50)
    const found = pending.some(m => m.id === id)
    expect(found).toBe(true)
  }, 15_000)

  it('scrollTemp filtra por proyecto correctamente', async () => {
    const idA = uuidv4()
    const idB = uuidv4()
    insertedIds.push(idA, idB)

    await Promise.all([
      upsertTemp(idA, 'Memoria de proyecto A', PROJECT_A),
      upsertTemp(idB, 'Memoria de proyecto B', PROJECT_B),
    ])

    const pendingA = await scrollTemp(PROJECT_A, 'pending', 50)
    const pendingB = await scrollTemp(PROJECT_B, 'pending', 50)

    expect(pendingA.some(m => m.id === idA)).toBe(true)
    expect(pendingB.some(m => m.id === idB)).toBe(true)

    // Aislamiento: A no contiene memorias de B
    expect(pendingA.some(m => m.id === idB)).toBe(false)
    expect(pendingB.some(m => m.id === idA)).toBe(false)
  }, 15_000)

  it('deleteTempIds elimina correctamente del buffer', async () => {
    const id = uuidv4()
    await upsertTemp(id, 'Para eliminar', PROJECT_A)

    let pending = await scrollTemp(PROJECT_A, 'pending', 50)
    expect(pending.some(m => m.id === id)).toBe(true)

    await deleteTempIds([id])
    await new Promise(r => setTimeout(r, 300))

    pending = await scrollTemp(PROJECT_A, 'pending', 50)
    expect(pending.some(m => m.id === id)).toBe(false)
  }, 15_000)

  it('searchTempByKeyword encuentra por texto exacto', async () => {
    const uniqueWord = `UNIQUETOKEN${TS}`
    const id = uuidv4()
    insertedIds.push(id)
    await upsertTemp(id, `Decisión crítica: ${uniqueWord} para identificar esta memoria`, PROJECT_A)

    // searchTempByKeyword(query, projectName, limit)
    const results = await searchTempByKeyword(uniqueWord, PROJECT_A, 5)
    const found = results.some(r => r.content?.includes(uniqueWord))
    expect(found).toBe(true)
  }, 15_000)

  it('batch de memorias en el buffer — todas se guardan', async () => {
    const ids: string[] = []
    const memories = [
      'Batch memoria 1: configuración de ESLint',
      'Batch memoria 2: prettier con tab-width 2',
      'Batch memoria 3: husky para pre-commit hooks',
    ]

    for (const content of memories) {
      const id = uuidv4()
      ids.push(id)
      insertedIds.push(id)
      await upsertTemp(id, content, PROJECT_A)
    }

    const pending = await scrollTemp(PROJECT_A, 'pending', 100)
    const foundCount = ids.filter(id => pending.some(m => m.id === id)).length
    expect(foundCount).toBe(3)
  }, 20_000)

  it('scrollTemp sin projectName retorna memorias de múltiples proyectos', async () => {
    const ids: string[] = []
    for (const [proj, content] of [[PROJECT_A, 'Memo A global'], [PROJECT_B, 'Memo B global']] as const) {
      const id = uuidv4()
      ids.push(id)
      insertedIds.push(id)
      await upsertTemp(id, content, proj)
    }

    const allPending = await scrollTemp(undefined, 'pending', 200)
    const foundAll = ids.every(id => allPending.some(m => m.id === id))
    expect(foundAll).toBe(true)
  }, 20_000)

  it('TempMemory retorna los campos esperados (contenido, proyecto, status)', async () => {
    const id = uuidv4()
    insertedIds.push(id)
    await upsertTemp(id, 'Contenido de prueba de campos', PROJECT_A)

    const pending = await scrollTemp(PROJECT_A, 'pending', 50)
    const mem = pending.find(m => m.id === id)

    expect(mem).toBeDefined()
    expect(mem!.content).toBe('Contenido de prueba de campos')
    expect(mem!.projectName).toBe(PROJECT_A)
    expect(mem!.status).toBe('pending')
    expect(mem!.createdAt).toBeGreaterThan(0)
  }, 15_000)
})
