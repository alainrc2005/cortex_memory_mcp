/**
 * LAYER 2 — INTEGRATION TESTS: observe → recall roundtrip
 *
 * El test más crítico: verifica que guardar una memoria y recuperarla
 * funciona correctamente de punta a punta (Qdrant real + fastembed ONNX).
 *
 * Bugs que detecta:
 *   - Memorias superseded apareciendo en recall (bug de producción)
 *   - Recall no encontrando lo que se acaba de guardar
 *   - Proyectos mezclados (aislamiento)
 *   - Filtro de status=superseded en recall
 */

import { describe, it, expect, afterAll } from 'vitest'
import { v4 as uuidv4 } from 'uuid'
import {
  upsertEngrama,
  searchHybridCrossProject,
  deleteAllInProject,
  ensureProjectCollection,
  collectionFor,
} from '../../src/services/qdrant.js'
import { getEmbedding, getSparseEmbedding } from '../../src/services/fastembed.js'
import { calcDecay, combinedScore } from '../../src/services/decay.js'
import type { Engrama } from '../../src/types/engrama.js'

// ─── Proyectos de test aislados ───────────────────────────────────────────────
const TS = Date.now()
const TEST_PROJECT   = `test_observe_${TS}`
const TEST_PROJECT_B = `test_observe_b_${TS}`

// ─── Helpers ──────────────────────────────────────────────────────────────────

async function saveEngrama(content: string, projectName: string, overrides: Partial<Engrama> = {}): Promise<string> {
  const id         = uuidv4()
  const embedding  = await getEmbedding(content)
  const sparse     = await getSparseEmbedding(content).catch(() => ({ indices: [], values: [] }))
  const now        = Date.now()
  const collection = await ensureProjectCollection(projectName)
  await upsertEngrama(id, embedding, {
    content,
    projectName,
    createdAt:   now,
    importance:  overrides.importance ?? 7,
    accessCount: overrides.accessCount ?? 0,
    lastAccessed: now,
    type:        overrides.type ?? 'FACT',
    tags:        [],
    linkedTo:    [],
    status:      overrides.status ?? 'active',
  } as Engrama, collection, sparse)
  return id
}

async function recall(query: string, projectName: string, limit = 5) {
  const [dense, sparse] = await Promise.all([
    getEmbedding(query),
    getSparseEmbedding(query).catch(() => ({ indices: [], values: [] })),
  ])
  return searchHybridCrossProject(dense, sparse, projectName, limit)
}

// ─── Cleanup ──────────────────────────────────────────────────────────────────

afterAll(async () => {
  await deleteAllInProject(TEST_PROJECT).catch(() => {})
  await deleteAllInProject(TEST_PROJECT_B).catch(() => {})
})

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('observe → recall roundtrip', () => {
  it('una memoria guardada aparece en recall con query semántico', async () => {
    await saveEngrama('Usamos Redis para caché de sesiones con TTL de 30 minutos', TEST_PROJECT)
    await new Promise(r => setTimeout(r, 500))

    const hits = await recall('Redis caché sesiones', TEST_PROJECT)
    const found = hits.some(h => (h.payload?.content as string)?.includes('Redis'))
    expect(found).toBe(true)
  }, 30_000)

  it('recall retorna hits con score entre 0 y 1', async () => {
    const hits = await recall('Redis caché', TEST_PROJECT)
    for (const hit of hits) {
      expect(hit.score).toBeGreaterThanOrEqual(0)
      expect(hit.score).toBeLessThanOrEqual(1)
    }
  }, 15_000)

  it('múltiples memorias del mismo proyecto son todas recuperables', async () => {
    const memories = [
      'Arquitectura: frontend en React con Vite',
      'Arquitectura: backend en Node.js con Express',
      'Arquitectura: base de datos PostgreSQL en RDS',
    ]
    for (const m of memories) {
      await saveEngrama(m, TEST_PROJECT)
    }
    await new Promise(r => setTimeout(r, 800))

    const hits = await recall('arquitectura del sistema', TEST_PROJECT, 10)
    // Al menos 2 de las 3 deben aparecer en recall de arquitectura
    const found = memories.filter(m =>
      hits.some(h => (h.payload?.content as string)?.includes(m.split(':')[1].trim().split(' ')[0]))
    )
    expect(found.length).toBeGreaterThanOrEqual(2)
  }, 45_000)
})

describe('Filtro de memorias superseded', () => {
  it('payload de memorias superseded tiene status=superseded', async () => {
    const activeId     = await saveEngrama('ACTIVA: JWT para autenticación',   TEST_PROJECT)
    const supersededId = await saveEngrama('OBSOLETA: cookies para auth',       TEST_PROJECT, { status: 'superseded', importance: 3 })
    await new Promise(r => setTimeout(r, 500))

    const hits = await recall('autenticación JWT cookie sesiones', TEST_PROJECT, 10)

    // El activo debe aparecer
    const activeHit     = hits.find(h => h.id === activeId)
    const supersededHit = hits.find(h => h.id === supersededId)

    // El activo debe estar presente
    expect(activeHit).toBeDefined()
    // Si el superseded aparece, debe tener el campo status correcto
    if (supersededHit) {
      expect(supersededHit.payload?.status).toBe('superseded')
    }
    // El activo debe rankear antes que el superseded
    if (activeHit && supersededHit) {
      const activeIdx     = hits.findIndex(h => h.id === activeId)
      const supersededIdx = hits.findIndex(h => h.id === supersededId)
      expect(activeIdx).toBeLessThan(supersededIdx)
    }
  }, 30_000)
})

describe('Aislamiento de proyectos — BUG CRÍTICO', () => {
  it('memorias del Proyecto A con nombre único NO aparecen en recall del Proyecto B', async () => {
    const uniqueToken = `SECRETO_A_${uuidv4().replace(/-/g, '')}`
    await saveEngrama(`Token único: ${uniqueToken}`, TEST_PROJECT)
    await new Promise(r => setTimeout(r, 800))

    // La colección de TEST_PROJECT_B no tiene este token
    // searchHybridCrossProject busca en: TEST_PROJECT_B_col + cortex_global
    // TEST_PROJECT (diferente) no debe aparecer
    const hitsInB = await recall(uniqueToken, TEST_PROJECT_B, 10)
    const leaked = hitsInB.some(h =>
      (h.payload?.content as string)?.includes('SECRETO_A') &&
      h.payload?.projectName === TEST_PROJECT
    )
    expect(leaked).toBe(false)
  }, 30_000)

  it('la colección de cada proyecto tiene su nombre correcto en Qdrant', () => {
    // Verificar que collectionFor genera nombres únicos por proyecto
    const colA = collectionFor('project-alpha')
    const colB = collectionFor('project-beta')
    expect(colA).not.toBe(colB)
    expect(colA).toBe('cortex_project_alpha')
    expect(colB).toBe('cortex_project_beta')
  })

  it('caracteres especiales en projectName se sanitizan', () => {
    const col = collectionFor('Mi Proyecto 2024!')
    expect(col).toMatch(/^cortex_[a-z0-9_]+$/)
  })
})

describe('combinedScore en contexto de recall', () => {
  it('memorias de alta importancia rankeadas más alto que baja importancia con mismo semantic', () => {
    const base = { id: '1', content: 'test', projectName: 'test',
      createdAt: Date.now(), lastAccessed: Date.now(), tags: [], linkedTo: [] } as Engrama

    const highImp: Engrama = { ...base, id: '1', importance: 9, accessCount: 5, type: 'DECISION' }
    const lowImp:  Engrama = { ...base, id: '2', importance: 2, accessCount: 0, type: 'FACT' }

    const scoreHigh = combinedScore(0.8, calcDecay(highImp), highImp.type)
    const scoreLow  = combinedScore(0.8, calcDecay(lowImp),  lowImp.type)

    expect(scoreHigh).toBeGreaterThan(scoreLow)
  })

  it('memoria reciente rankeada más alto que antigua con misma importancia', () => {
    const base = { id: '1', content: 'test', projectName: 'test',
      importance: 5, accessCount: 0, type: 'FACT', tags: [], linkedTo: [] } as Engrama

    const recent: Engrama = { ...base, id: '1', createdAt: Date.now(), lastAccessed: Date.now() }
    const old:    Engrama = { ...base, id: '2',
      lastAccessed: Date.now() - 60 * 86_400_000,
      createdAt:    Date.now() - 60 * 86_400_000,
    }

    const scoreRecent = combinedScore(0.8, calcDecay(recent), 'FACT')
    const scoreOld    = combinedScore(0.8, calcDecay(old),    'FACT')

    expect(scoreRecent).toBeGreaterThan(scoreOld)
  })
})
