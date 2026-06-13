import { QdrantClient } from '@qdrant/js-client-rest'
import * as dotenv from 'dotenv'
import type { Engrama, EngramaPayload } from '../types/engrama.js'
import type { SparseVector } from './fastembed.js'

dotenv.config()

export const VECTOR_SIZE = 384  // all-minilm output size

export const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL || 'http://localhost:6333',
  apiKey: process.env.QDRANT_API_KEY,
})

// ─── Nombres de colecciones ──────────────────────────────────────────────────

/** Colección por proyecto: cortex_my-api, cortex_backend, cortex_global, etc. */
export function collectionFor(projectName: string): string {
  const safe = (projectName || 'global').toLowerCase().replace(/[^a-z0-9]/g, '_')
  return `cortex_${safe}`
}

/** Colección global para Operator Profile y patrones cross-proyecto */
export const GLOBAL_COLLECTION = 'cortex_global'

// Legado: colección única donde vivían las memorias antes de Fase 3
export const LEGACY_COLLECTION = 'work_memories'

/** Buffer temporal sin LLM — recibe memorias sin embedding real hasta indexación manual */
export const TEMP_COLLECTION = 'temp_memories'

// ─── Ensure collection exists ─────────────────────────────────────────────────

export async function ensureCollection(name?: string): Promise<void> {
  const col = name || LEGACY_COLLECTION
  try {
    await qdrant.getCollection(col)
  } catch {
    await qdrant.createCollection(col, {
      vectors: { size: VECTOR_SIZE, distance: 'Cosine' },
    })
  }
}

export async function ensureProjectCollection(projectName: string): Promise<string> {
  const col = collectionFor(projectName)
  try {
    await qdrant.getCollection(col)
  } catch {
    // Crear con dense + sparse (BM25/IDF) desde el inicio
    // Qdrant 1.17+: sparse_vectors debe declararse en la creación, no se puede agregar después
    await qdrant.createCollection(col, {
      vectors: { size: VECTOR_SIZE, distance: 'Cosine' },
      sparse_vectors: {
        [SPARSE_VECTOR_NAME]: {
          index: { on_disk: false },
          modifier: 'idf',
        },
      },
    } as Parameters<typeof qdrant.createCollection>[1])
    process.stderr.write(`[qdrant] Colección creada con sparse index: ${col}\n`)
  }
  return col
}

// ─── Sparse index ────────────────────────────────────────────────────────────────────

/** Nombre del vector sparse que usamos en todas las colecciones */
export const SPARSE_VECTOR_NAME = 'bm25'

/**
 * Añade el índice sparse 'bm25' a una colección si todavía no lo tiene.
 * Operación no destructiva: los puntos existentes no se tocan,
 * simplemente no tendrán vector sparse hasta que se re-upserten.
 *
 * Usa IDF weighting para mejorar la precisión de búsqueda.
 */
export async function ensureSparseIndex(collection: string): Promise<void> {
  try {
    const info = await qdrant.getCollection(collection)
    const sparseVectors = (info.config?.params as Record<string, unknown>)?.sparse_vectors as Record<string, unknown> | undefined
    if (sparseVectors?.[SPARSE_VECTOR_NAME]) return  // ya existe
  } catch {
    return  // colección no existe, nada que hacer
  }

  try {
    await qdrant.updateCollection(collection, {
      sparse_vectors: {
        [SPARSE_VECTOR_NAME]: {
          index: {
            on_disk: false,
          },
          modifier: 'idf',  // IDF weighting: penaliza tokens muy frecuentes
        },
      },
    } as Parameters<typeof qdrant.updateCollection>[1])
  } catch (e) {
    // No bloquear si falla — la búsqueda degradará a solo-dense
    process.stderr.write(`[qdrant] ensureSparseIndex warn: ${e}\n`)
  }
}

// ─── Upsert ────────────────────────────────────────────────────────────────────

export async function upsertEngrama(
  id: string,
  vector: number[],
  payload: EngramaPayload,
  collection?: string,
  sparse?: SparseVector,
): Promise<void> {
  const col = collection || LEGACY_COLLECTION

  // Si tenemos sparse vector, lo incluimos como named vector
  const vectors: Record<string, unknown> = sparse
    ? {
        '': vector,                    // vector dense sin nombre (compatibilidad con legacy)
        [SPARSE_VECTOR_NAME]: sparse,  // vector sparse nombrado
      }
    : vector as unknown as Record<string, unknown>

  await qdrant.upsert(col, {
    wait: true,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    points: [{ id, vector: vectors as any, payload: payload as Record<string, unknown> }],
  })
}

// ─── Search ───────────────────────────────────────────────────────────────────

export interface SearchHit {
  id: string
  score: number
  payload: EngramaPayload
  collection?: string
}

/** Búsqueda en una colección con filtro opcional por projectName (para legado) */
export async function searchSimilar(
  vector: number[],
  projectName: string,
  limit = 5,
  collection?: string,
): Promise<SearchHit[]> {
  const col = collection || LEGACY_COLLECTION

  const results = await qdrant.search(col, {
    vector,
    limit,
    // En colección legada: filtra por projectName O sin etiqueta
    // En colecciones nuevas (cortex_*): sin filtro, toda la colección es del proyecto
    filter: col === LEGACY_COLLECTION ? {
      should: [
        { key: 'projectName', match: { value: projectName } },
        { is_empty: { key: 'projectName' } },
      ],
    } : undefined,
    with_payload: true,
  })

  return results.map((r) => ({
    id: String(r.id),
    score: r.score,
    payload: r.payload as unknown as EngramaPayload,
    collection: col,
  }))
}

/** Cross-project: busca en colección del proyecto + global, combina resultados */
export async function searchCrossProject(
  vector: number[],
  projectName: string,
  limit = 6,
): Promise<SearchHit[]> {
  const projectCol = collectionFor(projectName)
  const results: SearchHit[] = []

  // Buscar en colección del proyecto (si existe)
  try {
    await qdrant.getCollection(projectCol)
    const hits = await searchSimilar(vector, projectName, Math.ceil(limit * 0.7), projectCol)
    results.push(...hits)
  } catch { /* colección no existe aún */ }

  // Buscar en colección legada (work_memories) con filtro por proyecto
  try {
    const legacyHits = await searchSimilar(vector, projectName, limit, LEGACY_COLLECTION)
    results.push(...legacyHits)
  } catch { /* ignorar */ }

  // Buscar en global (patrones del operador)
  try {
    await qdrant.getCollection(GLOBAL_COLLECTION)
    const globalHits = await searchSimilar(vector, 'global', Math.ceil(limit * 0.3), GLOBAL_COLLECTION)
    results.push(...globalHits)
  } catch { /* colección no existe aún */ }

  // Dedup por id y ordenar por score
  const seen = new Set<string>()
  return results
    .filter(h => { if (seen.has(h.id)) return false; seen.add(h.id); return true })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
}

export async function searchGlobal(
  vector: number[],
  limit = 5,
): Promise<SearchHit[]> {
  const results = await qdrant.search(LEGACY_COLLECTION, {
    vector,
    limit,
    with_payload: true,
  })
  return results.map((r) => ({
    id: String(r.id),
    score: r.score,
    payload: r.payload as unknown as EngramaPayload,
  }))
}

// ─── Hybrid Search (dense + BM25 sparse via RRF) ──────────────────────────────
//
// Usa la Query API de Qdrant 1.7+ con prefetch + fusion:rrf.
// El pipeline:
//   prefetch[0]: dense vector  → top-N por similitud coseno
//   prefetch[1]: sparse (bm25) → top-N por score BM25 (solo si la col tiene sparse)
//   query: { fusion: 'rrf' }  → Reciprocal Rank Fusion de los dos pools
//
// Fallback: si la colección no tiene sparse index, cae a solo-dense automáticamente.

/**
 * Búsqueda híbrida (dense + sparse BM25) en una sola colección.
 * Si la colección no tiene sparse index → fallback a solo-dense.
 */
export async function searchHybrid(
  denseVec: number[],
  sparseVec: SparseVector,
  projectName: string,
  limit = 5,
  collection?: string,
): Promise<SearchHit[]> {
  const col = collection || LEGACY_COLLECTION

  // Filtro para colección legacy (multi-proyecto)
  const filter = col === LEGACY_COLLECTION ? {
    should: [
      { key: 'projectName', match: { value: projectName } },
      { is_empty: { key: 'projectName' } },
    ],
  } : undefined

  // Verificar si la colección tiene sparse index
  let hasSparse = false
  try {
    const info = await qdrant.getCollection(col)
    const sp = (info.config?.params as Record<string, unknown>)?.sparse_vectors as Record<string, unknown> | undefined
    hasSparse = !!(sp?.[SPARSE_VECTOR_NAME])
  } catch { /* si falla la verificación, asumimos sin sparse */ }

  if (!hasSparse) {
    // Fallback: solo dense
    return searchSimilar(denseVec, projectName, limit, col)
  }

  // Pool 2× para que RRF tenga más candidatos para re-ordenar
  const pool = limit * 2

  const results = await (qdrant as unknown as {
    query: (col: string, params: Record<string, unknown>) => Promise<{ points: Array<{ id: string | number; score: number; payload?: Record<string, unknown> }> }>
  }).query(col, {
    prefetch: [
      {
        query: denseVec,
        using: '',           // '' = vector dense sin nombre (legacy unnamed)
        limit: pool,
        filter,
        with_payload: false,
      },
      {
        query: { indices: sparseVec.indices, values: sparseVec.values },
        using: SPARSE_VECTOR_NAME,
        limit: pool,
        filter,
        with_payload: false,
      },
    ],
    query: { fusion: 'rrf' },
    limit,
    with_payload: true,
  })

  return results.points.map((r) => ({
    id: String(r.id),
    score: r.score,
    payload: (r.payload ?? {}) as unknown as EngramaPayload,
    collection: col,
  }))
}

/**
 * Búsqueda híbrida cross-project: colección del proyecto + legado + global.
 * Misma lógica que searchCrossProject pero usando searchHybrid en cada colección.
 */
export async function searchHybridCrossProject(
  denseVec: number[],
  sparseVec: SparseVector,
  projectName: string,
  limit = 6,
): Promise<SearchHit[]> {
  const projectCol = collectionFor(projectName)
  const results: SearchHit[] = []

  // Colección dedicada del proyecto
  try {
    await qdrant.getCollection(projectCol)
    const hits = await searchHybrid(denseVec, sparseVec, projectName, Math.ceil(limit * 0.7), projectCol)
    results.push(...hits)
  } catch { /* colección no existe aún */ }

  // Colección legada con filtro por proyecto
  try {
    const legacyHits = await searchHybrid(denseVec, sparseVec, projectName, limit, LEGACY_COLLECTION)
    results.push(...legacyHits)
  } catch { /* ignorar */ }

  // Colección global (patrones del operador)
  try {
    await qdrant.getCollection(GLOBAL_COLLECTION)
    const globalHits = await searchHybrid(denseVec, sparseVec, 'global', Math.ceil(limit * 0.3), GLOBAL_COLLECTION)
    results.push(...globalHits)
  } catch { /* ignorar */ }

  // Dedup y ordenar por score (RRF ya ordena, pero hay que fundir los 3 resultsets)
  const seen = new Set<string>()
  return results
    .filter(h => { if (seen.has(h.id)) return false; seen.add(h.id); return true })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
}


// ─── Patch payload ────────────────────────────────────────────────────────────

export async function patchPayload(
  id: string,
  patch: Partial<EngramaPayload>,
  collection?: string,
): Promise<void> {
  const col = collection || LEGACY_COLLECTION
  await qdrant.setPayload(col, {
    payload: patch as Record<string, unknown>,
    points: [id],
    wait: true,
  })
}

// ─── Scroll ───────────────────────────────────────────────────────────────────

export async function scrollAll(projectName: string): Promise<Engrama[]> {
  // Primero intentar colección dedicada
  const projectCol = collectionFor(projectName)
  let points: Array<{ id: string | number; payload?: Record<string, unknown> | null }> = []

  try {
    await qdrant.getCollection(projectCol)
    const result = await qdrant.scroll(projectCol, {
      limit: 200,
      with_payload: true,
      with_vector: false,
    })
    points = result.points
  } catch {
    // Fallback a colección legada con filtro
    const result = await qdrant.scroll(LEGACY_COLLECTION, {
      filter: { must: [{ key: 'projectName', match: { value: projectName } }] },
      limit: 200,
      with_payload: true,
      with_vector: false,
    })
    points = result.points
  }

  return points.map((p) => ({
    id: String(p.id),
    ...(p.payload as unknown as EngramaPayload),
  }))
}

// ─── Operator Profile ─────────────────────────────────────────────────────────

export interface OperatorProfile {
  codingPreferences: string[]
  activeProjects: string[]
  detectedPatterns: string[]
  lastUpdated: number
}

export async function getOperatorProfile(): Promise<OperatorProfile | null> {
  try {
    await qdrant.getCollection(GLOBAL_COLLECTION)
    const result = await qdrant.scroll(GLOBAL_COLLECTION, {
      filter: { must: [{ key: 'type', match: { value: 'OPERATOR_PROFILE' } }] },
      limit: 1,
      with_payload: true,
      with_vector: false,
    })
    if (result.points.length === 0) return null
    const payload = result.points[0].payload as Record<string, unknown>
    return payload?.profile as OperatorProfile ?? null
  } catch {
    return null
  }
}

export async function saveOperatorProfile(profile: OperatorProfile): Promise<void> {
  await ensureCollection(GLOBAL_COLLECTION)
  // Usar id fijo para el perfil (upsert lo sobreescribe)
  const PROFILE_ID = '00000000-0000-0000-0000-000000000001'
  await qdrant.upsert(GLOBAL_COLLECTION, {
    wait: true,
    points: [{
      id: PROFILE_ID,
      vector: new Array(VECTOR_SIZE).fill(0),  // vector dummy para el perfil
      payload: {
        type: 'OPERATOR_PROFILE',
        profile,
        updatedAt: Date.now(),
      },
    }],
  })
}

// ─── List collections ─────────────────────────────────────────────────────────

export async function listCortexCollections(): Promise<string[]> {
  const result = await qdrant.getCollections()
  return result.collections
    .map(c => c.name)
    .filter(n => n.startsWith('cortex_') || n === LEGACY_COLLECTION)
}

// ─── Delete ───────────────────────────────────────────────────────────────────

export async function deleteEngramas(ids: string[], collection?: string): Promise<void> {
  const col = collection || LEGACY_COLLECTION
  await qdrant.delete(col, { wait: true, points: ids })
}

// ─── Delete all in project ────────────────────────────────────────────────────

export async function deleteAllInProject(projectName: string): Promise<number> {
  const all = await scrollAll(projectName)
  if (all.length === 0) return 0
  const ids = all.map(e => e.id).filter(Boolean) as string[]
  const col = LEGACY_COLLECTION  // por ahora todo está en legado
  await qdrant.delete(col, { wait: true, points: ids })
  return ids.length
}

// ─── Export / Import ──────────────────────────────────────────────────────────

export interface ExportedProject {
  version: string
  exportedAt: number
  projectName: string
  count: number
  engramas: Array<{
    id: string
    content: string
    type: string
    importance: number
    tags: string[]
    linkedTo: string[]
    createdAt: number
    lastAccessed: number
    accessCount: number
  }>
}

export async function exportProject(projectName: string): Promise<ExportedProject> {
  const all = await scrollAll(projectName)
  return {
    version: '3.0.0',
    exportedAt: Date.now(),
    projectName,
    count: all.length,
    engramas: all.map(e => ({
      id: e.id as string,
      content: e.content ?? '',
      type: e.type ?? 'FACT',
      importance: e.importance ?? 5,
      tags: e.tags ?? [],
      linkedTo: e.linkedTo ?? [],
      createdAt: e.createdAt ?? 0,
      lastAccessed: e.lastAccessed ?? 0,
      accessCount: e.accessCount ?? 0,
    })),
  }
}

// ─── Temp Memories ────────────────────────────────────────────────────────────
// Colección ligera sin embedding semántico.
// Las memorias aquí se guardan con vector dummy (ceros) y se indexan
// manualmente con index_temp cuando el operador lo solicita.

export interface TempMemory {
  id: string
  content: string
  projectName: string
  createdAt: number
  status: 'pending' | 'indexed'
  type: string
  importance: number
  tags: string[]
}

/** Crea la colección temp_memories si no existe */
export async function ensureTempCollection(): Promise<void> {
  try {
    await qdrant.getCollection(TEMP_COLLECTION)
  } catch {
    await qdrant.createCollection(TEMP_COLLECTION, {
      vectors: { size: VECTOR_SIZE, distance: 'Cosine' },
    })
  }
}

/**
 * Guarda una memoria en temp_memories con vector dummy (ceros).
 * No llama a Ollama ni a fastembed — es instantáneo.
 */
export async function upsertTemp(
  id: string,
  content: string,
  projectName: string,
): Promise<void> {
  await ensureTempCollection()
  const dummyVector = new Array(VECTOR_SIZE).fill(0)
  const payload: Record<string, unknown> = {
    content,
    projectName,
    createdAt: Date.now(),
    status: 'pending',
    type: 'FACT',
    importance: 5,
    tags: [],
  }
  await qdrant.upsert(TEMP_COLLECTION, {
    wait: true,
    points: [{ id, vector: dummyVector, payload }],
  })
}

/**
 * Lista memorias de temp_memories.
 * Filtra opcionalmente por proyecto y/o status.
 */
export async function scrollTemp(
  projectName?: string,
  status?: 'pending' | 'indexed',
  limit = 200,
): Promise<TempMemory[]> {
  await ensureTempCollection()

  // Construir filtro Qdrant
  const mustClauses: Record<string, unknown>[] = []
  if (projectName) {
    mustClauses.push({ key: 'projectName', match: { value: projectName } })
  }
  if (status) {
    mustClauses.push({ key: 'status', match: { value: status } })
  }

  const result = await qdrant.scroll(TEMP_COLLECTION, {
    filter: mustClauses.length > 0 ? { must: mustClauses } : undefined,
    limit,
    with_payload: true,
    with_vector: false,
  })

  return result.points.map((p) => ({
    id: String(p.id),
    ...(p.payload as unknown as Omit<TempMemory, 'id'>),
  }))
}

/**
 * Búsqueda por keyword en temp_memories.
 * Hace scroll de todas las memorias del proyecto y filtra por substring.
 * Sin embedding, sin Ollama.
 */
export async function searchTempByKeyword(
  query: string,
  projectName: string,
  limit = 10,
): Promise<TempMemory[]> {
  const all = await scrollTemp(projectName, undefined, 500)
  const q = query.toLowerCase()
  return all
    .filter(m => m.content.toLowerCase().includes(q))
    .slice(0, limit)
}

/** Elimina memorias de temp_memories por ID */
export async function deleteTempIds(ids: string[]): Promise<void> {
  if (ids.length === 0) return
  await qdrant.delete(TEMP_COLLECTION, { wait: true, points: ids })
}

/** Marca una memoria en temp como 'indexed' (alternativa a borrarla) */
export async function markTempIndexed(id: string): Promise<void> {
  await qdrant.setPayload(TEMP_COLLECTION, {
    payload: { status: 'indexed' },
    points: [id],
    wait: true,
  })
}
