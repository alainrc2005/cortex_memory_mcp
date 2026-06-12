import { EmbeddingModel, FlagEmbedding, SparseTextEmbedding, SparseEmbeddingModel } from 'fastembed'
import * as dotenv from 'dotenv'
import { mkdirSync } from 'fs'

dotenv.config({ path: '/home/alainrc2005/IA/memory-mcp/.env' })

// ─── Cache dir ────────────────────────────────────────────────────────────────

const CACHE_DIR =
  process.env.FASTEMBED_CACHE_DIR ||
  '/home/alainrc2005/IA/memory-mcp/.fastembed_cache'

try { mkdirSync(CACHE_DIR, { recursive: true }) } catch {}

// ─── Singleton ────────────────────────────────────────────────────────────────

// all-MiniLM-L6-v2 → 384 dimensiones (igual que all-minilm en Ollama)
// Compatible 100% con los vectores existentes en Qdrant.

let _model: FlagEmbedding | null = null
let _loading: Promise<FlagEmbedding> | null = null

async function getModel(): Promise<FlagEmbedding> {
  if (_model) return _model
  if (_loading) return _loading

  _loading = FlagEmbedding.init({
    model: EmbeddingModel.AllMiniLML6V2,
    cacheDir: CACHE_DIR,
  }).then((m) => {
    _model = m
    _loading = null
    return m
  })

  return _loading
}

// ─── Sparse (SPLADE) ────────────────────────────────────────────────────────
//
// prithivida/Splade_PP_en_v1 — modelo SPLADE de producción (~110 MB).
// Genera vectores láxicos dispersos: indices=token_ids, values=pesos BM25-like.
// Complementa el dense (semántico) con búsqueda exacta de keywords.
//
// El modelo se descarga automáticamente en el primer uso.
// Comparte CACHE_DIR con el modelo dense.

export interface SparseVector {
  indices: number[]
  values: number[]
}

let _sparseModel: SparseTextEmbedding | null = null
let _sparseLoading: Promise<SparseTextEmbedding> | null = null

async function getSparseModel(): Promise<SparseTextEmbedding> {
  if (_sparseModel) return _sparseModel
  if (_sparseLoading) return _sparseLoading

  _sparseLoading = SparseTextEmbedding.init({
    model: SparseEmbeddingModel.SpladePPEnV1,
    cacheDir: CACHE_DIR,
  }).then((m) => {
    _sparseModel = m
    _sparseLoading = null
    return m
  })

  return _sparseLoading
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Genera un embedding usando ONNX local (fastembed).
 * Sin servidor externo, sin Ollama, sin bloqueo del LLM.
 * Primera llamada: carga el modelo (~22 MB, ~1-3 s).
 * Llamadas siguientes: instantáneas (modelo en memoria).
 */
export async function getEmbedding(text: string): Promise<number[]> {
  const model = await getModel()
  const iter = model.embed([text], 1)
  for await (const batch of iter) {
    return Array.from(batch[0])
  }
  throw new Error('fastembed: no embedding returned')
}

/**
 * Genera embeddings en lote — más eficiente que llamar getEmbedding N veces.
 * Útil para batch_observe y consolidate.
 */
export async function getEmbeddingBatch(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return []
  const model = await getModel()
  const results: number[][] = []
  const iter = model.embed(texts, Math.min(texts.length, 8))
  for await (const batch of iter) {
    for (const vec of batch) {
      results.push(Array.from(vec))
    }
  }
  return results
}

/**
 * Genera un sparse embedding (SPLADE) para búsqueda híbrida BM25.
 * Retorna {indices, values} compatibles con Qdrant sparse vectors.
 */
export async function getSparseEmbedding(text: string): Promise<SparseVector> {
  const model = await getSparseModel()
  const iter = model.embed([text], 1)
  for await (const batch of iter) {
    const vec = batch[0] as unknown as { indices: ArrayLike<number>, values: ArrayLike<number> }
    return {
      indices: Array.from(vec.indices),
      values:  Array.from(vec.values),
    }
  }
  throw new Error('fastembed sparse: no embedding returned')
}

/**
 * Genera sparse embeddings en lote.
 */
export async function getSparseEmbeddingBatch(texts: string[]): Promise<SparseVector[]> {
  if (texts.length === 0) return []
  const model = await getSparseModel()
  const results: SparseVector[] = []
  const iter = model.embed(texts, Math.min(texts.length, 8))
  for await (const batch of iter) {
    for (const vec of batch) {
      const sv = vec as unknown as { indices: ArrayLike<number>, values: ArrayLike<number> }
      results.push({ indices: Array.from(sv.indices), values: Array.from(sv.values) })
    }
  }
  return results
}

/**
 * Pre-calienta ambos modelos (dense + sparse) en background al arrancar.
 * Evita el delay en la primera llamada real.
 */
export function warmupEmbedding(): void {
  getModel()
    .then(() => process.stderr.write('[fastembed] Dense all-MiniLM-L6-v2 listo.\n'))
    .catch((e) => process.stderr.write(`[fastembed] Error cargando dense: ${e}\n`))
  getSparseModel()
    .then(() => process.stderr.write('[fastembed] Sparse SPLADE_PP_en listo.\n'))
    .catch((e) => process.stderr.write(`[fastembed] Error cargando sparse: ${e}\n`))
}
