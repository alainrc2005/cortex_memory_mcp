import { searchSimilar } from '../../../services/qdrant.js'
import { detectContradictions } from '../../../services/ollama.js'
import { patchPayload } from '../../../services/qdrant.js'
import type { ObserveState } from '../state.js'

/**
 * Umbral de similitud para buscar candidatos a invalidación.
 * Más alto que LINK_THRESHOLD (0.75) porque queremos solo los MUY similares.
 * Solo hechos muy cercanos semánticamente pueden ser contradictorios.
 */
const INVALIDATION_THRESHOLD = 0.88

/**
 * Pool de candidatos a evaluar. El LLM recibe máx 8 (definido en ollama.ts),
 * pero buscamos 10 para tener margen después del filtro por threshold.
 */
const INVALIDATION_POOL = 10

/**
 * Nodo 3.5: Detecta si el nuevo engrama contradice memorias existentes.
 *
 * Flujo:
 *  1. Busca los N más similares (score > 0.88) en la colección del proyecto.
 *  2. Si no hay candidatos → pasa sin hacer nada (caso común, ruta rápida).
 *  3. Llama a qwen3:8b en batch: "¿cuáles de estos son contradictorios con el nuevo?"
 *  4. Marca los contradictorios con status:'superseded' en Qdrant (patchPayload).
 *  5. Propaga supersededIds al state para que persist los registre en el nuevo engrama.
 *
 * Fallback: si Ollama falla → supersededIds=[], el engrama se guarda normal.
 * Nunca bloquea el pipeline.
 */
export async function invalidateNode(state: ObserveState): Promise<Partial<ObserveState>> {
  if (state.embedding.length === 0) {
    return { supersededIds: [], status: 'invalidate_skipped' }
  }

  // 1. Buscar candidatos muy similares (excluyendo el propio engrama si ya existía)
  const similar = await searchSimilar(
    state.embedding,
    state.projectName,
    INVALIDATION_POOL,
  ).catch(() => [])

  const candidates = similar
    .filter(h => h.score >= INVALIDATION_THRESHOLD)
    .filter(h => h.id !== state.engramaId)                    // no auto-invalidarse
    .filter(h => h.payload.status !== 'superseded')           // no re-evaluar ya obsoletos

  if (candidates.length === 0) {
    return { supersededIds: [], status: 'invalidate_clean' }
  }

  // 2. Preguntar al LLM cuáles son contradictorios
  const { supersededIds } = await detectContradictions(
    state.content,
    candidates.map(h => ({ id: h.id, content: h.payload.content ?? '' })),
  )

  if (supersededIds.length === 0) {
    return { supersededIds: [], status: 'invalidate_clean' }
  }

  // 3. Marcar en Qdrant como superseded (en paralelo, sin bloquear si alguno falla)
  const now = Date.now()
  await Promise.allSettled(
    supersededIds.map(id =>
      patchPayload(id, {
        status: 'superseded',
        supersededBy: state.engramaId,
        supersededAt: now,
      }),
    ),
  )

  console.error(
    `[invalidate] ${supersededIds.length} engrama(s) marcado(s) como superseded: ${supersededIds.join(', ')}`,
  )

  return { supersededIds, status: 'invalidated' }
}
