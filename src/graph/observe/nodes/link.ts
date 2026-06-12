import { searchSimilar, patchPayload } from '../../../services/qdrant.js'
import type { ObserveState } from '../state.js'

const LINK_THRESHOLD = 0.75  // Cosine similarity mínima para crear vínculo (calibrado para all-minilm)
const LINK_LIMIT = 3         // Máximo de vínculos por engrama

/**
 * Nodo 3: Busca los engramas más similares en Qdrant y crea vínculos bidireccionales.
 * También actualiza el campo linkedTo de los engramas existentes.
 */
export async function linkNode(state: ObserveState): Promise<Partial<ObserveState>> {
  if (state.embedding.length === 0) {
    return { linkedTo: [], status: 'link_skipped' }
  }

  // Buscar similares en el mismo proyecto
  const similar = await searchSimilar(state.embedding, state.projectName, LINK_LIMIT + 1)

  // Filtrar por threshold y excluir el mismo engrama si ya existe
  const candidates = similar
    .filter(h => h.score >= LINK_THRESHOLD)
    .slice(0, LINK_LIMIT)

  const linkedTo = candidates.map(h => h.id)

  // Actualizar vínculos en los engramas existentes (vínculo bidireccional)
  for (const candidate of candidates) {
    const existingLinks: string[] = candidate.payload.linkedTo ?? []
    if (!existingLinks.includes(state.engramaId)) {
      await patchPayload(candidate.id, {
        linkedTo: [...existingLinks, state.engramaId],
      }).catch(() => {/* no bloquear si falla un patch */})
    }
  }

  return { linkedTo, status: 'linked' }
}
