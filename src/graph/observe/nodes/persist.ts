import { upsertEngrama, ensureCollection, ensureSparseIndex, ensureProjectCollection, collectionFor } from '../../../services/qdrant.js'
import { getSparseEmbedding } from '../../../services/fastembed.js'
import type { ObserveState } from '../state.js'
import type { EngramaPayload } from '../../../types/engrama.js'

/** Nodo 5: Persiste el engrama enriquecido en Qdrant con dense + sparse vectors */
export async function persistNode(state: ObserveState): Promise<Partial<ObserveState>> {
  // Usar colección dedicada del proyecto; fallback a work_memories si algo falla
  const targetCol = await ensureProjectCollection(state.projectName).catch(async () => {
    await ensureCollection()
    return undefined as unknown as string
  })

  // Agregar índice sparse BM25 si la colección no lo tiene — falla silenciosamente
  await ensureSparseIndex(targetCol).catch(() => {
    process.stderr.write(`[persist] ensureSparseIndex warn en ${targetCol} — degradando a solo-dense\n`)
  })

  const now = Date.now()
  const payload: EngramaPayload = {
    content:      state.content,
    projectName:  state.projectName,
    createdAt:    now,
    importance:   state.importance,
    accessCount:  0,
    lastAccessed: now,
    type:         state.type,
    tags:         state.tags,
    linkedTo:     state.linkedTo,
    // Registrar qué engramas obsoletó este nuevo hecho (si los hay)
    ...(state.supersededIds.length > 0 && { supersedes: state.supersededIds }),
  }

  // Generar sparse embedding para BM25 (falla silenciosamente si SPLADE no está listo)
  let sparse: { indices: number[]; values: number[] } | undefined
  try {
    sparse = await getSparseEmbedding(state.content)
  } catch {
    // Fallback: guardar solo con dense — búsqueda híbrida degradará a solo-dense
    process.stderr.write('[persist] sparse embedding no disponible, guardando solo dense\n')
  }

  await upsertEngrama(state.engramaId, state.embedding, payload, targetCol, sparse)

  return { status: 'persisted' }
}
