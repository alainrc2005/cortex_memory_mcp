import { getEmbedding } from '../../../services/fastembed.js'
import type { ObserveState } from '../state.js'

/** Nodo 2: Genera el vector embedding del contenido con all-minilm */
export async function embedNode(state: ObserveState): Promise<Partial<ObserveState>> {
  const embedding = await getEmbedding(state.content)
  return { embedding, status: 'embedded' }
}
