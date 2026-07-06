import { scoreAndTag } from '../../../services/llm.js'
import type { ObserveState } from '../state.js'

/** Nodo 1: Puntúa el contenido con qwen3:8b → importance, type, tags */
export async function scoreNode(state: ObserveState): Promise<Partial<ObserveState>> {
  const result = await scoreAndTag(state.content)
  return {
    importance: result.importance,
    type: result.type,
    tags: result.tags,
    status: 'scored',
  }
}
