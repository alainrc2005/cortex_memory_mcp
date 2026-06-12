import { StateGraph, START, END } from '@langchain/langgraph'
import { observeStateChannels, type ObserveState } from './state.js'
import { scoreNode }      from './nodes/score.js'
import { embedNode }      from './nodes/embed.js'
import { linkNode }       from './nodes/link.js'
import { invalidateNode } from './nodes/invalidate.js'
import { persistNode }    from './nodes/persist.js'

/**
 * Pipeline: score → embed → link → invalidate → persist
 *
 * - score:      qwen3:8b asigna importance, type, tags
 * - embed:      all-minilm genera el vector
 * - link:       busca similares en Qdrant y crea vínculos bidireccionales
 * - invalidate: detecta contradicciones con similares (score>0.88) y marca obsoletos
 * - persist:    guarda el engrama enriquecido en Qdrant
 */
const workflow = new StateGraph<ObserveState>({ channels: observeStateChannels })
  .addNode('score',      scoreNode)
  .addNode('embed',      embedNode)
  .addNode('link',       linkNode)
  .addNode('invalidate', invalidateNode)
  .addNode('persist',    persistNode)
  .addEdge(START,        'score')
  .addEdge('score',      'embed')
  .addEdge('embed',      'link')
  .addEdge('link',       'invalidate')
  .addEdge('invalidate', 'persist')
  .addEdge('persist',    END)

export const observeGraph = workflow.compile()
