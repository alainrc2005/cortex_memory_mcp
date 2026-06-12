import { StateGraph, START, END } from '@langchain/langgraph'
import { observeStateChannels, type ObserveState } from './state.js'
import { scoreNode }        from './nodes/score.js'
import { embedNode }        from './nodes/embed.js'
import { linkNode }         from './nodes/link.js'
import { invalidateNode }   from './nodes/invalidate.js'
import { persistNode }      from './nodes/persist.js'
import { extractGraphNode } from './nodes/extract_graph.js'

/**
 * Pipeline: score → embed → link → invalidate → persist → extract_graph
 *
 * - score:         qwen3:8b asigna importance, type, tags
 * - embed:         all-minilm genera el vector
 * - link:          busca similares en Qdrant y crea vínculos bidireccionales
 * - invalidate:    detecta contradicciones y marca obsoletos
 * - persist:       guarda el engrama enriquecido en Qdrant
 * - extract_graph: extrae entidades/relaciones con qwen3 → Kuzu (no crítico)
 */
const workflow = new StateGraph<ObserveState>({ channels: observeStateChannels })
  .addNode('score',         scoreNode)
  .addNode('embed',         embedNode)
  .addNode('link',          linkNode)
  .addNode('invalidate',    invalidateNode)
  .addNode('persist',       persistNode)
  .addNode('extract_graph', extractGraphNode)
  .addEdge(START,           'score')
  .addEdge('score',         'embed')
  .addEdge('embed',         'link')
  .addEdge('link',          'invalidate')
  .addEdge('invalidate',    'persist')
  .addEdge('persist',       'extract_graph')
  .addEdge('extract_graph', END)

export const observeGraph = workflow.compile()

