export type EpisodeEventType =
  | 'DECISION'
  | 'ERROR'
  | 'SOLUTION'
  | 'INSIGHT'
  | 'CONTEXT_CHANGE'

export interface EpisodeEvent {
  timestamp: number          // Unix ms
  eventType: EpisodeEventType
  description: string        // qué pasó
  linkedEngramaId?: string   // si generó un engrama semántico
}

export interface Episode {
  // Identidad
  id: string
  projectName: string

  // Temporal
  startedAt: number          // Unix ms — inicio de sesión
  endedAt?: number           // Unix ms — fin (undefined = sesión abierta)
  status: 'open' | 'closed'

  // Contexto
  context: string            // tema/objetivo de la sesión
  summary: string            // concatenación de eventos (sin LLM)

  // Eventos
  events: EpisodeEvent[]
  eventCount: number
}

/** Payload que se guarda en Qdrant (sin el id) */
export type EpisodePayload = Omit<Episode, 'id'>
