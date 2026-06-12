export type EngramaType =
  | 'DECISION'
  | 'CONTEXT'
  | 'PREFERENCE'
  | 'FACT'
  | 'ERROR'
  | 'PATTERN'

export interface Engrama {
  // Identidad
  id: string
  content: string
  projectName: string
  createdAt: number        // Unix ms

  // Ciclo de vida
  importance: number       // 1–10, asignado por LLM
  accessCount: number      // veces recuperado
  lastAccessed: number     // Unix ms del último recall

  // Categorización
  type: EngramaType
  tags: string[]           // 3–5 keywords semánticos

  // Vínculos
  linkedTo: string[]       // IDs de engramas relacionados

  // Invalidación (opcional — solo presente en engramas superseded o que superseden)
  status?: 'active' | 'superseded'  // undefined = active (compatibilidad con engramas viejos)
  supersededBy?: string             // ID del engrama que lo invalida
  supersededAt?: number             // Unix ms
  supersedes?: string[]             // IDs que este engrama invalidó al crearse
}

/** Payload que se guarda en Qdrant (sin el vector) */
export type EngramaPayload = Omit<Engrama, 'id'>

/** Resultado de recall con score de relevancia calculado en query-time */
export interface RecallResult {
  engrama: Engrama
  score: number            // score semántico de Qdrant
  decayScore: number       // score ponderado por decay
  rerankScore?: number     // score cross-encoder (0–1), presente si se ejecutó reranking
  combined: number         // score final fusionado (semántico + decay [+ rerank])
}
