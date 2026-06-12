import type { Engrama } from '../types/engrama.js'

// ─── Clasificación por motor ────────────────────────────────────────────────
//
// Bayesiano  → memorias técnicas: no deben olvidarse a menos que sean
//              contradichas o nunca accedidas con baja importancia.
//
// FSRS       → memorias conversacionales: decaen como memoria humana,
//              se estabilizan con repetición (práctica espaciada).

const BAYESIAN_TYPES = new Set(['DECISION', 'FACT', 'ERROR', 'PATTERN'])
const FSRS_TYPES     = new Set(['PREFERENCE', 'CONTEXT'])

// ─── Motor Bayesiano (Beta Distribution) ────────────────────────────────────
//
//   alpha   = accessCount + importance/2
//             Crece con cada acceso y con la relevancia asignada.
//
//   beta    = max(0.1, daysSince × (10 / importance))
//             El tiempo erosiona, pero la importancia lo frena.
//             Una DECISION de imp:9 puede estar 60 días sin acceso
//             y seguir siendo altamente confiable.
//
//   utility = E[θ] = alpha / (alpha + beta)   → 0–1
//
function bayesianDecay(engrama: Engrama): number {
  const now         = Date.now()
  const lastAccessed = engrama.lastAccessed ?? engrama.createdAt ?? now
  const accessCount  = engrama.accessCount  ?? 0
  const importance   = engrama.importance   ?? 5
  const daysSince    = (now - lastAccessed) / 86_400_000

  const alpha = accessCount + importance / 2
  const beta  = Math.max(0.1, daysSince * (10 / importance))
  return alpha / (alpha + beta)
}

// ─── Motor FSRS-inspired (Spaced Repetition) ────────────────────────────────
//
//   stability     = log(1 + accessCount) × (importance / 5)
//                   Crece con la práctica; la importancia amplifica la estabilidad.
//
//   retrievability = exp(−daysSince / max(stability, 1))   → 0–1
//                   Curva de olvido de Ebbinghaus modulada por estabilidad.
//
function fsrsDecay(engrama: Engrama): number {
  const now          = Date.now()
  const lastAccessed = engrama.lastAccessed ?? engrama.createdAt ?? now
  const accessCount  = engrama.accessCount  ?? 0
  const importance   = engrama.importance   ?? 5
  const daysSince    = (now - lastAccessed) / 86_400_000

  const stability     = Math.log1p(accessCount) * (importance / 5)
  return Math.exp(-daysSince / Math.max(stability, 1))
}

// ─── Router público ──────────────────────────────────────────────────────────

/**
 * Calcula el score de vitalidad de un engrama (0–1).
 *
 * - DECISION / FACT / ERROR / PATTERN  → Motor Bayesiano
 *   Las memorias técnicas no decaen arbitrariamente.
 *
 * - PREFERENCE / CONTEXT               → Motor FSRS
 *   Las memorias conversacionales decaen como memoria humana.
 *
 * - Cualquier otro tipo                → Bayesiano (default seguro)
 */
export function calcDecay(engrama: Engrama): number {
  const type = engrama.type ?? 'FACT'
  if (BAYESIAN_TYPES.has(type)) return bayesianDecay(engrama)
  if (FSRS_TYPES.has(type))     return fsrsDecay(engrama)
  return bayesianDecay(engrama)   // default → técnico
}

// ─── Pesos por tipo de engrama ────────────────────────────────────────────────
//
// Cada tipo de memoria tiene una naturaleza distinta de relevancia:
//
//   DECISION / ERROR / PATTERN  → el decay pesa más porque son hechos duraderos.
//                                  Si llevan mucho sin accederse probablemente
//                                  ya no son relevantes para la sesión actual.
//
//   FACT                        → balanceado: ni demasiado efímero ni permanente.
//
//   CONTEXT / PREFERENCE        → el semántico pesa más porque su utilidad depende
//                                  de cuán cerca están del query actual, no de cuánto
//                                  tiempo llevan en memoria.

interface TypeWeights { semantic: number; decay: number }

const TYPE_WEIGHTS: Record<string, TypeWeights> = {
  DECISION:   { semantic: 0.55, decay: 0.45 },
  ERROR:      { semantic: 0.55, decay: 0.45 },
  PATTERN:    { semantic: 0.50, decay: 0.50 },
  FACT:       { semantic: 0.65, decay: 0.35 },
  CONTEXT:    { semantic: 0.75, decay: 0.25 },
  PREFERENCE: { semantic: 0.75, decay: 0.25 },
}
const DEFAULT_WEIGHTS: TypeWeights = { semantic: 0.65, decay: 0.35 }

/**
 * Score combinado para reranking en recall.
 *
 * @param semanticScore  0–1  Qdrant cosine similarity
 * @param decayScore     0–1  calcDecay — vitalidad temporal del engrama
 * @param engramaType    Tipo del engrama para seleccionar pesos dinámicos
 * @param rerankScore    0–1  (opcional) Score cross-encoder del LLM.
 *                           Cuando está presente desplaza los pesos hacia él
 *                           (40% semantic · 40% rerank · 20% decay) para máxima
 *                           precisión bidireccional.
 */
export function combinedScore(
  semanticScore: number,
  decayScore: number,
  engramaType?: string,
  rerankScore?: number,
): number {
  if (rerankScore !== undefined) {
    // Con cross-encoder: el rerank domina la señal de relevancia
    // decay sigue participando para penalizar memorias muy viejas y poco accedidas
    return 0.40 * semanticScore + 0.40 * rerankScore + 0.20 * decayScore
  }

  // Sin cross-encoder: pesos dinámicos por tipo
  const weights = TYPE_WEIGHTS[engramaType ?? ''] ?? DEFAULT_WEIGHTS
  return weights.semantic * semanticScore + weights.decay * decayScore
}

/**
 * Candidato a compresión en consolidación:
 * vitalidad muy baja (<0.1) e importancia baja.
 */
export function isCompressCandidate(engrama: Engrama): boolean {
  return calcDecay(engrama) < 0.1 && (engrama.importance ?? 5) <= 3
}
