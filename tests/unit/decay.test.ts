/**
 * LAYER 1 — UNIT TESTS: Decay algorithms
 *
 * Tests puros sobre calcDecay() y combinedScore().
 * Sin Qdrant, sin LLM, sin fastembed. Corren en microsegundos.
 *
 * Qué cubren:
 *   - Bayesian decay para tipos técnicos (DECISION, FACT, ERROR, PATTERN)
 *   - FSRS decay para tipos conversacionales (PREFERENCE, CONTEXT)
 *   - Invariantes: score ∈ [0, 1]
 *   - Comportamiento temporal: memorias viejas < nuevas
 *   - Importancia: alta importancia → más resistencia al olvido
 *   - combinedScore con y sin reranker
 *   - Pesos por tipo de engrama
 *   - isCompressCandidate
 */

import { describe, it, expect } from 'vitest'
import { calcDecay, combinedScore, isCompressCandidate } from '../../src/services/decay.js'
import type { Engrama } from '../../src/types/engrama.js'

// ─── Helpers ──────────────────────────────────────────────────────────────────

const DAY_MS = 86_400_000

function makeEngrama(overrides: Partial<Engrama> = {}): Engrama {
  return {
    id:          'test-id',
    content:     'test content',
    projectName: 'test',
    type:        'FACT',
    importance:  5,
    accessCount: 0,
    createdAt:   Date.now(),
    lastAccessed: Date.now(),
    tags:        [],
    linkedTo:    [],
    ...overrides,
  } as Engrama
}

// ─── calcDecay ────────────────────────────────────────────────────────────────

describe('calcDecay — invariantes fundamentales', () => {
  it('siempre retorna un número en [0, 1]', () => {
    const types = ['DECISION', 'FACT', 'ERROR', 'PATTERN', 'PREFERENCE', 'CONTEXT', 'UNKNOWN']
    for (const type of types) {
      const score = calcDecay(makeEngrama({ type }))
      expect(score).toBeGreaterThanOrEqual(0)
      expect(score).toBeLessThanOrEqual(1)
    }
  })

  it('un engrama recién creado tiene score alto (> 0.7)', () => {
    const e = makeEngrama({ lastAccessed: Date.now(), accessCount: 0, importance: 5 })
    expect(calcDecay(e)).toBeGreaterThan(0.7)
  })

  it('un engrama de 30 días sin acceso tiene score menor que uno reciente', () => {
    const recent = makeEngrama({ lastAccessed: Date.now() })
    const old    = makeEngrama({ lastAccessed: Date.now() - 30 * DAY_MS })
    expect(calcDecay(recent)).toBeGreaterThan(calcDecay(old))
  })

  it('mayor importancia → más resistencia al olvido (DECISION, 60 días)', () => {
    const base = { lastAccessed: Date.now() - 60 * DAY_MS, accessCount: 0, type: 'DECISION' }
    const lowImp  = makeEngrama({ ...base, importance: 2 })
    const highImp = makeEngrama({ ...base, importance: 9 })
    expect(calcDecay(highImp)).toBeGreaterThan(calcDecay(lowImp))
  })

  it('mayor accessCount → score más alto (DECISION)', () => {
    const base = { lastAccessed: Date.now() - 7 * DAY_MS, type: 'DECISION', importance: 5 }
    const neverAccessed = makeEngrama({ ...base, accessCount: 0 })
    const oftenAccessed = makeEngrama({ ...base, accessCount: 20 })
    expect(calcDecay(oftenAccessed)).toBeGreaterThan(calcDecay(neverAccessed))
  })
})

describe('calcDecay — routing por tipo (Bayesian vs FSRS)', () => {
  const baseOld = { lastAccessed: Date.now() - 10 * DAY_MS, accessCount: 0, importance: 5 }

  it('DECISION usa Bayesian (no colapsa a 0 rápido)', () => {
    const score = calcDecay(makeEngrama({ ...baseOld, type: 'DECISION' }))
    // Bayesian con alpha pequeño pero no 0 → no colapsa del todo
    expect(score).toBeGreaterThan(0.1)
  })

  it('PREFERENCE usa FSRS (decae más rápido que DECISION)', () => {
    const decision   = calcDecay(makeEngrama({ ...baseOld, type: 'DECISION' }))
    const preference = calcDecay(makeEngrama({ ...baseOld, type: 'PREFERENCE' }))
    // FSRS decae más rápido para memorias sin refuerzo reciente
    expect(decision).toBeGreaterThanOrEqual(preference)
  })

  it('CONTEXT usa FSRS', () => {
    const score = calcDecay(makeEngrama({ ...baseOld, type: 'CONTEXT' }))
    expect(score).toBeGreaterThanOrEqual(0)
    expect(score).toBeLessThanOrEqual(1)
  })

  it('tipo desconocido usa Bayesian (fallback seguro)', () => {
    const score = calcDecay(makeEngrama({ ...baseOld, type: 'WEIRD_TYPE' }))
    expect(score).toBeGreaterThan(0)
    expect(score).toBeLessThanOrEqual(1)
  })

  it('PATTERN usa Bayesian', () => {
    const score = calcDecay(makeEngrama({ ...baseOld, type: 'PATTERN' }))
    expect(score).toBeGreaterThan(0.1)
  })

  it('ERROR usa Bayesian', () => {
    const score = calcDecay(makeEngrama({ ...baseOld, type: 'ERROR' }))
    expect(score).toBeGreaterThan(0.1)
  })
})

describe('calcDecay — manejo de campos undefined/null', () => {
  it('sin lastAccessed ni createdAt → usa Date.now() como fallback', () => {
    const e = { ...makeEngrama(), lastAccessed: undefined, createdAt: undefined } as unknown as Engrama
    const score = calcDecay(e)
    expect(score).toBeGreaterThan(0)
    expect(score).toBeLessThanOrEqual(1)
  })

  it('sin importance → usa default 5', () => {
    const e = { ...makeEngrama(), importance: undefined } as unknown as Engrama
    const score = calcDecay(e)
    expect(score).toBeGreaterThanOrEqual(0)
    expect(score).toBeLessThanOrEqual(1)
  })

  it('sin accessCount → usa default 0', () => {
    const e = { ...makeEngrama(), accessCount: undefined } as unknown as Engrama
    const score = calcDecay(e)
    expect(score).toBeGreaterThanOrEqual(0)
    expect(score).toBeLessThanOrEqual(1)
  })
})

// ─── combinedScore ────────────────────────────────────────────────────────────

describe('combinedScore — sin reranker', () => {
  it('retorna número en [0, 1] para todos los tipos', () => {
    const types = ['DECISION', 'FACT', 'ERROR', 'PATTERN', 'PREFERENCE', 'CONTEXT']
    for (const t of types) {
      const score = combinedScore(0.8, 0.6, t)
      expect(score).toBeGreaterThanOrEqual(0)
      expect(score).toBeLessThanOrEqual(1)
    }
  })

  it('tipo CONTEXT / PREFERENCE: peso semántico dominante (0.75)', () => {
    // semantic=1.0, decay=0.0 → result debe estar cerca de 0.75
    const scoreCtx  = combinedScore(1.0, 0.0, 'CONTEXT')
    const scorePref = combinedScore(1.0, 0.0, 'PREFERENCE')
    expect(scoreCtx).toBeCloseTo(0.75, 2)
    expect(scorePref).toBeCloseTo(0.75, 2)
  })

  it('tipo DECISION: pesos 0.55 semantic + 0.45 decay', () => {
    const score = combinedScore(1.0, 0.0, 'DECISION')
    expect(score).toBeCloseTo(0.55, 2)
  })

  it('tipo FACT: pesos 0.65 semantic + 0.35 decay', () => {
    const score = combinedScore(1.0, 0.0, 'FACT')
    expect(score).toBeCloseTo(0.65, 2)
  })

  it('tipo desconocido → DEFAULT pesos 0.65/0.35', () => {
    const score = combinedScore(1.0, 0.0, 'UNKNOWN_TYPE')
    expect(score).toBeCloseTo(0.65, 2)
  })

  it('mayor semantic score → mayor combined (mismo decay y tipo)', () => {
    const low  = combinedScore(0.3, 0.5, 'FACT')
    const high = combinedScore(0.9, 0.5, 'FACT')
    expect(high).toBeGreaterThan(low)
  })
})

describe('combinedScore — con reranker', () => {
  it('con rerankScore: fórmula 0.40·sem + 0.40·rerank + 0.20·decay', () => {
    const score = combinedScore(1.0, 0.0, 'FACT', 1.0)
    // 0.4*1 + 0.4*1 + 0.2*0 = 0.8
    expect(score).toBeCloseTo(0.8, 2)
  })

  it('rerankScore=0 penaliza aunque semantic sea alto', () => {
    const score = combinedScore(1.0, 1.0, 'FACT', 0.0)
    // 0.4*1 + 0.4*0 + 0.2*1 = 0.6
    expect(score).toBeCloseTo(0.6, 2)
  })

  it('reranker presente ignora los pesos por tipo', () => {
    // Mismo semantic+decay+rerank, tipos diferentes → mismo resultado
    const scoreDecision = combinedScore(0.7, 0.5, 'DECISION', 0.8)
    const scoreContext  = combinedScore(0.7, 0.5, 'CONTEXT',  0.8)
    expect(scoreDecision).toBeCloseTo(scoreContext, 5)
  })

  it('retorna [0,1] con valores extremos', () => {
    expect(combinedScore(0, 0, 'FACT', 0)).toBeGreaterThanOrEqual(0)
    expect(combinedScore(1, 1, 'FACT', 1)).toBeLessThanOrEqual(1)
  })
})

// ─── isCompressCandidate ──────────────────────────────────────────────────────

describe('isCompressCandidate', () => {
  it('engrama muy viejo con importancia baja → candidato a comprimir', () => {
    const e = makeEngrama({
      lastAccessed: Date.now() - 180 * DAY_MS,  // 6 meses sin acceso
      accessCount:  0,
      importance:   2,
      type:         'FACT',
    })
    expect(isCompressCandidate(e)).toBe(true)
  })

  it('engrama importante (imp >= 4) nunca es candidato aunque sea viejo', () => {
    const e = makeEngrama({
      lastAccessed: Date.now() - 180 * DAY_MS,
      accessCount:  0,
      importance:   4,   // umbral es <= 3
      type:         'FACT',
    })
    expect(isCompressCandidate(e)).toBe(false)
  })

  it('engrama reciente con importancia baja → NO candidato (decay alto)', () => {
    const e = makeEngrama({
      lastAccessed: Date.now(),   // ahora mismo
      accessCount:  0,
      importance:   1,
    })
    expect(isCompressCandidate(e)).toBe(false)
  })

  it('DECISION de alta importancia, 90 días sin acceso → NO candidato', () => {
    const e = makeEngrama({
      lastAccessed: Date.now() - 90 * DAY_MS,
      accessCount:  0,
      importance:   8,
      type:         'DECISION',
    })
    expect(isCompressCandidate(e)).toBe(false)
  })
})
