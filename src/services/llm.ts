/**
 * CORTEX — LLM Router
 *
 * Punto de entrada unificado para todas las operaciones LLM.
 * Enruta a OpenRouter, Ollama o modo "none" según CORTEX_LLM_BACKEND.
 *
 * Variable de entorno:
 *   CORTEX_LLM_BACKEND = "openrouter" | "ollama" | "none"
 *   (default: "openrouter" si OPENROUTER_API_KEY está definida, si no "ollama")
 */

import * as dotenv from 'dotenv'
dotenv.config()  // Cargar .env antes de leer las variables de entorno

import type { EngramaType } from '../types/engrama.js'

export interface ScoreResult {
  importance: number
  type: EngramaType
  tags: string[]
}

export interface RerankCandidate {
  id: string
  content: string
}

export interface RerankResult {
  id: string
  rerankScore: number
}

export interface ContradictionResult {
  supersededIds: string[]
}

// ─── Detectar backend activo ──────────────────────────────────────────────────

function resolveBackend(): 'openrouter' | 'ollama' | 'none' {
  const raw = (process.env.CORTEX_LLM_BACKEND ?? '').toLowerCase().trim()
  if (raw === 'openrouter') return 'openrouter'
  if (raw === 'ollama')     return 'ollama'
  if (raw === 'none')       return 'none'

  // Auto-detect: si hay API key de OpenRouter, usarla
  if (process.env.OPENROUTER_API_KEY) return 'openrouter'

  // Fallback a Ollama
  return 'ollama'
}

const BACKEND = resolveBackend()

// Log al arrancar para saber qué backend está activo
const modelLabel = BACKEND === 'openrouter'
  ? `OpenRouter (${process.env.OPENROUTER_MODEL ?? 'google/gemma-4-27b-it:free'})`
  : BACKEND === 'ollama'
    ? 'Ollama (qwen3:8b)'
    : 'none (solo fastembed)'

process.stderr.write(`[CORTEX] LLM backend: ${modelLabel}\n`)

// ─── Importaciones dinámicas según backend ────────────────────────────────────

let _scoreAndTag: (content: string) => Promise<ScoreResult>
let _batchScoreAndTag: (contents: string[]) => Promise<ScoreResult[]>
let _rerankWithLLM: (query: string, candidates: RerankCandidate[]) => Promise<RerankResult[]>
let _detectContradictions: (newContent: string, candidates: RerankCandidate[]) => Promise<ContradictionResult>
let _generateText: (prompt: string) => Promise<string>

if (BACKEND === 'openrouter') {
  const or = await import('./openrouter.js')
  _scoreAndTag           = or.scoreAndTagOR
  _batchScoreAndTag      = or.batchScoreAndTagOR
  _rerankWithLLM         = or.rerankWithOpenRouter
  _detectContradictions  = or.detectContradictionsOR
  _generateText          = or.generateTextOR

} else if (BACKEND === 'ollama') {
  const ol = await import('./ollama.js')
  _scoreAndTag           = ol.scoreAndTag
  _batchScoreAndTag      = ol.batchScoreAndTag
  _rerankWithLLM         = ol.rerankWithLLM
  _detectContradictions  = ol.detectContradictions
  _generateText          = ol.generateText

} else {
  // Backend "none" — todo cae a valores por defecto sin llamada LLM
  const noop = async (): Promise<ScoreResult> => ({ importance: 5, type: 'FACT', tags: [] })
  _scoreAndTag           = noop
  _batchScoreAndTag      = async (cs) => cs.map(() => ({ importance: 5, type: 'FACT' as const, tags: [] }))
  _rerankWithLLM         = async (_, cands) => cands.map(c => ({ id: c.id, rerankScore: 0.5 }))
  _detectContradictions  = async () => ({ supersededIds: [] })
  _generateText          = async () => ''
}

// ─── Exports públicos ─────────────────────────────────────────────────────────

export const scoreAndTag          = _scoreAndTag
export const batchScoreAndTag     = _batchScoreAndTag
export const rerankWithLLM        = _rerankWithLLM
export const detectContradictions = _detectContradictions
export const generateText         = _generateText
export { BACKEND as LLM_BACKEND }
