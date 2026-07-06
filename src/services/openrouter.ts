/**
 * CORTEX — OpenRouter LLM Service
 *
 * Reemplaza a Ollama para scoring, tagging, reranking y detección de contradicciones.
 * Compatible con la API de OpenAI (chat completions).
 *
 * Variables de entorno:
 *   OPENROUTER_API_KEY   — API key de OpenRouter (obligatoria para usar este backend)
 *   OPENROUTER_MODEL     — Modelo a usar (default: google/gemma-4-27b-it:free)
 *   OPENROUTER_BASE_URL  — Base URL (default: https://openrouter.ai/api/v1)
 */

import type { EngramaType } from '../types/engrama.js'
import type { ScoreResult, RerankCandidate, RerankResult, ContradictionResult } from './ollama.js'

const OPENROUTER_BASE_URL = process.env.OPENROUTER_BASE_URL ?? 'https://openrouter.ai/api/v1'
const OPENROUTER_API_KEY  = process.env.OPENROUTER_API_KEY ?? ''
const OPENROUTER_MODEL    = process.env.OPENROUTER_MODEL ?? 'google/gemma-4-27b-it:free'

const RERANK_MAX_CANDIDATES = 12
const RERANK_CONTENT_LIMIT  = 200
const CONTRADICT_MAX_CANDIDATES = 8
const CONTRADICT_CONTENT_LIMIT  = 250

// ─── HTTP helper ─────────────────────────────────────────────────────────────

async function chatComplete(systemPrompt: string, userPrompt: string): Promise<string> {
  if (!OPENROUTER_API_KEY) {
    throw new Error('OPENROUTER_API_KEY no configurada')
  }

  const res = await fetch(`${OPENROUTER_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
      'HTTP-Referer': 'https://github.com/cortex-memory-mcp',
      'X-Title': 'CORTEX Memory MCP',
    },
    body: JSON.stringify({
      model: OPENROUTER_MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user',   content: userPrompt   },
      ],
      temperature: 0.1,
      max_tokens: 512,
    }),
  })

  if (!res.ok) {
    const body = await res.text()
    throw new Error(`OpenRouter error ${res.status}: ${body}`)
  }

  const data = await res.json() as {
    choices: Array<{ message: { content: string } }>
  }
  return data.choices?.[0]?.message?.content ?? ''
}

/** Extrae el primer bloque JSON `{...}` de un string. */
function extractObject(text: string): string | null {
  const m = text.match(/\{[\s\S]*?\}/)
  return m ? m[0] : null
}

/** Extrae el primer bloque JSON `[...]` de un string. */
function extractArray(text: string): string | null {
  const m = text.match(/\[[\s\S]*\]/)
  return m ? m[0] : null
}

// ─── Score & Tag ──────────────────────────────────────────────────────────────

export async function scoreAndTagOR(content: string): Promise<ScoreResult> {
  try {
    const system = 'Eres un clasificador de hechos técnicos. Responde ÚNICAMENTE con JSON válido, sin texto adicional, sin markdown.'
    const user   = `Analiza este hecho y devuelve JSON:

Hecho: "${content}"

JSON requerido:
{
  "importance": <número 1-10, donde 10=decisión arquitectónica crítica, 1=comentario trivial>,
  "type": <una de: DECISION, CONTEXT, PREFERENCE, FACT, ERROR, PATTERN>,
  "tags": [<3 a 5 keywords técnicos en minúsculas>]
}`

    const raw    = await chatComplete(system, user)
    const json   = extractObject(raw)
    if (!json) throw new Error('No JSON in response')

    const parsed = JSON.parse(json) as ScoreResult
    return {
      importance: Math.min(10, Math.max(1, Math.round(parsed.importance ?? 5))),
      type:       parsed.type ?? 'FACT',
      tags:       Array.isArray(parsed.tags) ? parsed.tags.slice(0, 5) : [],
    }
  } catch {
    return { importance: 5, type: 'FACT', tags: [] }
  }
}

// ─── Batch Score & Tag ────────────────────────────────────────────────────────

export async function batchScoreAndTagOR(contents: string[]): Promise<ScoreResult[]> {
  if (contents.length === 0) return []
  if (contents.length === 1) return [await scoreAndTagOR(contents[0])]

  try {
    const system = 'Eres un clasificador de memorias técnicas. Responde ÚNICAMENTE con un array JSON válido, sin texto adicional, sin markdown.'
    const user   = `Analiza estas ${contents.length} memorias y clasifica cada una:

${contents.map((m, i) => `${i + 1}. "${m.replace(/"/g, "'")}"`).join('\n')}

Array JSON requerido (exactamente ${contents.length} elementos, mismo orden):
[
  {"importance": <1-10>, "type": <"DECISION"|"CONTEXT"|"PREFERENCE"|"FACT"|"ERROR"|"PATTERN">, "tags": [<3-5 keywords>]},
  ...
]`

    const raw    = await chatComplete(system, user)
    const json   = extractArray(raw)
    if (!json) throw new Error('No JSON array in batch response')

    const parsed = JSON.parse(json) as ScoreResult[]
    if (!Array.isArray(parsed)) throw new Error('Response is not an array')

    return contents.map((_, i) => {
      const r = parsed[i]
      if (!r) return { importance: 5, type: 'FACT' as const, tags: [] }
      return {
        importance: Math.min(10, Math.max(1, Math.round(r.importance ?? 5))),
        type:       r.type ?? 'FACT',
        tags:       Array.isArray(r.tags) ? r.tags.slice(0, 5) : [],
      }
    })
  } catch {
    return Promise.all(contents.map(c => scoreAndTagOR(c)))
  }
}

// ─── Reranking ────────────────────────────────────────────────────────────────

export async function rerankWithOpenRouter(
  query: string,
  candidates: RerankCandidate[],
): Promise<RerankResult[]> {
  if (candidates.length === 0) return []
  if (candidates.length <= 2) {
    return candidates.map(c => ({ id: c.id, rerankScore: 0.5 }))
  }

  const batch = candidates.slice(0, RERANK_MAX_CANDIDATES)
  const candidateList = batch
    .map((c, i) => `${i + 1}. "${c.content.slice(0, RERANK_CONTENT_LIMIT).replace(/"/g, "'")}"`)
    .join('\n')

  try {
    const system = 'Eres un sistema de reranking de memoria. Responde ÚNICAMENTE con JSON válido, sin texto adicional.'
    const user   = `Evalúa qué tan relevante es cada fragmento de memoria para la consulta dada.

Consulta: "${query.slice(0, 300)}"

Fragmentos de memoria:
${candidateList}

Asigna un score entre 0.0 y 1.0 a cada fragmento:
- 1.0 = directamente responde o es crucial para la consulta
- 0.7 = muy relacionado, aporta contexto valioso
- 0.4 = parcialmente relacionado
- 0.1 = poco relevante
- 0.0 = completamente irrelevante

Responde ÚNICAMENTE con JSON válido:
{"scores": [<score_1>, <score_2>, ...]}

El array debe tener exactamente ${batch.length} números.`

    const raw    = await chatComplete(system, user)
    const json   = extractObject(raw)
    if (!json) throw new Error('No JSON in rerank response')

    const parsed = JSON.parse(json) as { scores: number[] }
    const scores = Array.isArray(parsed.scores) ? parsed.scores : []

    return batch.map((c, i) => ({
      id:          c.id,
      rerankScore: typeof scores[i] === 'number'
        ? Math.min(1, Math.max(0, scores[i]))
        : 0.5,
    }))
  } catch {
    return batch.map(c => ({ id: c.id, rerankScore: 0.5 }))
  }
}

// ─── Contradiction Detection ──────────────────────────────────────────────────

export async function detectContradictionsOR(
  newContent: string,
  candidates: RerankCandidate[],
): Promise<ContradictionResult> {
  if (candidates.length === 0) return { supersededIds: [] }

  const batch = candidates.slice(0, CONTRADICT_MAX_CANDIDATES)
  const candidateList = batch
    .map((c, i) => `${i + 1}. [ID:${c.id}] "${c.content.slice(0, CONTRADICT_CONTENT_LIMIT).replace(/"/g, "'")}"`)
    .join('\n')

  try {
    const system = 'Eres un sistema de gestión de memoria para un agente IA. Tu tarea es detectar contradicciones. Responde ÚNICAMENTE con JSON válido.'
    const user   = `Determina si el nuevo hecho CONTRADICE o HACE OBSOLETO alguno de los hechos previos.

Nuevo hecho: "${newContent.slice(0, 400)}"

Hechos previos:
${candidateList}

Reglas:
- SOLO marca como contradictorios hechos que el nuevo hecho reemplaza, revierte o hace falsos.
- Si el hecho previo es complementario o similar → NO es contradictorio.
- Contradicción: "usamos TypeORM" vs "migramos a Prisma"
- No-contradicción: "usamos Vue" y "añadimos Pinia a Vue"

Responde ÚNICAMENTE con JSON:
{"contradicts": [<índices base-1 de hechos contradictorios>]}

Si no hay contradicciones: {"contradicts": []}`

    const raw    = await chatComplete(system, user)
    const json   = extractObject(raw)
    if (!json) throw new Error('No JSON in contradiction response')

    const parsed = JSON.parse(json) as { contradicts: number[] }
    const indices = Array.isArray(parsed.contradicts) ? parsed.contradicts : []

    const supersededIds = indices
      .filter(i => typeof i === 'number' && i >= 1 && i <= batch.length)
      .map(i => batch[i - 1].id)

    return { supersededIds }
  } catch {
    return { supersededIds: [] }
  }
}

// ─── Generate Text ────────────────────────────────────────────────────────────

export async function generateTextOR(prompt: string): Promise<string> {
  return chatComplete('Eres un asistente técnico de memoria.', prompt)
}

// ─── Health check ─────────────────────────────────────────────────────────────

export function isOpenRouterConfigured(): boolean {
  return Boolean(OPENROUTER_API_KEY)
}

export { type ScoreResult, type RerankCandidate, type RerankResult, type ContradictionResult }
