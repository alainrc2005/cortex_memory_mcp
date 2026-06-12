import type { EngramaType } from '../types/engrama.js'

const OLLAMA_BASE = process.env.OLLAMA_URL || 'http://localhost:11434'
const SCORE_MODEL = 'qwen3:8b'

// ─── Scoring / Tagging ───────────────────────────────────────────────────────

export interface ScoreResult {
  importance: number
  type: EngramaType
  tags: string[]
}

// ─── Cross-Encoder Reranking ──────────────────────────────────────────────────

export interface RerankCandidate {
  id: string
  content: string
}

export interface RerankResult {
  id: string
  rerankScore: number   // 0.0 – 1.0, mayor = más relevante para el query
}

const SCORE_PROMPT = (content: string) => `Analiza este hecho técnico y responde ÚNICAMENTE con JSON válido, sin texto adicional, sin markdown:

Hecho: "${content}"

JSON requerido:
{
  "importance": <número 1-10, donde 10=decisión arquitectónica crítica, 1=comentario trivial>,
  "type": <una de: DECISION, CONTEXT, PREFERENCE, FACT, ERROR, PATTERN>,
  "tags": [<3 a 5 keywords técnicos en minúsculas>]
}`

export async function scoreAndTag(content: string): Promise<ScoreResult> {
  try {
    const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: SCORE_MODEL,
        prompt: SCORE_PROMPT(content),
        stream: false,
        options: { temperature: 0.1 },
      }),
    })
    if (!res.ok) throw new Error(`Ollama generate error: ${res.status}`)
    const data = await res.json() as { response: string }

    // Extraer JSON del response (puede venir con texto extra del thinking)
    const jsonMatch = data.response.match(/\{[\s\S]*?\}/)
    if (!jsonMatch) throw new Error('No JSON found in response')

    const parsed = JSON.parse(jsonMatch[0]) as ScoreResult
    return {
      importance: Math.min(10, Math.max(1, Math.round(parsed.importance ?? 5))),
      type: parsed.type ?? 'FACT',
      tags: Array.isArray(parsed.tags) ? parsed.tags.slice(0, 5) : [],
    }
  } catch {
    // Fallback seguro si el LLM falla
    return { importance: 5, type: 'FACT', tags: [] }
  }
}

// ─── Batch Scoring ───────────────────────────────────────────────────────────
//
// Envía N memorias en un ÚNICO prompt → recibe array JSON con N resultados.
// Elimina N-1 roundtrips HTTP a Ollama y el overhead de prefill repetido.
// Mucho más rápido que llamar scoreAndTag() N veces secuencialmente.
//
// Fallback: si el batch falla o el JSON está malformado, cae a scoreAndTag()
// individual por cada memoria para garantizar resultados siempre.

const BATCH_SCORE_PROMPT = (memories: string[]) =>
  `Analiza estas ${memories.length} memorias técnicas y clasifica cada una. Responde ÚNICAMENTE con un array JSON válido, sin texto adicional, sin markdown.

${memories.map((m, i) => `${i + 1}. "${m.replace(/"/g, "'")}"`).join('\n')}

Array JSON requerido (exactamente ${memories.length} elementos, en el mismo orden):
[
  {"importance": <1-10>, "type": <"DECISION"|"CONTEXT"|"PREFERENCE"|"FACT"|"ERROR"|"PATTERN">, "tags": [<3-5 keywords>]},
  ...
]`

export async function batchScoreAndTag(contents: string[]): Promise<ScoreResult[]> {
  if (contents.length === 0) return []
  if (contents.length === 1) return [await scoreAndTag(contents[0])]

  try {
    const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: SCORE_MODEL,
        prompt: BATCH_SCORE_PROMPT(contents),
        stream: false,
        options: { temperature: 0.1 },
      }),
    })
    if (!res.ok) throw new Error(`Ollama batch score error: ${res.status}`)
    const data = await res.json() as { response: string }

    // Extraer array JSON — puede venir con texto extra del thinking de qwen3
    const arrayMatch = data.response.match(/\[[\s\S]*\]/)
    if (!arrayMatch) throw new Error('No JSON array in batch response')

    const parsed = JSON.parse(arrayMatch[0]) as ScoreResult[]
    if (!Array.isArray(parsed)) throw new Error('Response is not an array')

    // Normalizar cada resultado y rellenar si el array viene corto
    return contents.map((_, i) => {
      const r = parsed[i]
      if (!r) return { importance: 5, type: 'FACT' as const, tags: [] }
      return {
        importance: Math.min(10, Math.max(1, Math.round(r.importance ?? 5))),
        type: r.type ?? 'FACT',
        tags: Array.isArray(r.tags) ? r.tags.slice(0, 5) : [],
      }
    })
  } catch {
    // Fallback: scoring individual por cada memoria
    return Promise.all(contents.map(c => scoreAndTag(c)))
  }
}


// ─── Cross-Encoder Rerank ────────────────────────────────────────────────────
//
// Evalúa (query, candidato) como par bidireccional usando el LLM.
// Un solo prompt con todos los candidatos → respuesta JSON con scores.
// Más preciso que cosine similarity porque el modelo entiende la relación
// semántica contextual entre la pregunta y cada resultado.
//
// Fallback: si el LLM falla, retorna scores neutros (0.5) para no romper el recall.

const RERANK_MAX_CANDIDATES = 12   // límite para no exceder el contexto del prompt
const RERANK_CONTENT_LIMIT  = 200  // chars por candidato en el prompt

export async function rerankWithLLM(
  query: string,
  candidates: RerankCandidate[],
): Promise<RerankResult[]> {
  if (candidates.length === 0) return []

  // Si solo hay 1-2 candidatos no vale la pena la llamada al LLM
  if (candidates.length <= 2) {
    return candidates.map((c) => ({ id: c.id, rerankScore: 0.5 }))
  }

  const batch = candidates.slice(0, RERANK_MAX_CANDIDATES)

  const candidateList = batch
    .map((c, i) => `${i + 1}. "${c.content.slice(0, RERANK_CONTENT_LIMIT).replace(/"/g, "'")}"`) 
    .join('\n')

  const prompt = `Eres un sistema de reranking de memoria. Tu tarea es evaluar qué tan relevante es cada fragmento de memoria para responder la consulta dada.

Consulta: "${query.slice(0, 300)}"

Fragmentos de memoria a evaluar:
${candidateList}

Para cada fragmento, asigna un score de relevancia entre 0.0 y 1.0:
- 1.0 = directamente responde o es crucial para la consulta
- 0.7 = muy relacionado, aporta contexto valioso
- 0.4 = parcialmente relacionado
- 0.1 = poco relevante o no relacionado
- 0.0 = completamente irrelevante

Responde ÚNICAMENTE con JSON válido, sin texto adicional:
{"scores": [<score_1>, <score_2>, ...]}

El array debe tener exactamente ${batch.length} números.`

  try {
    const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: SCORE_MODEL,
        prompt,
        stream: false,
        options: { temperature: 0.0 },  // determinístico para scoring
      }),
    })

    if (!res.ok) throw new Error(`Ollama rerank error: ${res.status}`)
    const data = await res.json() as { response: string }

    // Extraer JSON (puede venir con texto extra del thinking de qwen3)
    const jsonMatch = data.response.match(/\{[\s\S]*?\}/)
    if (!jsonMatch) throw new Error('No JSON in rerank response')

    const parsed = JSON.parse(jsonMatch[0]) as { scores: number[] }
    const scores = Array.isArray(parsed.scores) ? parsed.scores : []

    return batch.map((c, i) => ({
      id: c.id,
      rerankScore: typeof scores[i] === 'number'
        ? Math.min(1, Math.max(0, scores[i]))
        : 0.5,  // fallback neutro si el LLM no cubrió este índice
    }))
  } catch {
    // Fallback: scores neutros, no rompe el recall
    return batch.map((c) => ({ id: c.id, rerankScore: 0.5 }))
  }
}

// ─── Contradiction Detection ─────────────────────────────────────────────────
//
// Dado un hecho nuevo y una lista de candidatos similares (score > 0.88),
// pregunta al LLM cuáles son contradictorios con el nuevo.
// Retorna los IDs que deben marcarse como 'superseded'.
// Fallback: array vacío si Ollama falla (conservativo — mejor no invalidar que invalidar mal).

const CONTRADICT_MAX_CANDIDATES = 8   // no sobrecargar el contexto
const CONTRADICT_CONTENT_LIMIT  = 250 // chars por candidato

export interface ContradictionResult {
  supersededIds: string[]  // IDs a marcar como superseded
}

export async function detectContradictions(
  newContent: string,
  candidates: RerankCandidate[],
): Promise<ContradictionResult> {
  if (candidates.length === 0) return { supersededIds: [] }

  const batch = candidates.slice(0, CONTRADICT_MAX_CANDIDATES)

  const candidateList = batch
    .map((c, i) => `${i + 1}. [ID:${c.id}] "${c.content.slice(0, CONTRADICT_CONTENT_LIMIT).replace(/"/g, "'")}"` )
    .join('\n')

  const prompt = `Eres un sistema de gestión de memoria para un agente IA. Tu única tarea es detectar si el nuevo hecho CONTRADICE o HACE OBSOLETO alguno de los hechos previos.

Nuevo hecho: "${newContent.slice(0, 400)}"

Hechos previos a evaluar:
${candidateList}

Reglas estrictas:
- SOLO marca como contradictorios hechos que el nuevo hecho reemplaza, revierte o hace falsos.
- Si el hecho previo es complementario, relacionado, o simplemente similar → NO es contradictorios.
- Ejemplos de CONTRADICCIÓN: "usamos TypeORM" vs "migramos a Prisma", "el puerto es 3000" vs "el puerto es 8080".
- Ejemplos de NO-CONTRADICCIÓN: "usamos Vue" y "añadimos Pinia a Vue".

Responde ÚNICAMENTE con JSON válido, sin texto adicional:
{"contradicts": [<índices de los hechos que son contradictorios, base-1>]}

Si no hay contradicciones, responde: {"contradicts": []}`

  try {
    const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: SCORE_MODEL,
        prompt,
        stream: false,
        options: { temperature: 0.0 },  // determinístico: vida o muerte de un engrama
      }),
    })

    if (!res.ok) throw new Error(`Ollama contradiction error: ${res.status}`)
    const data = await res.json() as { response: string }

    const jsonMatch = data.response.match(/\{[\s\S]*?\}/)
    if (!jsonMatch) throw new Error('No JSON in contradiction response')

    const parsed = JSON.parse(jsonMatch[0]) as { contradicts: number[] }
    const indices = Array.isArray(parsed.contradicts) ? parsed.contradicts : []

    // Convertir índices base-1 a IDs reales
    const supersededIds = indices
      .filter(i => typeof i === 'number' && i >= 1 && i <= batch.length)
      .map(i => batch[i - 1].id)

    return { supersededIds }
  } catch {
    // Fallback conservativo: no invalidar nada si el LLM falla
    return { supersededIds: [] }
  }
}


export async function generateText(prompt: string): Promise<string> {
  const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: SCORE_MODEL,
      prompt,
      stream: false,
      options: { temperature: 0.3 },
    }),
  })
  if (!res.ok) throw new Error(`Ollama generate error: ${res.status}`)
  const data = await res.json() as { response: string }
  return data.response
}
