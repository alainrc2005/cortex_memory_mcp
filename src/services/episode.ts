/**
 * episode.ts — Servicio de Memoria Episódica
 *
 * 100% local sin LLM:
 *   - Almacenamiento: Qdrant (colección por proyecto)
 *   - Embeddings:     fastembed ONNX (mismo modelo que work_memories)
 *   - Resumen:        concatenación de eventos (sin Ollama)
 *
 * Colección: cortex_episodes_{projectName}
 * Vector:    384d all-MiniLM-L6-v2 (ceros mientras open, embedding real al cerrar)
 */

import { QdrantClient } from '@qdrant/js-client-rest'
import * as dotenv from 'dotenv'
import { getEmbedding } from './fastembed.js'
import type { Episode, EpisodeEvent, EpisodePayload } from '../types/episode.js'

dotenv.config({ path: '/home/alainrc2005/IA/memory-mcp/.env' })

const VECTOR_SIZE = 384

const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL || 'http://localhost:6333',
  apiKey: process.env.QDRANT_API_KEY,
})

// ─── Nombre de colección ──────────────────────────────────────────────────────

export function episodeCollectionFor(projectName: string): string {
  const safe = (projectName || 'global').toLowerCase().replace(/[^a-z0-9]/g, '_')
  return `cortex_episodes_${safe}`
}

// ─── Ensure collection ────────────────────────────────────────────────────────

export async function ensureEpisodeCollection(projectName: string): Promise<string> {
  const col = episodeCollectionFor(projectName)
  try {
    await qdrant.getCollection(col)
  } catch {
    await qdrant.createCollection(col, {
      vectors: { size: VECTOR_SIZE, distance: 'Cosine' },
    })
  }
  return col
}

// ─── Crear episodio ───────────────────────────────────────────────────────────

export async function createEpisode(
  projectName: string,
  context: string,
  sessionId: string,
): Promise<Episode> {
  const col = await ensureEpisodeCollection(projectName)
  const now = Date.now()

  const episode: Episode = {
    id: sessionId,
    projectName,
    startedAt: now,
    status: 'open',
    context,
    summary: '',
    events: [],
    eventCount: 0,
  }

  // Vector cero mientras la sesión esté abierta
  const zeroVector = new Array(VECTOR_SIZE).fill(0)

  await qdrant.upsert(col, {
    wait: true,
    points: [{
      id: sessionId,
      vector: zeroVector,
      payload: episode as unknown as Record<string, unknown>,
    }],
  })

  return episode
}

// ─── Añadir evento ────────────────────────────────────────────────────────────

export async function appendEvent(
  projectName: string,
  sessionId: string,
  event: EpisodeEvent,
): Promise<Episode | null> {
  const col = episodeCollectionFor(projectName)

  // Recuperar el episodio actual
  const current = await getEpisodeById(projectName, sessionId)
  if (!current) return null

  const updatedEvents = [...current.events, event]
  const updatedSummary = buildSummary(current.context, updatedEvents)

  await qdrant.setPayload(col, {
    wait: true,
    points: [sessionId],
    payload: {
      events: updatedEvents as unknown as Record<string, unknown>[],
      eventCount: updatedEvents.length,
      summary: updatedSummary,
    },
  })

  return { ...current, events: updatedEvents, eventCount: updatedEvents.length, summary: updatedSummary }
}

// ─── Obtener sesión abierta ───────────────────────────────────────────────────

export async function getOpenSession(projectName: string): Promise<Episode | null> {
  const col = episodeCollectionFor(projectName)

  try {
    await qdrant.getCollection(col)
  } catch {
    return null  // colección no existe aún → no hay sesión
  }

  const result = await qdrant.scroll(col, {
    filter: {
      must: [
        { key: 'status', match: { value: 'open' } },
        { key: 'projectName', match: { value: projectName } },
      ],
    },
    limit: 1,
    with_payload: true,
    with_vector: false,
  })

  if (result.points.length === 0) return null

  const p = result.points[0]
  return { id: String(p.id), ...(p.payload as unknown as EpisodePayload) }
}

// ─── Cerrar episodio ──────────────────────────────────────────────────────────
// Sin LLM: el resumen es la concatenación de eventos.
// El embedding se genera con fastembed (ONNX local, sin GPU).

export async function closeEpisode(
  projectName: string,
  sessionId: string,
): Promise<Episode | null> {
  const col = episodeCollectionFor(projectName)
  const episode = await getEpisodeById(projectName, sessionId)
  if (!episode) return null

  const now = Date.now()
  const summary = buildSummary(episode.context, episode.events)

  // Generar embedding del resumen final (fastembed ONNX, sin GPU)
  const textToEmbed = `${episode.context}. ${summary}`
  const vector = textToEmbed.trim()
    ? await getEmbedding(textToEmbed)
    : new Array(VECTOR_SIZE).fill(0)

  // Upsert completo con vector real + payload actualizado
  const closedEpisode: Episode = {
    ...episode,
    endedAt: now,
    status: 'closed',
    summary,
  }

  await qdrant.upsert(col, {
    wait: true,
    points: [{
      id: sessionId,
      vector,
      payload: closedEpisode as unknown as Record<string, unknown>,
    }],
  })

  return closedEpisode
}

// ─── Buscar episodios (semántica sobre summaries cerrados) ────────────────────

export async function searchEpisodes(
  projectName: string,
  queryVector: number[],
  limit = 3,
): Promise<Episode[]> {
  const col = episodeCollectionFor(projectName)

  try {
    await qdrant.getCollection(col)
  } catch {
    return []
  }

  const results = await qdrant.search(col, {
    vector: queryVector,
    limit,
    filter: {
      must: [
        { key: 'status', match: { value: 'closed' } },
        { key: 'projectName', match: { value: projectName } },
      ],
    },
    with_payload: true,
  })

  return results.map(r => ({ id: String(r.id), ...(r.payload as unknown as EpisodePayload) }))
}

// ─── Episodios recientes (sin búsqueda semántica) ─────────────────────────────

export async function getRecentEpisodes(
  projectName: string,
  limit = 3,
): Promise<Episode[]> {
  const col = episodeCollectionFor(projectName)

  try {
    await qdrant.getCollection(col)
  } catch {
    return []
  }

  const result = await qdrant.scroll(col, {
    filter: {
      must: [
        { key: 'status', match: { value: 'closed' } },
        { key: 'projectName', match: { value: projectName } },
      ],
    },
    limit: 50,  // traer más y ordenar por fecha
    with_payload: true,
    with_vector: false,
  })

  return result.points
    .map(p => ({ id: String(p.id), ...(p.payload as unknown as EpisodePayload) }))
    .sort((a, b) => (b.startedAt ?? 0) - (a.startedAt ?? 0))
    .slice(0, limit)
}

// ─── Helpers internos ─────────────────────────────────────────────────────────

async function getEpisodeById(projectName: string, sessionId: string): Promise<Episode | null> {
  const col = episodeCollectionFor(projectName)
  try {
    const result = await qdrant.retrieve(col, {
      ids: [sessionId],
      with_payload: true,
      with_vector: false,
    })
    if (result.length === 0) return null
    return { id: String(result[0].id), ...(result[0].payload as unknown as EpisodePayload) }
  } catch {
    return null
  }
}

/**
 * buildSummary — Construye el resumen de la sesión SIN LLM.
 * Formato: "Sesión sobre <context>: [EVENT1] desc1 | [EVENT2] desc2 ..."
 */
function buildSummary(context: string, events: EpisodeEvent[]): string {
  if (events.length === 0) return `Sesión sobre: ${context}`

  const eventLines = events
    .map(e => `[${e.eventType}] ${e.description}`)
    .join(' | ')

  return `${context} → ${eventLines}`
}
