import { scrollAll, searchGlobal, deleteEngramas, upsertEngrama, patchPayload, collectionFor, LEGACY_COLLECTION } from '../../services/qdrant.js'
import { getEmbedding } from '../../services/fastembed.js'
import { generateText } from '../../services/ollama.js'
import { isCompressCandidate } from '../../services/decay.js'
import { v4 as uuidv4 } from 'uuid'
import type { Engrama, EngramaPayload } from '../../types/engrama.js'

const DEDUPE_THRESHOLD = 0.92  // Cosine similarity para considerar duplicados

export interface ConsolidateResult {
  scanned: number
  merged: number
  deleted: number
  repaired: number
  report: string
}

/**
 * Nodo 1: Obtiene todos los engramas de un proyecto y filtra candidatos a compresión.
 */
export async function scanNode(projectName: string): Promise<Engrama[]> {
  const all = await scrollAll(projectName)
  return all.filter(isCompressCandidate)
}

/**
 * Nodo 2: Para cada candidato, busca duplicados semánticos en Qdrant.
 * Retorna grupos de engramas duplicados.
 */
export async function findDuplicatesNode(candidates: Engrama[]): Promise<Engrama[][]> {
  const groups: Engrama[][] = []
  const processed = new Set<string>()

  for (const candidate of candidates) {
    if (processed.has(candidate.id)) continue

    const embedding = await getEmbedding(candidate.content)
    const similar = await searchGlobal(embedding, 10)

    const duplicates = similar
      .filter(h => h.score >= DEDUPE_THRESHOLD && h.id !== candidate.id && !processed.has(h.id))
      .map(h => ({ id: h.id, ...h.payload } as Engrama))

    if (duplicates.length > 0) {
      const group = [candidate, ...duplicates]
      groups.push(group)
      group.forEach(e => processed.add(e.id))
    }
  }

  return groups
}

/**
 * Nodo 3: Fusiona un grupo de engramas duplicados en uno solo usando qwen3:8b.
 */
export async function mergeGroupNode(group: Engrama[]): Promise<Engrama | null> {
  if (group.length === 0) return null
  if (group.length === 1) return group[0]

  const contents = group.map((e, i) => `${i + 1}. ${e.content}`).join('\n')
  const prompt = `Fusiona estos hechos técnicos relacionados en UNO SOLO que preserve toda la información relevante. Sé conciso. Devuelve SOLO el hecho fusionado, sin explicaciones:\n\n${contents}`

  const merged = await generateText(prompt)
  const mergedContent = merged.trim().replace(/^[-•]\s*/, '')

  // El engrama fusionado hereda la importancia máxima y los tags combinados
  const maxImportance = Math.max(...group.map(e => e.importance ?? 5))
  const allTags = [...new Set(group.flatMap(e => e.tags ?? []))]
  const allLinks = [...new Set(group.flatMap(e => e.linkedTo ?? []))]

  const now = Date.now()
  const newId = uuidv4()
  const newEmbedding = await getEmbedding(mergedContent)

  const payload: EngramaPayload = {
    content:      mergedContent,
    projectName:  group[0].projectName,
    createdAt:    now,
    importance:   maxImportance,
    accessCount:  group.reduce((sum, e) => sum + (e.accessCount ?? 0), 0),
    lastAccessed: now,
    type:         group[0].type ?? 'FACT',
    tags:         allTags.slice(0, 5),
    linkedTo:     allLinks.filter(id => !group.map(e => e.id).includes(id)),
  }

  await upsertEngrama(newId, newEmbedding, payload)
  return { id: newId, ...payload }
}

/**
 * Ejecuta el pipeline completo de consolidación para un proyecto.
 */
export async function runConsolidation(projectName: string): Promise<ConsolidateResult> {
  const candidates = await scanNode(projectName)

  if (candidates.length === 0) {
    return { scanned: 0, merged: 0, deleted: 0, repaired: 0, report: 'No hay engramas candidatos a compresión.' }
  }

  const groups = await findDuplicatesNode(candidates)
  let merged = 0
  let deleted = 0
  let repaired = 0

  // Cargar todos los engramas del proyecto una sola vez para reparar vínculos
  const allEngramas = await scrollAll(projectName)
  const projectCol = collectionFor(projectName)

  for (const group of groups) {
    const mergedEngrama = await mergeGroupNode(group)
    if (!mergedEngrama) continue

    const deletedIds = new Set(group.map(e => e.id))

    // ── Reparar vínculos rotos ───────────────────────────────────────────────
    // Buscar engramas externos que apunten a algún ID que se va a eliminar
    // y reemplazar esas referencias por el ID del engrama fusionado.
    for (const engrama of allEngramas) {
      if (deletedIds.has(engrama.id)) continue  // saltar los que se van a borrar
      const links = engrama.linkedTo ?? []
      if (!links.some(id => deletedIds.has(id))) continue

      const repairedLinks = [
        ...new Set([
          ...links.filter(id => !deletedIds.has(id)),
          mergedEngrama.id,
        ])
      ]

      // Patch en ambas colecciones: Qdrant ignora IDs inexistentes, es seguro
      await patchPayload(engrama.id, { linkedTo: repairedLinks }, projectCol)
      await patchPayload(engrama.id, { linkedTo: repairedLinks }, LEGACY_COLLECTION)
      repaired++
    }

    // ── Eliminar los originales ──────────────────────────────────────────────
    const toDelete = group.map(e => e.id)
    await deleteEngramas(toDelete)
    deleted += toDelete.length
    merged += 1
  }

  return {
    scanned: candidates.length,
    merged,
    deleted,
    repaired,
    report: `Escaneados: ${candidates.length} candidatos. Grupos fusionados: ${merged}. Engramas eliminados: ${deleted}. Vínculos reparados: ${repaired}.`,
  }
}
