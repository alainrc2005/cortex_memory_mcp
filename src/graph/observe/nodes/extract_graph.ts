import { generateText } from '../../../services/ollama.js'
import { upsertEntities, upsertRelations, ensureSchema } from '../../../services/kuzu.js'
import type { ObserveState } from '../state.js'

export interface ExtractedGraph {
  entities: Array<{ name: string; type: string }>
  relations: Array<{ from: string; to: string; relType: string; label: string }>
}

/**
 * Nodo extract_graph: extrae entidades y relaciones del contenido del engrama
 * usando qwen3:8b y las persiste en Kuzu.
 *
 * Corre DESPUÉS de persist para no bloquear el almacenamiento principal.
 * Si falla → log + continúa sin interrumpir el pipeline.
 */
export async function extractGraphNode(state: ObserveState): Promise<Partial<ObserveState>> {
  const { content, projectName, engramaId } = state

  try {
    await ensureSchema()

    const prompt = `Analiza este hecho técnico y extrae entidades y relaciones en JSON.

Hecho: "${content}"

Tipos de entidad válidos: Technology, Module, Person, Config, Concept, Error, Decision
Tipos de relación válidos: USA, DEPENDE_DE, CAUSA, RESUELVE, CONTRADICE, DEFINE, CORRE_EN, REEMPLAZA

Responde SOLO con JSON válido, sin explicaciones:
{
  "entities": [
    { "name": "NombreExacto", "type": "TipoEntidad" }
  ],
  "relations": [
    { "from": "EntidadA", "to": "EntidadB", "relType": "TIPO", "label": "descripción corta" }
  ]
}

Reglas:
- Solo entidades concretas y relevantes (máx 5 entidades, máx 4 relaciones)
- Nombres cortos y precisos (ej: "PostgreSQL", "reactor-netty", "JWT")
- Si no hay entidades claras, devuelve {"entities":[],"relations":[]}`

    const raw = await generateText(prompt)

    // Extraer JSON de la respuesta (puede venir con markdown)
    const jsonMatch = raw.match(/\{[\s\S]*\}/)
    if (!jsonMatch) return {}

    const extracted: ExtractedGraph = JSON.parse(jsonMatch[0])

    // Validar estructura mínima
    if (!Array.isArray(extracted.entities) || !Array.isArray(extracted.relations)) return {}

    // Persistir en Kuzu
    await upsertEntities(extracted.entities.map(e => ({ ...e, projectName })), projectName)
    await upsertRelations(
      extracted.relations.map(r => ({ ...r, projectName, engramaId })),
      projectName,
    )

    process.stderr.write(
      `[kuzu] ${extracted.entities.length} entidades, ${extracted.relations.length} relaciones → proyecto ${projectName}\n`
    )
  } catch (err) {
    // Fallo silencioso — el grafo es opcional, no debe romper observe
    process.stderr.write(`[kuzu] extract_graph error (no crítico): ${err}\n`)
  }

  return {}  // No modifica el estado del pipeline
}
