import kuzu from 'kuzu'
import * as path from 'path'
import * as fs from 'fs'

// ─── Configuración ────────────────────────────────────────────────────────────

const DB_PATH = process.env.CORTEX_KUZU_DIR ?? path.resolve(process.cwd(), 'kuzu_db')

// Singleton: una sola instancia de DB/Connection por proceso
let _db: kuzu.Database | null = null
let _conn: kuzu.Connection | null = null

function getConnection(): kuzu.Connection {
  if (!_conn) {
    if (!fs.existsSync(DB_PATH)) fs.mkdirSync(DB_PATH, { recursive: true })
    _db   = new kuzu.Database(DB_PATH)
    _conn = new kuzu.Connection(_db)
  }
  return _conn
}

// ─── Schema ───────────────────────────────────────────────────────────────────
//
// Nodos:
//   Entity(name, type, projectName, createdAt, updatedAt)
//
// Relaciones:
//   RELATED_TO(from Entity, to Entity, relType, label, projectName, engramaId, createdAt)
//
// Tipos de entidad: Technology, Person, Module, Config, Concept, Error, Decision
// Tipos de relación: USA, DEPENDE_DE, CAUSA, RESUELVE, CONTRADICE, DEFINE, CORRE_EN

export async function ensureSchema(): Promise<void> {
  const conn = getConnection()

  // Nodos
  await conn.query(`
    CREATE NODE TABLE IF NOT EXISTS Entity (
      name       STRING,
      type       STRING,
      projectName STRING,
      createdAt  INT64,
      updatedAt  INT64,
      PRIMARY KEY (name, projectName)
    )
  `).catch(() => {}) // Ignora si ya existe

  // Relaciones
  await conn.query(`
    CREATE REL TABLE IF NOT EXISTS RELATED_TO (
      FROM Entity TO Entity,
      relType   STRING,
      label     STRING,
      projectName STRING,
      engramaId STRING,
      createdAt INT64
    )
  `).catch(() => {})
}

// ─── Tipos ────────────────────────────────────────────────────────────────────

export interface KuzuEntity {
  name: string
  type: string
  projectName: string
}

export interface KuzuRelation {
  from: string
  to: string
  relType: string
  label: string
  projectName: string
  engramaId: string
}

// ─── Upsert entidades y relaciones ───────────────────────────────────────────

export async function upsertEntities(
  entities: KuzuEntity[],
  projectName: string,
): Promise<void> {
  if (entities.length === 0) return
  const conn = getConnection()
  const now  = Date.now()

  for (const e of entities) {
    // MERGE manual: intentar insert, si falla (PK duplicada) hacer update
    await conn.query(`
      MERGE (n:Entity {name: '${esc(e.name)}', projectName: '${esc(projectName)}'})
      ON CREATE SET n.type = '${esc(e.type)}', n.createdAt = ${now}, n.updatedAt = ${now}
      ON MATCH  SET n.updatedAt = ${now}
    `).catch(async () => {
      // Fallback si MERGE no está disponible en esta versión
      await conn.query(`
        CREATE (n:Entity {
          name: '${esc(e.name)}',
          type: '${esc(e.type)}',
          projectName: '${esc(projectName)}',
          createdAt: ${now},
          updatedAt: ${now}
        })
      `).catch(() => {}) // Ignora duplicados
    })
  }
}

export async function upsertRelations(
  relations: KuzuRelation[],
  projectName: string,
): Promise<void> {
  if (relations.length === 0) return
  const conn = getConnection()
  const now  = Date.now()

  for (const r of relations) {
    await conn.query(`
      MATCH (a:Entity {name: '${esc(r.from)}', projectName: '${esc(projectName)}'}),
            (b:Entity {name: '${esc(r.to)}',   projectName: '${esc(projectName)}'})
      CREATE (a)-[:RELATED_TO {
        relType:     '${esc(r.relType)}',
        label:       '${esc(r.label)}',
        projectName: '${esc(projectName)}',
        engramaId:   '${esc(r.engramaId)}',
        createdAt:   ${now}
      }]->(b)
    `).catch(() => {}) // Ignora si los nodos no existen aún
  }
}

// ─── Queries ─────────────────────────────────────────────────────────────────

/** Vecinos directos de una entidad (1 hop) */
export async function queryNeighbors(
  entityName: string,
  projectName: string,
  limit = 10,
): Promise<Array<{ neighbor: string; relType: string; label: string; direction: string }>> {
  const conn = getConnection()

  const outRes = await conn.query(`
    MATCH (a:Entity {name: '${esc(entityName)}', projectName: '${esc(projectName)}'})-[r:RELATED_TO]->(b:Entity)
    RETURN b.name AS neighbor, r.relType AS relType, r.label AS label, 'out' AS direction
    LIMIT ${limit}
  `).catch(() => null)

  const inRes = await conn.query(`
    MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity {name: '${esc(entityName)}', projectName: '${esc(projectName)}'})
    RETURN a.name AS neighbor, r.relType AS relType, r.label AS label, 'in' AS direction
    LIMIT ${limit}
  `).catch(() => null)

  const rows: Array<{ neighbor: string; relType: string; label: string; direction: string }> = []
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if (outRes) rows.push(...(await (outRes as any).getAll() as typeof rows))
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if (inRes)  rows.push(...(await (inRes  as any).getAll() as typeof rows))
  return rows
}

/** Timeline de una entidad: todas las relaciones ordenadas por createdAt */
export async function queryTimeline(
  entityName: string,
  projectName: string,
  limit = 20,
): Promise<Array<{ neighbor: string; relType: string; label: string; createdAt: number }>> {
  const conn = getConnection()

  const res = await conn.query(`
    MATCH (a:Entity {name: '${esc(entityName)}', projectName: '${esc(projectName)}'})-[r:RELATED_TO]->(b:Entity)
    RETURN b.name AS neighbor, r.relType AS relType, r.label AS label, r.createdAt AS createdAt
    ORDER BY r.createdAt DESC
    LIMIT ${limit}
  `).catch(() => null)

  if (!res) return []
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return await (res as any).getAll() as Array<{ neighbor: string; relType: string; label: string; createdAt: number }>
}

/** Cypher directo — para graph_query */
export async function rawQuery(
  cypher: string,
): Promise<unknown[]> {
  const conn = getConnection()
  const res  = await conn.query(cypher)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return await (res as any).getAll()
}

/** Contexto de grafo para una query semántica: entidades mencionadas y sus vecinos */
export async function queryGraphContext(
  keywords: string[],
  projectName: string,
  limit = 5,
): Promise<string> {
  if (keywords.length === 0) return ''
  const conn = getConnection()

  const conditions = keywords
    .map(k => `n.name CONTAINS '${esc(k)}'`)
    .join(' OR ')

  const res = await conn.query(`
    MATCH (n:Entity {projectName: '${esc(projectName)}'})-[r:RELATED_TO]->(m:Entity)
    WHERE ${conditions}
    RETURN n.name AS from, r.relType AS relType, m.name AS to, r.label AS label
    LIMIT ${limit * 3}
  `).catch(() => null)

  if (!res) return ''
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const rows = await (res as any).getAll() as Array<{ neighbor: string; relType: string; to: string; label: string; direction: string }>
  if (rows.length === 0) return ''

  const lines = rows.map(r => {
    const arrow = r.direction === 'out' ? `→ ${r.neighbor}` : `← ${r.neighbor}`
    return `• [${r.relType}] ${arrow}${r.label ? ` — ${r.label}` : ''}`
  })
  return `## 🕸️ Grafo relacionado\n${lines.join('\n')}`
}

/** Listado de entidades del proyecto */
export async function listEntities(
  projectName: string,
  limit = 30,
): Promise<Array<{ name: string; type: string }>> {
  const conn = getConnection()
  const res  = await conn.query(`
    MATCH (n:Entity {projectName: '${esc(projectName)}'})
    RETURN n.name AS name, n.type AS type
    ORDER BY n.updatedAt DESC
    LIMIT ${limit}
  `).catch(() => null)

  if (!res) return []
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return await (res as any).getAll() as Array<{ name: string; type: string }>
}

// ─── Util ─────────────────────────────────────────────────────────────────────

/** Escapa comillas simples para Cypher */
function esc(s: string): string {
  return s.replace(/'/g, "\\'").replace(/\\/g, '\\\\').slice(0, 200)
}
