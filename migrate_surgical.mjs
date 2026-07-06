#!/usr/bin/env node
/**
 * migrate_surgical.mjs — Migración Quirúrgica work_memories → cortex_*
 *
 * Fase 1: Migrar memorias con projectName etiquetado (114 puntos)
 * Fase 2: Clasificar por keywords y migrar las clasificables (~1,400)
 * Fase 3: Purga total del remanente (~15K)
 *
 * Uso:
 *   node migrate_surgical.mjs --dry-run     # solo muestra qué haría
 *   node migrate_surgical.mjs --phase 1     # ejecuta solo fase 1
 *   node migrate_surgical.mjs --phase 2     # ejecuta solo fase 2
 *   node migrate_surgical.mjs --phase 3     # ejecuta solo fase 3 (purga)
 *   node migrate_surgical.mjs --all         # ejecuta todo
 */

import { QdrantClient } from '@qdrant/js-client-rest'
import dotenv from 'dotenv'
dotenv.config()

const DRY_RUN = process.argv.includes('--dry-run')
const RUN_ALL = process.argv.includes('--all')
const PHASE_ARG = process.argv.find((a, i) => process.argv[i - 1] === '--phase')
const PHASE = PHASE_ARG ? parseInt(PHASE_ARG) : (RUN_ALL ? 0 : null)

const VECTOR_SIZE = 384
const LEGACY = 'work_memories'
const SPARSE_NAME = 'bm25'

const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL || 'http://localhost:6333',
  apiKey: process.env.QDRANT_API_KEY,
})

// ─── Mapeo de projectName → colección destino ────────────────────────────────

const PROJECT_MAP = {
  'cortex':                'cortex_cortex',
  'school':                'cortex_school',
  'SGA':                   'cortex_school',           // fusionar con school
  'rentetrix':             'cortex_rentetrix',         // nueva
  'tickets-venue-editor':  'cortex_tickets_venue_editor',
  'hotetec':               'cortex_hotetec',
}

const DELETE_PROJECTS = ['test-cortex']

// ─── Keywords para clasificación de memorias sin projectName ─────────────────

// Cada regla usa regex word-boundary para evitar falsos positivos
// (e.g. "cat" matcheaba con "indicates", "compilation", etc.)
// Orden: más específicos primero, más genéricos después
const KEYWORD_RULES = [
  {
    target: 'cortex_cortex',
    projectName: 'cortex',
    // Todas son bastante específicas del dominio CORTEX
    keywords: [
      'cortex', 'memory-mcp', 'qdrant', 'engrama', 'fastembed', 'langgraph',
      'work_memories', 'temp_memories', 'cortex_status', 'operator profile',
      'memory manager', 'quick_observe', 'cross-encoder',
      'cortex-dashboard', 'decay score',
    ],
  },
  {
    target: 'cortex_pets',
    projectName: 'pets',
    // Solo keywords inequívocas — eliminados 'cat', 'dog', 'pet', 'animal'
    keywords: [
      'mascota', 'mascotas', 'perro', 'perros', 'veterinario', 'veterinaria',
      'vacuna', 'vacunas', 'pets app', 'pets module',
    ],
  },
  {
    target: 'cortex_rentetrix',
    projectName: 'rentetrix',
    keywords: [
      'rentetrix', 'alquiler', 'inquilino', 'arrendamiento',
      'inmueble', 'inmuebles',
    ],
  },
  {
    target: 'cortex_hotetec',
    projectName: 'hotetec',
    // Eliminados 'room', 'rate', 'guest' (demasiado genéricos)
    keywords: [
      'hotetec', 'hotel', 'reserva', 'reservation', 'booking',
      'habitacion', 'habitación', 'check-in', 'checkout', 'hospedaje',
      'tarifa', 'disponibilidad', 'huésped', 'huesped',
      'channel manager', 'folio',
    ],
  },
  {
    target: 'cortex_school',
    projectName: 'school',
    // Eliminados 'subject', 'schedule', 'notas' (ambiguos)
    keywords: [
      'school', 'escuela', 'calificacion', 'calificación',
      'estudiante', 'enrollment', 'matrícula', 'matricula',
      'docente', 'asignatura', 'aula', 'classroom',
      'boletin', 'boletín', 'trimestre', 'semestre', 'executeCmd',
    ],
  },
  {
    target: 'cortex_tickets_venue_editor',
    projectName: 'tickets-venue-editor',
    // Eliminados 'event', 'editor' (demasiado genéricos)
    keywords: [
      'venue-editor', 'venue editor', 'tickets-venue',
      'asiento', 'entrada',
    ],
  },
]

// ─── Utilidades ──────────────────────────────────────────────────────────────

function collectionName(projectName) {
  const safe = (projectName || 'global').toLowerCase().replace(/[^a-z0-9]/g, '_')
  return `cortex_${safe}`
}

async function ensureCollection(name) {
  try {
    await qdrant.getCollection(name)
  } catch {
    await qdrant.createCollection(name, {
      vectors: { size: VECTOR_SIZE, distance: 'Cosine' },
      sparse_vectors: {
        [SPARSE_NAME]: { index: { on_disk: false }, modifier: 'idf' },
      },
    })
    log(`  ✅ Colección creada: ${name}`)
  }
}

async function scrollAll(collection, filter, withVector = true) {
  const all = []
  let offset = null
  while (true) {
    const result = await qdrant.scroll(collection, {
      limit: 100,
      with_payload: true,
      with_vector: withVector,
      filter: filter || undefined,
      offset,
    })
    all.push(...result.points)
    if (!result.next_page_offset) break
    offset = result.next_page_offset
  }
  return all
}

function log(msg) {
  const prefix = DRY_RUN ? '[DRY-RUN] ' : ''
  console.log(`${prefix}${msg}`)
}

function classifyByKeywords(content) {
  const lower = content.toLowerCase()
  for (const rule of KEYWORD_RULES) {
    for (const kw of rule.keywords) {
      // Multi-word keywords: substring match is fine (e.g. "channel manager")
      // Single-word keywords: use word boundary regex to avoid false positives
      if (kw.includes(' ')) {
        if (lower.includes(kw)) return rule
      } else {
        const regex = new RegExp(`\\b${kw.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&')}\\b`, 'i')
        if (regex.test(lower)) return rule
      }
    }
  }
  return null
}

// ─── Fase 1: Migrar memorias con projectName etiquetado ─────────────────────

async function phase1() {
  log('\n══════════════════════════════════════════════════════')
  log('  FASE 1: Migrar memorias etiquetadas')
  log('══════════════════════════════════════════════════════\n')

  const stats = { migrated: 0, deleted: 0, skipped: 0 }

  // 1a. Borrar test-cortex
  for (const proj of DELETE_PROJECTS) {
    const points = await scrollAll(LEGACY, {
      must: [{ key: 'projectName', match: { value: proj } }],
    }, false)
    log(`  🗑️  ${proj}: ${points.length} memorias para borrar`)
    if (!DRY_RUN && points.length > 0) {
      const ids = points.map(p => String(p.id))
      await qdrant.delete(LEGACY, { wait: true, points: ids })
      stats.deleted += ids.length
    }
  }

  // 1b. Migrar por projectName
  for (const [srcProject, targetCol] of Object.entries(PROJECT_MAP)) {
    const points = await scrollAll(LEGACY, {
      must: [{ key: 'projectName', match: { value: srcProject } }],
    }, true)

    if (points.length === 0) {
      log(`  ⏭️  ${srcProject}: 0 memorias — skip`)
      continue
    }

    log(`  📦 ${srcProject} → ${targetCol}: ${points.length} memorias`)

    if (!DRY_RUN) {
      await ensureCollection(targetCol)

      // Migrar en batches de 50
      for (let i = 0; i < points.length; i += 50) {
        const batch = points.slice(i, i + 50)
        const upsertPoints = batch.map(p => {
          const payload = { ...(p.payload || {}) }
          // Normalizar projectName al del destino
          const mappedProject = srcProject === 'SGA' ? 'school' : srcProject
          payload.projectName = mappedProject
          // Marcar como migrado del legado
          payload.migratedFrom = 'work_memories'
          payload.migratedAt = Date.now()

          // Manejar vectores: pueden ser named o unnamed
          let vector = p.vector
          if (vector && typeof vector === 'object' && !Array.isArray(vector)) {
            // Named vectors — extraer el dense ('' key)
            vector = vector[''] || Object.values(vector)[0]
          }
          if (!vector || (Array.isArray(vector) && vector.every(v => v === 0))) {
            // Vector dummy/cero — no podemos migrar sin re-embedear
            log(`    ⚠️  ${p.id} tiene vector cero — marcando para re-embed`)
            payload._needsReembed = true
          }

          return {
            id: String(p.id),
            vector: vector || new Array(VECTOR_SIZE).fill(0),
            payload,
          }
        })

        await qdrant.upsert(targetCol, { wait: true, points: upsertPoints })
        stats.migrated += batch.length
      }

      // Borrar del legado
      const ids = points.map(p => String(p.id))
      await qdrant.delete(LEGACY, { wait: true, points: ids })
      log(`    ✅ ${points.length} migradas y borradas del legado`)
    } else {
      stats.migrated += points.length
    }
  }

  log(`\n  📊 Fase 1 completada: ${stats.migrated} migradas, ${stats.deleted} borradas`)
  return stats
}

// ─── Fase 2: Clasificar y migrar memorias sin projectName ───────────────────

async function phase2() {
  log('\n══════════════════════════════════════════════════════')
  log('  FASE 2: Clasificar por keywords y migrar')
  log('══════════════════════════════════════════════════════\n')

  const stats = {}
  let total = 0
  let classified = 0
  let unclassified = 0

  // Scroll de todas las memorias sin projectName
  log('  🔍 Leyendo memorias sin projectName...')
  const points = await scrollAll(LEGACY, {
    must: [{ is_empty: { key: 'projectName' } }],
  }, true)

  log(`  📊 Total sin projectName: ${points.length}`)

  // Clasificar
  const buckets = {} // targetCol → [points]
  const unclassifiedPoints = []

  for (const p of points) {
    total++
    const content = (p.payload?.content || '') + ' ' + JSON.stringify(p.payload?.tags || [])
    const rule = classifyByKeywords(content)

    if (rule) {
      classified++
      if (!buckets[rule.target]) buckets[rule.target] = { projectName: rule.projectName, points: [] }
      buckets[rule.target].points.push(p)
    } else {
      unclassified++
      unclassifiedPoints.push(p)
    }
  }

  log(`  📊 Clasificadas: ${classified} | Sin clasificar: ${unclassified}`)
  log('')

  // Migrar por bucket
  for (const [targetCol, bucket] of Object.entries(buckets)) {
    const count = bucket.points.length
    stats[targetCol] = count
    log(`  📦 ${targetCol}: ${count} memorias a migrar`)

    if (!DRY_RUN && count > 0) {
      await ensureCollection(targetCol)

      // Migrar en batches de 50
      for (let i = 0; i < bucket.points.length; i += 50) {
        const batch = bucket.points.slice(i, i + 50)
        const upsertPoints = batch.map(p => {
          const payload = { ...(p.payload || {}) }
          payload.projectName = bucket.projectName
          payload.migratedFrom = 'work_memories'
          payload.migratedAt = Date.now()
          payload.tags = [...(payload.tags || []), 'LEGACY']

          let vector = p.vector
          if (vector && typeof vector === 'object' && !Array.isArray(vector)) {
            vector = vector[''] || Object.values(vector)[0]
          }
          if (!vector || (Array.isArray(vector) && vector.every(v => v === 0))) {
            payload._needsReembed = true
          }

          return {
            id: String(p.id),
            vector: vector || new Array(VECTOR_SIZE).fill(0),
            payload,
          }
        })

        await qdrant.upsert(targetCol, { wait: true, points: upsertPoints })
        process.stdout.write(`    [${i + batch.length}/${count}]\r`)
      }

      // Borrar del legado
      const ids = bucket.points.map(p => String(p.id))
      for (let i = 0; i < ids.length; i += 100) {
        await qdrant.delete(LEGACY, { wait: true, points: ids.slice(i, i + 100) })
      }
      log(`    ✅ ${count} migradas a ${targetCol}`)
    }
  }

  log(`\n  📊 Fase 2 completada: ${classified} clasificadas, ${unclassified} sin clasificar (quedan en work_memories)`)
  return { classified, unclassified, stats }
}

// ─── Fase 3: Purga total del remanente ──────────────────────────────────────

async function phase3() {
  log('\n══════════════════════════════════════════════════════')
  log('  FASE 3: Purga total de work_memories')
  log('══════════════════════════════════════════════════════\n')

  // Contar lo que queda
  const info = await qdrant.getCollection(LEGACY)
  const remaining = info.points_count
  log(`  📊 Puntos restantes en work_memories: ${remaining}`)

  if (remaining === 0) {
    log('  ✅ work_memories ya está vacía — nada que purgar')
    return { purged: 0 }
  }

  if (!DRY_RUN) {
    // Borrar en batches usando scroll + delete
    let purged = 0
    while (true) {
      const result = await qdrant.scroll(LEGACY, {
        limit: 100,
        with_payload: false,
        with_vector: false,
      })

      if (result.points.length === 0) break

      const ids = result.points.map(p => String(p.id))
      await qdrant.delete(LEGACY, { wait: true, points: ids })
      purged += ids.length
      process.stdout.write(`    Purgadas: ${purged}/${remaining}\r`)
    }

    log(`\n  🗑️  ${purged} memorias purgadas de work_memories`)

    // Verificar
    const finalInfo = await qdrant.getCollection(LEGACY)
    log(`  📊 Puntos finales en work_memories: ${finalInfo.points_count}`)

    return { purged }
  } else {
    log(`  🗑️  Se purgarían ${remaining} memorias`)
    return { purged: remaining }
  }
}

// ─── Verificación final ─────────────────────────────────────────────────────

async function verify() {
  log('\n══════════════════════════════════════════════════════')
  log('  VERIFICACIÓN FINAL')
  log('══════════════════════════════════════════════════════\n')

  const collections = await qdrant.getCollections()
  const relevant = collections.collections
    .filter(c => c.name.startsWith('cortex_') || c.name === LEGACY)

  for (const c of relevant) {
    const info = await qdrant.getCollection(c.name)
    const emoji = c.name === LEGACY ? '🔴' : '✅'
    log(`  ${emoji} ${c.name}: ${info.points_count} puntos`)
  }
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  log('╔══════════════════════════════════════════════════════╗')
  log('║  MIGRACIÓN QUIRÚRGICA work_memories → cortex_*      ║')
  log('╚══════════════════════════════════════════════════════╝')

  if (DRY_RUN) log('\n⚠️  MODO DRY-RUN — no se ejecutará ningún cambio\n')

  if (PHASE === null) {
    console.error('\nUso:')
    console.error('  node migrate_surgical.mjs --dry-run     # ver qué haría')
    console.error('  node migrate_surgical.mjs --phase 1     # solo etiquetadas')
    console.error('  node migrate_surgical.mjs --phase 2     # solo keywords')
    console.error('  node migrate_surgical.mjs --phase 3     # solo purga')
    console.error('  node migrate_surgical.mjs --all         # todo')
    process.exit(1)
  }

  try {
    if (PHASE === 0 || PHASE === 1) await phase1()
    if (PHASE === 0 || PHASE === 2) await phase2()
    if (PHASE === 0 || PHASE === 3) await phase3()

    await verify()

    log('\n✅ Migración completada.')
  } catch (err) {
    console.error('\n❌ Error durante la migración:', err)
    process.exit(1)
  }
}

main()
