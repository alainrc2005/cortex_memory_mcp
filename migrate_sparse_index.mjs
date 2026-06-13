#!/usr/bin/env node
/**
 * migrate_sparse_index.mjs
 *
 * Migra todas las colecciones Qdrant para añadirles sparse index 'bm25'.
 *
 * Estrategia (Qdrant no permite añadir sparse vectors a colecciones existentes):
 *   1. Hacer scroll de todos los puntos (payload + vectores densos)
 *   2. Borrar la colección original
 *   3. Recrearla con sparse_vectors: { bm25: { modifier: 'idf' } }
 *   4. Re-insertar todos los puntos (sin vector sparse — se añadirán en upserts futuros)
 *
 * Flags:
 *   --dry-run   Solo muestra lo que haría, sin tocar nada
 *   --only=col  Migra solo la colección indicada (ej. --only=work_memories)
 *   --skip=col  Omite esa colección (se puede repetir)
 *
 * Uso:
 *   node migrate_sparse_index.mjs
 *   node migrate_sparse_index.mjs --dry-run
 *   node migrate_sparse_index.mjs --only=cortex_global
 */

import * as dotenv from 'dotenv'
import { QdrantClient } from '@qdrant/js-client-rest'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

const __dirname = dirname(fileURLToPath(import.meta.url))
dotenv.config({ path: join(__dirname, '.env') })

// ─── Config ───────────────────────────────────────────────────────────────────
const SPARSE_VECTOR_NAME = 'bm25'
const BATCH_SIZE = 100          // puntos por lote en upsert
const SCROLL_LIMIT = 250        // puntos por página en scroll

const args = process.argv.slice(2)
const DRY_RUN   = args.includes('--dry-run')
const ONLY_COL  = args.find(a => a.startsWith('--only='))?.slice(7)
const SKIP_COLS = args.filter(a => a.startsWith('--skip=')).map(a => a.slice(7))

// ─── Cliente ──────────────────────────────────────────────────────────────────
const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL || 'http://localhost:6333',
  apiKey: process.env.QDRANT_API_KEY,
})

// ─── Colores ──────────────────────────────────────────────────────────────────
const G = '\x1b[32m', Y = '\x1b[33m', R = '\x1b[31m', C = '\x1b[36m', B = '\x1b[1m', X = '\x1b[0m'
const ok   = msg => process.stdout.write(`${G}✓${X} ${msg}\n`)
const skip = msg => process.stdout.write(`${Y}→${X} ${msg}\n`)
const fail = msg => process.stdout.write(`${R}✗${X} ${msg}\n`)
const info = msg => process.stdout.write(`${C}ℹ${X} ${msg}\n`)
const log  = msg => process.stdout.write(msg + '\n')
const prog = msg => process.stdout.write(`\r  ${msg}              `)

// ─── Verificar sparse index ───────────────────────────────────────────────────
async function hasSparseIndex(colName) {
  const info = await qdrant.getCollection(colName)
  const sp = info.config?.params?.sparse_vectors
  return !!(sp && sp[SPARSE_VECTOR_NAME])
}

// ─── Obtener schema de la colección ──────────────────────────────────────────
async function getCollectionConfig(colName) {
  const info = await qdrant.getCollection(colName)
  return info.config?.params ?? {}
}

// ─── Scroll completo de todos los puntos ─────────────────────────────────────
async function scrollAllPoints(colName) {
  const points = []
  let offset = null
  let page = 0

  while (true) {
    page++
    const result = await qdrant.scroll(colName, {
      limit: SCROLL_LIMIT,
      offset: offset ?? undefined,
      with_payload: true,
      with_vector: true,   // necesitamos los vectores densos para re-insertar
    })

    points.push(...result.points)
    prog(`Leyendo ${colName}: ${points.length} puntos leídos...`)

    if (!result.next_page_offset) break
    offset = result.next_page_offset
  }

  process.stdout.write('\n')
  return points
}

// ─── Migrar una colección ────────────────────────────────────────────────────
async function migrateCollection(colName) {
  log(`\n${B}── ${colName} ──${X}`)

  // 1. Verificar si ya tiene sparse
  if (await hasSparseIndex(colName)) {
    skip(`${colName} — ya tiene sparse index '${SPARSE_VECTOR_NAME}'`)
    return 'skip'
  }

  // 2. Leer config
  const config = await getCollectionConfig(colName)
  const vectorsConfig = config.vectors

  // Determinar parámetros del vector denso
  let denseParams
  if (vectorsConfig && typeof vectorsConfig === 'object' && 'size' in vectorsConfig) {
    // unnamed vector (legacy format)
    denseParams = { size: vectorsConfig.size, distance: vectorsConfig.distance }
  } else if (vectorsConfig && typeof vectorsConfig === 'object') {
    // named vectors — tomamos el primero (o el vacío '')
    const keys = Object.keys(vectorsConfig)
    const key = keys.includes('') ? '' : keys[0]
    denseParams = vectorsConfig[key]
  } else {
    fail(`${colName} — no se puede determinar la configuración del vector`)
    return 'fail'
  }

  info(`  Vector denso: size=${denseParams.size}, distance=${denseParams.distance}`)

  // 3. Scroll completo
  info(`  Leyendo puntos...`)
  const points = await scrollAllPoints(colName)
  info(`  Total: ${points.length} puntos`)

  if (DRY_RUN) {
    skip(`  [DRY-RUN] Se recrearían ${points.length} puntos con sparse index`)
    return 'dryrun'
  }

  // 4. Borrar colección original
  info(`  Eliminando colección original...`)
  await qdrant.deleteCollection(colName)

  // 5. Recrear con sparse index declarado
  info(`  Recreando con sparse_vectors.${SPARSE_VECTOR_NAME} (IDF)...`)
  await qdrant.createCollection(colName, {
    vectors: { size: denseParams.size, distance: denseParams.distance },
    sparse_vectors: {
      [SPARSE_VECTOR_NAME]: {
        index: { on_disk: false },
        modifier: 'idf',
      },
    },
  })

  // 6. Re-insertar en lotes (solo vector denso, sin sparse por ahora)
  info(`  Re-insertando ${points.length} puntos en lotes de ${BATCH_SIZE}...`)
  let inserted = 0

  for (let i = 0; i < points.length; i += BATCH_SIZE) {
    const batch = points.slice(i, i + BATCH_SIZE).map(p => {
      // El vector puede venir como array (unnamed) o como objeto (named vectors)
      let vector = p.vector
      if (vector && typeof vector === 'object' && !Array.isArray(vector)) {
        // named vectors — extraer el denso (key '' o primer key)
        const keys = Object.keys(vector)
        const key = keys.includes('') ? '' : keys[0]
        vector = vector[key]
      }
      return {
        id: p.id,
        vector: vector ?? new Array(denseParams.size).fill(0),
        payload: p.payload ?? {},
      }
    })

    await qdrant.upsert(colName, { wait: true, points: batch })
    inserted += batch.length
    prog(`  Insertados: ${inserted}/${points.length}`)
  }

  process.stdout.write('\n')
  ok(`${colName} — migrado ✓ (${points.length} puntos)`)
  return 'ok'
}

// ─── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  log('')
  log(`${B}╔══════════════════════════════════════════════════════════════╗${X}`)
  log(`${B}║   Migración Sparse Index BM25 (IDF) — Qdrant Collections    ║${X}`)
  log(`${B}╚══════════════════════════════════════════════════════════════╝${X}`)
  log('')

  if (DRY_RUN) info(`${Y}[DRY RUN]${X} Solo se mostrará el plan, sin cambios reales.\n`)
  if (ONLY_COL) info(`Migrando solo: ${ONLY_COL}\n`)
  if (SKIP_COLS.length) info(`Omitiendo: ${SKIP_COLS.join(', ')}\n`)

  // Listar colecciones
  const { collections } = await qdrant.getCollections()
  let names = collections.map(c => c.name).sort()

  if (ONLY_COL) {
    names = names.filter(n => n === ONLY_COL)
    if (names.length === 0) {
      fail(`Colección '${ONLY_COL}' no encontrada`)
      process.exit(1)
    }
  }

  if (SKIP_COLS.length) {
    names = names.filter(n => !SKIP_COLS.includes(n))
  }

  info(`Colecciones a procesar: ${names.length}`)

  const stats = { ok: 0, skip: 0, fail: 0, dryrun: 0 }

  for (const name of names) {
    try {
      const result = await migrateCollection(name)
      stats[result]++
    } catch (e) {
      fail(`${name} — ERROR inesperado: ${e?.message ?? e}`)
      stats.fail++
    }
  }

  log('')
  log(`${B}─────────────────────── Resumen ───────────────────────${X}`)
  log(`  ${G}Migradas OK         :${X} ${stats.ok}`)
  log(`  ${Y}Ya tenían sparse    :${X} ${stats.skip}`)
  log(`  ${C}Dry-run (sin cambio):${X} ${stats.dryrun}`)
  log(`  ${R}Errores             :${X} ${stats.fail}`)
  log('')

  if (!DRY_RUN && stats.ok > 0) {
    info('Los puntos existentes tienen vector denso pero NO sparse.')
    info('El vector BM25 se añadirá automáticamente en el próximo upsert de cada memoria.')
    log('')
  }

  process.exit(stats.fail > 0 ? 1 : 0)
}

main().catch(e => {
  fail(`Error fatal: ${e?.message ?? e}`)
  process.exit(1)
})
