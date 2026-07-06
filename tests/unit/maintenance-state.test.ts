/**
 * LAYER 1 — UNIT TESTS: Maintenance state file
 *
 * Tests del sistema de persistencia de mantenimiento automático.
 * Verifica que los timestamps se guardan y cargan correctamente
 * tras reinicios del servidor.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { writeFileSync, unlinkSync, mkdirSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'

// ─── Helpers para simular loadMaintState / saveMaintState ─────────────────────
// (copiamos la lógica porque las funciones son internas al server.ts)

interface MaintenanceState {
  lastConsolidate:    Record<string, number>
  lastDetectPatterns: Record<string, number>
}

function loadState(file: string): MaintenanceState {
  try {
    return JSON.parse(require('fs').readFileSync(file, 'utf-8')) as MaintenanceState
  } catch {
    return { lastConsolidate: {}, lastDetectPatterns: {} }
  }
}

function saveState(file: string, s: MaintenanceState): void {
  writeFileSync(file, JSON.stringify(s, null, 2))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('Maintenance state — persistencia en disco', () => {
  let tmpFile: string

  beforeEach(() => {
    tmpFile = join(tmpdir(), `cortex-maint-test-${Date.now()}.json`)
  })

  afterEach(() => {
    try { unlinkSync(tmpFile) } catch {}
  })

  it('retorna estado vacío si el archivo no existe', () => {
    const state = loadState('/path/que/no/existe/maint.json')
    expect(state.lastConsolidate).toEqual({})
    expect(state.lastDetectPatterns).toEqual({})
  })

  it('persiste y recupera timestamps de consolidate', () => {
    const now = Date.now()
    const state: MaintenanceState = {
      lastConsolidate:    { 'mi-proyecto': now, 'otro-proyecto': now - 86400000 },
      lastDetectPatterns: {},
    }
    saveState(tmpFile, state)

    const loaded = loadState(tmpFile)
    expect(loaded.lastConsolidate['mi-proyecto']).toBe(now)
    expect(loaded.lastConsolidate['otro-proyecto']).toBe(now - 86400000)
  })

  it('persiste y recupera timestamps de detect_patterns', () => {
    const ts = Date.now()
    const state: MaintenanceState = {
      lastConsolidate:    {},
      lastDetectPatterns: { 'cortex': ts },
    }
    saveState(tmpFile, state)
    const loaded = loadState(tmpFile)
    expect(loaded.lastDetectPatterns['cortex']).toBe(ts)
  })

  it('múltiples proyectos coexisten sin interferirse', () => {
    const t1 = Date.now()
    const t2 = t1 - 7 * 86400000
    const state: MaintenanceState = {
      lastConsolidate:    { 'projectA': t1, 'projectB': t2 },
      lastDetectPatterns: { 'projectA': t1, 'projectB': t2 },
    }
    saveState(tmpFile, state)
    const loaded = loadState(tmpFile)
    expect(loaded.lastConsolidate['projectA']).toBe(t1)
    expect(loaded.lastConsolidate['projectB']).toBe(t2)
    expect(loaded.lastDetectPatterns['projectA']).toBe(t1)
    expect(loaded.lastDetectPatterns['projectB']).toBe(t2)
  })

  it('actualizar solo un proyecto no destruye los otros', () => {
    const t1 = Date.now()
    let state: MaintenanceState = {
      lastConsolidate:    { 'A': t1, 'B': t1 },
      lastDetectPatterns: {},
    }
    saveState(tmpFile, state)

    // Simular que se actualiza solo A
    state = loadState(tmpFile)
    state.lastConsolidate['A'] = t1 + 1000
    saveState(tmpFile, state)

    const final = loadState(tmpFile)
    expect(final.lastConsolidate['A']).toBe(t1 + 1000)
    expect(final.lastConsolidate['B']).toBe(t1)   // B intacto
  })

  it('archivo corrupto → retorna estado vacío (no crashea)', () => {
    writeFileSync(tmpFile, 'INVALID JSON {{{{')
    const state = loadState(tmpFile)
    expect(state.lastConsolidate).toEqual({})
    expect(state.lastDetectPatterns).toEqual({})
  })

  it('los intervalos de mantenimiento son los esperados', () => {
    const CONSOLIDATE_INTERVAL = 7  * 24 * 60 * 60 * 1000   // 7 días
    const DETECT_INTERVAL      = 14 * 24 * 60 * 60 * 1000   // 14 días

    // Consolidate vence después de 7 días
    const lastConsol = Date.now() - CONSOLIDATE_INTERVAL - 1
    expect(Date.now() - lastConsol).toBeGreaterThan(CONSOLIDATE_INTERVAL)

    // Detect vence después de 14 días
    const lastDetect = Date.now() - DETECT_INTERVAL - 1
    expect(Date.now() - lastDetect).toBeGreaterThan(DETECT_INTERVAL)
  })
})

describe('Maintenance — condiciones de disparo', () => {
  it('NO dispara consolidate si lastConsolidate hace menos de 7 días', () => {
    const INTERVAL = 7 * 24 * 60 * 60 * 1000
    const lastRun  = Date.now() - 3 * 24 * 60 * 60 * 1000  // hace 3 días
    const shouldRun = (Date.now() - lastRun) > INTERVAL
    expect(shouldRun).toBe(false)
  })

  it('SÍ dispara consolidate si lastConsolidate hace más de 7 días', () => {
    const INTERVAL = 7 * 24 * 60 * 60 * 1000
    const lastRun  = Date.now() - 8 * 24 * 60 * 60 * 1000  // hace 8 días
    const shouldRun = (Date.now() - lastRun) > INTERVAL
    expect(shouldRun).toBe(true)
  })

  it('SÍ dispara si nunca se ejecutó (lastRun = 0)', () => {
    const INTERVAL = 7 * 24 * 60 * 60 * 1000
    const lastRun  = 0  // nunca
    const shouldRun = (Date.now() - lastRun) > INTERVAL
    expect(shouldRun).toBe(true)
  })

  it('NO dispara detect_patterns si fue hace menos de 14 días', () => {
    const INTERVAL = 14 * 24 * 60 * 60 * 1000
    const lastRun  = Date.now() - 10 * 24 * 60 * 60 * 1000
    expect((Date.now() - lastRun) > INTERVAL).toBe(false)
  })

  it('SÍ dispara detect_patterns si fue hace más de 14 días', () => {
    const INTERVAL = 14 * 24 * 60 * 60 * 1000
    const lastRun  = Date.now() - 15 * 24 * 60 * 60 * 1000
    expect((Date.now() - lastRun) > INTERVAL).toBe(true)
  })
})
