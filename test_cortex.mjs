#!/usr/bin/env node
/**
 * CORTEX MCP — Test Suite v3.0.0
 * Cubre los 13 tools con casos válidos, inválidos y edge cases
 */

import { spawn } from 'child_process'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dir = dirname(fileURLToPath(import.meta.url))
const SERVER = resolve(__dir, 'dist/server.js')

// ─── Runner ───────────────────────────────────────────────────────────────────

let passed = 0
let failed = 0
let reqId = 0

async function callTool(name, args = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn('node', [SERVER], { stdio: ['pipe', 'pipe', 'pipe'] })
    const id = ++reqId
    const req = JSON.stringify({ jsonrpc: '2.0', id, method: 'tools/call', params: { name, arguments: args } })
    let out = ''
    proc.stdout.on('data', d => out += d)
    proc.on('close', () => {
      try { resolve(JSON.parse(out)) } catch { reject(new Error(`Bad JSON: ${out.slice(0, 100)}`)) }
    })
    proc.stdin.write(req + '\n')
    proc.stdin.end()
    setTimeout(() => { proc.kill(); reject(new Error('timeout')) }, 10000)
  })
}

async function listTools() {
  return new Promise((resolve, reject) => {
    const proc = spawn('node', [SERVER], { stdio: ['pipe', 'pipe', 'pipe'] })
    const req = JSON.stringify({ jsonrpc: '2.0', id: 0, method: 'tools/list', params: {} })
    let out = ''
    proc.stdout.on('data', d => out += d)
    proc.on('close', () => {
      try { resolve(JSON.parse(out)) } catch { reject(new Error(`Bad JSON: ${out.slice(0, 100)}`)) }
    })
    proc.stdin.write(req + '\n')
    proc.stdin.end()
    setTimeout(() => { proc.kill(); reject(new Error('timeout')) }, 6000)
  })
}

function assert(condition, label, detail = '') {
  if (condition) {
    console.log(`  ✅ ${label}`)
    passed++
  } else {
    console.log(`  ❌ FAIL: ${label}${detail ? '\n     → ' + detail : ''}`)
    failed++
  }
}

function getText(res) {
  return res?.result?.content?.[0]?.text ?? ''
}

// ─── Tests ────────────────────────────────────────────────────────────────────

console.log('\n🧠 CORTEX Test Suite v3.0.0')
console.log('═══════════════════════════════════════\n')

// ── 1. tools/list ─────────────────────────────────────────────────────────────
console.log('[ tools/list ]')
try {
  const res = await listTools()
  const tools = res.result?.tools?.map(t => t.name) ?? []
  const expected = [
    'observe', 'recall', 'consolidate', 'get_context_for',
    'detect_patterns', 'get_operator_profile', 'cortex_status',
    'delete_memory', 'update_memory', 'get_all_memories',
    'delete_all_memories', 'batch_observe', 'export_memories',
  ]
  assert(tools.length === 13, `13 tools registrados (encontrados: ${tools.length})`)
  for (const t of expected) {
    assert(tools.includes(t), `tool "${t}" registrado`)
  }
} catch (e) {
  assert(false, 'tools/list responde', e.message)
}

// ── 2. observe — validación ───────────────────────────────────────────────────
console.log('\n[ observe — validación ]')
try {
  const r1 = await callTool('observe', {})
  assert(getText(r1).includes('Parámetro requerido'), 'sin args → error de validación')

  const r2 = await callTool('observe', { projectName: 'test' })
  assert(getText(r2).includes('Parámetro requerido'), 'sin content → error de validación')
} catch (e) {
  assert(false, 'observe validación', e.message)
}

// ── 3. recall — validación ────────────────────────────────────────────────────
console.log('\n[ recall — validación ]')
try {
  const r = await callTool('recall', { projectName: 'test' })
  assert(getText(r).includes('Parámetro requerido'), 'sin query → error de validación')
} catch (e) {
  assert(false, 'recall validación', e.message)
}

// ── 4. get_context_for — validación ──────────────────────────────────────────
console.log('\n[ get_context_for — validación ]')
try {
  const r = await callTool('get_context_for', {})
  // Devuelve contexto vacío o contexto válido (no error de crash)
  const text = getText(r)
  assert(text.includes('CORTEX') || text.includes('Sin memorias') || text.includes('global'), 'respuesta válida sin projectName (fallback a global)')
} catch (e) {
  assert(false, 'get_context_for sin args', e.message)
}

// ── 5. cortex_status ─────────────────────────────────────────────────────────
console.log('\n[ cortex_status ]')
try {
  const r = await callTool('cortex_status')
  const text = getText(r)
  assert(text.includes('CORTEX'), 'responde con header CORTEX')
  assert(text.includes('engramas'), 'incluye conteo de engramas')
  assert(text.includes('Operator Profile'), 'incluye estado del profile')
} catch (e) {
  assert(false, 'cortex_status', e.message)
}

// ── 6. get_operator_profile ──────────────────────────────────────────────────
console.log('\n[ get_operator_profile ]')
try {
  const r = await callTool('get_operator_profile')
  const text = getText(r)
  assert(text.includes('Operator Profile'), 'responde con header correcto')
} catch (e) {
  assert(false, 'get_operator_profile', e.message)
}

// ── 7. delete_memory — validación ────────────────────────────────────────────
console.log('\n[ delete_memory — validación ]')
try {
  const r = await callTool('delete_memory', {})
  assert(getText(r).includes('Parámetro requerido'), 'sin id → error de validación')
} catch (e) {
  assert(false, 'delete_memory validación', e.message)
}

// ── 8. update_memory — validación ────────────────────────────────────────────
console.log('\n[ update_memory — validación ]')
try {
  const r = await callTool('update_memory', { id: 'some-id' })
  assert(getText(r).includes('Parámetro requerido') || getText(r).includes('Error'), 'sin content → error')
} catch (e) {
  assert(false, 'update_memory validación', e.message)
}

// ── 9. get_all_memories — validación ─────────────────────────────────────────
console.log('\n[ get_all_memories — validación ]')
try {
  const r = await callTool('get_all_memories', {})
  assert(getText(r).includes('Parámetro requerido'), 'sin projectName → error de validación')
} catch (e) {
  assert(false, 'get_all_memories validación', e.message)
}

// ── 10. get_all_memories — válido ─────────────────────────────────────────────
console.log('\n[ get_all_memories — con proyecto real ]')
try {
  const r = await callTool('get_all_memories', { projectName: 'school', limit: 3 })
  const text = getText(r)
  assert(text.includes('engramas') || text.includes('school'), 'respuesta válida')
} catch (e) {
  assert(false, 'get_all_memories school', e.message)
}

// ── 11. delete_all_memories — seguridad ──────────────────────────────────────
console.log('\n[ delete_all_memories — safety gate ]')
try {
  const r1 = await callTool('delete_all_memories', { projectName: 'test' })
  assert(getText(r1).includes('cancelada'), 'sin confirm → operación cancelada')

  const r2 = await callTool('delete_all_memories', { projectName: 'test', confirm: false })
  assert(getText(r2).includes('cancelada'), 'confirm:false → operación cancelada')
} catch (e) {
  assert(false, 'delete_all_memories safety', e.message)
}

// ── 12. batch_observe — validación ───────────────────────────────────────────
console.log('\n[ batch_observe — validación ]')
try {
  const r1 = await callTool('batch_observe', { projectName: 'test' })
  assert(getText(r1).includes('Parámetro requerido') || getText(r1).includes('Error'), 'sin memories → error')

  const r2 = await callTool('batch_observe', { projectName: 'test', memories: [] })
  assert(getText(r2).includes('Error') || getText(r2).includes('vacío'), 'memories vacío → error')
} catch (e) {
  assert(false, 'batch_observe validación', e.message)
}

// ── 13. export_memories ──────────────────────────────────────────────────────
console.log('\n[ export_memories ]')
try {
  const r = await callTool('export_memories', { projectName: 'school' })
  const text = getText(r)
  assert(text.includes('export') || text.includes('json') || text.includes('"version"'), 'retorna JSON válido')
  assert(text.includes('school') || text.includes('"projectName"'), 'incluye nombre del proyecto')
} catch (e) {
  assert(false, 'export_memories', e.message)
}

// ── 14. error genérico — tool desconocido ────────────────────────────────────
console.log('\n[ error genérico ]')
try {
  const r = await callTool('tool_inexistente', {})
  const text = getText(r)
  assert(text.includes('Error') || text.includes('no encontrado'), 'tool desconocido → error limpio')
} catch (e) {
  assert(false, 'tool desconocido', e.message)
}

// ─── Resumen ──────────────────────────────────────────────────────────────────

const total = passed + failed
console.log('\n═══════════════════════════════════════')
console.log(`  Resultado: ${passed}/${total} tests pasaron`)
if (failed > 0) {
  console.log(`  ⚠️  ${failed} fallaron`)
  process.exit(1)
} else {
  console.log('  🎉 Todos los tests pasaron')
}
console.log('═══════════════════════════════════════\n')
