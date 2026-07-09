---
name: memory_manager
description: Memoria persistente CORTEX v3.2 (local-first). Cold start automático inyecta contexto del proyecto + perfil del operador. Hook post-turno obligatorio con quick_observe. 22 tools.
---

# CORTEX — Memory Manager v3.2

Memoria persistente local-first (LangGraph.js + Qdrant + KuzuDB + fastembed ONNX).
Este archivo define **solo tu comportamiento operativo** con las 22 tools; la arquitectura
real (4 capas de memoria, búsqueda híbrida BM25+SPLADE→RRF→rerank, knowledge graph,
memoria episódica tipada, dual-engine Bayesian/FSRS) está en el `README.md` del repo.
**No juzgues las capacidades de CORTEX por este archivo — lee el README.**

---

## 1. Cold start (obligatorio, antes de responder nada)

Ejecuta EN PARALELO: `get_context_for({projectName, message})` + `get_operator_profile({})`.

- `get_context_for` → inyecta los engramas en tu razonamiento; actúa como si ya lo supieras.
- `get_operator_profile` → aplica `codingPreferences` y `detectedPatterns` proactivamente, sin anunciarlo.
- **No repitas** el contexto inyectado salvo que sea relevante para la pregunta.

**Determinar `projectName`** (en orden): (1) campo `name` de `<workspace_root>/.project`;
(2) lo que el operador mencione explícitamente; (3) nombre del directorio; sin contexto → `global`.

**Si no existe `.project`:** pregunta antes de responder ("¿Cómo se llama el proyecto?"),
crea el archivo `{ "name": ..., "workspace": ... }` con la respuesta, y sigue el cold start.

> ⚠️ Si `get_context_for` revela pendientes en el buffer, **solo alerta** ("Hay N memorias
> pendientes de indexar en [proyecto]. ¿Ejecuto `index_temp`?") y espera confirmación.
> NUNCA llames `index_temp` en el cold start.

---

## 2. Hook post-turno — `quick_observe` (LA REGLA MÁS IMPORTANTE)

**Al final de CADA turno**, evalúa el checklist. Si UNA condición es verdadera →
`quick_observe({projectName, content})` inmediatamente (instantáneo, sin LLM ni embedding;
no esperes confirmación).

| # | Condición |
|---|---|
| A | Decisión técnica/arquitectónica ("usaremos Redis", "descartamos Prisma") |
| B | Se resolvió un error o bug |
| C | Preferencia explícita del operador ("prefiero async/await") |
| D | Hito/avance concreto (módulo completado, PR mergeado, deploy) |
| E | Patrón de trabajo nuevo ("siempre empieza por el schema") |
| F | Configuración del sistema (puerto, env var sin valor, path clave) |
| G | Riesgo o deuda técnica ("esto es temporal", "hay que refactorizar Y") |

- Si **ninguna** aplica → NO llames `quick_observe`. No guardes trivialidades, preguntas ni código desechable.
- Varios hechos → una llamada por hecho, **máx 3 por turno** (agrupa el resto en una frase).

---

## 3. Guardado con scoring (no es el flujo normal)

- **`observe`** — solo para: cierre de sesión larga (>1h), preferencia crítica que debe sobrevivir,
  o cuando el operador lo pide. El 90% del guardado va por `quick_observe`.
- **`batch_observe`** — varios hechos al cierre de sesión o importación masiva (**máx 20**).

---

## 4. Recuperación

- **`recall({projectName, query, limit})`** — el operador pregunta por el pasado o `get_context_for` no bastó. Busca proyecto + global.
- **`recall_hybrid`** — igual pero incluye el buffer sin indexar (`temp_memories`) además de `work_memories`. Úsalo para algo guardado hace poco.
- **`get_all_memories({projectName, limit})`** — "muéstrame todo lo que recuerdas de X" (ordenado por decay).

---

## 5. Mantenimiento

- **`consolidate`** — fusiona duplicados; marca `superseded` (nunca borra). Fin de sesión larga o a petición.
- **`detect_patterns`** — analiza patrones del operador; cada 3+ sesiones o a petición.
- **`update_memory({id, content})`** — corrige un engrama (recalcula embedding). Obtén el `id` con `recall`/`get_all_memories`.
- **`cortex_status`** / **`list_pending`** — diagnóstico y buffer pendiente por proyecto.
- **`export_memories`** — backup/migración a JSON.

---

## 6. Operaciones sensibles

- **`index_temp({projectName, batchSize=5, skipScoring})`** — promueve buffer → `work_memories` con embedding + scoring.
  Costoso. **⛔ NUNCA auto**: solo cuando el operador lo confirma tras tu alerta.
- **`delete_memory({id})`** — solo a petición o si sabes con certeza que es incorrecto.
- **`delete_all_memories({projectName, confirm:true})`** — **⚠️ IRREVERSIBLE.** Nunca sin que el operador diga literalmente "borra toda la memoria de [proyecto]".

---

## Interpretación de engramas

| Tipo | Cómo tratarlo |
|---|---|
| `DECISION` | Verdad establecida; no la re-debatas sin petición. |
| `PATTERN` / `PREFERENCE` | Aplícalos proactivamente (código, tono, formato, herramientas). |
| `FACT` | Dato contextual para enriquecer respuestas. |
| `ERROR` | Error ya resuelto; no lo repitas. |
| `CONTEXT` | Contexto de sesión; mantén continuidad. |

**Importancia 8-10** → verdad inamovible. **Decay < 2** → muy antiguo; verifica vigencia antes de usar.

---

## Referencia rápida — 17 tools

| Tool | Cuándo | Trigger |
|---|---|---|
| `get_context_for` | Cold start | 🔴 AUTO |
| `get_operator_profile` | Cold start + "¿qué sabes de mí?" | 🔴 AUTO |
| `quick_observe` | Hook post-turno (A–G) | 🔴 AUTO |
| `index_temp` | Buffer → work | 🔴 Nunca auto — solo alertar |
| `delete_all_memories` | Reset de proyecto | 🔴 Nunca auto — confirmación literal |
| `observe` | Cierre sesión / preferencia crítica | 🟡 Semi-auto |
| `recall` | Preguntas del pasado | 🟡 Semi-auto |
| `recall_hybrid` | Búsqueda incluyendo buffer | 🟡 Semi-auto |
| `batch_observe` | Fin de sesión / importación | 🟠 Manual |
| `list_pending` | Ver buffer sin indexar | 🟠 Manual |
| `detect_patterns` | Análisis de patrones | 🟠 Manual |
| `consolidate` | Duplicados / fin de sesión | 🟠 Manual |
| `cortex_status` | Diagnóstico | 🟠 Manual |
| `get_all_memories` | Auditoría | 🟠 Manual |
| `update_memory` | Corregir engrama | 🟠 Manual |
| `delete_memory` | Borrar engrama | 🟠 Manual |
| `export_memories` | Backup / migración | 🟠 Manual |
