# CORTEX vs Red Pill — Comparativa de Sistemas de Memoria

> Reconstruida desde engramas CORTEX · Proyecto: cortex · Junio 2026

---

## Bases comunes (ambos tienen)

| Característica | CORTEX | Red Pill |
|---|---|---|
| Vector DB | Qdrant local | Qdrant local |
| Embeddings sin GPU | fastembed ONNX | fastembed ONNX |
| Retrieval Priority dual | Bayesian + FSRS | Bayesian + FSRS |
| Buffer de dos fases | ✅ `quick_observe` → `index_temp` | ✅ |
| Perfil del operador | ✅ `get_operator_profile` | ✅ |

---

## Dónde CORTEX gana

### 1. Embeddings híbridos BM25+dense sin GPU
**Único MCP local** con SPLADE_PP_en_v1 (sparse) + all-MiniLM-L6-v2 (dense) vía fastembed ONNX
en un solo pipeline. Ningún competidor (Mem0, Zep, Letta) hace esto sin API externa.

Pipeline completo de recall:
```
Dense (cosine) + Sparse (BM25/IDF)
  → Qdrant Prefetch paralelo
  → Reciprocal Rank Fusion
  → Bayesian/FSRS decay scoring
  → Cross-encoder rerank (Qwen3, batch único)
  → Score final: 40% semántico · 40% rerank · 20% decay
```

### 2. Spaced Repetition (FSRS) aplicado a memoria AI
A junio 2026, **ningún competidor** del mercado implementa FSRS para priorización de recuperación.
- `DECISION/FACT/ERROR/PATTERN` → motor Bayesian α/(α+β)
- `PREFERENCE/CONTEXT` → motor FSRS-inspired con stability acumulada

### 3. 4 capas de memoria ortogonales en un servidor
Ningún competidor tiene las 4 en un solo proceso:

| Capa | Herramientas |
|---|---|
| ⚡ Buffer rápido (sin LLM, instantáneo) | `quick_observe`, `batch_observe` |
| 🧠 Semántica (embedding + scoring) | `observe`, `recall`, `recall_hybrid` |
| 📼 Episódica (timeline de sesiones) | `start_session`, `log_event`, `recall_sessions` |
| 🕸️ Knowledge Graph (Cypher/KuzuDB) | `graph_neighbors`, `graph_timeline`, `graph_query` |

### 4. LLM completamente opcional
`CORTEX_LLM_BACKEND=none` → zero-LLM mode. Los embeddings ONNX siguen funcionando.
El scoring cae a defaults seguros (`importance: 5`, `type: FACT`).

**Mem0, Zep y Letta no arrancan sin LLM API.** Red Pill requiere LLM local para el sleep cycle.

### 5. Filosofía de no-borrado
El "decay" en CORTEX **no es olvido**: es un Retrieval Priority Score (0–1).
Una memoria con score bajo sigue existiendo — solo rankea más abajo.
Solo `consolidate` puede marcar memorias como `status: superseded`.
**Nunca se borra nada por edad.**

Red Pill puede hacer DELETE físico en Qdrant (filosofía diferente).

### 6. Knowledge Graph con timeline temporal
`graph_timeline` muestra la **historia cronológica** de cómo evolucionaron las relaciones
entre entidades. Único en el mercado de memory-MCPs a junio 2026.

### 7. Cross-project search
`recall` busca proyecto + global en un solo pass. Sin configuración extra.

### 8. Zero fricción de setup
Un `.env`, `npm install`, `npm run build`, `npm start`. Sin instaladores de agentes,
sin swarms que configurar, sin modelos locales obligatorios.

---

## Dónde Red Pill gana

### 1. Motor emocional ACE
Red Pill tiene un sistema de estado emocional (Affective Computing Engine) que influye
en cómo el agente responde. CORTEX no tiene nada equivalente.

### 2. Grafo sináptico avanzado
Red Pill implementa:
- **Axones** con pesos direccionales entre nodos
- **N-hop traversal** (caminos de profundidad N en el grafo)
- **Evocative Cascade** — una memoria activa cascadas de memorias relacionadas

El Knowledge Graph de CORTEX (KuzuDB + Cypher) es más simple: relaciones directas,
sin pesos ni cascadas automáticas.

### 3. Sleep cycle autónomo
Red Pill tiene un proceso de consolidación nocturna autónomo que reorganiza memorias,
detecta patrones y actualiza el grafo sin intervención del usuario.
CORTEX requiere llamar `consolidate` y `detect_patterns` manualmente.

### 4. Swarm de agentes especializados
Red Pill tiene agentes internos con roles distintos:
- **Smith** — eliminación de memorias contradictorias
- **Healer** — reparación de inconsistencias
- **Oracle** — predicción y planificación
- **Keymaker** — acceso a recursos bloqueados
- **Gru** — coordinación general del swarm

CORTEX es un servidor single-process, sin swarm.

### 5. Hardware awareness
Red Pill detecta GPU/NPU disponibles y adapta su pipeline (embeddings en GPU, Ollama en NPU).
CORTEX asume CPU y fastembed ONNX sin adaptación de hardware.

### 6. Seguridad formal
Red Pill tiene especificaciones de seguridad formales (SEC-001 a SEC-008).
CORTEX no tiene un modelo de seguridad formal definido.

### 7. Suite de tests
Red Pill: **770+ tests** cubriendo comportamiento del sistema.
CORTEX: tests mínimos (`test_cortex.mjs`), sin suite formal.

---

## Posicionamiento de mercado

| Sistema | Fortaleza principal | Público objetivo |
|---|---|---|
| **CORTEX** | Local-first, zero-GPU, 4 capas, setup mínimo | Devs que quieren memoria persistente sin fricción |
| **Red Pill** | Agente autónomo completo con emociones y swarm | Proyectos de agentes sofisticados, investigación |
| **Mem0** | Cloud, hosted, API simple | Startups que no quieren infraestructura propia |
| **Zep** | Memoria conversacional para LLM apps | Chatbots y asistentes con contexto de sesión |
| **Letta** | Agente con estado persistente, open source | Investigación académica, agentes con larga vida |

---

## Brecha crítica identificada

> **CORTEX no tiene benchmarks públicos.**
>
> OMEGA (competidor) se posiciona con **95.4% en LongMemEval**.
> Sin números públicos, CORTEX es invisible aunque técnicamente superior.
>
> Pendiente: correr `eval/longmemeval_cortex.mjs` y publicar resultados.
> El scorer oficial requiere OpenAI API (GPT-4o como juez LLM).
> Comando: `python3 evaluate_qa.py gpt-4o <hypothesis.jsonl> <dataset.json>`

---

*Última actualización: Junio 2026 · Fuente: engramas CORTEX proyecto cortex*
