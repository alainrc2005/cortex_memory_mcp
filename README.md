# 🧠 CORTEX Memory MCP

> **Persistent semantic memory for AI agents.**
> TypeScript · LangGraph.js · Qdrant · fastembed ONNX · Ollama — 100% local, zero mandatory cloud.

[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![LangGraph.js](https://img.shields.io/badge/LangGraph.js-0.0.25-1a1a2e?logo=langchain)](https://github.com/langchain-ai/langgraphjs)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.9-DC244C?logo=data:image/svg+xml;base64,)](https://qdrant.tech/)
[![MCP](https://img.shields.io/badge/MCP-SDK_1.29-6366f1)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

---

## What is CORTEX?

CORTEX is a **Model Context Protocol (MCP) server** that gives AI agents a persistent, semantically searchable long-term memory. Unlike simple key-value stores, CORTEX understands *what* information is important, *how* memories relate to each other, and *which* memories are becoming stale over time.

Built for agents running in **CPU-only environments** — no GPU required, no cloud dependencies.

---

## Architecture — 3 Memory Layers

```
┌─────────────────────────────────────────────────────────┐
│                    CORTEX v3.2                          │
│                                                         │
│  ① WORKING MEMORY    temp_memories (Qdrant)             │
│     └─ quick_observe → instant write, no LLM           │
│                                                         │
│  ② SEMANTIC MEMORY   cortex_<project> (Qdrant)          │
│     └─ dense (all-MiniLM-L6-v2) + sparse (SPLADE)      │
│     └─ scored by qwen3 · linked · decay-weighted        │
│                                                         │
│  ③ EPISODIC MEMORY   cortex_episodes_<project> (Qdrant) │
│     └─ sessions with timestamped events, no LLM        │
└─────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Details |
|---|---|
| **LLM scoring on ingest** | `qwen3` assigns importance (1-10), type, and tags to every memory |
| **Hybrid search** | Dense (all-MiniLM-L6-v2) + Sparse (SPLADE_PP_en_v1) via Qdrant RRF fusion |
| **Cross-encoder reranking** | Single `qwen3` call evaluates all (query, candidate) pairs in batch |
| **Dual decay engine** | Bayesian (DECISION/FACT/ERROR) + FSRS-inspired (PREFERENCE/CONTEXT) |
| **Contradiction detection** | Auto-marks superseded memories on ingest |
| **Episodic sessions** | Zero-LLM session tracking with typed events |
| **Operator Profile** | Persistent coding preferences and work patterns |
| **20 MCP tools** | Complete CRUD + search + analytics surface |
| **100% local** | fastembed ONNX for embeddings, Ollama for LLM — no API keys needed |

---

## Competitive Landscape (June 2026)

| | **CORTEX** | mem0 | Graphiti | Basic Memory | MCP Official |
|---|---|---|---|---|---|
| **LLM scoring on ingest** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Hybrid search (dense+sparse)** | ✅ | ⚠️ cloud | ❌ | ❌ | ❌ |
| **Cross-encoder reranking** | ✅ | ⚠️ cloud | ❌ | ❌ | ❌ |
| **Episodic layer (no LLM)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Dual decay engine** | ✅ | ✅ | ✅ bi-temporal | ❌ | ❌ |
| **Contradiction detection** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **TypeScript + LangGraph.js** | ✅ | ❌ Python | ❌ Python | ❌ Python | ✅ |
| **100% local** | ✅ | ⚠️ MCP=cloud | ✅ | ✅ | ✅ |

---

## Prerequisites

```bash
# 1. Qdrant (Docker)
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# 2. Ollama + models
ollama pull qwen3:8b      # scoring, tagging, reranking, consolidation
# optional faster alternative:
# ollama pull qwen3:1.7b  # 3-5× faster on CPU, slightly lower accuracy
```

> **fastembed** (embeddings) is bundled as an npm dependency — no separate installation needed.
> Models download automatically to `.fastembed_cache/` on first use (~22 MB dense, ~110 MB sparse).

---

## Installation

```bash
git clone git@github.com:alainrc2005/cortex_memory_mcp.git
cd cortex_memory_mcp
npm install
npm run build
```

### Environment Configuration

Create a `.env` file in the project root:

```env
QDRANT_URL=http://localhost:6333
FASTEMBED_CACHE_DIR=/absolute/path/to/cortex_memory_mcp/.fastembed_cache

# Optional — only needed if Qdrant has auth enabled
# QDRANT_API_KEY=your_key

# Optional — defaults to http://localhost:11434
# OLLAMA_URL=http://localhost:11434
```

---

## MCP Configuration

Add to your MCP client config (e.g. `~/.gemini/config/mcp_config.json`):

```json
{
  "mcpServers": {
    "langgraph-memory-mcp": {
      "command": "/absolute/path/to/cortex_memory_mcp/cortex-mcp.sh"
    }
  }
}
```

---

## Tool Reference — 20 Tools

### Semantic Memory (17 tools)

| Tool | Description | Trigger |
|---|---|---|
| `observe` | Store a memory through the full pipeline: score → embed → link → persist | Manual / session close |
| `recall` | Hybrid BM25+dense search with LLM cross-encoder reranking | On demand |
| `get_context_for` | RAG-style context injection for a project + message | Auto (cold start) |
| `consolidate` | Merge duplicate/similar memories using LLM | End of long session |
| `detect_patterns` | Extract operator behavior patterns, update Operator Profile | Periodic |
| `get_operator_profile` | Read coding preferences and detected patterns | Auto (cold start) |
| `cortex_status` | System health: collections, engram counts, pending buffer | Diagnostic |
| `delete_memory` | Delete a single engram by ID | On demand |
| `update_memory` | Update engram content, recalculate embedding + score | On demand |
| `get_all_memories` | List all engrams for a project sorted by decay score | Audit |
| `delete_all_memories` | ⚠️ Irreversible reset of a project (requires `confirm: true`) | Explicit only |
| `batch_observe` | Store up to 20 memories in one call | Bulk import |
| `export_memories` | Export project as JSON (backup/migration) | On demand |
| `quick_observe` | Write to working buffer instantly — no LLM, no embedding | Auto (post-turn hook) |
| `list_pending` | View working buffer contents by project | Diagnostic |
| `index_temp` | Promote buffer → semantic memory with ONNX embedding + LLM scoring | Auto (next cold start) |
| `recall_hybrid` | Search both buffer (keyword) and indexed memories (semantic) simultaneously | On demand |

### Episodic Memory (3 tools)

| Tool | Description |
|---|---|
| `start_session` | Open an episodic session for a project. Auto-closes any previous open session. Returns `sessionId`. |
| `log_event` | Record a typed event in the active session. Types: `DECISION` `ERROR` `SOLUTION` `INSIGHT` `CONTEXT_CHANGE` |
| `recall_sessions` | Semantic search over past session summaries using fastembed ONNX |

---

## Indexing Pipeline

```
quick_observe(content)
      │
      ▼  (instant, no LLM, no embedding)
temp_memories  ◄──── working buffer (Qdrant, dummy vectors)
      │
      │  index_temp() — called at next session cold start
      ▼
  fastembed ONNX
  ├─ AllMiniLML6V2   → dense vector 384d
  └─ SpladePPEnV1    → sparse BM25 vector
      │
  qwen3 (Ollama)
  ├─ importance: 1-10
  ├─ type: DECISION | FACT | ERROR | PATTERN | PREFERENCE | CONTEXT
  └─ tags: [keyword, ...]
      │
  Contradiction detection (qwen3)
  └─ marks superseded memories if similarity > 0.88
      │
      ▼
  Qdrant upsert (dense + sparse vectors)
  └─ bidirectional links to related engrams
```

---

## Decay Engine

CORTEX uses a **dual decay model** tuned per memory type:

**Bayesian (DECISION · FACT · ERROR · PATTERN)**
```
utility = alpha / (alpha + beta)
alpha += importance on access
beta  += 1 per day without access
```

**FSRS-inspired (PREFERENCE · CONTEXT)**
```
stability     = log1p(accessCount) × (importance / 5)
retrievability = exp(-daysSinceAccess / stability)
```

Memories accessed frequently become more stable. Stale, unaccessed memories decay toward zero and eventually become candidates for consolidation.

---

## Project Structure

```
src/
├── server.ts                    # MCP server — 20 tools, ~1500 LOC
├── bootstrap.ts                 # Qdrant collection init on startup
├── graph/
│   ├── observe/
│   │   ├── workflow.ts          # LangGraph pipeline: score→embed→link→persist
│   │   ├── state.ts             # Graph state types
│   │   └── nodes/
│   │       ├── score.ts         # qwen3: importance + type + tags
│   │       ├── embed.ts         # fastembed ONNX: dense + sparse vectors
│   │       ├── link.ts          # Bidirectional links in Qdrant
│   │       └── persist.ts       # Final upsert
│   └── consolidate/
│       └── nodes.ts             # LLM merge of duplicate engrams
├── services/
│   ├── qdrant.ts                # Qdrant client, collections, hybrid search
│   ├── fastembed.ts             # ONNX embeddings: dense (AllMiniLM) + sparse (SPLADE)
│   ├── ollama.ts                # qwen3: scoring, reranking, contradiction detection
│   ├── decay.ts                 # Dual decay engine: Bayesian + FSRS-inspired
│   └── episode.ts               # Episodic session management
└── types/
    ├── engrama.ts               # Engram TypeScript types
    └── episode.ts               # Episode/event types
```

---

## Running Tests

```bash
npm test
# Covers all 20 tools with valid, invalid, and connectivity test cases
```

---

## Memory Lifecycle

```
Session N (active)
    Agent detects storable fact
         ↓
    quick_observe()  ← instant, no CPU cost
         ↓
    Written to temp_memories

Session N+1 cold start
    get_context_for() + get_operator_profile()  ← parallel
         ↓
    Pending in temp_memories? → index_temp()
         ↓
    fastembed ONNX + qwen3 scoring applied
         ↓
    Promoted to cortex_<project> with full embeddings
         ↓
    Available for recall() and get_context_for()
```

---

## License

MIT © 2026 — Built by Zeus with [Antigravity](https://antigravity.ai)
