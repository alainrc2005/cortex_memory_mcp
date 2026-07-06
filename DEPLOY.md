# 🚀 CORTEX — Guía de Despliegue

Guía completa para instalar CORTEX en un entorno nuevo desde cero.

---

## 📋 Prerrequisitos

| Requisito | Versión mínima | Verificar |
|---|---|---|
| Node.js | ≥ 18 | `node --version` |
| npm | ≥ 9 | `npm --version` |
| Docker | cualquier reciente | `docker --version` |
| Git | cualquier reciente | `git --version` |

> **Sin GPU.** CORTEX no necesita GPU. Los embeddings corren en CPU con ONNX local.
> La única razón para querer GPU es si usas Ollama como backend LLM (opcional).

---

## Paso 1 — Levantar Qdrant (base de datos vectorial)

Qdrant es el único servicio externo obligatorio. Se levanta con Docker:

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

> ⚠️ **El volumen `-v` es crítico.** Ahí viven todos los engramas (memorias).
> Sin él, pierdes todo al reiniciar el container.

Verifica que está corriendo:

```bash
curl http://localhost:6333/healthz
# Respuesta esperada: {"title":"qdrant - version x.x.x","version":"x.x.x"}
```

---

## Paso 2 — Clonar e instalar dependencias

```bash
git clone <URL-DEL-REPOSITORIO>
cd <nombre-del-directorio>
npm install
```

---

## Paso 3 — Elegir tu backend LLM

Esta es **la decisión más importante** del despliegue.
CORTEX usa un LLM para scoring, tagging, reranking y consolidación de memorias.

### ✅ Opción A — OpenRouter (recomendado para CPU-only)

**Sin GPU. Sin infraestructura extra. Modelos gratuitos disponibles.**

1. Crear cuenta gratuita en [openrouter.ai](https://openrouter.ai)
2. Generar una API key
3. Usar un modelo con sufijo `:free` (sin costo por token)

```env
CORTEX_LLM_BACKEND=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=google/gemma-4-27b-it:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Modelos gratuitos recomendados:
- `google/gemma-4-27b-it:free` — rápido, buena calidad ✅
- `deepseek/deepseek-v4-flash:free` — alternativa DeepSeek

### Opción B — Ollama (local, offline)

Requiere [Ollama](https://ollama.com) instalado y un modelo descargado.

```bash
ollama pull qwen3
```

```env
CORTEX_LLM_BACKEND=ollama
OLLAMA_URL=http://localhost:11434
```

> ⚠️ En CPU puro, Ollama puede tardar 90–130 segundos por llamada LLM.
> Usar OpenRouter si la latencia importa.

### Opción C — Sin LLM (`none`)

Cero dependencias externas. Embeddings ONNX funcionan igual.
El scoring cae a defaults seguros (`importance: 5`, `type: FACT`).

```env
CORTEX_LLM_BACKEND=none
```

Ideal para probar CORTEX rápidamente o entornos muy restringidos.

---

## Paso 4 — Crear el archivo `.env`

```bash
cp .env.example .env
# Editar con tu editor favorito
```

### `.env` mínimo (OpenRouter, recomendado)

```env
# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                             # dejar vacío si sin autenticación

# ── fastembed ONNX (modelos de embeddings locales) ────────────────────────────
FASTEMBED_CACHE_DIR=./.fastembed_cache      # ~130 MB, se descarga automáticamente

# ── LLM Backend ───────────────────────────────────────────────────────────────
CORTEX_LLM_BACKEND=openrouter

# ── OpenRouter ────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=google/gemma-4-27b-it:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# ── Reranker ──────────────────────────────────────────────────────────────────
# true  = reranking LLM después de fastembed (mejor precisión)
# false = solo fastembed ONNX (<1s, suficiente para <200 engramas por proyecto)
CORTEX_RERANKER_ENABLED=true
```

---

## Paso 5 — Compilar y arrancar

```bash
npm run build   # Compila TypeScript → dist/
npm start       # Inicia el servidor MCP (stdio)
```

### Primer arranque

En el primer inicio, CORTEX descarga los modelos ONNX automáticamente:

- `all-MiniLM-L6-v2` (~22 MB) — embeddings densos, 384 dimensiones
- `SPLADE_PP_en_v1` (~110 MB) — embeddings sparse (BM25-like)

**Total: ~130 MB.** Solo ocurre una vez. Los modelos quedan cacheados en `.fastembed_cache/`.

Espera hasta ver en el log que el servidor está listo antes de conectar el cliente MCP.

---

## Paso 6 — Conectar a tu cliente MCP

### Antigravity / Claude Desktop

Edita el archivo de configuración MCP del cliente (normalmente en `~/.gemini/settings.json` o equivalente):

```json
{
  "mcpServers": {
    "langgraph-memory-mcp": {
      "command": "node",
      "args": ["/ruta/absoluta/al/repo/dist/server.js"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "",
        "FASTEMBED_CACHE_DIR": "/ruta/absoluta/al/repo/.fastembed_cache",
        "CORTEX_LLM_BACKEND": "openrouter",
        "OPENROUTER_API_KEY": "sk-or-v1-...",
        "OPENROUTER_MODEL": "google/gemma-4-27b-it:free",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "CORTEX_RERANKER_ENABLED": "true"
      }
    }
  }
}
```

> ⚠️ **Importante:**
> - Las rutas deben ser **absolutas** (no usar `~/` ni rutas relativas).
> - El nombre del servidor debe ser exactamente **`langgraph-memory-mcp`**.

### Verificar que funciona

Después de reiniciar el cliente MCP, pide al agente:

```
¿Cuál es el estado de CORTEX?
```

El agente debe llamar `cortex_status` y responder con las colecciones activas.

---

## Paso 7 — Instalar el SKILL.md (instrucciones del agente)

Esta es la pieza más importante para que tu agente AI sepa **cómo usar CORTEX**.
Sin este archivo, el servidor corre pero el agente no sabrá:
- Hacer el cold start automático al inicio de cada sesión
- Ejecutar el hook post-turno (`quick_observe`)
- Detectar el proyecto activo desde `.project`

El archivo `skill/SKILL.md` que viene en este repo contiene todas las instrucciones.
Cópialo a la carpeta de skills de tu cliente AI:

### Antigravity

```bash
# Crear la carpeta del skill
mkdir -p ~/.gemini/config/skills/memory_manager

# Copiar el SKILL.md del repo
cp /ruta/al/repo/skill/SKILL.md ~/.gemini/config/skills/memory_manager/SKILL.md
```

### Claude Desktop / Cursor / otro cliente compatible con skills

Consulta la documentación de tu cliente para saber dónde instalar archivos de skill/instrucciones del sistema.
El contenido de `skill/SKILL.md` debe quedar accesible para el agente como contexto de sistema.

### Verificar que el skill está activo

En tu próxima conversación, el agente debería:
1. Leer el `.project` de tu workspace automáticamente
2. Llamar `get_context_for` y `get_operator_profile` al inicio
3. Guardar hechos importantes con `quick_observe` al final de cada turno

---

## Paso 8 — Configurar el proyecto en tu workspace

Para que el cold start automático funcione correctamente, crea un archivo `.project`
en la raíz de cada directorio de trabajo:

```bash
# En la raíz de tu proyecto
cat > .project << 'EOF'
{
  "name": "mi-proyecto",
  "workspace": "/ruta/absoluta/al/proyecto"
}
EOF
```

El agente leerá este archivo al inicio de cada sesión para saber en qué
colección de CORTEX trabajar.

---

## Estructura de datos persistentes

```
<directorio-del-repo>/
├── .fastembed_cache/    # Modelos ONNX (~130 MB) — NO borrar
├── kuzu_db/             # Knowledge Graph (KuzuDB) — NO mover
└── ...

<donde-levantaste-qdrant>/
└── qdrant_storage/      # Todos los engramas vectorizados — HACER BACKUP
```

> 💡 El único directorio que realmente necesitas respaldar es `qdrant_storage/`.
> Es la fuente de verdad de todas las memorias.

---

## Troubleshooting

### Qdrant no responde

```bash
# Verificar que el container está corriendo
docker ps | grep qdrant

# Si no está, levantarlo
docker start qdrant

# Ver logs
docker logs qdrant
```

### Error al descargar modelos ONNX

Los modelos se descargan de [Hugging Face](https://huggingface.co). Si hay problemas de red:

```bash
# Probar conectividad
curl -I https://huggingface.co

# Si estás detrás de proxy, configurar:
export HTTPS_PROXY=http://tu-proxy:puerto
npm start
```

### El servidor MCP no aparece en el cliente

1. Verificar que la ruta en `args` apunta al `dist/server.js` correcto.
2. Verificar que hiciste `npm run build` (el `dist/` no viene en el repo).
3. Verificar que Node.js ≥ 18: `node --version`.
4. Reiniciar el cliente MCP después de editar la configuración.

### `CORTEX_LLM_BACKEND` no definido

Si no defines la variable, CORTEX auto-detecta:
1. Si `OPENROUTER_API_KEY` está definida → usa **openrouter**
2. Si no → intenta **ollama** en `http://localhost:11434`

---

## Actualizar CORTEX

```bash
git pull
npm install      # por si cambiaron dependencias
npm run build    # recompilar
# reiniciar el cliente MCP
```

---

## Resumen rápido (happy path)

```bash
# 1. Qdrant
docker run -d --name qdrant -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# 2. Clonar
git clone <URL> && cd <repo>
npm install

# 3. Configurar
cp .env.example .env
# Editar .env con tu OPENROUTER_API_KEY

# 4. Build y arrancar
npm run build && npm start

# 5. Configurar cliente MCP con la ruta absoluta a dist/server.js

# 6. Instalar SKILL.md para que el agente sepa usar CORTEX
mkdir -p ~/.gemini/config/skills/memory_manager
cp skill/SKILL.md ~/.gemini/config/skills/memory_manager/SKILL.md

# 7. Crear .project en cada workspace de proyecto
echo '{"name": "mi-proyecto", "workspace": "'$(pwd)'"}' > /ruta/al/proyecto/.project
```

---

*CORTEX v3.0.0 — LangGraph.js + Qdrant + fastembed ONNX*
