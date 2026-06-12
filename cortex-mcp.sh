#!/bin/bash
# CORTEX MCP Server wrapper
# Garantiza que el proceso node se ejecuta con el entorno correcto

export NODE_PATH="/home/alainrc2005/IA/memory-mcp/node_modules"
cd /home/alainrc2005/IA/memory-mcp

# ── ONNX Runtime CPU threading (fastembed) ──────────────────────────────────
# 8 hilos para ONNX — deja headroom para Ollama (qwen3) corriendo en paralelo.
# intra_op: paralelismo dentro de una operación (matmul, etc.)
# inter_op: paralelismo entre operaciones independientes del grafo
export OMP_NUM_THREADS=8
export ORT_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

exec node dist/server.js 2>>/home/alainrc2005/IA/logs/cortex-mcp.log
