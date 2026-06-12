#!/bin/bash
# CORTEX MCP Server wrapper
# Garantiza que el proceso node se ejecuta con el entorno correcto

export NODE_PATH="/home/alainrc2005/IA/memory-mcp/node_modules"
cd /home/alainrc2005/IA/memory-mcp

exec node dist/server.js 2>>/home/alainrc2005/IA/logs/cortex-mcp.log
