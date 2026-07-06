#!/usr/bin/env bash
# =============================================================================
#  CORTEX Memory MCP — Installer
#  https://github.com/alainrc2005/cortex_memory_mcp
# =============================================================================
set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
RESET="\033[0m"
BOLD="\033[1m"
DIM="\033[2m"
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
WHITE="\033[0;37m"

# ── Helpers ───────────────────────────────────────────────────────────────────
info()    { echo -e "${CYAN}${BOLD}[→]${RESET} $*"; }
success() { echo -e "${GREEN}${BOLD}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[!]${RESET} $*"; }
error()   { echo -e "${RED}${BOLD}[✗]${RESET} $*"; }
step()    { echo -e "\n${BLUE}${BOLD}━━━ $* ━━━${RESET}"; }
ask()     { echo -e "${WHITE}${BOLD}    $*${RESET}"; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
#  BANNER
# =============================================================================
echo ""
echo -e "${CYAN}${BOLD}"
echo "  ██████╗ ██████╗ ██████╗ ████████╗███████╗██╗  ██╗"
echo " ██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝"
echo " ██║     ██║   ██║██████╔╝   ██║   █████╗   ╚███╔╝ "
echo " ██║     ██║   ██║██╔══██╗   ██║   ██╔══╝   ██╔██╗ "
echo " ╚██████╗╚██████╔╝██║  ██║   ██║   ███████╗██╔╝ ██╗"
echo "  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝"
echo -e "${RESET}"
echo -e "${DIM}  Persistent Semantic Memory MCP for AI Agents${RESET}"
echo -e "${DIM}  LangGraph.js · Qdrant · fastembed ONNX · KuzuDB${RESET}"
echo ""

# =============================================================================
#  STEP 0 — Detect OS
# =============================================================================
OS="linux"
case "$(uname -s)" in
  Darwin) OS="macos" ;;
  Linux)  OS="linux" ;;
  MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
esac

# =============================================================================
#  STEP 1 — Check prerequisites
# =============================================================================
step "Checking prerequisites"

MISSING=()

check_cmd() {
  local cmd=$1 label=$2
  if command -v "$cmd" &>/dev/null; then
    success "$label found: $(command -v "$cmd")"
  else
    error "$label not found"
    MISSING+=("$label")
  fi
}

check_node() {
  if command -v node &>/dev/null; then
    local ver
    ver=$(node --version | sed 's/v//')
    local major
    major=$(echo "$ver" | cut -d. -f1)
    if [ "$major" -ge 18 ]; then
      success "Node.js $ver ✓"
    else
      error "Node.js $ver is too old — need ≥ 18"
      MISSING+=("Node.js ≥ 18")
    fi
  else
    error "Node.js not found"
    MISSING+=("Node.js ≥ 18")
  fi
}

check_node
check_cmd npm   "npm"
check_cmd docker "Docker"

if [ ${#MISSING[@]} -gt 0 ]; then
  echo ""
  error "Missing prerequisites: ${MISSING[*]}"
  echo ""
  echo -e "  ${YELLOW}Install guides:${RESET}"
  echo -e "  • Node.js: ${CYAN}https://nodejs.org${RESET}  (use nvm for easy management)"
  echo -e "  • Docker:  ${CYAN}https://docs.docker.com/get-docker/${RESET}"
  echo ""
  exit 1
fi

# =============================================================================
#  STEP 2 — npm install + build
# =============================================================================
step "Installing dependencies and building CORTEX"

info "Running npm install..."
npm install --silent
success "Dependencies installed"

info "Compiling TypeScript → dist/..."
npm run build
success "Build successful (dist/server.js ready)"

# =============================================================================
#  STEP 3 — Qdrant
# =============================================================================
step "Setting up Qdrant (vector database)"

QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY_INPUT=""

# ── First: check if something is already answering on port 6333 ───────────────
if curl -sf "${QDRANT_URL}/healthz" > /dev/null 2>&1; then
  success "Qdrant already running at ${QDRANT_URL} — skipping installation"

else
  # ── Nothing on 6333 — ask the user what to do ────────────────────────────
  echo ""
  echo -e "  ${BOLD}No Qdrant instance detected at ${QDRANT_URL}.${RESET}"
  echo -e "  ${BOLD}How would you like to set it up?${RESET}"
  echo ""
  echo -e "  ${GREEN}1)${RESET} Install via Docker  ${DIM}(creates a local container — recommended)${RESET}"
  echo -e "  ${YELLOW}2)${RESET} Use custom URL      ${DIM}(Qdrant Cloud, remote server, different port...)${RESET}"
  echo -e "  ${WHITE}3)${RESET} Skip                ${DIM}(I'll set it up myself)${RESET}"
  echo ""
  ask "Enter choice [1/2/3] (default: 1):"
  read -r QDRANT_CHOICE
  QDRANT_CHOICE="${QDRANT_CHOICE:-1}"

  case "$QDRANT_CHOICE" in
    1)
      # Docker install
      if ! command -v docker &>/dev/null; then
        error "Docker not found. Install it from https://docs.docker.com/get-docker/"
        exit 1
      fi

      # Check if a stopped container with any name is using port 6333
      EXISTING=$(docker ps -a --format '{{.Names}} {{.Ports}}' 2>/dev/null \
                 | grep "6333" | awk '{print $1}' | head -1)

      if [ -n "$EXISTING" ]; then
        info "Found existing container using port 6333: ${EXISTING} — starting it..."
        docker start "$EXISTING" > /dev/null
        success "Container '${EXISTING}' started"
      else
        info "Creating new Qdrant container..."
        QDRANT_STORAGE="${REPO_DIR}/qdrant_storage"
        mkdir -p "$QDRANT_STORAGE"
        docker run -d \
          --name qdrant \
          --restart unless-stopped \
          -p 6333:6333 \
          -p 6334:6334 \
          -v "${QDRANT_STORAGE}:/qdrant/storage" \
          qdrant/qdrant \
          > /dev/null
        success "Qdrant container created (storage: ${QDRANT_STORAGE})"
      fi

      # Wait for it to be ready
      info "Waiting for Qdrant to be ready..."
      for i in {1..20}; do
        if curl -sf "${QDRANT_URL}/healthz" > /dev/null 2>&1; then
          success "Qdrant is up at ${QDRANT_URL}"
          break
        fi
        if [ "$i" -eq 20 ]; then
          error "Qdrant didn't respond after 20s."
          error "Check logs: docker logs qdrant"
          exit 1
        fi
        sleep 1
      done
      ;;

    2)
      # Custom URL
      echo ""
      ask "Enter your Qdrant URL (e.g. https://xyz.qdrant.io or http://192.168.1.10:6333):"
      read -r CUSTOM_QDRANT_URL
      QDRANT_URL="${CUSTOM_QDRANT_URL:-http://localhost:6333}"

      ask "Qdrant API key (press Enter if none):"
      read -r QDRANT_API_KEY_INPUT

      # Verify connectivity
      if curl -sf "${QDRANT_URL}/healthz" > /dev/null 2>&1; then
        success "Connected to Qdrant at ${QDRANT_URL}"
      else
        warn "Could not reach ${QDRANT_URL} — continuing anyway (check .env after install)"
      fi
      ;;

    3)
      warn "Skipping Qdrant setup. Set QDRANT_URL in .env before starting CORTEX."
      ;;

    *)
      warn "Invalid choice — skipping Qdrant setup"
      ;;
  esac
fi

# =============================================================================
#  STEP 4 — Choose LLM backend
# =============================================================================
step "Configuring LLM backend"

echo ""
echo -e "  ${BOLD}Choose your LLM backend:${RESET}"
echo ""
echo -e "  ${GREEN}1)${RESET} OpenRouter ${DIM}(recommended — cloud, free models available, no GPU needed)${RESET}"
echo -e "  ${YELLOW}2)${RESET} Ollama     ${DIM}(local, offline — GPU recommended, slow on CPU)${RESET}"
echo -e "  ${WHITE}3)${RESET} None       ${DIM}(fastest — ONNX embeddings only, no LLM scoring)${RESET}"
echo ""
ask "Enter choice [1/2/3] (default: 1):"
read -r LLM_CHOICE
LLM_CHOICE="${LLM_CHOICE:-1}"

LLM_BACKEND=""
OPENROUTER_API_KEY=""
OPENROUTER_MODEL="google/gemma-4-27b-it:free"
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
OLLAMA_URL="http://localhost:11434"

case "$LLM_CHOICE" in
  1)
    LLM_BACKEND="openrouter"
    echo ""
    echo -e "  ${DIM}Get a free API key at: ${CYAN}https://openrouter.ai${RESET}"
    echo -e "  ${DIM}Free models (no cost): google/gemma-4-27b-it:free, deepseek/deepseek-v4-flash:free${RESET}"
    echo ""
    ask "Paste your OpenRouter API key (sk-or-v1-...):"
    read -r OPENROUTER_API_KEY
    if [ -z "$OPENROUTER_API_KEY" ]; then
      warn "No API key provided — you can add it manually to .env later"
    fi
    echo ""
    ask "OpenRouter model (press Enter for default: google/gemma-4-27b-it:free):"
    read -r MODEL_INPUT
    OPENROUTER_MODEL="${MODEL_INPUT:-google/gemma-4-27b-it:free}"
    success "Backend: OpenRouter (${OPENROUTER_MODEL})"
    ;;
  2)
    LLM_BACKEND="ollama"
    if ! command -v ollama &>/dev/null; then
      warn "Ollama not found. Install it from https://ollama.com"
      warn "After installing, run: ollama pull qwen3"
    else
      success "Ollama found: $(command -v ollama)"
      info "Checking for qwen3 model..."
      if ollama list 2>/dev/null | grep -q "qwen3"; then
        success "qwen3 model available"
      else
        warn "qwen3 not found. Run: ollama pull qwen3"
      fi
    fi
    success "Backend: Ollama (${OLLAMA_URL})"
    ;;
  3)
    LLM_BACKEND="none"
    success "Backend: none (ONNX-only mode, no LLM)"
    ;;
  *)
    warn "Invalid choice — defaulting to OpenRouter"
    LLM_BACKEND="openrouter"
    ;;
esac

# =============================================================================
#  STEP 5 — Create .env
# =============================================================================
step "Creating .env configuration"

ENV_FILE="${REPO_DIR}/.env"

if [ -f "$ENV_FILE" ]; then
  warn ".env already exists — creating backup at .env.backup"
  cp "$ENV_FILE" "${ENV_FILE}.backup"
fi

cat > "$ENV_FILE" << EOF
# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_URL=${QDRANT_URL}
QDRANT_API_KEY=${QDRANT_API_KEY_INPUT}

# ── fastembed ONNX Model Cache ────────────────────────────────────────────────
FASTEMBED_CACHE_DIR=${REPO_DIR}/.fastembed_cache

# ── LLM Backend ───────────────────────────────────────────────────────────────
CORTEX_LLM_BACKEND=${LLM_BACKEND}

# ── OpenRouter ────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
OPENROUTER_MODEL=${OPENROUTER_MODEL}
OPENROUTER_BASE_URL=${OPENROUTER_BASE_URL}

# ── Ollama (if CORTEX_LLM_BACKEND=ollama) ─────────────────────────────────────
OLLAMA_URL=${OLLAMA_URL}

# ── Reranker ──────────────────────────────────────────────────────────────────
CORTEX_RERANKER_ENABLED=true
EOF

success ".env created"

# =============================================================================
#  STEP 6 — Install SKILL.md
# =============================================================================
step "Installing SKILL.md (agent behavior instructions)"

SKILL_SRC="${REPO_DIR}/skill/SKILL.md"
SKILL_DST=""

# Detect Antigravity config location
if [ -d "${HOME}/.gemini/config/skills" ]; then
  SKILL_DST="${HOME}/.gemini/config/skills/memory_manager"
  info "Detected Antigravity config at ~/.gemini/config/skills"
elif [ -d "${HOME}/.config/gemini/config/skills" ]; then
  SKILL_DST="${HOME}/.config/gemini/config/skills/memory_manager"
  info "Detected Antigravity config at ~/.config/gemini/config/skills"
else
  warn "Could not auto-detect agent skills directory"
  echo ""
  ask "Enter the path to your agent's skills directory (or press Enter to skip):"
  read -r CUSTOM_SKILL_DIR
  if [ -n "$CUSTOM_SKILL_DIR" ]; then
    SKILL_DST="${CUSTOM_SKILL_DIR}/memory_manager"
  fi
fi

if [ -n "$SKILL_DST" ]; then
  mkdir -p "$SKILL_DST"
  cp "$SKILL_SRC" "${SKILL_DST}/SKILL.md"
  success "SKILL.md installed → ${SKILL_DST}/SKILL.md"
else
  warn "Skipping SKILL.md install — copy it manually:"
  echo -e "  ${DIM}cp ${SKILL_SRC} <your-agent-skills-dir>/memory_manager/SKILL.md${RESET}"
fi

# =============================================================================
#  STEP 7 — Configure MCP client
# =============================================================================
step "Configuring MCP client"

echo ""
echo -e "  ${BOLD}Which AI client are you using?${RESET}"
echo ""
echo -e "  ${GREEN}1)${RESET} Antigravity  ${DIM}(Google Gemini agent)${RESET}"
echo -e "  ${YELLOW}2)${RESET} Claude Desktop"
echo -e "  ${CYAN}3)${RESET} Cursor"
echo -e "  ${WHITE}4)${RESET} Manual       ${DIM}(I'll edit the config myself)${RESET}"
echo ""
ask "Enter choice [1/2/3/4] (default: 1):"
read -r CLIENT_CHOICE
CLIENT_CHOICE="${CLIENT_CHOICE:-1}"

MCP_CONFIG_FILE=""

case "$CLIENT_CHOICE" in
  1)
    # Antigravity — try common locations
    for candidate in \
      "${HOME}/.gemini/settings.json" \
      "${HOME}/.gemini/antigravity/settings.json" \
      "${HOME}/.config/gemini/settings.json"; do
      if [ -f "$candidate" ]; then
        MCP_CONFIG_FILE="$candidate"
        break
      fi
    done
    if [ -z "$MCP_CONFIG_FILE" ]; then
      # Create default if not found
      MCP_CONFIG_FILE="${HOME}/.gemini/settings.json"
      mkdir -p "$(dirname "$MCP_CONFIG_FILE")"
      echo '{}' > "$MCP_CONFIG_FILE"
      info "Created new Antigravity settings at ${MCP_CONFIG_FILE}"
    fi
    ;;
  2)
    case "$OS" in
      macos)   MCP_CONFIG_FILE="${HOME}/Library/Application Support/Claude/claude_desktop_config.json" ;;
      linux)   MCP_CONFIG_FILE="${HOME}/.config/Claude/claude_desktop_config.json" ;;
      windows) MCP_CONFIG_FILE="${APPDATA}/Claude/claude_desktop_config.json" ;;
    esac
    ;;
  3)
    case "$OS" in
      macos)   MCP_CONFIG_FILE="${HOME}/Library/Application Support/Cursor/User/settings.json" ;;
      linux)   MCP_CONFIG_FILE="${HOME}/.config/Cursor/User/settings.json" ;;
      windows) MCP_CONFIG_FILE="${APPDATA}/Cursor/User/settings.json" ;;
    esac
    ;;
  4)
    MCP_CONFIG_FILE=""
    ;;
esac

if [ -n "$MCP_CONFIG_FILE" ]; then
  info "MCP config: ${MCP_CONFIG_FILE}"

  # Backup existing config
  if [ -f "$MCP_CONFIG_FILE" ]; then
    cp "$MCP_CONFIG_FILE" "${MCP_CONFIG_FILE}.cortex-backup"
    success "Backup created: ${MCP_CONFIG_FILE}.cortex-backup"
  else
    mkdir -p "$(dirname "$MCP_CONFIG_FILE")"
    echo '{}' > "$MCP_CONFIG_FILE"
  fi

  # Use Python to safely merge JSON (avoids jq dependency)
  python3 - <<PYEOF
import json, sys, os

config_path  = "${MCP_CONFIG_FILE}"
repo_dir     = "${REPO_DIR}"
qdrant_url   = "${QDRANT_URL}"
qdrant_key   = "${QDRANT_API_KEY_INPUT}"
api_key      = "${OPENROUTER_API_KEY}"
llm_backend  = "${LLM_BACKEND}"
model        = "${OPENROUTER_MODEL}"
ollama_url   = "${OLLAMA_URL}"

try:
    with open(config_path, "r") as f:
        config = json.load(f)
except Exception:
    config = {}

config.setdefault("mcpServers", {})

config["mcpServers"]["langgraph-memory-mcp"] = {
    "command": "node",
    "args": [os.path.join(repo_dir, "dist", "server.js")],
    "env": {
        "QDRANT_URL": qdrant_url,
        "QDRANT_API_KEY": qdrant_key,
        "FASTEMBED_CACHE_DIR": os.path.join(repo_dir, ".fastembed_cache"),
        "CORTEX_LLM_BACKEND": llm_backend,
        "OPENROUTER_API_KEY": api_key,
        "OPENROUTER_MODEL": model,
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "OLLAMA_URL": ollama_url,
        "CORTEX_RERANKER_ENABLED": "true"
    }
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")

print("  JSON config updated successfully")
PYEOF

  success "MCP server added to ${MCP_CONFIG_FILE}"
else
  warn "Skipping MCP config — add this manually to your client's config:"
  echo ""
  cat << MANUALEOF
  "mcpServers": {
    "langgraph-memory-mcp": {
      "command": "node",
      "args": ["${REPO_DIR}/dist/server.js"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "FASTEMBED_CACHE_DIR": "${REPO_DIR}/.fastembed_cache",
        "CORTEX_LLM_BACKEND": "${LLM_BACKEND}",
        "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}",
        "OPENROUTER_MODEL": "${OPENROUTER_MODEL}",
        "CORTEX_RERANKER_ENABLED": "true"
      }
    }
  }
MANUALEOF
fi

# =============================================================================
#  STEP 8 — Create .project in workspace (optional)
# =============================================================================
step "Setting up workspace project"

echo ""
ask "Create a .project file in the current directory? [y/N]:"
read -r CREATE_PROJECT
if [[ "$CREATE_PROJECT" =~ ^[Yy]$ ]]; then
  ask "Project name (used as the memory namespace in CORTEX):"
  read -r PROJECT_NAME
  if [ -n "$PROJECT_NAME" ]; then
    cat > "${REPO_DIR}/.project" << EOF
{
  "name": "${PROJECT_NAME}",
  "workspace": "${REPO_DIR}"
}
EOF
    success ".project created with name: ${PROJECT_NAME}"
  else
    warn "No name provided — skipping .project"
  fi
fi

# =============================================================================
#  DONE
# =============================================================================
echo ""
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}${BOLD}  CORTEX installed successfully!${RESET}"
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
echo -e "  ${BOLD}What was installed:${RESET}"
echo -e "  ${GREEN}✓${RESET} Qdrant at ${QDRANT_URL}"
echo -e "  ${GREEN}✓${RESET} CORTEX server built → ${REPO_DIR}/dist/server.js"
echo -e "  ${GREEN}✓${RESET} Configuration → ${REPO_DIR}/.env"
if [ -n "$SKILL_DST" ]; then
  echo -e "  ${GREEN}✓${RESET} Agent skill → ${SKILL_DST}/SKILL.md"
fi
if [ -n "$MCP_CONFIG_FILE" ]; then
  echo -e "  ${GREEN}✓${RESET} MCP config updated → ${MCP_CONFIG_FILE}"
fi
echo ""
echo -e "  ${BOLD}${YELLOW}⚠  One manual step required:${RESET}"
echo -e "  ${YELLOW}Restart your AI client to activate the CORTEX MCP server.${RESET}"
echo ""
echo -e "  ${BOLD}First time verification — ask your agent:${RESET}"
echo -e "  ${DIM}\"¿Cuál es el estado de CORTEX?\" / \"What is CORTEX status?\"${RESET}"
echo -e "  ${DIM}The agent should call cortex_status and list active collections.${RESET}"
echo ""
echo -e "  ${BOLD}Note on first startup:${RESET}"
echo -e "  ${DIM}CORTEX will download ONNX embedding models (~130 MB) on first use.${RESET}"
echo -e "  ${DIM}This only happens once. Subsequent starts are instant.${RESET}"
echo ""
echo -e "  ${DIM}Docs: ${CYAN}https://github.com/alainrc2005/cortex_memory_mcp${RESET}"
echo ""
