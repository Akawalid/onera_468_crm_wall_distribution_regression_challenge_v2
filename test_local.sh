#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# test_local.sh  –  Run the full Codabench pipeline locally (ingestion → scoring)
#
# Place this script at the ROOT of your bundle folder (next to competition.yaml).
#
# Usage:
#   bash test_local.sh --data /path/to/your/data/folder
#
# Your data folder must contain:
#   input_data/
#     X9_train_fl32.npy
#     rho_fl32.npy
#     X9_test_fl32.npy
#   ref/
#     rho_test_fl32.npy
#     fullfiles_PiMinfAoA_with_scores.csv
#
# Mirrors these Codabench container paths exactly:
#   /app/input_data/        ← raw .npy files (train + test)
#   /app/program/           ← ingestion.py
#   /app/ingested_program/  ← model.py + requirements.txt (your submission)
#   /app/output/            ← Yhat.npy + metadata.json  (written by ingestion)
#   /app/input/ref/         ← ground truth for scoring
#   /app/input/res/         ← symlink → /app/output/    (scoring reads here)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[✔]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
error()   { echo -e "${RED}[✗]${NC} $*"; exit 1; }
section() { echo; echo "──────────────────────────────────────────"; echo "  $*"; echo "──────────────────────────────────────────"; }

# ── Parse arguments ───────────────────────────────────────────────────────────
DATA_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_DIR="$2"; shift 2 ;;
    *) error "Unknown option: $1\nUsage: bash test_local.sh --data /path/to/data" ;;
  esac
done
[[ -z "$DATA_DIR" ]] && error "Missing --data argument.\nUsage: bash test_local.sh --data /path/to/data"

DATA_DIR="$(realpath "$DATA_DIR")"
BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Validate data folder ──────────────────────────────────────────────────────
section "Validating data folder: $DATA_DIR"

MISSING=0
for f in \
  "input_data/X9_train_fl32.npy" \
  "input_data/rho_fl32.npy" \
  "input_data/X9_test_fl32.npy" \
  "ref/rho_test_fl32.npy" \
  "ref/fullfiles_PiMinfAoA_with_scores.csv"; do
  if [[ ! -f "$DATA_DIR/$f" ]]; then
    warn "Missing: $DATA_DIR/$f"
    MISSING=$((MISSING + 1))
  else
    info "Found: $f"
  fi
done
[[ $MISSING -gt 0 ]] && error "$MISSING required file(s) missing. See above."

# ── Build /app sandbox (requires sudo for the real /app path) ────────────────
section "Building /app sandbox"

# We create a real /app directory so the hardcoded paths in ingestion.py and
# scoring.py resolve without any patching.
if [[ -d /app ]]; then
  warn "/app already exists — backing it up to /app.bak_$(date +%s)"
  sudo mv /app "/app.bak_$(date +%s)"
fi

sudo mkdir -p \
  /app/input_data \
  /app/program \
  /app/ingested_program \
  /app/output \
  /app/input/ref \
  /app/input/res

# Make output world-writable so ingestion.py can write without sudo
sudo chmod -R 777 /app/output /app/input

info "Sandbox created at /app"

# ── Copy files into sandbox ───────────────────────────────────────────────────
section "Copying files"

# Raw data for ingestion
sudo cp -r "$DATA_DIR/input_data/." /app/input_data/
info "input_data/ → /app/input_data/"

# Ground truth for scoring
sudo cp -r "$DATA_DIR/ref/." /app/input/ref/
info "ref/ → /app/input/ref/"

# ingestion.py goes in /app/program/
sudo cp "$BUNDLE_DIR/ingestion_program/ingestion.py" /app/program/
info "ingestion.py → /app/program/"

# model.py + requirements.txt (submission) go in /app/ingested_program/
sudo cp "$BUNDLE_DIR/solution/model.py" /app/ingested_program/
[[ -f "$BUNDLE_DIR/solution/requirements.txt" ]] && \
  sudo cp "$BUNDLE_DIR/solution/requirements.txt" /app/ingested_program/
info "model.py (+ requirements.txt) → /app/ingested_program/"

# scoring.py goes in its own folder (we run it from there)
sudo cp "$BUNDLE_DIR/scoring_program/scoring.py" /app/program/
info "scoring.py → /app/program/"

# /app/input/res/ must point at /app/output/ so scoring finds Yhat.npy
# (scoring reads from /app/input/res/, ingestion writes to /app/output/)
sudo rm -rf /app/input/res
sudo ln -s /app/output /app/input/res
info "/app/input/res → /app/output (symlink)"

# ── Install dependencies ──────────────────────────────────────────────────────
section "Installing dependencies"

if [[ -f "$BUNDLE_DIR/ingestion_program/requirements.txt" ]]; then
  echo "  → ingestion_program/requirements.txt"
  pip install -q -r "$BUNDLE_DIR/ingestion_program/requirements.txt"
fi
# solution/requirements.txt is handled by ingestion.py itself at runtime
# (check_and_install_dependencies), but we install it upfront too for speed
if [[ -f "$BUNDLE_DIR/solution/requirements.txt" ]]; then
  echo "  → solution/requirements.txt"
  pip install -q -r "$BUNDLE_DIR/solution/requirements.txt"
fi
if [[ -f "$BUNDLE_DIR/scoring_program/requirements.txt" ]]; then
  echo "  → scoring_program/requirements.txt"
  pip install -q -r "$BUNDLE_DIR/scoring_program/requirements.txt"
fi
info "Dependencies installed"

# ── Run ingestion ─────────────────────────────────────────────────────────────
section "Running ingestion"
echo "  reads  : /app/input_data/"
echo "  writes : /app/output/Yhat.npy + metadata.json"
echo ""

python /app/program/ingestion.py 2>&1 | tee /app/ingestion.log

echo ""
if [[ ! -f /app/output/Yhat.npy ]]; then
  error "Yhat.npy was NOT produced. Check /app/ingestion.log"
fi
info "Yhat.npy produced at /app/output/Yhat.npy"

if [[ ! -f /app/output/metadata.json ]]; then
  warn "metadata.json not found — duration will be -1 in scores"
fi

# ── Run scoring ───────────────────────────────────────────────────────────────
section "Running scoring"
echo "  reads predictions : /app/input/res/  (→ /app/output/)"
echo "  reads reference   : /app/input/ref/"
echo "  writes            : /app/output/scores.json + detailed_results.html"
echo ""

python /app/program/scoring.py 2>&1 | tee /app/scoring.log

# ── Print results ─────────────────────────────────────────────────────────────
section "Results"

if [[ -f /app/output/scores.json ]]; then
  python3 - << 'PY'
import json
with open('/app/output/scores.json') as f:
    d = json.load(f)
width = max(len(k) for k in d)
for k, v in d.items():
    val = f"{v:.6f}" if isinstance(v, float) else str(v)
    print(f"  {k:<{width}} : {val}")
PY
else
  warn "scores.json not found — check /app/scoring.log"
fi

echo ""
echo "  Logs    : /app/ingestion.log"
echo "            /app/scoring.log"
echo "  HTML    : /app/output/detailed_results.html"
echo "  Scores  : /app/output/scores.json"
echo ""
info "Done."
