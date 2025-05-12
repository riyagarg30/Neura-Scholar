#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="./venv"
LOGFILE="./streamlit.log"
PORT=8501

# 0) Kill anything already using port $PORT
if lsof -iTCP:${PORT} -sTCP:LISTEN -t >/dev/null; then
  echo "→ Killing process on port ${PORT}..."
  kill -9 $(lsof -tiTCP:${PORT} -sTCP:LISTEN)
fi

# 1) Ensure Python3
if ! command -v python3 &> /dev/null; then
  echo "Error: python3 not found." >&2
  exit 1
fi

# 2) Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
  echo "→ Creating virtualenv in ${VENV_DIR}..."
  python3 -m venv "$VENV_DIR"
fi

# 3) Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 4) Install / upgrade deps
echo "→ Upgrading pip & installing dependencies..."
pip install --upgrade pip >/dev/null
pip install streamlit pandas plotly numpy >/dev/null

# 5) Launch Streamlit on localhost only
echo "→ Starting Streamlit on 127.0.0.1:${PORT} (logs → ${LOGFILE})"
nohup streamlit run dashboard.py \
    --server.address 127.0.0.1 \
    --server.port ${PORT} \
    > "${LOGFILE}" 2>&1 &

echo "✅ Streamlit launched with PID $!; tail -f ${LOGFILE} to follow logs."