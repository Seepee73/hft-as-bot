#!/usr/bin/env bash
# =============================================================================
# HFT AS Bot — Pre-flight Dependency Installer
# Run this ONCE before starting the build. Handles all compiled C/Rust deps.
# Usage: chmod +x preflight_install.sh && ./preflight_install.sh
# =============================================================================

set -e  # exit on any error

PYTHON=${PYTHON:-python3}
PIP="$PYTHON -m pip"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║        HFT Avellaneda-Stoikov Bot — Pre-flight           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 0. Check Python version (needs 3.11+) ──────────────────────────────────
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "► Python version: $PY_VERSION"

$PYTHON -c "
import sys
if sys.version_info < (3, 11):
    print('  WARNING: Python 3.11+ recommended. Some wheels may be unavailable.')
else:
    print('  OK')
"

# ── 1. Create virtual environment ──────────────────────────────────────────
echo ""
echo "► Creating virtual environment at ./venv ..."
$PYTHON -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel -q

# ── 2. System-level C library for TA-Lib ───────────────────────────────────
echo ""
echo "► Installing TA-Lib C library (requires sudo) ..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &>/dev/null; then
        brew install ta-lib
    else
        echo "  ERROR: Homebrew not found. Install from https://brew.sh then re-run."
        exit 1
    fi
elif [[ -f /etc/debian_version ]]; then
    # Ubuntu / Debian
    sudo apt-get update -q
    sudo apt-get install -y -q libta-lib-dev
elif [[ -f /etc/redhat-release ]]; then
    # RHEL / CentOS / Fedora
    sudo yum install -y ta-lib-devel
else
    echo "  WARNING: Unknown OS. Install TA-Lib C headers manually."
    echo "  See: https://github.com/TA-Lib/ta-lib-python#dependencies"
fi

# ── 3. Python TA-Lib wrapper ────────────────────────────────────────────────
echo ""
echo "► Installing Python ta-lib wrapper ..."
pip install ta-lib -q

# ── 4. uvloop (libuv-backed event loop) ────────────────────────────────────
echo ""
echo "► Installing uvloop ..."
pip install uvloop -q

# ── 5. bmoscon/orderbook (C extension) ─────────────────────────────────────
echo ""
echo "► Installing orderbook (C extension) ..."
pip install orderbook -q

# ── 6. All pure-Python / pre-compiled wheel dependencies ───────────────────
echo ""
echo "► Installing all other project dependencies ..."
pip install -q \
    cryptofeed \
    hftbacktest \
    numba \
    numpy \
    pandas \
    aiohttp \
    websockets \
    prometheus-client \
    pydantic \
    pyyaml \
    orjson \
    scipy \
    python-dotenv \
    pytest \
    pytest-asyncio

# ── 7. Verify everything imports cleanly ───────────────────────────────────
echo ""
echo "► Verifying imports ..."

$PYTHON -c "
failed = []
checks = [
    'talib', 'uvloop', 'orderbook',
    'cryptofeed', 'hftbacktest', 'numba',
    'numpy', 'pandas', 'aiohttp', 'websockets',
    'prometheus_client', 'pydantic', 'yaml',
    'orjson', 'scipy',
]
for mod in checks:
    try:
        __import__(mod)
        print(f'  OK  {mod}')
    except ImportError as e:
        print(f'  FAIL {mod}: {e}')
        failed.append(mod)

print()
if failed:
    print(f'FAILED: {failed}')
    exit(1)
else:
    print('All dependencies verified. Ready to build.')
"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Pre-flight complete. Activate env with:                 ║"
echo "║    source venv/bin/activate                              ║"
echo "║  Then hand off to Claude Code:                           ║"
echo "║    claude                                                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
