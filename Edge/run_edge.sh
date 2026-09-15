#!/usr/bin/env bash
# run_edge.sh — Start health_agent + pipeline in a single command.
#
# Normal usage:
#   ./run_edge.sh                        # rtsp_push mode (default)
#   ./run_edge.sh --mode display
#   ./run_edge.sh --mode rtsp_push --rtsp-push-url rtsp://host:8554/jetson_A
#   ./run_edge.sh --load-policy predict_with_base --load-model formula
#   ./run_edge.sh --telemetry-interval 1.0   # the only supported cadence
#
# Background / SSH-safe run (survives disconnect):
#   nohup ./run_edge.sh >/dev/null 2>&1 &
#
# Press Ctrl+C once to gracefully stop all processes.

set -euo pipefail
export PYTHONUNBUFFERED=1

EDGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EDGE_DIR"

PID_FILE="${EDGE_PID_FILE:-${EDGE_LOG_DIR:-logs}/run_edge.pid}"

# Guard against duplicate instances running simultaneously
if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[run_edge] ERROR: Another run_edge process is already running (PID $OLD_PID)." >&2
        echo "[run_edge] Stop it first: kill $OLD_PID" >&2
        exit 1
    else
        # Stale PID file; remove safely
        rm -f "$PID_FILE"
    fi
fi

mkdir -p "${EDGE_LOG_DIR:-logs}"
echo "$$" > "$PID_FILE"

RUN_LOG="${RUN_LOG:-${EDGE_LOG_DIR:-logs}/run_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$RUN_LOG") 2>&1
echo "[run_edge] Runtime log: $RUN_LOG"
echo "[run_edge] Process PID: $$ (recorded in $PID_FILE)"

PYTHON="${PYTHON:-python3}"
MODE="${MODE:-rtsp_push}"
LOAD_POLICY="${LOAD_POLICY:-actual}"
LOAD_MODEL="${LOAD_MODEL:-formula}"
# ponytail: 1.0 s is locked — the single operational cadence.
TELEMETRY_INTERVAL="1.0"

# ── Node Identity & Deployment Guard ───────────────────────────────────────
# Edge/.env holds node-specific identity (NODE_ID, ADVERTISE_IP, RTSP_PUSH_URL).
# setup_system.sh provisions per-node settings (jetson_A: 192.168.212.20,
# jetson_B: 192.168.212.21, jetson_C: 192.168.212.22).
# Fleet deployment must explicitly exclude Edge/.env from rsync/scp/copy logic
# (e.g. rsync -avz --exclude='Edge/.env' ...) so per-device config is preserved.
if [[ -z "${NODE_ID:-}" ]] && [[ -f .env ]]; then
    NODE_ID="$(grep -E '^NODE_ID=' .env | cut -d= -f2- | tr -d '\r\n ' || true)"
fi
NODE_ID="${NODE_ID:-edge}"

# Guard: detect if fleet copy overwrote per-device .env with tracked jetson_A template
DETECTED_HOST=""
HOST_IPS="$(hostname -I 2>/dev/null || ip addr show 2>/dev/null || true)"

if [[ "$HOST_IPS" =~ 192\.168\.212\.21 ]]; then
    DETECTED_HOST="jetson_B"
elif [[ "$HOST_IPS" =~ 192\.168\.212\.22 ]]; then
    DETECTED_HOST="jetson_C"
elif [[ "$HOST_IPS" =~ 192\.168\.212\.20 ]]; then
    DETECTED_HOST="jetson_A"
fi

if [[ -n "$DETECTED_HOST" && "$DETECTED_HOST" != "jetson_A" ]]; then
    if [[ "$NODE_ID" == "jetson_A" ]]; then
        echo "[run_edge] ERROR: Detected host is $DETECTED_HOST, but NODE_ID is 'jetson_A'." >&2
        echo "[run_edge] Edge/.env was likely overwritten with tracked jetson_A template during fleet deployment." >&2
        echo "[run_edge] Exclude Edge/.env when syncing (rsync --exclude='Edge/.env') and re-provision:" >&2
        echo "[run_edge]   sudo ./setup_system.sh $DETECTED_HOST" >&2
        exit 1
    fi
    if [[ -f .env ]]; then
        ENV_ADV_IP="$(grep -E '^ADVERTISE_IP=' .env | cut -d= -f2- | tr -d '\r\n ' || true)"
        if [[ "$ENV_ADV_IP" == "192.168.212.20" ]]; then
            echo "[run_edge] ERROR: Detected host is $DETECTED_HOST, but .env has ADVERTISE_IP=192.168.212.20." >&2
            echo "[run_edge] Edge/.env was overwritten with tracked jetson_A template during fleet deployment." >&2
            echo "[run_edge] Exclude Edge/.env when syncing (rsync --exclude='Edge/.env') and re-provision:" >&2
            echo "[run_edge]   sudo ./setup_system.sh $DETECTED_HOST" >&2
            exit 1
        fi
    fi
fi

# Parse args — pipeline flags consumed here; rest forwarded to main.py
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --load-policy)
            [[ $# -lt 2 ]] && { echo "[run_edge] ERROR: --load-policy requires a value" >&2; exit 1; }
            LOAD_POLICY="$2"; shift 2 ;;
        --load-model)
            [[ $# -lt 2 ]] && { echo "[run_edge] ERROR: --load-model requires a value" >&2; exit 1; }
            LOAD_MODEL="$2"; shift 2 ;;
        --telemetry-interval)
            [[ $# -lt 2 ]] && { echo "[run_edge] ERROR: --telemetry-interval requires a value" >&2; exit 1; }
            TELEMETRY_INTERVAL="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# --- Normal pipeline path below this point ---

case "$LOAD_POLICY" in
    actual|predict_no_base|predict_with_base) ;;
    *)
        echo "[run_edge] ERROR: LOAD_POLICY must be: actual | predict_no_base | predict_with_base" >&2
        exit 1 ;;
esac

case "$LOAD_MODEL" in
    formula|dl) ;;
    *)
        echo "[run_edge] ERROR: LOAD_MODEL must be: formula | dl" >&2
        exit 1 ;;
esac

# ponytail: single 1.0 s cadence — reject any override attempt.
if [[ "$TELEMETRY_INTERVAL" != "1.0" ]]; then
    echo "[run_edge] ERROR: --telemetry-interval only accepts 1.0 (full cadence)" >&2
    exit 1
fi

export LOAD_POLICY LOAD_MODEL TELEMETRY_INTERVAL
echo "[run_edge] LOAD_POLICY=$LOAD_POLICY  LOAD_MODEL=$LOAD_MODEL  TELEMETRY_INTERVAL=${TELEMETRY_INTERVAL}s"

# CUDA scheduling hardening — Jetson B/C hard-freeze mitigation.
# Root cause: CUDA fence never signals when the GPU context wedges during NVDEC
# teardown, so cuda-EvtHandlr busy-spins on CPU0 → RCU stall → hard reset.
# blocking schedule + single connection reduce GPU-context wedge spin.
export CUDA_DEVICE_SCHEDULE=blocking
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---------------------------------------------------------------------------
# Cleanup every Edge command started by run_edge.sh.
# ---------------------------------------------------------------------------
_pids=()
_cleanup_done=0  # re-entrancy guard

_cleanup() {
    # ponytail: prevent recursive/double-run from nested signals.
    if [[ $_cleanup_done -ne 0 ]]; then
        return 0
    fi
    _cleanup_done=1
    # Clear traps so a second signal during cleanup does not re-enter.
    trap - EXIT INT TERM

    echo ""
    echo "[run_edge] Stopping Edge processes..."
    pkill -f "health_agent.py" 2>/dev/null || true
    pkill -f "main.py" 2>/dev/null || true

    # Give Python and GStreamer a bounded grace period before forcing exit.
    local deadline=$(($(date +%s) + 3))
    while [[ $(date +%s) -lt $deadline ]] && {
        pgrep -f "health_agent.py" >/dev/null ||
        pgrep -f "main.py" >/dev/null
    }; do
        sleep 0.2
    done

    pkill -KILL -f "health_agent.py" 2>/dev/null || true
    pkill -KILL -f "main.py" 2>/dev/null || true
    for pid in "${_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    rm -f "$PID_FILE"
    echo "[run_edge] Done."
}
trap _cleanup EXIT
trap '_cleanup; exit 130' INT
trap '_cleanup; exit 143' TERM

# ===========================================================================
# Start main.py pipeline (starts internal HealthAgent + PeerOrch)
# ===========================================================================
echo "[run_edge] Starting pipeline (mode=$MODE)..."

if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    taskset -c 1-7 "$PYTHON" main.py --mode "$MODE" &
else
    taskset -c 1-7 "$PYTHON" main.py "${EXTRA_ARGS[@]}" &
fi
_pids+=("$!")
PIPELINE_PID=${_pids[-1]}
echo "[run_edge] pipeline PID=$PIPELINE_PID"

# ---------------------------------------------------------------------------
# Watch loop — supervises the pipeline indefinitely, restarting on unexpected
# exit up to MAX_PIPELINE_RESTARTS before dying (so systemd can restart it).
# ---------------------------------------------------------------------------
MAX_PIPELINE_RESTARTS=5
PIPELINE_RESTART_COUNT=0
PIPELINE_BACKOFF=3

while true; do
    sleep 2

    if ! kill -0 "$PIPELINE_PID" 2>/dev/null; then
        echo "[run_edge] WARNING: pipeline exited unexpectedly. Restart attempt $((PIPELINE_RESTART_COUNT + 1))/$MAX_PIPELINE_RESTARTS..." >&2
        if [[ $PIPELINE_RESTART_COUNT -ge $MAX_PIPELINE_RESTARTS ]]; then
            echo "[run_edge] Max pipeline restarts reached ($MAX_PIPELINE_RESTARTS). Exiting to let systemd handle restart." >&2
            exit 1
        else
            sleep $PIPELINE_BACKOFF
            ((PIPELINE_RESTART_COUNT++)) || true
            PIPELINE_BACKOFF=$((PIPELINE_BACKOFF * 2))
            [[ $PIPELINE_BACKOFF -gt 30 ]] && PIPELINE_BACKOFF=30
        fi

        if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
            taskset -c 1-7 "$PYTHON" main.py --mode "$MODE" &
        else
            taskset -c 1-7 "$PYTHON" main.py "${EXTRA_ARGS[@]}" &
        fi
        PIPELINE_PID=$!
        _pids+=("$PIPELINE_PID")
        echo "[run_edge] pipeline restarted with PID=$PIPELINE_PID (restart count: $PIPELINE_RESTART_COUNT)"
        continue
    fi

    done
