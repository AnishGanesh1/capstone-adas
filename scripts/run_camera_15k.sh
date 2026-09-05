#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_camera_15k.sh - drive the camera dataset to 15,048 adverse-weather frames.
#
#   22 adverse presets x 4 towns = 88 cells
#   88 x 171 frames = 15,048 total
#
# Resumable: every (town, condition) already at 171 is skipped instantly, so
# rerunning after a crash, a reboot or a power cut costs nothing.
#
#   ~/capstone/scripts/run_camera_15k.sh
#   FRAMES=200 ~/capstone/scripts/run_camera_15k.sh      # different target
# ---------------------------------------------------------------------------
set -u

CP=${CP:-4000}
FRAMES=${FRAMES:-171}
VEHICLES=${VEHICLES:-45}
WALKERS=${WALKERS:-20}
MAX_TRIES=${MAX_TRIES:-25}
MIN_FREE_GB=${MIN_FREE_GB:-4}
OUT=${OUT:-$HOME/dataset/train}
TOWNS=${TOWNS:-"Town02 Town01 Town03 Town10HD_Opt"}

SCRIPTS="$HOME/capstone/scripts"
LOGS="$HOME/logs"
mkdir -p "$LOGS"

exec 9>/tmp/camera15k.lock
flock -n 9 || { echo "another run_camera_15k.sh is already running"; exit 0; }

log() { echo "$(date '+%F %T') | $*"; }

cd "$SCRIPTS" || { echo "no $SCRIPTS"; exit 1; }
# shellcheck disable=SC1090
source "$HOME/venv-carla/bin/activate"

# --- preflight -------------------------------------------------------------
FREE=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
log "free disk ${FREE} GB, need ${MIN_FREE_GB}"
[ "$FREE" -lt "$MIN_FREE_GB" ] && { log "ABORT: not enough disk"; exit 1; }

if ! ss -tln 2>/dev/null | grep -q ":${CP} "; then
  log "ABORT: nothing listening on port ${CP}. Start CARLA first."
  exit 2
fi
log "CARLA reachable on port ${CP}"

if ls "$OUT"/meta/*_clear_*.json >/dev/null 2>&1; then
  n=$(ls "$OUT"/meta/*_clear_*.json | wc -l)
  log "ABORT: $n clear frames still in $OUT - move them to ~/dataset/reference first"
  exit 3
fi

log "target: 88 cells x ${FRAMES} = $((88 * FRAMES)) frames"
START=$(ls "$OUT"/images 2>/dev/null | wc -l)
log "starting from ${START} images"

python3 reset_world.py --port "$CP" >>"$LOGS/cam15k.log" 2>&1 || true

# --- capture ---------------------------------------------------------------
T0=$(date +%s)
for T in $TOWNS; do
  log "########## $T ##########"
  for i in $(seq 1 "$MAX_TRIES"); do
    python3 capture_drive.py --port "$CP" --town "$T" --tier adverse \
            --frames "$FRAMES" --no-lidar \
            --vehicles "$VEHICLES" --walkers "$WALKERS" \
            --out "$OUT" 2>&1 | tee -a "$LOGS/cam15k_$T.log"
    rc=${PIPESTATUS[0]}
    if [ $rc -eq 0 ]; then
      log "$T complete ($(ls "$OUT"/images | wc -l) images total)"
      break
    fi
    if [ $rc -eq 3 ]; then
      log "STOPPED: disk guard tripped. Free space and rerun."
      exit 3
    fi
    log "$T died rc=$rc on attempt $i - resetting world"
    python3 reset_world.py --port "$CP" >/dev/null 2>&1 || true
    sleep 8
  done
done

# --- report ----------------------------------------------------------------
END=$(ls "$OUT"/images | wc -l)
EL=$(( $(date +%s) - T0 ))
log "=========================================="
log "started with ${START}, now ${END}  (+$((END - START)) new)"
log "elapsed $((EL / 60)) min"
df -h "$HOME" | tail -1
python3 count_dataset.py --root "$OUT" | tail -5
