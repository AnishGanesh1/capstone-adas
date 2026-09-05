#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_tl_15k.sh - build 15,048 traffic-light images using teleport capture.
#
#   22 adverse presets x 4 towns = 88 cells,  88 x 171 = 15,048
#
# Every frame is captured from a pose on a traffic light's approach lane, so
# almost all of them contain a labelled light. Light states are frozen and
# cycled R/Y/G so the three classes come out balanced.
#
#   ~/capstone/scripts/run_tl_15k.sh
#   FRAMES=200 ~/capstone/scripts/run_tl_15k.sh
#   OUT=~/dataset/train ~/capstone/scripts/run_tl_15k.sh    # top up the old set
# ---------------------------------------------------------------------------
set -u

CP=${CP:-4000}
FRAMES=${FRAMES:-171}
VEHICLES=${VEHICLES:-35}
WALKERS=${WALKERS:-15}
MAX_TRIES=${MAX_TRIES:-25}
MIN_FREE_GB=${MIN_FREE_GB:-4}
OUT=${OUT:-$HOME/dataset/train_tl}
TOWNS=${TOWNS:-"Town02 Town01 Town03 Town10HD_Opt"}
BACKGROUND=${BACKGROUND:-0.03}

SCRIPTS="$HOME/capstone/scripts"
LOGS="$HOME/logs"
mkdir -p "$LOGS"

exec 9>/tmp/tl15k.lock
flock -n 9 || { echo "another run_tl_15k.sh is already running"; exit 0; }

log() { echo "$(date '+%F %T') | $*"; }

cd "$SCRIPTS" || { echo "no $SCRIPTS"; exit 1; }
# shellcheck disable=SC1090
source "$HOME/venv-carla/bin/activate"

# ---- preflight ------------------------------------------------------------
FREE=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
log "free disk ${FREE} GB (need ${MIN_FREE_GB})"
[ "$FREE" -lt "$MIN_FREE_GB" ] && { log "ABORT: not enough disk"; exit 1; }

if ! ss -tln 2>/dev/null | grep -q ":${CP} "; then
  log "ABORT: nothing listening on port ${CP} - start CARLA first"
  exit 2
fi
log "CARLA reachable on port ${CP}"

if pgrep -af 'capture_drive.py|capture_lidar.py' | grep -qv grep; then
  log "ABORT: another capture client is running. Two clients purge each other's actors."
  pgrep -af 'capture_drive.py|capture_lidar.py'
  exit 4
fi

mkdir -p "$OUT"/{images,labels,meta}
if ls "$OUT"/meta/*_clear_*.json >/dev/null 2>&1; then
  n=$(ls "$OUT"/meta/*_clear_*.json | wc -l)
  log "ABORT: $n clear frames in $OUT - move them to ~/dataset/reference first"
  exit 3
fi

log "target: 88 cells x ${FRAMES} = $((88 * FRAMES)) images"
log "output: $OUT"
START=$(ls "$OUT"/images 2>/dev/null | wc -l)
log "starting from ${START} images"

python3 reset_world.py --port "$CP" >>"$LOGS/tl15k.log" 2>&1 || true

# ---- capture --------------------------------------------------------------
T0=$(date +%s)
for T in $TOWNS; do
  log "########## $T ##########"
  for i in $(seq 1 "$MAX_TRIES"); do
    python3 capture_tl_focused.py --port "$CP" --town "$T" --tier adverse \
            --frames "$FRAMES" --vehicles "$VEHICLES" --walkers "$WALKERS" \
            --background "$BACKGROUND" --min-free-gb "$MIN_FREE_GB" \
            --out "$OUT" 2>&1 | tee -a "$LOGS/tl15k_$T.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
      log "$T complete - $(ls "$OUT"/images | wc -l) images total"
      break
    fi
    if [ "$rc" -eq 3 ]; then
      log "STOPPED: disk guard tripped. Free space and rerun."
      exit 3
    fi
    log "$T died rc=$rc on attempt $i - resetting world"
    python3 reset_world.py --port "$CP" >/dev/null 2>&1 || true
    sleep 8
  done
done

# ---- report ---------------------------------------------------------------
END=$(ls "$OUT"/images | wc -l)
EL=$(( $(date +%s) - T0 ))
log "=========================================="
log "started ${START}, now ${END}  (+$((END - START)) new)"
log "elapsed $((EL / 60)) min   rate $(echo "scale=2; ($END-$START)/$EL" | bc) img/s"

cd "$OUT/labels" || exit 0
tot=$(ls | wc -l); emp=$(find . -maxdepth 1 -size 0 | wc -l)
log "positives: $((tot - emp)) of $tot  ($(( (tot-emp)*100/(tot>0?tot:1) ))%)"
log "class balance (0=red 1=yellow 2=green 3=off):"
cat ./*.txt 2>/dev/null | awk '{print $1}' | sort | uniq -c

cd "$SCRIPTS"
python3 count_dataset.py --root "$OUT" | tail -5
df -h "$HOME" | tail -1
