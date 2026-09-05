#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_dataset_15k.sh - 15,000 images: ~12,000 with traffic lights, ~3,000 street.
#
#   6 towns x 22 adverse presets, captured at 1600x900
#
#   train  Town01 Town03 Town04 Town05      94 pos + 24 neg per preset
#   val    Town02                           94 pos + 24 neg per preset
#   test   Town10HD_Opt  (reduced share)    73 pos + 18 neg per preset
#
#   positives : capture_tl_focused.py  - teleported to light approach lanes,
#               states frozen and cycled so R/Y/G come out balanced
#   negatives : capture_drive.py --negatives-only - real driving through the
#               map, keeping only frames with no light in view
#
# Resumable per (town, condition, mode). Rerun after any crash.
#
#   ~/capstone/scripts/run_dataset_15k.sh
#   POS=120 NEG=30 ~/capstone/scripts/run_dataset_15k.sh     # scale up
# ---------------------------------------------------------------------------
set -u

CP=${CP:-4000}
OUT=${OUT:-$HOME/dataset/final}
POS=${POS:-94}                 # positives per preset, train/val towns
NEG=${NEG:-24}                 # negatives per preset, train/val towns
POS_TEST=${POS_TEST:-73}       # positives per preset, Town10
NEG_TEST=${NEG_TEST:-18}       # negatives per preset, Town10
TOWNS=${TOWNS:-"Town01 Town02 Town03 Town04 Town05"}
TEST_TOWN=${TEST_TOWN:-Town10HD_Opt}
VEHICLES=${VEHICLES:-35}
WALKERS=${WALKERS:-15}
MAX_TRIES=${MAX_TRIES:-25}
MIN_FREE_GB=${MIN_FREE_GB:-5}

SCRIPTS="$HOME/capstone/scripts"
LOGS="$HOME/logs"; mkdir -p "$LOGS"

exec 9>/tmp/dataset15k.lock
flock -n 9 || { echo "another run_dataset_15k.sh is running"; exit 0; }
log() { echo "$(date '+%F %T') | $*"; }

cd "$SCRIPTS" || exit 1
# shellcheck disable=SC1090
source "$HOME/venv-carla/bin/activate"

# ---- preflight ------------------------------------------------------------
FREE=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
log "free disk ${FREE} GB (need ${MIN_FREE_GB})"
[ "$FREE" -lt "$MIN_FREE_GB" ] && { log "ABORT: not enough disk"; exit 1; }

ss -tln 2>/dev/null | grep -q ":${CP} " || { log "ABORT: CARLA not on port ${CP}"; exit 2; }
log "CARLA reachable on ${CP}"

if pgrep -af 'capture_drive.py|capture_lidar.py|capture_tl_focused.py' | grep -qv grep; then
  log "ABORT: another capture client is running - two clients purge each other"
  exit 4
fi

python3 - <<'EOF' || exit 5
import sys; sys.path.insert(0, '.')
import config as C
print('  resolution : %dx%d' % (C.IMAGE_W, C.IMAGE_H))
print('  train      : %s' % C.TOWNS_TRAIN)
print('  val        : %s' % C.TOWNS_VAL)
print('  test       : %s' % C.TOWNS_TEST)
print('  presets    : %d adverse' % len(C.CONDITIONS_ADVERSE))
assert C.IMAGE_W == 1600, 'config.py not patched to 1600x900'
assert 'Town05' in C.ALL_TOWNS, 'config.py missing Town04/Town05'
EOF

mkdir -p "$OUT"/{images,labels,meta}
log "target: 5 towns x 22 x ($POS+$NEG) + $TEST_TOWN x 22 x ($POS_TEST+$NEG_TEST)"
log "      = $((5*22*(POS+NEG) + 22*(POS_TEST+NEG_TEST))) images"
log "output: $OUT"
START=$(ls "$OUT"/images 2>/dev/null | wc -l)
log "starting from $START"

python3 reset_world.py --port "$CP" >>"$LOGS/ds15k.log" 2>&1 || true

# ---- helpers --------------------------------------------------------------
retry () {                     # retry <label> <logfile> <cmd...>
  local label=$1 lf=$2; shift 2
  for i in $(seq 1 "$MAX_TRIES"); do
    "$@" 2>&1 | tee -a "$lf"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && { log "  $label OK"; return 0; }
    [ "$rc" -eq 3 ] && { log "  STOPPED: disk guard"; return 3; }
    log "  $label died rc=$rc attempt $i - resetting"
    python3 reset_world.py --port "$CP" >/dev/null 2>&1 || true
    sleep 8
  done
  log "  $label gave up after $MAX_TRIES"
  return 1
}

positives () {                 # positives <town> <n>
  retry "$1 POS" "$LOGS/ds15k_$1_pos.log" \
    python3 capture_tl_focused.py --port "$CP" --town "$1" --tier adverse \
            --frames "$2" --vehicles "$VEHICLES" --walkers "$WALKERS" \
            --background 0.0 --min-lights 1 --min-free-gb "$MIN_FREE_GB" \
            --out "$OUT"
}

negatives () {                 # negatives <town> <n>
  retry "$1 NEG" "$LOGS/ds15k_$1_neg.log" \
    python3 capture_drive.py --port "$CP" --town "$1" --tier adverse \
            --frames "$2" --no-lidar --negatives-only --tag neg \
            --vehicles "$VEHICLES" --walkers "$WALKERS" \
            --out "$OUT"
}

# ---- capture --------------------------------------------------------------
T0=$(date +%s)
for T in $TOWNS; do
  log "########## $T  (train/val) ##########"
  positives "$T" "$POS"; rc=$?
  if [ "$rc" -eq 3 ]; then log "disk guard - stopping"; exit 3; fi
  negatives "$T" "$NEG"; rc=$?
  if [ "$rc" -eq 3 ]; then log "disk guard - stopping"; exit 3; fi
done

log "########## $TEST_TOWN  (test, reduced share) ##########"
positives "$TEST_TOWN" "$POS_TEST"
negatives "$TEST_TOWN" "$NEG_TEST"

# ---- report ---------------------------------------------------------------
END=$(ls "$OUT"/images | wc -l)
EL=$(( $(date +%s) - T0 ))
log "=========================================="
log "started $START, now $END  (+$((END-START)) new)  in $((EL/60)) min"

cd "$OUT/labels" || exit 0
tot=$(ls | wc -l); emp=$(find . -maxdepth 1 -size 0 | wc -l)
log "with a light : $((tot-emp))"
log "street only  : $emp"
log "class counts (0=red 1=yellow 2=green 3=off):"
cat ./*.txt 2>/dev/null | awk '{print $1}' | sort | uniq -c

cd "$SCRIPTS"
python3 count_dataset.py --root "$OUT" | tail -6
df -h "$HOME" | tail -1
