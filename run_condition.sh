#!/bin/bash
# S3 driver: run 6 rounds sequentially for one condition.
# seed = 2000 + round_index, identical across conditions (paired design).
# A round that fails is pruned of its partial records and retried, because a
# transient backend outage (claude -p rc=1) took out all three conditions at
# round 2 on 2026-08-26 and left orphaned env/episode/memory rows behind.
COND="$1"; OUT="$2"; MUT="$3"
MAX_TRIES=4
cd /home/user/spade-zero-gradient
for r in 0 1 2 3 4 5; do
  if [ -f "$OUT/.round_${r}_done" ]; then echo "round $r already done, skipping"; continue; fi
  try=1
  while [ $try -le $MAX_TRIES ]; do
    echo "=== $COND round $r try $try/$MAX_TRIES (seed $((2000+r))) start $(date -u +%H:%M:%S) ==="
    python3 -m szg.cli round --condition "$COND" --round-index "$r" \
        --n-envs 6 --G 3 --rng-seed $((2000+r)) \
        --backend claude-cli --designer-model sonnet --solver-model haiku \
        $MUT --out "$OUT"
    rc=$?
    if [ $rc -eq 0 ]; then break; fi
    echo "=== $COND round $r try $try FAILED rc=$rc; pruning partial records ==="
    python3 -m szg.prune_round "$OUT" "$COND" "$r"
    try=$((try+1))
    if [ $try -le $MAX_TRIES ]; then echo "backing off 120s before retry"; sleep 120; fi
  done
  if [ $rc -ne 0 ]; then echo "=== $COND round $r EXHAUSTED after $MAX_TRIES tries ==="; exit $rc; fi
  touch "$OUT/.round_${r}_done"
  echo "=== $COND round $r done $(date -u +%H:%M:%S) ==="
done
echo "=== $COND ALL ROUNDS COMPLETE ==="
