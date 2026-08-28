#!/bin/bash
# S3 driver: run 6 rounds sequentially for one condition.
# seed = 2000 + round_index, identical across conditions (paired design).
#
# A round is retried if it EXITS nonzero, or if it exits 0 but lost
# environments to backend failure at the designer call. The second case is
# the dangerous one: cmd_round catches design failures per-env, so a round
# where every env died still exits 0, scores fitness 0.0, and looks
# complete. That happened to C0 r4 and C1 r4 in the 2026-08-27 usage-limit
# outage and would have entered the primary endpoint as real data.
COND="$1"; OUT="$2"; MUT="$3"
MAX_TRIES=6
cd /home/user/spade-zero-gradient
for r in 0 1 2 3 4 5; do
  if [ -f "$OUT/.round_${r}_done" ]; then echo "round $r already done, skipping"; continue; fi
  try=1; ok=0
  while [ $try -le $MAX_TRIES ]; do
    echo "=== $COND round $r try $try/$MAX_TRIES (seed $((2000+r))) start $(date -u +%H:%M:%S) ==="
    python3 -m szg.cli round --condition "$COND" --round-index "$r" \
        --n-envs 6 --G 3 --rng-seed $((2000+r)) \
        --backend claude-cli --designer-model sonnet --solver-model haiku \
        $MUT --out "$OUT"
    rc=$?
    if [ $rc -eq 0 ] && python3 -m szg.round_health "$OUT" "$COND" "$r"; then ok=1; break; fi
    if [ $rc -eq 0 ]; then
      echo "=== $COND round $r try $try produced an UNUSABLE round; pruning and retrying ==="
    else
      echo "=== $COND round $r try $try FAILED rc=$rc; pruning partial records ==="
    fi
    python3 -m szg.prune_round "$OUT" "$COND" "$r"
    rm -f "$OUT"/envs/${COND}_r$(printf '%03d' $r)_*.py
    try=$((try+1))
    if [ $try -le $MAX_TRIES ]; then echo "backing off 300s before retry"; sleep 300; fi
  done
  if [ $ok -ne 1 ]; then echo "=== $COND round $r EXHAUSTED after $MAX_TRIES tries ==="; exit 1; fi
  touch "$OUT/.round_${r}_done"
  echo "=== $COND round $r done $(date -u +%H:%M:%S) ==="
done
echo "=== $COND ALL ROUNDS COMPLETE ==="
