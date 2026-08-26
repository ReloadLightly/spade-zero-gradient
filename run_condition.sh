#!/bin/bash
# S3 driver: run 6 rounds sequentially for one condition.
# seed = 2000 + round_index, identical across conditions (paired design).
COND="$1"; OUT="$2"; MUT="$3"
cd /home/user/spade-zero-gradient
for r in 0 1 2 3 4 5; do
  if [ -f "$OUT/.round_${r}_done" ]; then echo "round $r already done, skipping"; continue; fi
  echo "=== $COND round $r (seed $((2000+r))) start $(date -u +%H:%M:%S) ==="
  python3 -m szg.cli round --condition "$COND" --round-index "$r" \
      --n-envs 6 --G 3 --rng-seed $((2000+r)) \
      --backend claude-cli --designer-model sonnet --solver-model haiku \
      $MUT --out "$OUT"
  rc=$?
  if [ $rc -ne 0 ]; then echo "=== $COND round $r FAILED rc=$rc ==="; exit $rc; fi
  touch "$OUT/.round_${r}_done"
  echo "=== $COND round $r done $(date -u +%H:%M:%S) ==="
done
echo "=== $COND ALL ROUNDS COMPLETE ==="
