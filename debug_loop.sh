#!/bin/bash
# Run debug_one.py N times sequentially with different seeds.
# Each run takes ~5 min and appends to dashboard.

set -e
N=${1:-5}
LABEL=${2:-R9-prompts}
LEADER_REF=${3:-5dc2dab}
OURS_REF=${4:-HEAD}

cd "$(dirname "$0")"
echo "=== debug loop: $N runs · label=$LABEL · $LEADER_REF vs $OURS_REF ==="

for i in $(seq 1 $N); do
  SEED=$((RANDOM * 7919 + i))
  echo
  echo "─── run $i / $N · seed=$SEED ───"
  python debug_one.py --seed "$SEED" \
    --leader-ref "$LEADER_REF" --ours-ref "$OURS_REF" --label "$LABEL" \
    --judge-url http://localhost:8003/v1 \
    --judge-model zai-org/GLM-4.6V-Flash \
    2>&1 | tee -a /tmp/dashboard/debug_run.log | grep -E "category|subject|complex|challenge|best_score|status=|vote |AGGREGATE|debug page|=== "
  echo
done
echo "=== all $N runs done ==="
