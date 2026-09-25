#!/usr/bin/env bash
# Step 2 throughput matrix for the GPU PC (see scripts/bench_train.py). Run from the repo root:
#   bash scripts/bench_gpupc.sh results/throughput_gpupc
# Phases: single (one job at a time, idle machine), concurrency (N identical jobs at once).
set -euo pipefail
OUT=${1:-results/throughput_gpupc}
PY=${PY:-.venv/Scripts/python}
mkdir -p "$OUT"
bench() { "$PY" scripts/bench_train.py "$@" 2>/dev/null | tail -1; }

# 1. Teacher forcing, default model, chunk 256: CPU (1 thread = ablation setting; 4 threads for reference) vs CUDA.
for b in 32 64 128; do
  bench --device cuda --batch-size $b --warmup 50 --steps 400 --label single --json "$OUT/single.jsonl"
  bench --device cpu --threads 1 --batch-size $b --warmup 5 --steps 40 --label single --json "$OUT/single.jsonl"
  bench --device cpu --threads 4 --batch-size $b --warmup 5 --steps 40 --label single --json "$OUT/single.jsonl"
done
# 2. Reconstructed-context modes (sequential per-bit rollout inside each chunk).
for m in rollout scheduled; do
  for b in 32 64 128; do
    bench --device cuda --mode $m --batch-size $b --warmup 3 --steps 20 --label single --json "$OUT/single.jsonl"
    bench --device cpu --threads 1 --mode $m --batch-size $b --warmup 2 --steps 8 --label single --json "$OUT/single.jsonl"
  done
done
# 3. Position-feature model (later stages use it), teacher forcing, CUDA.
for b in 32 64 128; do
  bench --device cuda --batch-size $b --plane-dim 8 --row-dim 8 --col-dim 8 --warmup 50 --steps 400 \
    --label single_rowcol --json "$OUT/single.jsonl"
done

# 4. Concurrency: N identical jobs started together; per-job throughput vs N.
concurrent() {  # name n args...
  local name=$1 n=$2; shift 2
  for i in $(seq 1 "$n"); do
    bench "$@" --label "${name}_n${n}" --json "$OUT/conc_${name}_n${n}_$i.jsonl" > /dev/null &
  done
  wait
  cat "$OUT"/conc_"${name}"_n"${n}"_*.jsonl >> "$OUT/concurrency.jsonl"
  rm "$OUT"/conc_"${name}"_n"${n}"_*.jsonl
}
for n in 1 2 4 6 8; do concurrent cuda_tf_b32 $n --device cuda --batch-size 32 --warmup 100 --steps 2000; done
for n in 1 2 4; do concurrent cuda_tf_b128 $n --device cuda --batch-size 128 --warmup 50 --steps 1000; done
for n in 1 2 4 6 8; do concurrent cuda_rollout_b32 $n --device cuda --mode rollout --batch-size 32 --warmup 3 --steps 30; done
for n in 1 4 8 10 12; do concurrent cpu_tf_b32 $n --device cpu --threads 1 --batch-size 32 --warmup 5 --steps 150; done
echo done
