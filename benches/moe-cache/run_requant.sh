#!/usr/bin/env bash
# run_requant.sh - requantize ONLY the 144 expert tensors to iq2_xxs, leaving the other 1080
# tensors at their exact current types. This is the lossy-cached-tier experiment: fewer bytes
# per miss -> less PCIe traffic per token. Quality is priced separately with llama-perplexity.
set -u
export PATH=/opt/software/cuda/13.2.1/bin:$PATH
export LD_LIBRARY_PATH=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin:/opt/software/cuda/13.2.1/lib64:${LD_LIBRARY_PATH:-}
Q=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin/llama-quantize
M=/mnt/SSD/qwen3.8-flash-next-apex-mini/Qwen3.8-Flash-Next-APEX-I-Mini-00001-of-00006.gguf
OUT=/mnt/SSD/qwen3.8-flash-next-apex-mini-q2k-exp/Qwen3.8-Flash-Next-APEX-I-Mini-q2k-00001-of-00006.gguf
T=/mnt/WorkDisk/harness/multiturn-ctx/expert-quant-all.txt
mkdir -p "$(dirname "$OUT")"
echo "########## DRY (confirm the size actually drops) ##########"
"$Q" --dry-run --allow-requantize --tensor-type-file "$T" "$M" Q2_K 2>&1 | tail -6
echo "########## REAL ##########"
date
"$Q" --allow-requantize --tensor-type-file "$T" "$M" "$OUT" Q2_K $(nproc) 2>&1 | tail -8
echo "exit=$? at $(date)"
du -sh "$(dirname "$OUT")"
echo "########## DONE ##########"
