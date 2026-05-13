#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

UV_BIN="${UV_BIN:-/home/sunwenhao/.local/bin/uv}"
AGENT_CMD="${AGENT_CMD:-}"
MAX_KERNELS="${MAX_KERNELS:-1}"
QUICK="${QUICK:-1}"
RECORD_BASELINE="${RECORD_BASELINE:-1}"
PROMPT_ONLY="${PROMPT_ONLY:-0}"

PROMPT_DIR="$SCRIPT_DIR/workspace/agent_prompts"
LOG_DIR="$SCRIPT_DIR/workspace/agent_logs"
mkdir -p "$PROMPT_DIR" "$LOG_DIR" "$SCRIPT_DIR/workspace/results"

if [[ ! -x "$UV_BIN" ]]; then
  echo "ERROR: uv not found at $UV_BIN"
  exit 1
fi

if [[ -z "$AGENT_CMD" ]]; then
  if [[ "$PROMPT_ONLY" != "1" ]]; then
    if command -v codex >/dev/null 2>&1; then
      AGENT_CMD="codex {prompt_file}"
    elif command -v claude >/dev/null 2>&1; then
      AGENT_CMD="claude {prompt_file}"
    else
      echo "ERROR: No external agent CLI found."
      echo 'Set AGENT_CMD explicitly, for example:'
      echo '  AGENT_CMD="codex {prompt_file}" ./auto_optimize_simple.sh'
      echo '  AGENT_CMD="claude {prompt_file}" ./auto_optimize_simple.sh'
      echo 'Or generate prompts only:'
      echo '  PROMPT_ONLY=1 ./auto_optimize_simple.sh'
      exit 1
    fi
  fi
fi

bench_cmd=("$UV_BIN" run bench.py)
if [[ "$QUICK" == "1" ]]; then
  bench_cmd+=("--quick")
fi

current_ts() {
  date +"%Y%m%d_%H%M%S"
}

parse_metric() {
  local pattern="$1"
  local log_file="$2"
  grep "^${pattern}:" "$log_file" | tail -n1 | awk -F': ' '{print $2}' | tr -d '\r'
}

parse_kernel_from_next() {
  echo "$1" | grep -oE 'kernel_[^ ]+\.py' | head -n1
}

kernel_stem() {
  basename "${1%.py}"
}

select_kernel() {
  if [[ $# -gt 0 ]]; then
    local input="$1"
    if [[ "$input" == workspace/* ]]; then
      echo "$input"
    else
      echo "workspace/$input"
    fi
    return 0
  fi

  local next_out
  next_out="$("$UV_BIN" run orchestrate.py next)"
  echo "$next_out" >&2

  local kernel_name
  kernel_name="$(parse_kernel_from_next "$next_out")"
  if [[ -z "$kernel_name" ]]; then
    return 1
  fi
  echo "workspace/$kernel_name"
}

results_path_for_kernel() {
  echo "$SCRIPT_DIR/workspace/results/$(kernel_stem "$1")_results.tsv"
}

has_recorded_results() {
  local kernel_rel="$1"
  local path
  path="$(results_path_for_kernel "$kernel_rel")"
  [[ -f "$path" ]] && [[ $(wc -l < "$path") -gt 1 ]]
}

run_and_record_baseline() {
  local kernel_rel="$1"
  local kernel_name
  kernel_name="$(basename "$kernel_rel")"
  local ts
  ts="$(current_ts)"
  local log_file="$LOG_DIR/$(kernel_stem "$kernel_name")_baseline_${ts}.log"

  echo
  echo "==> Baseline benchmark: $kernel_name"
  "${bench_cmd[@]}" >"$log_file" 2>&1 || true

  local correctness throughput
  correctness="$(parse_metric "correctness" "$log_file")"
  throughput="$(parse_metric "throughput_tflops" "$log_file")"
  throughput="${throughput:-0}"

  echo "    correctness=${correctness:-N/A} throughput=${throughput} TFLOPS"

  if [[ "$correctness" != "PASS" ]]; then
    echo "ERROR: baseline correctness failed for $kernel_name"
    echo "Check log: $log_file"
    exit 1
  fi

  "$UV_BIN" run orchestrate.py record "$kernel_rel" "$throughput" kept "baseline"
}

generate_prompt() {
  local kernel_rel="$1"
  local prompt_file="$2"
  local kernel_name
  kernel_name="$(basename "$kernel_rel")"
  local kernel_stem_name
  kernel_stem_name="$(kernel_stem "$kernel_name")"
  local results_path
  results_path="$(results_path_for_kernel "$kernel_rel")"
  local run_bench_cmd="${bench_cmd[*]} > run.log 2>&1"

  cat >"$prompt_file" <<EOF
You are the external optimization agent for AutoKernel. Follow the repository's official Phase B loop in program.md.

Repo root: $SCRIPT_DIR
Current kernel target: $kernel_rel
Working file to edit: kernel.py
Save best kernel to: workspace/${kernel_stem_name}_optimized.py
Per-kernel results TSV: $results_path

Required commands:
- Read instructions:
  sed -n '160,380p' program.md
- Benchmark:
  $run_bench_cmd
- Parse benchmark:
  grep "correctness\\|throughput_tflops\\|latency_us\\|speedup_vs_pytorch\\|pct_peak_compute\\|pct_peak_bandwidth\\|bottleneck\\|peak_vram_mb" run.log
- On crash:
  tail -n 50 run.log
- Record experiment:
   $UV_BIN run orchestrate.py record $kernel_rel <throughput_tflops> <kept|revert|failed> "<description>"
- Ask orchestrator after experiments:
  $UV_BIN run orchestrate.py next

Workflow:
1. Focus only on $kernel_rel.
2. Modify only kernel.py.
3. Run one focused experiment at a time.
4. Keep only correctness=PASS and throughput improvement.
5. Revert slower or broken experiments.
6. When plateaued or when orchestrate says move on, save:
   cp kernel.py workspace/${kernel_stem_name}_optimized.py
7. End with a short summary of:
   - best throughput_tflops
   - speedup_vs_pytorch
   - what changed
   - final saved file

Constraints:
- Do not modify bench.py, reference.py, prepare.py, extract.py, profile.py, or orchestrate.py.
- Do not ask the human questions.
- Use the existing uv environment in this repo.
EOF
}

launch_agent() {
  local prompt_file="$1"
  if [[ "$PROMPT_ONLY" == "1" ]]; then
    echo "Prompt written to: $prompt_file"
    return 0
  fi

  echo
  echo "==> Launching external agent"
  echo "    AGENT_CMD: $AGENT_CMD"
  echo "    prompt:    $prompt_file"
  echo

  local cmd="$AGENT_CMD"
  local prompt_text
  prompt_text="$(cat "$prompt_file")"
  cmd="${cmd//\{prompt_file\}/$prompt_file}"
  cmd="${cmd//\{prompt_text\}/$prompt_text}"

  if [[ "$cmd" == "$AGENT_CMD" ]]; then
    cmd="$AGENT_CMD \"$prompt_file\""
  fi

  eval "$cmd"
}

optimize_one_kernel() {
  local kernel_rel="$1"
  local kernel_abs="$SCRIPT_DIR/$kernel_rel"
  local kernel_name
  kernel_name="$(basename "$kernel_rel")"
  local ts
  ts="$(current_ts)"
  local prompt_file="$PROMPT_DIR/$(kernel_stem "$kernel_name")_${ts}.txt"

  if [[ ! -f "$kernel_abs" ]]; then
    echo "ERROR: kernel file not found: $kernel_abs"
    exit 1
  fi

  cp "$kernel_abs" "$SCRIPT_DIR/kernel.py"
  echo "Prepared kernel.py from $kernel_rel"

  if [[ "$RECORD_BASELINE" == "1" ]] && ! has_recorded_results "$kernel_rel"; then
    run_and_record_baseline "$kernel_rel"
  fi

  generate_prompt "$kernel_rel" "$prompt_file"
  launch_agent "$prompt_file"
}

main() {
  local optimized=0
  local requested="${1:-}"

  while [[ "$optimized" -lt "$MAX_KERNELS" ]]; do
    local kernel_rel
    if [[ -n "$requested" ]]; then
      kernel_rel="$(select_kernel "$requested")"
      requested=""
    else
      if ! kernel_rel="$(select_kernel)"; then
        echo "No kernel selected by orchestrate.py. Likely all done."
        break
      fi
    fi

    echo
    echo "============================================================"
    echo "Agent optimizing: $kernel_rel"
    echo "============================================================"

    optimize_one_kernel "$kernel_rel"
    optimized=$((optimized + 1))
  done

  echo
  echo "Finished external-agent optimization handoff."
  echo "Prompts: $PROMPT_DIR"
  echo "Logs:    $LOG_DIR"
}

main "$@"
