#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PIPELINE_OUTPUT_ROOT="/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/G_DFN监督基线/后台全流程"

run_name=""
pipeline_output_root="$DEFAULT_PIPELINE_OUTPUT_ROOT"
conda_env="gan-dfn"
help_requested="0"
forward_args=()

while (($# > 0)); do
  case "$1" in
    -h|--help)
      help_requested="1"
      forward_args+=("$1")
      shift
      ;;
    --run-name)
      if (($# < 2)); then
        echo "missing value for --run-name" >&2
        exit 1
      fi
      run_name="$2"
      forward_args+=("$1" "$2")
      shift 2
      ;;
    --pipeline-output-root)
      if (($# < 2)); then
        echo "missing value for --pipeline-output-root" >&2
        exit 1
      fi
      pipeline_output_root="$2"
      forward_args+=("$1" "$2")
      shift 2
      ;;
    --conda-env)
      if (($# < 2)); then
        echo "missing value for --conda-env" >&2
        exit 1
      fi
      conda_env="$2"
      forward_args+=("$1" "$2")
      shift 2
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$run_name" ]]; then
  run_name="layerwise_full_pipeline_$(date +%Y%m%d_%H%M%S)"
  forward_args=(--run-name "$run_name" "${forward_args[@]}")
fi

run_dir="${pipeline_output_root}/${run_name}"
log_dir="${run_dir}/logs"
log_path="${log_dir}/orchestrator_nohup.log"
pid_path="${run_dir}/orchestrator.pid"
state_path="${run_dir}/pipeline_state.json"
tmux_session_file="${run_dir}/tmux_session.txt"
runner_script="${run_dir}/run_in_tmux.sh"
mkdir -p "$log_dir"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found in PATH" >&2
  exit 1
fi

conda_cmd="${CONDA_EXE:-$(command -v conda)}"

if [[ "$help_requested" == "1" ]]; then
  exec "$conda_cmd" run --no-capture-output -n "$conda_env" python \
    "${SCRIPT_DIR}/run_layerwise_full_pipeline.py" "${forward_args[@]}"
fi

cmd=(
  "$conda_cmd" run --no-capture-output -n "$conda_env" python
  "${SCRIPT_DIR}/run_layerwise_full_pipeline.py"
  "${forward_args[@]}"
)

printf '%q ' "${cmd[@]}" > "${run_dir}/nohup_command.txt"
printf '\n' >> "${run_dir}/nohup_command.txt"

tmux_session="$(printf '%s' "gan_dfn_${run_name}" | tr -cs '[:alnum:]_-' '_')"
if tmux has-session -t "$tmux_session" 2>/dev/null; then
  echo "tmux session already exists: $tmux_session" >&2
  exit 1
fi

cat > "$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$PWD")
exec $(printf '%q ' "${cmd[@]}") >> $(printf '%q' "$log_path") 2>&1
EOF
chmod +x "$runner_script"

tmux new-session -d -s "$tmux_session" "bash $(printf '%q' "$runner_script")"
tmux_pid="$(tmux list-panes -t "${tmux_session}:0.0" -F '#{pane_pid}' | head -n 1)"
echo "${tmux_pid:-}" > "$pid_path"
echo "$tmux_session" > "$tmux_session_file"

echo "run_dir: $run_dir"
echo "pid: ${tmux_pid:-unknown}"
echo "log: $log_path"
echo "state: $state_path"
echo "command: ${run_dir}/nohup_command.txt"
echo "tmux_session: $tmux_session"
echo "tmux_session_file: $tmux_session_file"
echo "tmux_attach: tmux attach -t \"$tmux_session\""
echo "tail_cmd: tail -f \"$log_path\""
