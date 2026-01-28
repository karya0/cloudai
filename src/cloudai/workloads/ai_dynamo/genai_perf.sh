function launch_genai_perf()
{
  local genai_perf_arguments=$(array_to_args genai_perf_args)
  log "Launching genai-perf with args: $genai_perf_arguments ${genai_perf_args["--extra-args"]}"

  ${dynamo_args["genai-perf-cmd"]} ${genai_perf_arguments} ${genai_perf_args["--extra-args"]} > ${RESULTS_DIR}/genai_perf.log 2>&1

  log "Done with genai-perf run"

  local num_gpus="$(_gpus_per_node)"
  local total_gpus=$(( $num_gpus * $SLURM_JOB_NUM_NODES ))

  profile_path=$(find . -type f -name "profile_genai_perf.csv" -print -quit)
  if [[ -f "$profile_path" ]]; then
    python3 /cloudai_install/calc_percentile_csv.py $profile_path -o $RESULTS_DIR/genai_perf_report.csv
    output_tokens_per_second=$(grep "output_tokens_per_second" $profile_path | awk '{print $2}')
    output_tokens_per_second_per_gpu=$(( $output_tokens_per_second / $total_gpus ))
    grep ".*,.*,.*,.*" $profile_path > $RESULTS_DIR/genai_perf_report.csv
    echo "Output tokens per second per gpu,$output_tokens_per_second_per_gpu,0,0,0,0,0,0,0,0,0,0,0" >> $RESULTS_DIR/genai_perf_report.csv
  fi
}
