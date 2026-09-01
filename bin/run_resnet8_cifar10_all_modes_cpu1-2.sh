#!/usr/bin/env bash
set -u

cd "$(dirname "$0")"
mkdir -p run_log

LOG_DATE="$(date '+%y_%m_%d')"
CPU_CORES="1,2"
DIM_TAG="8x8x1"
GS="4096"
SIM="4000000"
THREAD_TAG="pre2_pe2_cpu1-2"
SUMMARY_LOG="run_log/run_log_${LOG_DATE}_resnet8_cifar10_batch_${THREAD_TAG}_${DIM_TAG}_gs${GS}_sim4e6.txt"

echo "[batch] start $(date '+%F %T')" | tee -a "$SUMMARY_LOG"
echo "[batch] cwd=$(pwd)" | tee -a "$SUMMARY_LOG"
echo "[batch] cores=${CPU_CORES}" | tee -a "$SUMMARY_LOG"
echo "[batch] order=exact abdtr fas sap-rle sap-rle-v2" | tee -a "$SUMMARY_LOG"

COMMON_ARGS=(
  -dimx 8 -dimy 8 -dimz 1
  -NNmodel resnet8_cifar10/resnet8_model.txt
  -NNweight resnet8_cifar10/resnet8_weight_fc_wb2parser.txt
  -NNweight_scale resnet8_cifar10/resnet8_weight_scale.txt
  -NNinput resnet8_cifar10/resnet8_input.txt
  -NNlabel resnet8_cifar10/resnet8_label.txt
  -mapping dir_x
  -groupsize "${GS}"
  -pe_log 0
  -sim "${SIM}"
  -stop_on_infer_done 1
  -precompute_threads 2
  -pe_compute_threads 2
  -thermal_update 0
)

FAS_TABLE_ARGS=(
  -NNfas_threshold_file resnet8_cifar10/resnet8_fas_threshold.txt
  -NNfas_level_table_file resnet8_cifar10/resnet8_fas_level_table.txt
)

SAP_TABLE_ARGS=(
  -sap_threshold_file resnet8_cifar10/resnet8_sap_threshold.txt
  -sap_level_table_file resnet8_cifar10/resnet8_sap_level_table.txt
)

run_case() {
  local name="$1"
  local logfile="$2"
  shift 2

  echo "[batch] ${name} start $(date '+%F %T') -> ${logfile}" | tee -a "$SUMMARY_LOG"
  taskset -c "${CPU_CORES}" ./build/noxim "${COMMON_ARGS[@]}" "$@" > "$logfile" 2>&1
  local rc=$?
  echo "[batch] ${name} exit_code=${rc} end $(date '+%F %T')" | tee -a "$SUMMARY_LOG"
  return "$rc"
}

overall=0

run_case "exact" \
  "run_log/run_log_${LOG_DATE}_resnet8_cifar10_exact_${THREAD_TAG}_${DIM_TAG}_gs${GS}_sim4e6.txt" \
  "${FAS_TABLE_ARGS[@]}" \
  -isapprox 0 -allzeropacket 0 -zero_skip 0 -acdc_abdtr 0 -is_sap_rle 0 -is_sap_rle_v2 0 \
  || overall=1

run_case "abdtr" \
  "run_log/run_log_${LOG_DATE}_resnet8_cifar10_abdtr_edgecfg_${THREAD_TAG}_${DIM_TAG}_gs${GS}_sim4e6.txt" \
  "${FAS_TABLE_ARGS[@]}" \
  -isapprox 0 -allzeropacket 0 -zero_skip 0 -acdc_abdtr 1 \
  -abdtr_drop_file resnet8_cifar10/resnet8_drop.txt \
  -abdtr_edge_drop_file resnet8_cifar10/resnet8_abdtr_edge_drop.txt \
  -is_sap_rle 0 -is_sap_rle_v2 0 \
  || overall=1

run_case "fas" \
  "run_log/run_log_${LOG_DATE}_resnet8_cifar10_fas_edgecfg_${THREAD_TAG}_${DIM_TAG}_gs${GS}_sim4e6.txt" \
  "${FAS_TABLE_ARGS[@]}" \
  -NNfas_edge_approx resnet8_cifar10/resnet8_fas_edge_approx.txt \
  -config_sel 0 \
  -isapprox 1 -allzeropacket 0 -zero_skip 0 -acdc_abdtr 0 -is_sap_rle 0 -is_sap_rle_v2 0 \
  || overall=1

run_case "sap-rle" \
  "run_log/run_log_${LOG_DATE}_resnet8_cifar10_saprle_edgecfg_${THREAD_TAG}_${DIM_TAG}_gs${GS}_sim4e6.txt" \
  "${SAP_TABLE_ARGS[@]}" \
  -sap_edge_approx resnet8_cifar10/resnet8_sap_edge_approx.txt \
  -config_sel 0 \
  -isapprox 0 -allzeropacket 0 -zero_skip 0 -acdc_abdtr 0 -is_sap_rle 1 -is_sap_rle_v2 0 \
  || overall=1

run_case "sap-rle-v2" \
  "run_log/run_log_${LOG_DATE}_resnet8_cifar10_saprlev2_edgecfg_${THREAD_TAG}_${DIM_TAG}_gs${GS}_sim4e6.txt" \
  "${SAP_TABLE_ARGS[@]}" \
  -sap_edge_approx resnet8_cifar10/resnet8_sap_edge_approx.txt \
  -config_sel 0 \
  -isapprox 0 -allzeropacket 0 -zero_skip 0 -acdc_abdtr 0 -is_sap_rle 0 -is_sap_rle_v2 1 \
  || overall=1

echo "[batch] finish $(date '+%F %T') overall=${overall}" | tee -a "$SUMMARY_LOG"
exit "$overall"
