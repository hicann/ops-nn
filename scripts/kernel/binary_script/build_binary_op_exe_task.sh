#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -e

main() {
  echo "[INFO]excute file: $0"
  if [ $# -lt 2 ]; then
    echo "[ERROR]input error"
    echo "[ERROR]bash $0 {out_path} {task_id}"
    exit 1
  fi
  local output_path="$1"
  local idx="$2"
  local workdir=$(
    cd $(dirname $0)
    pwd
  )
  local task_path=$output_path/opc_cmd
  mkdir -p $output_path/build_logs/

  source build_env.sh
  opc_cmd_file="${task_path}/${OPC_TASK_NAME}"

  if [[ -f "${opc_cmd_file}" && "$idx" =~ ^[0-9]+$ && "$idx" -gt 0 ]]; then
    total_lines=$(wc -l < "${opc_cmd_file}")
    if [ "$idx" -gt "$total_lines" ]; then
      echo "[ERROR]task $idx is bigger than file:$opc_cmd_file lines: $total_lines, please check."
      exit 1
    fi
    # step1: do compile kernel
    set +e
    cmd_task=$(sed -n ''${idx}'p;' ${opc_cmd_file})
    key=$(echo "${cmd_task}" | grep -oP '\w*\.json_\d*')
    echo "[INFO]exe_task: begin to build kernel with cmd: ${cmd_task}."
    start_time=$(date +%s)

    log_file="${output_path}/build_logs/${key}.log"

    start_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$start_timestamp] Build started: ${cmd_task}" > "$log_file"
    timeout 7200 ${cmd_task} >> "$log_file" 2>&1
    compile_rc=$?
    set -e
    end_time=$(date +%s)
    exe_time=$((end_time - start_time))
    if [ ${compile_rc} -ne 0 ]; then
      if [ ${compile_rc} -eq 124 ]; then
        echo "[ERROR]exe_task: build kernel TIMEOUT, op name: ${key}. Run this command again for debug:"
        echo "[ERROR]${cmd_task}"
      else
        echo "[ERROR]exe_task: build kernel FAILED, op name: ${key}. Run this command again for debug:"
        echo "[ERROR]${cmd_task}"
      fi
      cat ${log_file}
      exit ${compile_rc}
    fi
    end_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$end_timestamp] exe_time: $exe_time s" >> "$log_file"
    echo "[INFO]exe_task: end to build kernel: ${key} --exe_time=${exe_time}"
  fi
}
set -o pipefail
main "$@" | gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
