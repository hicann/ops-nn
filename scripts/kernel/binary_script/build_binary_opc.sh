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

source get_threadnum_with_op.sh

main() {
  echo "[INFO]excute file: $0"
  if test $# -lt 3; then
    echo "[ERROR]input error"
    echo "[ERROR]bash $0 {op_type} {soc_version} {output_path} {opc_info_csv}(optional)"
    exit 1
  fi
  local op_type=$1
  local soc_version=$2
  local output_path=$3
  local enable_debug=$4
  local enable_oom=$5
  local workdir=$(
    cd $(dirname $0)
    pwd
  )

  local task_path=$output_path/opc_cmd

  test -d "$task_path/" || mkdir -p $task_path/
  local lock_file="$output_path/.$soc_version.cleaned.lock"
  if [[ ! -f "$lock_file" ]]; then
    rm -f $task_path/*.sh
    touch "$lock_file" || {
      echo "[WARNING] Failed to create lock file"
    }
  fi

  result=$(bash build_binary_opc_gen_task.sh $op_type $soc_version $output_path $task_path $enable_debug $enable_oom)
  local gen_res=$?
  if [ $gen_res -ne 0 ]; then
    echo -e "[ERROR] [$op_type]build binary single op gen task failed with ErrorCode[$gen_res]."
    echo -e "Command executed: build_binary_single_op_gen_task.sh $op_type $soc_version $output_path $task_path $enable_debug $enable_oom"
    echo -e "Error output: \n $result"
    return
  fi
  echo "$result\n"
}
set -o pipefail
main "$@" | gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
