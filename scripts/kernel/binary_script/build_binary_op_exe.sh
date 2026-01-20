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

main() {
  echo "[INFO]excute file: $0"
  if [ $# -lt 2 ]; then
    echo "[ERROR]input error"
    echo "[ERROR]bash $0 {out_path} {task_id}"
    exit 1
  fi
  local output_path="$1"
  local idx="$2"

  result=$(bash build_binary_op_exe_task.sh $output_path $idx)
  local gen_res=$?
  if [ $gen_res -ne 0 ]; then
    echo -e "[ERROR] build binary single op failed with ErrorCode[$gen_res]."
    echo -e "For more detail: please set env:"
    echo -e "export ASCEND_GLOBAL_LOG_LEVEL=0"
    echo -e "export ASCEND_SLOG_PRINT_TO_STDOUT=1"
    echo -e "and run again, then check kernel log in $output_path/build_logs"
    return
  fi
  echo "$result"
}
set -o pipefail
main "$@" | gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
