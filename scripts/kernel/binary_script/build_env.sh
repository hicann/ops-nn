#!/bin/bash
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

declare -A SOC_MAP
SOC_MAP=([Ascend310P]="Ascend310P3"
          [Ascend310B]="Ascend310B1"
          [Ascend910]="Ascend910A"
          [Ascend910B]="Ascend910B1"
          [Ascend910_93]="Ascend910_9391"
          [Ascend910_95]="Ascend910_9599"
)

OPC_TASK_NAME="opc_cmd.sh"
OUT_TASK_NAME="out_cmd.sh"
OP_CATEGORY_LIST="activation conv foreach vfusion index loss matmul norm optim pooling quant rnn control"

function trans_soc() {
  local _soc_input=$1
  local _soc_input_lower=${_soc_input,,}
  for key in ${!SOC_MAP[*]}; do
    key_lower=${key,,}
    if [ "${_soc_input_lower}" == "${key_lower}" ]; then
      echo ${SOC_MAP[$key]}
      return
    fi
  done

  echo ${_soc_input}
}
