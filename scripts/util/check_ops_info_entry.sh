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

# 校验算子是否已生成ops-info条目，缺失则明确报错并拦截，避免进入kernel编译流程后在深处报模糊错误。
# 典型原因：def中声明的soc支持被版本门禁/条件编译（如METADEF_VERSION_NUM门禁）裁剪，
# opbuild解析libascend_all_ops.so时该soc的注册信息不存在。
# usage: check_ops_info_entry.sh <ini_file> <op_type> <compute_unit>
set -u

ini_file=$1
op_type=$2
compute_unit=$3

if [ -z "${ini_file}" ] || [ -z "${op_type}" ] || [ -z "${compute_unit}" ]; then
  echo "[ERROR] usage: bash $0 <ini_file> <op_type> <compute_unit>" >&2
  exit 1
fi

if [ ! -f "${ini_file}" ]; then
  echo "[ERROR] op:${op_type} ops-info ini not found: ${ini_file} (compute_unit=${compute_unit})" >&2
  exit 1
fi

if grep -q "^\[${op_type}\]" "${ini_file}"; then
  exit 0
fi

echo "[ERROR] op:${op_type} 未在 ops-info 中生成信息 (ini: ${ini_file##*/}, compute_unit=${compute_unit})，已提前拦截，未进入kernel编译流程。" >&2
echo "[ERROR] def中声明的${compute_unit}支持可能被版本门禁/条件编译裁剪（如METADEF_VERSION_NUM门禁要求更高版本toolkit），请核对def注册或升级构建环境的CANN包。" >&2
exit 1
