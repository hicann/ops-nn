#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

echo "start run test case, please wait ..."
cd /home/taskspace
WORKSPACE=/home/taskspace

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0
source /usr/local/Ascend/cann/set_env.sh

log() {
  local dt
  dt=$(date '+%Y%m%d.%H%M%S')
  echo "===================================================================="
  echo "$dt : $*"
  echo "===================================================================="
}

log "init test case, please wait ..."
rm -rf /root/ascend/log

# ==============================
# 确定要测试的 ops 列表
# ==============================
declare -a ops
ops=("fatrelu_mul")

# ==============================
# 下载 single.tar.gz
# ==============================
log "start run test case, please wait ..."

DOWNLOAD_FILE=$(basename "${single_tar_url}")
echo "Starting to download file: ${DOWNLOAD_FILE}"
wget -nv --no-clobber "${single_tar_url}"

if [ ! -f "${DOWNLOAD_FILE}" ]; then
    echo "File ${DOWNLOAD_FILE} does not exist, no need to execute smoke test task"
    exit 0
fi

FILE_SIZE=$(stat -c%s "${DOWNLOAD_FILE}" 2>/dev/null || echo 0)
if [ "${FILE_SIZE}" -eq 0 ]; then
    echo "No compiled operators, no need to execute smoke test task"
    rm -f "${DOWNLOAD_FILE}"
    exit 0
fi
echo "File download completed, size ${FILE_SIZE}, starting decompression."

tar -ztf ${DOWNLOAD_FILE}
tar -zxf ${DOWNLOAD_FILE}

# ==============================
# 运行测试主循环
# ==============================

for op in "${ops[@]}"; do
  echo "Processing: $op"
  mode="eager"
  [ "$op" = "crop_and_resize" ] && mode="graph"
  source /usr/local/Ascend/cann/set_env.sh
  bash ${WORKSPACE}/scripts/ci/check_example.sh ${WORKSPACE}/pr_filelist.txt  2>&1 | tee -a ./run_test.log
  fatrel_arm_package=$(basename "${fatrelu_run_url}")
  wget -nv ${fatrelu_run_url}
  chmod +x ${fatrel_arm_package} && ./${fatrel_arm_package} 2>&1 | tee -a ./run_test.log
  bash build.sh --run_example "$op" "$mode" cust  2>&1 | tee -a ./run_test.log
  experimental_arm_package=$(basename "${experimental_run_url}")
  wget -nv "${experimental_run_url}" && chmod +x ${experimental_arm_package}
  echo 'y' | bash ${experimental_arm_package} --quiet && source /usr/local/Ascend/cann/set_env.sh && bash scripts/ci/check_experimental_example.sh ${WORKSPACE}/pr_filelist.txt 2>&1 | tee -a ./run_test.log
done

# ==============================
# 打包log
# ==============================
mkdir -p /root/ascend
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

# upload plog
if python3 /home/upload.py --bucket-name "ascend-ci" --action upload  --local-file "slog.tar.gz" --obs-object-key "${repo_name}/package/${pr_id}/${slog_name}"; then
  echo "::set-output var=plog_url:https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${repo_name}/package/${pr_id}/slog.tar.gz"
fi

# ==============================
# 检查 NPU 状态
# ==============================
log "checking NPU status ..."
mkdir -p ./npu_log
npu-smi info  2>&1 | tee ./npu_log/npu_info.log

# ==============================
# 检查测试结果
# ==============================
log "checking test results ..."

date_time=`date +%Y%m%d`"."`date +%H%M%S`
if grep -E '\b(FAIL|errors|fail|failed|error|ERROR:|Error|error:)\b' "./run_test.log" | grep -v "error)"; then
    echo "$date_time : run test case failed"
    exit 1
fi
