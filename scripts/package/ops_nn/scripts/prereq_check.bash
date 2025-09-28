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

MIN_PIP_VERSION=19
PYTHON_VERSION=3.7.5
FILE_NOT_EXIST="0x0080"

log() {
    local content cur_date
    content=$(echo "$@" | cut -d" " -f2-)
    cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    echo "[OpsNN] [$cur_date] [$1]: $content"
}

log "INFO" "OpsNN do pre check started."

log "INFO" "Check pip version."
which pip3 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    log "WARNING" "\033[33mpip3 is not found.\033[0m"
fi

log "INFO" "Check python version."
curpath="$(dirname ${BASH_SOURCE:-$0})"
install_dir="$(realpath $curpath/..)"
common_interface=$(realpath $install_dir/script*/common_interface.bash)
if [ -f "$common_interface" ]; then
    . "$common_interface"
    py_version_check
fi

log "INFO" "OpsNN do pre check finished."