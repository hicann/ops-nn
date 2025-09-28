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


if [ "$(id -u)" != "0" ]; then
    _LOG_PATH=$(echo "${HOME}")"/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
    _OPERATE_LOG_FILE="${_LOG_PATH}/operation.log"
else
    _LOG_PATH="/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
    _OPERATE_LOG_FILE="${_LOG_PATH}/operation.log"
fi

# log functions
getdate() {
    _cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    echo "${_cur_date}"
}

logandprint() {
    is_error_level=$(echo $1 | grep -E 'ERROR|WARN|INFO')
    if [ "${is_quiet}" != "y" ] || [ "${is_error_level}" != "" ]; then
        echo "[OpsNN] [$(getdate)] ""$1"
    fi
    echo "[OpsNN] [$(getdate)] ""$1" >> "${_INSTALL_LOG_FILE}"
}

# create opapi soft link
createrelativelysoftlink() {
    local src_path_="$1"
    local dst_path_="$2"
    local dst_parent_path_=$(dirname ${dst_path_})
    # echo "dst_parent_path_: ${dst_parent_path_}"
    local relative_path_=$(realpath --relative-to="$dst_parent_path_" "$src_path_")
    # echo "relative_path_: ${relative_path_}"
    if [ -L "$2" ]; then
        return 0
    fi
    ln -s "${relative_path_}" "${dst_path_}" 2> /dev/null
    if [ "$?" != "0" ]; then
        return 1
    else
        return 0
    fi
}

logwitherrorlevel() {
    _ret_status="$1"
    _level="$2"
    _msg="$3"
    if [ "${_ret_status}" != 0 ]; then
        if [ "${_level}" = "error" ]; then
            logandprint "${_msg}"
            exit 1
        else
            logandprint "${_msg}"
        fi
    fi
}