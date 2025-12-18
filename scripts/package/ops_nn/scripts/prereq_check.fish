#!/usr/bin/fish
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set ILE_NOT_EXIST "0x0080"

set cur_date (date +"%Y-%m-%d %H:%M:%S")
echo "[OpsNN][$cur_date][INFO]: Start pre installation check of ops_nn module."
which python3 >/dev/null
if test ! $status -eq 0
    set cur_date (date +"%Y-%m-%d %H:%M:%S")
    exit 0
end
set python_version (python3 --version 2>/dev/null)
set val (echo $python_version|grep -i '3.7.5')
if test ! $status -eq 0
    set cur_date (date +"%Y-%m-%d %H:%M:%S")
    exit 0
else
    set cur_date (date +"%Y-%m-%d %H:%M:%S")
    exit 0
end
