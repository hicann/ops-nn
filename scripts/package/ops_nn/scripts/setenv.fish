#!/usr/bin/fish
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set param_mult_ver $argv[1]
set REAL_SHELL_PATH (realpath (command -v $argv[0]))
set CANN_PATH (cd (dirname $REAL_SHELL_PATH)/../../ && pwd)
if test -d "$CANN_PATH/ops_nn" -a test -d "$CANN_PATH/../latest"
    set INSATLL_PATH (cd (dirname $REAL_SHELL_PATH)/../../../ && pwd)
    if test -L "$INSATLL_PATH/latest/ops_nn"
        set _ASCEND_OPS_NN_PATH (cd $CANN_PATH/ops_nn && pwd)
        if test "$param_mult_ver" = "multi_version"
            set _ASCEND_OPS_NN_PATH (cd $INSATLL_PATH/latest/ops_nn && pwd)
        end
    end
elseif test -d "$CANN_PATH/ops_nn"
    set _ASCEND_OPS_NN_PATH (cd $CANN_PATH/ops_nn && pwd)
end

set -x ASCEND_OPS_NN_PATH $_ASCEND_OPS_NN_PATH

