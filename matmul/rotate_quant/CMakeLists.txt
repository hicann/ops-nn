# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 设置算子定义时支持的芯片类型
set(SUPPORT_COMPUTE_UNIT "ascend910b" "ascend910_93" "ascend950")
# 设置每种芯片类型对应的tiling文件目录，即采用op_host目录下哪个文件夹下的tiling文件编译
set(SUPPORT_TILING_DIR "arch22" "arch22" "arch35")
add_all_modules_sources(
    ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
    TILING_DIR ${SUPPORT_TILING_DIR}
    DISABLE_IN_OPP TRUE
)

# [COMPUTE_UNITS ascendxx] soc版本，支持配置多个unit，缺省为配置所有的soc
# [SIMPLIFIED_KEY 0/None] 缺省为0，设置为None则不携带该参数
# [AUTO_SYNC false] 同步选项，缺省为true
# [IMPL_MODE high_performance] 高性能模式，缺省为high_performance[,optional]
# [OPTIONS "option1" "option2"] 其他编译选项
add_kernel_sources(
    COMPUTE_UNITS ascend910b ascend910_93
    SIMPLIFIED_KEY None
    AUTO_SYNC false
)

add_kernel_sources(
    KERNEL_SRC arch35/rotate_quant.cpp
    COMPUTE_UNITS ascend950
    SIMPLIFIED_KEY None
    AUTO_SYNC false
    OPTIONS "--cce-no-dcache-flush" "-DENABLE_CV_COMM_VIA_SSBUF=true"
)

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(
        "ascend910B1"
        ""
    )
endif()
