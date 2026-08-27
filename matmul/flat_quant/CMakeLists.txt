# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_all_modules_sources(
    ACLNNTYPE aclnn_exclude
)

# [COMPUTE_UNITS ascendxx] soc版本，支持配置多个unit，缺省为配置所有的soc
# [SIMPLIFIED_KEY 0/None] 缺省为0，设置为None则不携带该参数
# [AUTO_SYNC false] 同步选项，缺省为true
# [IMPL_MODE high_performance] 高性能模式，缺省为high_performance[,optional]
# [OPTIONS "option1" "option2"] 其他编译选项
add_kernel_sources(
    COMPUTE_UNITS ascend910b
    AUTO_SYNC false
)

add_kernel_sources(
    KERNEL_SRC arch35/flat_quant.cpp
    COMPUTE_UNITS ascend950 ascend350
    AUTO_SYNC false
    OPTIONS  "-DENABLE_CV_COMM_VIA_SSBUF=true"
)

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(
        "ascend910B1"
        "-DDTYPE_X=half"
        flat_quant
    )
endif()
