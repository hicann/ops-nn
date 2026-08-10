# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_all_modules_sources(
    ACLNNTYPE aclnn_exclude
    DEPENDENCIES batch_mat_mul_v3
)

# [COMPUTE_UNITS ascendxx] soc版本，支持配置多个unit，缺省为配置所有的soc
# [SIMPLIFIED_KEY 0/None] 缺省为0，设置为None则不携带该参数
# [AUTO_SYNC false] 同步选项，缺省为true
# [IMPL_MODE high_performance] 高性能模式，缺省为high_performance[,optional]
# [OPTIONS "option1" "option2"] 其他编译选项
add_kernel_sources(
    COMPUTE_UNITS ascend310p ascend910b ascend910_93 kirinx90
    SIMPLIFIED_KEY None
    AUTO_SYNC false
)

add_kernel_sources(
    KERNEL_SRC arch35/mat_mul_v3.cpp
    COMPUTE_UNITS ascend950 ascend350
    SIMPLIFIED_KEY None
    AUTO_SYNC false
    OPTIONS "--cce-no-dcache-flush" "-DENABLE_CV_COMM_VIA_SSBUF=true"
)

add_kernel_sources(
    KERNEL_SRC arch35/mat_mul_v3.cpp
    COMPUTE_UNITS mc62
)

add_kernel_sources(
    COMPUTE_UNITS kirin9030
    AUTO_SYNC false
)

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(
        "ascend910B1"
        "-DDTYPE_X1=half -DDTYPE_X2=half -DDTYPE_BIAS=half -DDTYPE_Y=half"
        batch_mat_mul_v3
    )
endif()
