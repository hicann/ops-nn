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
    DEPENDENCIES batch_mat_mul_v3 mat_mul_v3
)

add_kernel_sources(
    AUTO_SYNC false
)

add_kernel_sources(
    KERNEL_SRC arch35/transpose_batch_mat_mul.cpp
    COMPUTE_UNITS ascend950 ascend350
    AUTO_SYNC false
    OPTIONS "--cce-no-dcache-flush"
)

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(
        "ascend910B1"
        "-DKFC_L1_RESERVER_SIZE=0 -DDTYPE_X1=half -DDTYPE_X2=half -DDTYPE_BIAS=half -DDTYPE_Y=half"
        mat_mul_v3 batch_mat_mul_v3
    )
endif()
