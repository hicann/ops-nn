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
    DEPENDENCIES quant_batch_matmul_v3
)

add_kernel_sources(
    KERNEL_SRC arch35/quant_batch_matmul_inplace_add.cpp
    COMPUTE_UNITS ascend950 ascend350
    AUTO_SYNC false
)

add_kernel_sources(
    KERNEL_SRC arch35/quant_batch_matmul_inplace_add.cpp
    COMPUTE_UNITS ascend960dt
    AUTO_SYNC false
)

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(
        "ascend950pr_9599"
        "-D__CCE_AICORE__=310"
        MULTI_KERNEL_TARGET
    )

    add_custom_target(qbmmia_ut_clean_gcda
      COMMAND ${CMAKE_COMMAND} -E rm -f
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/opkernel_test_quant_batch_matmul_inplace_add.dir/arch35/test_quant_batch_matmul_inplace_add.cpp.gcda"
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/opkernel_test_quant_batch_matmul_inplace_add_apt_hif8.dir/arch35/test_quant_batch_matmul_inplace_add_apt_hif8.cpp.gcda"
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/opkernel_test_quant_batch_matmul_inplace_add_apt_mx.dir/arch35/test_quant_batch_matmul_inplace_add_apt_mx.cpp.gcda"
    )
    foreach(_qbmmia_ut_target
        opkernel_test_quant_batch_matmul_inplace_add
        opkernel_test_quant_batch_matmul_inplace_add_apt_hif8
        opkernel_test_quant_batch_matmul_inplace_add_apt_mx)
        if(TARGET ${_qbmmia_ut_target})
            add_dependencies(${_qbmmia_ut_target} qbmmia_ut_clean_gcda)
        endif()
    endforeach()
endif()
