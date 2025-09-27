# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
return()
if(UT_TEST_ALL OR FAST_OP_UT)

include_directories(
        ${ASCEND_DIR}/opensdk/opensdk/gtest/include
        ${ASCEND_DIR}/compiler/tikcpp/tikcfw
        ${ASCEND_DIR}/compiler/tikcpp/tikcfw/impl
        ${ASCEND_DIR}/compiler/tikcpp/tikcfw/interface
        ${CANN_ROOT}/ops/built-in/tbe/impl/ascendc/common/
        ${CANN_ROOT}/ops/built-in/tests/ut/op_tiling_test
        ${ASCEND_DIR}/runtime/highlevel_api
        ${ASCEND_DIR}/include/ascendc/highlevel_api
        ${CMAKE_CURRENT_SOURCE_DIR}/
        ${CANN_ROOT}/ops/common
        ${CANN_ROOT}/ops/built-in/tests/ut/fast_op_test
        )

SET(PRIVATE_L
    c_sec
    mmpa
    error_manager
    pem_davinci
    tikicpulib::ascend910B1
    pthread
)

add_library(quant_matmul_reduce_sum SHARED
        ${CANN_ROOT}/ops/matmul/quant_matmul_reduce_sum/op_kernel/quant_matmul_reduce_sum.cpp
            ./test_quant_matmul_reduce_sum.cpp
)

target_compile_options(quant_matmul_reduce_sum PRIVATE
    -DDTYPE_X1=half -DDTYPE_X2=half -DDTYPE_BIAS=half -DDTYPE_Y=half
    -g -include ${CMAKE_CURRENT_SOURCE_DIR}/quant_matmul_reduce_sum_tiling_def.h)

target_link_libraries(quant_matmul_reduce_sum PRIVATE
                      $<BUILD_INTERFACE:intf_llt_pub_asan>
                      -Wl,--whole-archive
                      -Wl,--no-as-needed
                      ${PRIVATE_L}
                      -Wl,--as-needed
                      -Wl,--no-whole-archive
)
endif()