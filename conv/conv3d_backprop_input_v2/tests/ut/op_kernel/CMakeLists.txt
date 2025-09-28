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
add_library(conv3d_backprop_input_v2 SHARED
            ${CANN_ROOT}/ops/ops-conv/conv3d_backprop_input_v2/op_kernel/conv3d_backprop_input_v2.cpp
            ${CANN_ROOT}/ops/ops-conv/conv3d_backprop_input_v2/tests/ut/op_kernel/test_conv3d_backprop_input_v2.cpp
)

target_compile_options(conv3d_backprop_input_v2 PRIVATE
                       -DDTYPE_FILTER=bfloat16_t
                       -DDTYPE_OUT_BACKPROP=bfloat16_t
                       -DDTYPE_Y=bfloat16_t
                       -DFORMAT_FILTER=FORMAT_FRACTAL_Z_3D
                       -DFORMAT_OUT_BACKPROP=FORMAT_NDC1HWC0
                       -DFORMAT_Y=FORMAT_NDC1HWC0
                       -g
                       -fvisibility=hidden
                       -include ${CANN_ROOT}/ops/ops-conv/conv3d_backprop_input_v2/tests/ut/op_kernel/conv3d_backprop_input_v2_tiling_def.h)

target_include_directories(conv3d_backprop_input_v2 PRIVATE
                          ${ASCEND_DIR}/opensdk/opensdk/gtest/include
                          ${ASCEND_DIR}/compiler/tikcpp/tikcfw
                          ${ASCEND_DIR}/compiler/tikcpp/tikcfw/impl
                          ${ASCEND_DIR}/compiler/tikcpp/tikcfw/interface
                          ${CMAKE_CURRENT_SOURCE_DIR}/
                          ${CANN_ROOT}/ops/ops-conv/conv3d_backprop_input_v2/op_kernel/
)

SET(PRIVATE_L
    c_sec
    mmpa
    error_manager
    pem_davinci
    tikicpulib::ascend910B1
    pthread
)

target_link_libraries(conv3d_backprop_input_v2 PRIVATE
                    $<BUILD_INTERFACE:intf_llt_pub_asan>
                    -Wl,--whole-archive
                    -Wl,--no-as-needed
                    ${PRIVATE_L}
                    -Wl,--as-needed
                    -Wl,--no-whole-archive
)
endif()
