# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
if(UT_TEST_ALL OR OP_GRAPH_UT)
    add_modules_ut_sources(HOSTNAME ${OP_GRAPH_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    # The current repository ES package does not generate the legacy LN/V3/V4
    # operators used to build the source graphs. Use the installed ES package
    # for these tests only; the product still uses the 8.5-compatible builder.
    if(EXISTS "${ASCEND_DIR}/include/es/es_nn/es_LayerNorm.h")
        set_property(SOURCE
            test_add_layer_norm_fusion_pass.cpp
            APPEND PROPERTY COMPILE_OPTIONS
            "-include" "${ASCEND_DIR}/include/es/es_nn/es_nn_ops.h")
        target_link_options(${OP_GRAPH_MODULE_NAME}_static_lib INTERFACE
            "-Wl,--no-as-needed" "${ASCEND_DIR}/lib64/libes_nn.so" "-Wl,--as-needed")
    endif()
endif()
