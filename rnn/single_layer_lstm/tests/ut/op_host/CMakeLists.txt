# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# arch35/ holds the cases that call ExecuteTiling. add_tiling_sources compiles
# op_host/arch35/single_layer_lstm_tiling.cpp only when ASCEND_COMPUTE_UNIT names a SoC that
# maps to arch35, and add_modules_ut_sources gates tests/ut/op_host/arch35/ on the same
# variable. A case that looks the tiling function up in the registry therefore has to sit on
# the same side of that gate as the file that registers it: at the root it compiles in a build
# that has no tiling, GetOpImpl returns a struct whose tiling member is null, and the harness
# calls through it. Cases that only exercise headers -- the budget and infershape ones -- stay
# at the root, where they run in every op_host UT build.
file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(HOSTNAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    add_modules_ut_sources(HOSTNAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
