# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

message(STATUS "=== Debug: start ops.norm.in_training_reduce_v2.tests.CMakeLists.txt ")

# 仅纳入 ut 子目录参与主构建（UT 集成到 nn_op_host_ut）。
# st/ 是独立自包含的 C++ ST 工程（自带 project() + Real 模式 FATAL_ERROR），
# 通过 tests/st/run.sh 单独构建，不并入主 UT 构建树。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ut/CMakeLists.txt")
    add_subdirectory(ut)
endif()
