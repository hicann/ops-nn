# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(UT_TEST_ALL OR OP_API_UT)
    add_modules_ut_sources(HOSTNAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    # The planning shim recompiles aclnn_lstm_backward.cpp under private link names, so it
    # references the same l0 ops the production body does -- Cast, ConcatD, Slice, Transpose
    # and nine more that live in libopapi_math_stub.so. That stub is linked PRIVATE into the
    # UT copy of the op_api shared library and is NOT on the test executable's link line, and
    # $<TARGET_OBJECTS:..._cases_obj> carries objects without the object library's link
    # interface. Building the shim into the library instead of the executable is what puts it
    # on the same side of that boundary as the body it duplicates. The guard keeps both files
    # out of every build that is not an op_api UT.
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/lstm_backward_plan_api.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lstm_backward_plan_spy.cpp)
endif()
