# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(UT_TEST_ALL OR OP_API_UT)
    add_modules_ut_sources(HOSTNAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    # The private L0 failure contract uses the exact production object in a local UT.
    get_target_property(_convolution_sources ${OPHOST_NAME}_opapi_obj SOURCES)
    list(FILTER _convolution_sources INCLUDE REGEX "/convolution[.]cpp$")
    list(LENGTH _convolution_sources _convolution_source_count)
    if(NOT _convolution_source_count EQUAL 1)
        message(FATAL_ERROR "Convolution L0 UT requires exactly one production convolution.cpp source")
    endif()
    set(_convolution_object "$<FILTER:$<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>,INCLUDE,/convolution[.]cpp[.]o$>")
    add_executable(convolution_forward_l0_ut
        test_convolution_l0.cpp
        test_aclnn_convolution_cpu_runtime_stub.cpp
        ${UT_PATH}/op_api/stub/opdev/platform.cpp
        ${UT_PATH}/op_api/stub/opdev/op_executor.cpp
        "${_convolution_object}")
    add_dependencies(convolution_forward_l0_ut ${OPHOST_NAME}_opapi_obj)

    # The normal API UT executable consumes cases_obj, including in selected builds.
    if(ENABLE_UT_EXEC)
        set(_convolution_l0_command "$<TARGET_FILE:convolution_forward_l0_ut>")
        if(ENABLE_ASAN)
            list(PREPEND _convolution_l0_command ${CMAKE_COMMAND} -E env ASAN_OPTIONS=detect_leaks=0)
        endif()
        add_custom_target(convolution_forward_l0_ut_run
            COMMAND ${_convolution_l0_command}
            DEPENDS convolution_forward_l0_ut
            COMMENT "Run convolution forward L0 UT"
            VERBATIM)
        add_dependencies(${OP_API_MODULE_NAME}_cases_obj convolution_forward_l0_ut_run)
    else()
        add_dependencies(${OP_API_MODULE_NAME}_cases_obj convolution_forward_l0_ut)
    endif()
    target_compile_definitions(convolution_forward_l0_ut PRIVATE LOG_CPP)
    target_include_directories(convolution_forward_l0_ut PRIVATE
        "$<TARGET_PROPERTY:${OPHOST_NAME}_opapi_obj,INCLUDE_DIRECTORIES>")
    target_link_directories(convolution_forward_l0_ut PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)
    target_link_libraries(convolution_forward_l0_ut PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        ${OPHOST_NAME}_opapi op_api_ut_common opapi_math_stub nnopbase graph gtest platform register
        opp_registry rt2_registry error_manager c_sec unified_dlog ${CMAKE_DL_LIBS}
        -Wl,--no-as-needed ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64/libopapi_math.so -Wl,--as-needed)
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/convolution_l0_ut_objects.txt"
        CONTENT "${_convolution_object}\n")
endif()
