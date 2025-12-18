# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
 
if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
    ## set variables
    set(AICPU_OP_KERNEL_UT_EXE ${PKG_NAME}_aicpu_op_kernel_ut)
 
    ## add opkernel ut cases dynamic lib: libcv_op_kernel_ut_cases.so
    add_aicpu_opkernel_ut_modules(${AICPU_OP_KERNEL_MODULE_NAME})
 
    message(STATUS ">>>> Defined targets 111: ${AICPU_OP_KERNEL_MODULE_NAME}_cases")
    add_executable(${AICPU_OP_KERNEL_UT_EXE}
        test_op_kernel_aicpu_main.cpp
    )
 
    get_target_property(tmp ${AICPU_OP_KERNEL_MODULE_NAME}_cases NAME)
    message(STATUS ">>> Linking target: ${tmp}")
 
    message(STATUS ">>>> Defined targets 222: ${AICPU_OP_KERNEL_MODULE_NAME}_cases")
    target_link_libraries(${AICPU_OP_KERNEL_UT_EXE} PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan>
        $<BUILD_INTERFACE:dlog_headers>
        -ldl
        -Wl,--whole-archive
            ${ASCEND_DIR}/lib64/libaicpu_context_host.a
            ${ASCEND_DIR}/lib64/libaicpu_nodedef_host.a
            ${ASCEND_DIR}/lib64/libhost_ascend_protobuf.a
        -Wl,--no-whole-archive
        -Wl,-Bsymbolic
        -Wl,--exclude-libs=libhost_ascend_protobuf.a
        -Wl,--no-as-needed
        ${AICPU_OP_KERNEL_MODULE_NAME}_cases
        -Wl,--as-needed
        gtest
        c_sec
        Eigen3::EigenNn
    )
 
    if(ENABLE_UT_EXEC)
        if(ENABLE_ASAN)
            execute_process(
                COMMAND ${CMAKE_C_COMPILER} -print-file-name=libasan.so
                OUTPUT_VARIABLE LIBASAN_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE result
            )
            if(NOT result EQUAL 0)
                message(FATAL_ERROR "compiler not support asan, please disable asan")
            endif()
            set(PRELOAD "LD_PRELOAD=${LIBASAN_PATH}:/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
            add_custom_command(
                    TARGET ${AICPU_OP_KERNEL_UT_EXE} POST_BUILD
                    COMMAND
                    COMMAND export LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} &&
                    ${PRELOAD} ASAN_OPTIONS=detect_leaks=0 ./${AICPU_OP_KERNEL_UT_EXE}
                    COMMENT "Run fast op utest with asan"
            )
        else()
            add_custom_command(
                    TARGET ${AICPU_OP_KERNEL_UT_EXE} POST_BUILD
                    COMMAND export LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} && ./${AICPU_OP_KERNEL_UT_EXE}
                    COMMENT "Run fast op utest"
            )
        endif()
    endif()
 
    if(ENABLE_VALGRIND)
        add_custom_command(
                TARGET ${AICPU_OP_KERNEL_UT_EXE} POST_BUILD
                COMMAND export LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} && valgrind --undef-value-errors=no --leak-check=full ./${AICPU_OP_KERNEL_UT_EXE}
                COMMENT "Run fast op utest with valgrind"
        )
    endif()
 
    # add_custom_target(${AICPU_OP_KERNEL_UT_EXE})
endif()
