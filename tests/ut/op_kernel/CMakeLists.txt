# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

if(UT_TEST_ALL OR OP_KERNEL_UT)
    ## set variables
    set(OP_KERNEL_UT_EXE ${PKG_NAME}_op_kernel_ut)

    ## add opkernel ut cases dynamic lib: libnn_op_kernel_ut_cases.so
    add_opkernel_ut_modules(${OP_KERNEL_MODULE_NAME})

    ## add op kernel ut exe for each socVersion
    foreach (socVersion ${fastOpTestSocVersions})
        ## append all exe
        set(AllSocVersion_OP_KERNEL_UT_EXE ${AllSocVersion_OP_KERNEL_UT_EXE} ${OP_KERNEL_UT_EXE}_${socVersion})

        ## add opkernel ut exe for socVersion
        add_executable(${OP_KERNEL_UT_EXE}_${socVersion}
            test_op_kernel_main.cpp
        )
        target_include_directories(${OP_KERNEL_UT_EXE}_${socVersion} PRIVATE
            ${ASCEND_DIR}/${SYSTEM_PREFIX}/pkg_inc
        )
        get_property(lib_list GLOBAL PROPERTY OPKERNEL_OBJECT_LIBS)
        foreach(lib_name ${lib_list})
            target_link_libraries(${OP_KERNEL_UT_EXE}_${socVersion} PRIVATE
                    -Wl,--no-as-needed
                    ${lib_name}
                    )
        endforeach()
        target_link_libraries(${OP_KERNEL_UT_EXE}_${socVersion} PRIVATE
                $<BUILD_INTERFACE:intf_llt_pub_asan>
                -Wl,--no-as-needed
                    ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases
                -Wl,--as-needed
                graph
                gtest
                tikicpulib::${socVersion}
                opp_registry
                metadef
                register
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
                        TARGET ${OP_KERNEL_UT_EXE}_${socVersion} POST_BUILD
                        COMMAND
                        COMMAND export LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} &&
                        ${PRELOAD} ASAN_OPTIONS=detect_leaks=0 ./${OP_KERNEL_UT_EXE}_${socVersion}
                        COMMENT "Run fast op utest with asan"
                )
            else()
                add_custom_command(
                        TARGET ${OP_KERNEL_UT_EXE}_${socVersion} POST_BUILD
                        COMMAND export LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} && ./${OP_KERNEL_UT_EXE}_${socVersion}
                        COMMENT "Run fast op utest"
                )
            endif()
        endif()
        if(ENABLE_VALGRIND)
            add_custom_command(
                    TARGET ${OP_KERNEL_UT_EXE}_${socVersion} POST_BUILD
                    COMMAND export LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} && valgrind --undef-value-errors=no --leak-check=full ./${OP_KERNEL_UT_EXE}_${socVersion}
                    COMMENT "Run fast op utest with valgrind"
            )
        endif()
    endforeach ()

    add_custom_target(${OP_KERNEL_UT_EXE} DEPENDS ${AllSocVersion_OP_KERNEL_UT_EXE})

endif()