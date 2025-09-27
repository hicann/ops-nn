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

if(UT_TEST_ALL OR OP_HOST_UT)
    file(GLOB SUBDIRECTORIES LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
    foreach(SUBDIR ${SUBDIRECTORIES})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/CMakeLists.txt)
            add_subdirectory(${SUBDIR})
        endif()
    endforeach()

    # add ophost source so: libophost_nn_ut.so
    add_library(${OPHOST_NAME}_ut SHARED
        $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
        $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
        $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
    )
    target_link_libraries(${OPHOST_NAME}_ut PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        -Wl,--no-as-needed
        $<$<TARGET_EXISTS:opsbase>:opsbase>
        $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:optiling>>
        -Wl,--as-needed
        -Wl,--whole-archive
        rt2_registry_static
        -Wl,--no-whole-archive
        tiling_api
    )

    ## add ophost ut exe:  nn_op_host_ut
    set(OP_HOST_UT_EXE ${PKG_NAME}_op_host_ut)
    add_executable(${OP_HOST_UT_EXE}
        test_op_host_main.cpp
    )
    target_compile_definitions(${OP_HOST_UT_EXE} PRIVATE LOG_CPP)
    target_compile_options(${OP_HOST_UT_EXE} PUBLIC -fPIE -fno-access-control)
    add_dependencies(${OP_HOST_UT_EXE} ${OPHOST_NAME}_ut)
    target_link_libraries(${OP_HOST_UT_EXE} PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        -Wl,--whole-archive
        $<$<TARGET_EXISTS:${OP_TILING_MODULE_NAME}_static_lib>:${OP_TILING_MODULE_NAME}_static_lib>
        $<$<TARGET_EXISTS:${OP_INFERSHAPE_MODULE_NAME}_static_lib>:${OP_INFERSHAPE_MODULE_NAME}_static_lib>
        -Wl,--no-whole-archive
        -Wl,--no-as-needed
        metadef
        -Wl,--as-needed
        error_manager
        exe_graph
        graph_base
        gtest
        graph
        platform
        register
        opp_registry
        opsbase
        tiling_api
        dlog
        dl
    )

    add_dependencies(${OP_HOST_UT_EXE} json)

    find_package(optiling MODULE REQUIRED)
    if(ENABLE_UT_EXEC)
        if(NOT LIB_OP_TILING_SO_PATH)
            message(FATAL_ERROR "Can not found optiling so!")
            return()
        endif()
        if(${ENABLE_ASAN} STREQUAL "TRUE")
            add_custom_command(
                TARGET ${OP_HOST_UT_EXE} POST_BUILD
                COMMAND export LD_LIBRARY_PATH=${LIB_OP_TILING_SO_PATH}:$ENV{LD_LIBRARY_PATH} && BUILD_PATH=${BUILD_PATH}
                LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.4:/usr/lib/x86_64-linux-gnu/libstdc++.so.6 ASAN_OPTIONS=detect_leaks=0 && ./${OP_HOST_UT_EXE}
                COMMENT "Run ops op_host utest with asan"
            )
        else()
            add_custom_command(
                TARGET ${OP_HOST_UT_EXE} POST_BUILD
                COMMAND export LD_LIBRARY_PATH=${LIB_OP_TILING_SO_PATH}:$ENV{LD_LIBRARY_PATH} && ./${OP_HOST_UT_EXE}
                COMMENT "Run ops op_host utest"
            )
        endif()
    endif()

    if(${ENABLE_VALGRIND} STREQUAL "TRUE")
        if(NOT LIB_OP_TILING_SO_PATH)
            message(FATAL_ERROR "Can not found optiling so!")
            return()
        endif()
        add_custom_command(
            TARGET ${OP_HOST_UT_EXE} POST_BUILD
            COMMAND export LD_LIBRARY_PATH=${LIB_OP_TILING_SO_PATH}:$ENV{LD_LIBRARY_PATH} &&
            valgrind --undef-value-errors=no --leak-check=full ./${OP_HOST_UT_EXE}
            COMMENT "Run ops op_host utest by valgrind"
        )
    endif()
endif()