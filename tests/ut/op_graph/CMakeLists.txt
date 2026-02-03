# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/

cmake_minimum_required(VERSION 3.16)

if(UT_TEST_ALL OR OP_GRAPH_UT)
    gen_es_nn_lib_ready()
    if(TARGET ${OP_GRAPH_MODULE_NAME}_cases_obj)
        add_dependencies(${OP_GRAPH_MODULE_NAME}_cases_obj json build_es_math build_es_nn)
        target_link_libraries(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE es_math es_nn)
    endif()

    file(GLOB SUBDIRECTORIES LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
    foreach(SUBDIR ${SUBDIRECTORIES})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/CMakeLists.txt)
            add_subdirectory(${SUBDIR})
        endif()
    endforeach()

    ## add opgraph ut exe:  nn_op_graph_ut
    set(OP_GRAPH_UT_EXE ${PKG_NAME}_op_graph_ut)
    
    set(INFERSHAPE_STUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/stub)
    file(GLOB INFERSHAPE_STUB_FILES ${INFERSHAPE_STUB_DIR}/*_infershape.cpp)

    add_executable(${OP_GRAPH_UT_EXE}
        ${INFERSHAPE_STUB_FILES}
        test_op_graph_main.cpp
    )

    # 当前桩函数实现不完整，优先从环境变量中链接so
    set_target_properties(${OP_GRAPH_UT_EXE} PROPERTIES
        SKIP_BUILD_RPATH TRUE
    )

    target_compile_definitions(${OP_GRAPH_UT_EXE} PRIVATE LOG_CPP)
    target_compile_options(${OP_GRAPH_UT_EXE} PUBLIC -fPIE -fno-access-control)
    add_dependencies(${OP_GRAPH_UT_EXE} 
        json 
        build_es_math 
        build_es_nn
        ophost_${PKG_NAME}_ut
    )
    target_link_directories(${OP_GRAPH_UT_EXE} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)
    target_link_libraries(${OP_GRAPH_UT_EXE} PRIVATE
        ophost_${PKG_NAME}_ut
        es_math
        es_nn
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        -Wl,--whole-archive
        $<$<TARGET_EXISTS:${OP_GRAPH_MODULE_NAME}_static_lib>:${OP_GRAPH_MODULE_NAME}_static_lib>
        -Wl,--no-whole-archive
        -Wl,--no-as-needed
        metadef
        -Wl,--as-needed
        error_manager
        unified_dlog
        exe_graph
        graph_base
        gtest
        graph
        platform
        register
        opp_registry
        rt2_registry
        c_sec
        mmpa
        dlog
        dl
        ascend_protobuf
    )

    if(ENABLE_UT_EXEC)
        if(${ENABLE_ASAN} STREQUAL "TRUE")
            add_custom_command(
                TARGET ${OP_GRAPH_UT_EXE} POST_BUILD
                COMMAND export LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/tests/ut/op_host:${CMAKE_BINARY_DIR}/es_packages_math/lib64:${CMAKE_BINARY_DIR}/es_packages/lib64:${LIB_OP_TILING_SO_PATH}:$ENV{LD_LIBRARY_PATH} && BUILD_PATH=${BUILD_PATH}
                LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.4:/usr/lib/x86_64-linux-gnu/libstdc++.so.6 ASAN_OPTIONS=detect_leaks=0 && ./${OP_GRAPH_UT_EXE}
                COMMENT "Run ops op_host utest with asan"
            )
        else()
            add_custom_command(
                TARGET ${OP_GRAPH_UT_EXE} POST_BUILD
                COMMAND export LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/tests/ut/op_host:${CMAKE_BINARY_DIR}/es_packages_math/lib64:${CMAKE_BINARY_DIR}/es_packages/lib64:${LIB_OP_TILING_SO_PATH}:$ENV{LD_LIBRARY_PATH}:${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64 && ./${OP_GRAPH_UT_EXE}
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
            TARGET ${OP_GRAPH_UT_EXE} POST_BUILD
            COMMAND export LD_LIBRARY_PATH=${LIB_OP_TILING_SO_PATH}:$ENV{LD_LIBRARY_PATH} &&
            valgrind --undef-value-errors=no --leak-check=full ./${OP_GRAPH_UT_EXE}
            COMMENT "Run ops op_host utest by valgrind"
        )
    endif()
endif()