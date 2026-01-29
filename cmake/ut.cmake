# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

function(add_optiling_ut_modules OP_TILING_MODULE_NAME)

    # set variables
    set(UT_COMMON_INC ${PROJECT_SOURCE_DIR}/tests/ut/common)
    set(UT_CONV3DV2_TILING_INC ${PROJECT_SOURCE_DIR}/common/inc/op_host
                               ${PROJECT_SOURCE_DIR}
                               ${PROJECT_SOURCE_DIR}/common/include)

    # add op tiling ut test cases obj
    add_library(${OP_TILING_MODULE_NAME}_cases_obj OBJECT)
    add_dependencies(${OP_TILING_MODULE_NAME}_cases_obj json)
    target_include_directories(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
        ${UT_COMMON_INC}
        ${UT_CONV3DV2_TILING_INC}
        ${JSON_INCLUDE}
        ${GTEST_INCLUDE}
        ${OPBASE_INC_DIRS}
        ${PROJECT_SOURCE_DIR}/common/inc
        ${ASCEND_DIR}/include
        ${ASCEND_DIR}/pkg_inc
        ${ASCEND_DIR}/include/external
        ${ASCEND_DIR}/include/exe_graph
        ${OP_TILING_INCLUDE}
        ${ASCEND_DIR}/include/base/context_builder
        ${OPS_TEST_DIR}/ut/common/utils
    )
    target_link_libraries(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        $<BUILD_INTERFACE:dlog_headers>
        gtest
    )

    target_compile_options(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
        -fno-access-control
    )

    target_compile_definitions(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
        LOG_CPP
    )

    # add op tiling ut static lib
    add_library(${OP_TILING_MODULE_NAME}_static_lib STATIC
        $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
        $<$<TARGET_EXISTS:${OP_TILING_MODULE_NAME}_common_obj>:$<TARGET_OBJECTS:${OP_TILING_MODULE_NAME}_common_obj>>
    )
    target_link_libraries(${OP_TILING_MODULE_NAME}_static_lib PRIVATE
        ${OP_TILING_MODULE_NAME}_common_obj
        ${OP_TILING_MODULE_NAME}_cases_obj
    )
endfunction()

function(add_infershape_ut_modules OP_INFERSHAPE_MODULE_NAME)
    # set variables
    set(UT_COMMON_INC ${PROJECT_SOURCE_DIR}/tests/ut/common)

    # add op tiling ut test cases obj
    add_library(${OP_INFERSHAPE_MODULE_NAME}_cases_obj OBJECT)
    add_dependencies(${OP_INFERSHAPE_MODULE_NAME}_cases_obj json)
    target_include_directories(${OP_INFERSHAPE_MODULE_NAME}_cases_obj PRIVATE
        ${UT_COMMON_INC}
        ${JSON_INCLUDE}
        ${GTEST_INCLUDE}
        ${OPBASE_INC_DIRS}
        ${PROJECT_SOURCE_DIR}/common/inc
        ${ASCEND_DIR}/include
        ${ASCEND_DIR}/pkg_inc
        ${ASCEND_DIR}/include/external
        ${ASCEND_DIR}/include/exe_graph
        ${ASCEND_DIR}/include/base/context_builder
    )
    target_link_libraries(${OP_INFERSHAPE_MODULE_NAME}_cases_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        $<BUILD_INTERFACE:dlog_headers>
        metadef
        graph
        gtest
    )

    target_compile_options(${OP_INFERSHAPE_MODULE_NAME}_cases_obj PRIVATE
        -fno-access-control
    )

    target_compile_definitions(${OP_INFERSHAPE_MODULE_NAME}_cases_obj PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
        LOG_CPP
    )

    # add infershape ut static lib
    add_library(${OP_INFERSHAPE_MODULE_NAME}_static_lib STATIC
        $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
        $<$<TARGET_EXISTS:${OP_INFERSHAPE_MODULE_NAME}_common_obj>:$<TARGET_OBJECTS:${OP_INFERSHAPE_MODULE_NAME}_common_obj>>
    )
    target_link_libraries(${OP_INFERSHAPE_MODULE_NAME}_static_lib PRIVATE
        ${OP_INFERSHAPE_MODULE_NAME}_common_obj
        ${OP_INFERSHAPE_MODULE_NAME}_cases_obj
    )
endfunction()

function(add_opapi_ut_modules OP_API_MODULE_NAME)
    ## add opapi ut L2 obj
    add_library(${OP_API_MODULE_NAME}_cases_obj OBJECT)
    add_dependencies(${OP_API_MODULE_NAME}_cases_obj json)
    target_sources(${OP_API_MODULE_NAME}_cases_obj PRIVATE
                    ${UT_PATH}/op_api/stub/opdev/platform.cpp
                    ${UT_PATH}/op_api/stub/opdev/op_executor.cpp
                    )
    target_include_directories(${OP_API_MODULE_NAME}_cases_obj PRIVATE
            ${JSON_INCLUDE}
            ${OPS_NN_COMMON_INC}
            ${OP_API_UT_COMMON_INC}
            ${UT_PATH}/op_api/stub
            ${ASCEND_DIR}/include
            ${ASCEND_DIR}/pkg_inc
            ${ASCEND_DIR}/include/aclnn
            ${ASCEND_DIR}/include/aclnnop
            ${OPAPI_INCLUDE}
            )
    target_link_libraries(${OP_API_MODULE_NAME}_cases_obj PRIVATE
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
            $<BUILD_INTERFACE:dlog_headers>
            gtest
            )
endfunction()

function(add_opkernel_ut_modules OP_KERNEL_MODULE_NAME)
    # set variables
    set(UT_COMMON_INC ${CMAKE_CURRENT_SOURCE_DIR}/../common CACHE STRING "ut common include path" FORCE)

    ## add opkernel ut common object: nn_op_kernel_ut_common_obj
    add_library(${OP_KERNEL_MODULE_NAME}_common_obj OBJECT)
    add_dependencies(${OP_KERNEL_MODULE_NAME}_common_obj json)
    file(GLOB OP_KERNEL_UT_COMMON_SRC
        ${UT_COMMON_INC}/test_cube_util.cpp
        ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel/data_utils.cpp
    )
    target_sources(${OP_KERNEL_MODULE_NAME}_common_obj PRIVATE
        ${OP_KERNEL_UT_COMMON_SRC}
    )
    target_include_directories(${OP_KERNEL_MODULE_NAME}_common_obj PRIVATE
        ${JSON_INCLUDE}
        ${GTEST_INCLUDE}
        ${ASCEND_DIR}/include/base/context_builder
    )
    target_link_libraries(${OP_KERNEL_MODULE_NAME}_common_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        gtest
        c_sec
    )

    foreach(socVersion ${fastOpTestSocVersions})
        ## add op kernel ut cases dynamic lib: libnn_op_kernel_ut_${socVersion}_cases.so
        add_library(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases SHARED
            $<TARGET_OBJECTS:${OP_KERNEL_MODULE_NAME}_common_obj>
            )
        target_link_libraries(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases PRIVATE
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
            ${OP_KERNEL_MODULE_NAME}_common_obj
            gcov
        )
    endforeach()
endfunction()

function(add_op_graph_ut_modules OP_GRAPH_MODULE_NAME)
    # set variables
    set(UT_COMMON_INC ${PROJECT_SOURCE_DIR}/tests/ut/common)

    if(TARGET ${GRAPH_PLUGIN_NAME}_obj)
        add_dependencies(${GRAPH_PLUGIN_NAME}_obj build_es_math build_es_nn)
        target_link_libraries(${GRAPH_PLUGIN_NAME}_obj PRIVATE es_math es_nn)
    endif()

    # add op graph ut test cases obj
    add_library(${OP_GRAPH_MODULE_NAME}_cases_obj OBJECT)
    
    target_include_directories(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
        ${UT_COMMON_INC}
        ${JSON_INCLUDE}
        ${GTEST_INCLUDE}
        ${OPBASE_INC_DIRS}
        ${PROJECT_SOURCE_DIR}/common/inc
        ${ASCEND_DIR}/include
        ${ASCEND_DIR}/include/external
        ${ASCEND_DIR}/include/exe_graph
        ${ASCEND_DIR}/include/base/context_builder
        ${ASCEND_DIR}/include/ge/
        ${CMAKE_BINARY_DIR}/es_packages/include/es_math/
        ${CMAKE_BINARY_DIR}/es_packages/include/es_nn/
    )
    target_link_libraries(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        $<BUILD_INTERFACE:dlog_headers>
        -Wl,--no-as-needed
        metadef
        graph
        gtest
        register
        ge_compiler
        ascendalog
 	    unified_dlog
    )

    target_compile_options(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
            -fno-access-control
    )

    # add infershape ut static lib
    add_library(${OP_GRAPH_MODULE_NAME}_static_lib STATIC)
    target_link_libraries(${OP_GRAPH_MODULE_NAME}_static_lib PRIVATE
            ${OP_GRAPH_MODULE_NAME}_cases_obj
            ${GRAPH_PLUGIN_NAME}_obj
            es_math
            es_nn
    )
    add_dependencies(${OP_GRAPH_MODULE_NAME}_static_lib build_es_math build_es_nn)
    target_link_libraries(${OP_GRAPH_MODULE_NAME}_static_lib PRIVATE es_math es_nn)
endfunction()

if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
  set(AICPU_OP_KERNEL_MODULE_NAME ${PKG_NAME}_aicpu_op_kernel_ut CACHE STRING "aicpu_op_kernel ut module name" FORCE)
  message("******************* AICPU_OP_KERNEL_MODULE_NAME is ${AICPU_OP_KERNEL_MODULE_NAME}" )
  function(add_aicpu_opkernel_ut_modules AICPU_OP_KERNEL_MODULE_NAME)
    ## add opkernel ut common object: nn_aicpu_op_kernel_ut_common_obj
    add_library(${AICPU_OP_KERNEL_MODULE_NAME}_common_obj OBJECT)
    file(GLOB OP_KERNEL_UT_COMMON_SRC
        ./stub/*.cpp
    )
    file(GLOB OP_KERNEL_AICPU_UT_UTILS_SRC
        utils/*.cpp
    )
    target_sources(${AICPU_OP_KERNEL_MODULE_NAME}_common_obj PRIVATE ${OP_KERNEL_UT_COMMON_SRC})
    target_include_directories(${AICPU_OP_KERNEL_MODULE_NAME}_common_obj PRIVATE
        ${GTEST_INCLUDE}
    )
    target_compile_definitions(${AICPU_OP_KERNEL_MODULE_NAME}_common_obj PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    target_link_libraries(${AICPU_OP_KERNEL_MODULE_NAME}_common_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        gtest
        c_sec
        Eigen3::EigenNn
    )

    ## add opkernel ut cases object: nn_aicpu_op_kernel_ut_cases_obj
    if(NOT TARGET ${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj)
        add_library(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj OBJECT ${UT_PATH}/empty.cpp)
    endif()
    target_link_libraries(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj PRIVATE gcov -ldl)
    target_sources(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj PRIVATE ${OP_KERNEL_AICPU_UT_UTILS_SRC})

    ## add opkernel ut cases shared lib: libnn_aicpu_op_kernel_ut_cases.so
    add_library(${AICPU_OP_KERNEL_MODULE_NAME}_cases SHARED
        $<TARGET_OBJECTS:${AICPU_OP_KERNEL_MODULE_NAME}_common_obj>
        $<TARGET_OBJECTS:${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj>
    )
    message(STATUS ">>>> ut.cmake Defined targets: ${AICPU_OP_KERNEL_MODULE_NAME}_cases, ${ASCEND_DIR}")

    # 链接静态库时使用 whole-archive，保证 RegistCpuKernel 被拉入
    target_link_libraries(${AICPU_OP_KERNEL_MODULE_NAME}_cases PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        gtest
        c_sec
        -ldl
        -Wl,--whole-archive
            ${ASCEND_DIR}/lib64/libaicpu_context_host.a
            ${ASCEND_DIR}/lib64/libaicpu_nodedef_host.a
            ${ASCEND_DIR}/lib64/libhost_ascend_protobuf.a
        -Wl,--no-whole-archive
        -Wl,-Bsymbolic
        -Wl,--exclude-libs=libhost_ascend_protobuf.a
        Eigen3::EigenNn
        ${AICPU_OP_KERNEL_MODULE_NAME}_common_obj
        ${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj
    )
  endfunction()
endif()

function(add_modules_llt_sources)
    add_modules_ut_sources(${ARGN})
endfunction()

function(add_modules_ut_sources)
    set(options OPTION_RESERVED)
    set(oneValueArgs HOSTNAME MODE DIR)
    set(multiValueArgs MULIT_RESERVED)

    cmake_parse_arguments(MODULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    message(STATUS "=== Debug<add_modules_ut_sources>: ${MODULE_HOSTNAME} ${MODULE_MODE} ${MODULE_DIR}")

    string(FIND "${MODULE_HOSTNAME}_cases_obj" "tiling" TILING_FOUND_INDEX)
    if (DEFINED ASCEND_COMPUTE_UNIT)
 	    map_compute_unit(${ASCEND_COMPUTE_UNIT} compute_unit_long)
 	    get_target_dir(${compute_unit_long} target_dir)
 	else()
 	    set(target_dir "")
 	endif()
    if("${target_dir}" STREQUAL "")
        file(GLOB OPHOST_TILING_SRCS ${MODULE_DIR}/test_*_tiling.cpp)
    else()
        file(GLOB OPHOST_TILING_SRCS ${MODULE_DIR}/test_*_tiling.cpp ${MODULE_DIR}/${target_dir}/test_*_tiling.cpp)
    endif()
    if(${TILING_FOUND_INDEX} GREATER_EQUAL 0 AND OPHOST_TILING_SRCS)
        if (NOT TARGET ${MODULE_HOSTNAME}_cases_obj)
            add_optiling_ut_modules(${OP_TILING_MODULE_NAME})
        endif()
        target_sources(${MODULE_HOSTNAME}_cases_obj ${MODULE_MODE} ${OPHOST_TILING_SRCS})
        message(STATUS "=== Debug<add_modules_ut_sources>: ${MODULE_HOSTNAME}_cases_obj ${OPHOST_TILING_SRCS}")
    endif()

    string(FIND "${MODULE_HOSTNAME}_cases_obj" "infershape" INFERSHAPE_FOUND_INDEX)
    if("${target_dir}" STREQUAL "")
        file(GLOB OPHOST_INFERSHAPE_SRCS ${MODULE_DIR}/test_*_infershape.cpp)
    else()
        file(GLOB OPHOST_INFERSHAPE_SRCS ${MODULE_DIR}/test_*_infershape.cpp ${MODULE_DIR}/${target_dir}/test_*_infershape.cpp)
    endif()
    if(${INFERSHAPE_FOUND_INDEX} GREATER_EQUAL 0 AND OPHOST_INFERSHAPE_SRCS)
        if (NOT TARGET ${MODULE_HOSTNAME}_cases_obj)
            add_infershape_ut_modules(${OP_INFERSHAPE_MODULE_NAME})
        endif()
        target_sources(${MODULE_HOSTNAME}_cases_obj ${MODULE_MODE} ${OPHOST_INFERSHAPE_SRCS})
        message(STATUS "=== Debug<add_modules_ut_sources>: ${MODULE_HOSTNAME}_cases_obj ${OPHOST_INFERSHAPE_SRCS}")
    endif()

    string(FIND "${MODULE_HOSTNAME}_cases_obj" "op_api" OPAPI_FOUND_INDEX)
    if(${OPAPI_FOUND_INDEX} GREATER_EQUAL 0)
        if(${target_dir} STREQUAL "")
            file(GLOB OPHOST_OPAPI_SRCS ${MODULE_DIR}/test_aclnn_*.cpp)
        else()
            file(GLOB OPHOST_OPAPI_SRCS ${MODULE_DIR}/test_aclnn_*.cpp ${MODULE_DIR}/${target_dir}/test_aclnn_*.cpp)
        endif()
        if (NOT TARGET ${MODULE_HOSTNAME}_cases_obj)
            add_opapi_ut_modules(${OP_API_MODULE_NAME})
        endif()
        target_sources(${MODULE_HOSTNAME}_cases_obj ${MODULE_MODE} ${OPHOST_OPAPI_SRCS})
        message(STATUS "=== Debug<add_modules_ut_sources>: ${MODULE_HOSTNAME}_cases_obj ${OPHOST_OPAPI_SRCS}")
    endif()
    
    string(FIND "${MODULE_HOSTNAME}_cases_obj" "op_graph" OPGRAPH_FOUND_INDEX)
    if(${OPGRAPH_FOUND_INDEX} GREATER_EQUAL 0)
        file(GLOB OPHOST_OPGRAPH_SRCS ${MODULE_DIR}/test_*_pass.cpp)
        if (NOT TARGET ${MODULE_HOSTNAME}_cases_obj)
            add_op_graph_ut_modules(${OP_GRAPH_MODULE_NAME})
        endif()
        target_sources(${MODULE_HOSTNAME}_cases_obj ${MODULE_MODE} ${OPHOST_OPGRAPH_SRCS})
        message(STATUS "=== Debug<add_modules_ut_sources>: ${MODULE_HOSTNAME}_cases_obj ${OPHOST_OPGRAPH_SRCS}")
    endif()
endfunction()

if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(fastOpTestSocVersions "" CACHE STRING "fastOp Test SocVersions")
endif()

# supportedSocVersion: ascend310p ascend910B1 ascend950pr_9599
function(AddOpTestCase opName supportedSocVersion otherCompileOptions)
    set(DEPENDENCY_OPS "")
    set(temp_ops "")
    foreach(arg ${ARGN})
        if("${arg}" STREQUAL "MULTI_KERNEL_TARGET")
            set(multi_kernel_target "TRUE")
        else()
            list(APPEND temp_ops ${arg})
        endif()
    endforeach()
    set(DEPENDENCY_OPS ${temp_ops})

    string(REPLACE "${OPS_NN_DIR}/" "" opCategoryDir ${CMAKE_CURRENT_SOURCE_DIR})
    string(FIND "${opCategoryDir}" "/" firstSlashPos)
    string(SUBSTRING "${opCategoryDir}" 0 ${firstSlashPos} opCategory)

    get_filename_component(UT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
    get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
    list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
    ## if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
        return()
    endif()

    set(COMPILED_OPS_UT "${opName}")
    set(COMPILED_OP_DIRS_UT ${OP_NAME_DIR})
    if(DEPENDENCY_OPS)
        foreach(depOp ${DEPENDENCY_OPS})
            list(APPEND COMPILED_OPS_UT ${depOp})
            get_filename_component(DEP_OP_NAME_DIR ${OP_NAME_DIR}/../${depOp} ABSOLUTE)
            list(APPEND COMPILED_OP_DIRS_UT ${DEP_OP_NAME_DIR})
        endforeach()
    endif()
    set(COMPILED_OPS_UT "${COMPILED_OPS_UT}" CACHE STRING "Compiled Ops" FORCE)

    set(KERNEL_COPY_TARGET "ascendc_kernel_ut_src_copy_${opName}")
    set(KERNEL_COMMON_COPY_TARGET "ascendc_kernel_ut_common_src_copy_${opName}")
    if(NOT TARGET ${KERNEL_COPY_TARGET})
        add_custom_target(${KERNEL_COMMON_COPY_TARGET}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tbe/ascendc/common/cmct
            COMMAND cp -r ${PROJECT_SOURCE_DIR}/matmul/common/cmct/* ${CMAKE_BINARY_DIR}/tbe/ascendc/common/cmct
            COMMAND cp -r ${PROJECT_SOURCE_DIR}/conv/common/op_kernel/* ${CMAKE_BINARY_DIR}/tbe/ascendc/common
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tbe/ascendc/inc
            COMMAND cp -r ${PROJECT_SOURCE_DIR}/common/inc/op_kernel/* ${CMAKE_BINARY_DIR}/tbe/ascendc/inc
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tbe/ascendc/pool_3d_common
            COMMAND cp -r ${PROJECT_SOURCE_DIR}/pooling/pool_3d_common/op_kernel/* ${CMAKE_BINARY_DIR}/tbe/ascendc/pool_3d_common
        )

        kernel_src_copy(
            TARGET ${KERNEL_COPY_TARGET}
            OP_LIST ${COMPILED_OPS_UT}
            IMPL_DIR ${COMPILED_OP_DIRS_UT}
            DST_DIR ${ASCEND_KERNEL_SRC_DST}
        )
        add_dependencies(${KERNEL_COPY_TARGET} ${KERNEL_COMMON_COPY_TARGET})
    endif()

    ## get/standardize opType
    set(opType "")
    file(GLOB jsonFiles "${PROJECT_SOURCE_DIR}/*/${opName}/op_host/config/*/${opName}_binary.json")
    list(LENGTH jsonFiles numFiles)
    if(numFiles EQUAL 0)
        string(REPLACE "_" ";" opTypeTemp "${opName}")
        foreach(word IN LISTS opTypeTemp)
            string(SUBSTRING "${word}" 0 1 firstLetter)
            string(SUBSTRING "${word}" 1 -1 restOfWord)
            string(TOUPPER "${firstLetter}" firstLetter)
            string(TOLOWER "${restOfWord}" restOfWord)
            set(opType "${opType}${firstLetter}${restOfWord}")
        endforeach()
    endif()
    if(NOT numFiles EQUAL 0)
        foreach(jsonFile ${jsonFiles})
            get_op_type_from_binary_json(${jsonFile} resultVar)
            set(opType ${resultVar})
            message(STATUS "Current file opType: ${opType}")
        endforeach()
    endif()

    set(kernel_dependency_files ${ASCEND_KERNEL_SRC_DST}/${opName})

    if (NOT TARGET pre_op_kernel_ut)
        add_custom_target(pre_op_kernel_ut)
    endif()

    foreach(oriSocVersion ${supportedSocVersion})
        ## standardize socVersion
        string(REPLACE "ascend" "Ascend" socVersion "${oriSocVersion}")
        string(TOLOWER ${oriSocVersion} lowerSocVersion)
        set(socMatch OFF)
        foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
            map_compute_unit(${compute_unit} compute_unit_long)
            if (${compute_unit_long} STREQUAL ${lowerSocVersion} OR ${compute_unit} STREQUAL ${lowerSocVersion})
                set(socMatch ON)

                set(opsInfo ${CMAKE_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/aic-${compute_unit}-info.json)
                if (EXISTS ${opsInfo})
                    execute_process(
                        COMMAND
                        ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/get_opfile_from_opsinfo.py
                        ${opsInfo} ${opType}
                        RESULT_VARIABLE EXEC_RESULT
                        OUTPUT_VARIABLE TEMP
                        ERROR_VARIABLE EXEC_ERROR
                    )

                    string(REGEX REPLACE "\n" "" kernel_cpp_name "${TEMP}")
                    message(STATUS "get_opfile_from_opsinfo kernel_cpp_name is ${kernel_cpp_name}")
                else()
                    gen_aclnn_with_opdef()
                    if(NOT TARGET opbuild_custom_gen_aclnn_all)
                        message(STATUS "no need build binary, for all the ops donot have any operator def")
                        break()
                    endif()

                    merge_ini_files(TARGET merge_ini_${compute_unit}_${opName}
                        OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
                        COMPUTE_UNIT ${compute_unit}
                    )

                    add_dependencies(merge_ini_${compute_unit}_${opName} opbuild_custom_gen_aclnn_all)

                    add_ops_info_target_v1(
                        TARGET ops_info_gen_${lowerSocVersion}_${opName}
                        OUTPUT ${opsInfo}
                        OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
                        COMPUTE_UNIT ${compute_unit}
                        INSTALL_DIR ${OPS_INFO_INSTALL_DIR}/${compute_unit}
                    )
                    add_dependencies(ops_info_gen_${lowerSocVersion}_${opName} merge_ini_${compute_unit}_${opName})

                    add_dependencies(pre_op_kernel_ut ops_info_gen_${lowerSocVersion}_${opName})
                endif()
                break()
            endif()
        endforeach()
        if (NOT ${socMatch})
            continue()
        endif()

        ## add tiling tmp so: ${opName}_${socVersion}_tiling_tmp.so
        add_library(${opName}_${socVersion}_tiling_tmp SHARED)
        add_dependencies(${opName}_${socVersion}_tiling_tmp json)
        target_include_directories(${opName}_${socVersion}_tiling_tmp PRIVATE
                ${OPBASE_INC_DIRS}
                ${OP_TILING_INCLUDE}
                ${PROJECT_SOURCE_DIR}/common/inc
                ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
                ${JSON_INCLUDE}
                )
        target_sources(${opName}_${socVersion}_tiling_tmp PUBLIC
            $<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>
            # 只要尝试跑ophost ut，legacy_common_manager.cpp就不会编译到tiling obj里，此时需要把打桩代码编译进去
            $<$<OR:$<BOOL:${UT_TEST_ALL}>,$<BOOL:${OP_HOST_UT}>>:${OPS_NN_DIR}/tests/ut/common/legacy_common_manager_stub.cpp>
        )

        target_compile_definitions(${opName}_${socVersion}_tiling_tmp PRIVATE
                LOG_CPP
                _GLIBCXX_USE_CXX11_ABI=0
                )
        target_link_libraries(${opName}_${socVersion}_tiling_tmp PRIVATE
                -Wl,--no-as-needed
                    $<$<TARGET_EXISTS:opsbase>:opsbase>
                -Wl,--as-needed
                $<BUILD_INTERFACE:dlog_headers>
                -Wl,--whole-archive
                    tiling_api
                -Wl,--no-whole-archive
                gcov
                metadef
                )

        ## gen ascendc tiling head files
        set(tilingFile ${CMAKE_CURRENT_BINARY_DIR}/${opName}_tiling_data.h)
        set(foreachTilingFile ${OPS_NN_DIR}/foreach/foreach_abs/tests/ut/op_kernel/foreach_abs_tiling_def.h)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${opName}_tiling_def.h")
            set(compileOptions -include "${CMAKE_CURRENT_SOURCE_DIR}/${opName}_tiling_def.h")
        else()
            if(${opCategory} STREQUAL "foreach")
                set(compileOptions -include ${foreachTilingFile})
            else()
                set(compileOptions -include ${tilingFile})
                if(NOT EXISTS ${tilingFile})
                    set(gen_tiling_head_file ${OPS_NN_DIR}/tests/ut/op_kernel/scripts/gen_tiling_head_file.sh)
                    set(gen_tiling_so_path ${CMAKE_CURRENT_BINARY_DIR}/lib${opName}_${socVersion}_tiling_tmp.so)
                    set(gen_tiling_head_tag ${opName}_${socVersion}_gen_head)
                    set(gen_cmd "bash ${gen_tiling_head_file} ${opType} ${opName} ${gen_tiling_so_path} ${CUSTOM_TILING_DATA_KEYS}")
                    message("gen tiling head file to ${tilingFile}, command:")
                    message("${gen_cmd}")
                    add_custom_command(OUTPUT ${tilingFile}
                        COMMAND rm -f ${tilingFile}
                        COMMAND bash -c ${gen_cmd}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        DEPENDS ${opName}_${socVersion}_tiling_tmp
                    )
                    add_custom_target(${gen_tiling_head_tag} ALL
                        DEPENDS ${tilingFile}
                    )
                endif()

            endif()
        endif()
        set(CUSTOM_TILING_DATA_KEYS "")
        string(REGEX MATCH "-DUT_CUSTOM_TILING_DATA_KEYS=([^ ]+)" matchedPart "${otherCompileOptions}")
        if(CMAKE_MATCH_1)
            set(CUSTOM_TILING_DATA_KEYS ${CMAKE_MATCH_1})
            string(REGEX REPLACE "-DUT_CUSTOM_TILING_DATA_KEYS=[^ ]+" "" modifiedString ${otherCompileOptions})
            set(otherCompileOptions ${modifiedString})
        endif()
        string(REPLACE " " ";" options "${otherCompileOptions}")
        foreach(option IN LISTS options)
            set(compileOptions ${compileOptions} ${option})
        endforeach()
        message("compileOptions: ${compileOptions}")

        if("${kernel_cpp_name}" STREQUAL "")
            set(kernel_cpp_name ${opName})
        endif()

        set(kernelFile ${ASCEND_KERNEL_SRC_DST}/${opName}/${kernel_cpp_name}.cpp)
        set_source_files_properties(${kernelFile} PROPERTIES GENERATED TRUE)

        set(testCaseFileName "test_${kernel_cpp_name}")

        ## add object: ${opName}_${socVersion}_cases_obj
        get_target_dir(${lowerSocVersion} target_dir)
        if(target_dir STREQUAL "")
            file(GLOB OPKERNEL_CASES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${testCaseFileName}*.cpp)
        else()
            file(GLOB OPKERNEL_CASES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${testCaseFileName}*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${target_dir}/${testCaseFileName}*.cpp)
        endif()

        get_property(LIBSEXIST GLOBAL PROPERTY OPKERNEL_OBJECT_LIBS)
        if(NOT LIBSEXIST)
            set_property(GLOBAL PROPERTY OPKERNEL_OBJECT_LIBS "")
        endif()
        if(NOT DEFINED multi_kernel_target)
            add_library(opkernel_${opName} OBJECT ${OPKERNEL_CASES_SRC} ${kernelFile})
            add_dependencies(opkernel_${opName} ${gen_tiling_head_tag} ${KERNEL_COPY_TARGET})
            target_compile_options(opkernel_${opName} PRIVATE
                -g ${compileOptions} -DUT_SOC_VERSION="${socVersion}" -DKERNELUT=1
            )
            target_link_libraries(opkernel_${opName} PRIVATE
                $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
                tikicpulib::${oriSocVersion}
                gtest
            )
            target_include_directories(opkernel_${opName} PRIVATE
                ${ASCEND_DIR}/include/base/context_builder
                ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel
                ${PROJECT_SOURCE_DIR}/tests/ut/common
                ${PROJECT_SOURCE_DIR}/conv/conv3d_backprop_input_v2/op_kernel
                ${kernel_dependency_files}
                ${CMAKE_BINARY_DIR}/tbe/ascendc/
                ${CMAKE_BINARY_DIR}/tbe/ascendc/common
            )
            set_property(GLOBAL APPEND PROPERTY OPKERNEL_OBJECT_LIBS opkernel_${opName})
        else()
            foreach(src_file ${OPKERNEL_CASES_SRC})
                get_filename_component(src_name ${src_file} NAME_WE)
                add_library(opkernel_${src_name} OBJECT ${src_file})
                add_dependencies(opkernel_${src_name} ${gen_tiling_head_tag} ${KERNEL_COPY_TARGET})
                target_compile_options(opkernel_${src_name} PRIVATE
                    -g ${compileOptions} -DUT_SOC_VERSION="${socVersion}"
                )
                target_link_libraries(opkernel_${src_name} PRIVATE
                    $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
                    tikicpulib::${oriSocVersion}
                    gtest
                )
                target_include_directories(opkernel_${src_name} PRIVATE
                    ${ASCEND_DIR}/include/base/context_builder
                    ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel
                    ${PROJECT_SOURCE_DIR}/tests/ut/common
                    ${PROJECT_SOURCE_DIR}/conv/conv3d_backprop_input_v2/op_kernel
                    ${kernel_dependency_files}
                    ${CMAKE_BINARY_DIR}/tbe/ascendc/
                    ${CMAKE_BINARY_DIR}/tbe/ascendc/common
                )
                set_property(GLOBAL APPEND PROPERTY OPKERNEL_OBJECT_LIBS opkernel_${src_name})
            endforeach()
        endif()
        list(FIND fastOpTestSocVersions "${oriSocVersion}" index)
        if(index EQUAL -1)
            set(fastOpTestSocVersions ${fastOpTestSocVersions} ${oriSocVersion} CACHE STRING "fastOp Test SocVersions" FORCE)
        endif()
    endforeach()
endfunction()

function(AddOpTestCaseV2 opName opFileValue supportedSocVersion otherCompileOptions)
    set(DEPENDENCY_OPS ${ARGN})

    get_filename_component(UT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
    get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
    list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
    ## if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
        return()
    endif()

    set(COMPILED_OPS_UT "${opName}")
    set(COMPILED_OP_DIRS_UT ${OP_NAME_DIR})
    if(DEPENDENCY_OPS)
        foreach(depOp ${DEPENDENCY_OPS})
            list(APPEND COMPILED_OPS_UT ${depOp})
            get_filename_component(DEP_OP_NAME_DIR ${OP_NAME_DIR}/../${depOp} ABSOLUTE)
            list(APPEND COMPILED_OP_DIRS_UT ${DEP_OP_NAME_DIR})
        endforeach()
    endif()
    set(COMPILED_OPS_UT "${COMPILED_OPS_UT}" CACHE STRING "Compiled Ops" FORCE)

    set(KERNEL_COPY_TARGET "ascendc_kernel_ut_src_copy_${opName}")
    ## kernel src copy
    if(NOT TARGET ${KERNEL_COPY_TARGET})
        kernel_src_copy(
            TARGET ${KERNEL_COPY_TARGET}
            OP_LIST ${COMPILED_OPS_UT}
            IMPL_DIR ${COMPILED_OP_DIRS_UT}
            DST_DIR ${ASCEND_KERNEL_SRC_DST}
        )
    endif()

    ## get/standardize opType
    set(opType "")
    file(GLOB jsonFiles "${PROJECT_SOURCE_DIR}/*/${opName}/op_host/config/*/${opName}_binary.json")
    list(LENGTH jsonFiles numFiles)
    if(numFiles EQUAL 0)
        string(REPLACE "_" ";" opTypeTemp "${opName}")
        foreach(word IN LISTS opTypeTemp)
            string(SUBSTRING "${word}" 0 1 firstLetter)
            string(SUBSTRING "${word}" 1 -1 restOfWord)
            string(TOUPPER "${firstLetter}" firstLetter)
            string(TOLOWER "${restOfWord}" restOfWord)
            set(opType "${opType}${firstLetter}${restOfWord}")
        endforeach()
    endif()
    if(NOT numFiles EQUAL 0)
        foreach(jsonFile ${jsonFiles})
            get_op_type_from_binary_json(${jsonFile} resultVar)
            set(opType ${resultVar})
            message(STATUS "Current file opType: ${opType}")
        endforeach()
    endif()

    foreach(oriSocVersion ${supportedSocVersion})
        ## standardize socVersion
        STRING(REPLACE "ascend" "Ascend" socVersion "${oriSocVersion}")

        ## add tiling tmp so: ${opName}_${socVersion}_tiling_tmp.so
        add_library(${opName}_${socVersion}_tiling_tmp SHARED)
        add_dependencies(${opName}_${socVersion}_tiling_tmp json)
        target_include_directories(${opName}_${socVersion}_tiling_tmp PRIVATE
                ${OPBASE_INC_DIRS}
                ${OP_TILING_INCLUDE}
                ${PROJECT_SOURCE_DIR}/common/inc
                ${JSON_INCLUDE}
                )
        target_sources(${opName}_${socVersion}_tiling_tmp PUBLIC
            $<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>
            # 只要尝试跑ophost ut，legacy_common_manager.cpp就不会编译到tiling obj里，此时需要把打桩代码编译进去
            $<$<OR:$<BOOL:${UT_TEST_ALL}>,$<BOOL:${OP_HOST_UT}>>:${OPS_NN_DIR}/tests/ut/common/legacy_common_manager_stub.cpp>
        )

        target_compile_definitions(${opName}_${socVersion}_tiling_tmp PRIVATE
                LOG_CPP
                _GLIBCXX_USE_CXX11_ABI=0
                )
        target_link_libraries(${opName}_${socVersion}_tiling_tmp PRIVATE
                -Wl,--no-as-needed
                    $<$<TARGET_EXISTS:opsbase>:opsbase>
                -Wl,--as-needed
                -Wl,--whole-archive
                    tiling_api
                -Wl,--no-whole-archive
                $<BUILD_INTERFACE:dlog_headers>
                gcov
                metadef
                )

        ## gen ascendc tiling head files
        set(tilingFile ${CMAKE_CURRENT_BINARY_DIR}/${opName}_tiling_data.h)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${opName}_tiling_def.h")
            set(compileOptions -include "${CMAKE_CURRENT_SOURCE_DIR}/${opName}_tiling_def.h")
        else()
            set(compileOptions -include ${tilingFile})
        endif()
        set(CUSTOM_TILING_DATA_KEYS "")
        string(REGEX MATCH "-DUT_CUSTOM_TILING_DATA_KEYS=([^ ]+)" matchedPart "${otherCompileOptions}")
        if(CMAKE_MATCH_1)
            set(CUSTOM_TILING_DATA_KEYS ${CMAKE_MATCH_1})
            string(REGEX REPLACE "-DUT_CUSTOM_TILING_DATA_KEYS=[^ ]+" "" modifiedString ${otherCompileOptions})
            set(otherCompileOptions ${modifiedString})
        endif()
        string(REPLACE " " ";" options "${otherCompileOptions}")
        foreach(option IN LISTS options)
            set(compileOptions ${compileOptions} ${option})
        endforeach()
        message("compileOptions: ${compileOptions}")
        set(gen_tiling_head_file ${OPS_NN_DIR}/tests/ut/op_kernel/scripts/gen_tiling_head_file.sh)
        set(gen_tiling_so_path ${CMAKE_CURRENT_BINARY_DIR}/lib${opName}_${socVersion}_tiling_tmp.so)
        set(gen_tiling_head_tag ${opName}_${socVersion}_gen_head)
        set(gen_cmd "bash ${gen_tiling_head_file} ${opType} ${opName} ${gen_tiling_so_path} ${CUSTOM_TILING_DATA_KEYS}")
        message("gen tiling head file to ${tilingFile}, command:")
        message("${gen_cmd}")
        add_custom_command(OUTPUT ${tilingFile}
                COMMAND rm -f ${tilingFile}
                COMMAND bash -c ${gen_cmd}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS ${opName}_${socVersion}_tiling_tmp
                )
        add_custom_target(${gen_tiling_head_tag} ALL
                DEPENDS ${tilingFile}
                )

        set(kernelFile ${ASCEND_KERNEL_SRC_DST}/${opName}/${opFileValue}.cpp)
        set_source_files_properties(${kernelFile} PROPERTIES GENERATED TRUE)

        ## add object: ${opName}_${socVersion}_cases_obj
        file(GLOB OPKERNEL_CASES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_${opName}*.cpp)
        add_library(${opName}_${socVersion}_cases_obj OBJECT)
        target_sources(${opName}_${socVersion}_cases_obj PRIVATE
            ${kernelFile}
            ${OPKERNEL_CASES_SRC}
        )
        add_dependencies(${opName}_${socVersion}_cases_obj ${gen_tiling_head_tag} ${KERNEL_COPY_TARGET})
        target_compile_options(${opName}_${socVersion}_cases_obj PRIVATE
            -g ${compileOptions} -DUT_SOC_VERSION="${socVersion}"
        )
        target_include_directories(${opName}_${socVersion}_cases_obj PRIVATE
                ${ASCEND_DIR}/include/base/context_builder
                ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel
                ${PROJECT_SOURCE_DIR}/tests/ut/common
                )
        target_link_libraries(${opName}_${socVersion}_cases_obj PRIVATE
                $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
                tikicpulib::${socVersion}
                gtest
                )

        ## add object: nn_op_kernel_ut_${oriSocVersion}_cases_obj
        if(NOT TARGET ${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj)
            add_library(${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj OBJECT
                $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
                )
        endif()
        target_link_libraries(${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj PRIVATE
                $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
                $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
                )

        list(FIND fastOpTestSocVersions "${oriSocVersion}" index)
        if(index EQUAL -1)
            set(fastOpTestSocVersions ${fastOpTestSocVersions} ${oriSocVersion} CACHE STRING "fastOp Test SocVersions" FORCE)
        endif()
    endforeach()
endfunction()

if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
  function(AddAicpuOpTestCase opName)
    get_filename_component(UT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${UT_DIR} NAME)
    list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
    # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
      return()
    endif()

    ## find kernel file
    file(GLOB KernelFile "${PROJECT_SOURCE_DIR}/*/${opName}/op_kernel_aicpu/${opName}_aicpu.cpp")

    ## add object: ${opName}_cases_obj
    file(GLOB OPKERNEL_CASES_SRC ${UT_DIR}/tests/ut/op_kernel_aicpu/test_${opName}*.cpp)

    message(STATUS "aicpu kernel info: ${opName}, ${KernelFile}, ${OPKERNEL_CASES_SRC}")

    add_library(${opName}_cases_obj OBJECT
            ${KernelFile}
            ${OPKERNEL_CASES_SRC}
            )
    target_compile_options(${opName}_cases_obj PRIVATE
            -g
            )
    message(STATUS "111******************** ${AICPU_INCLUDE}")
    ## add op_kernel_aicpu test header file search path, so that header files can be referenced based on relative path
    target_include_directories(${opName}_cases_obj PRIVATE
            ${AICPU_INCLUDE}
            ${OPBASE_INC_DIRS}
            ${AICPU_INC_DIRS}
            ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel_aicpu
            ${ASCEND_DIR}/pkg_inc/base
            )
    target_link_libraries(${opName}_cases_obj PRIVATE
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
            -ldl
            gtest
            c_sec
            Eigen3::EigenNn
            $<$<TARGET_EXISTS:opsbase>:opsbase>
            )

    ## add object: nn_op_kernel_ut_cases_obj
    if(NOT TARGET ${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj)
      add_library(
        ${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj OBJECT
        $<TARGET_OBJECTS:${opName}_cases_obj>
        )
    else()
      target_sources(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj PRIVATE $<TARGET_OBJECTS:${opName}_cases_obj>)
    endif()

    target_link_libraries(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan>
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
            -ldl
            $<TARGET_OBJECTS:${opName}_cases_obj>
            gtest
            c_sec
            Eigen3::EigenNn
      $<$<TARGET_EXISTS:opsbase>:opsbase>
            )
  endfunction()
endif()
