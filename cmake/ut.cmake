# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

function(add_optiling_ut_modules OP_TILING_MODULE_NAME)

    # set variables
    set(UT_COMMON_INC ${PROJECT_SOURCE_DIR}/tests/ut/common)
    set(UT_CONV3DV2_TILING_INC ${PROJECT_SOURCE_DIR}/common/stub/op_tiling
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

    target_compile_options(${OP_TILING_MODULE_NAME}_common_obj PRIVATE
        -fno-access-control
    )

    # add op tiling ut static lib
    add_library(${OP_TILING_MODULE_NAME}_static_lib STATIC
        $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
        $<$<TARGET_EXISTS:${OP_TILING_MODULE_NAME}_common_obj>:$<TARGET_OBJECTS:${OP_TILING_MODULE_NAME}_common_obj>>
    )
    target_link_libraries(${OP_TILING_MODULE_NAME}_static_lib PRIVATE
        ${OP_TILING_MODULE_NAME}_common_obj
        ${OP_TILING_MODULE_NAME}_cases_obj
        $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:optiling>>
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

    target_compile_options(${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE
        -fno-access-control
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
                    )
    target_include_directories(${OP_API_MODULE_NAME}_cases_obj PRIVATE
            ${JSON_INCLUDE}
            ${OPS_NN_COMMON_INC}
            ${OP_API_UT_COMMON_INC}
            ${UT_PATH}/op_api/stub
            ${ASCEND_DIR}/include
            ${ASCEND_DIR}/include/aclnn
            ${ASCEND_DIR}/include/aclnnop
            )
    set_source_files_properties(${ASCEND_DIR}/include/aclnn/opdev/make_op_executor.h PROPERTIES HEAD_FILE_ONLY TRUE)
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
        ## add op kernel ut cases obj: nn_op_kernel_ut_${socVersion}_cases
        if(NOT TARGET ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj)
            add_library(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj OBJECT)
        endif()
        target_link_libraries(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj PRIVATE
            gcov
            )

        ## add op kernel ut cases dynamic lib: libnn_op_kernel_ut_${socVersion}_cases.so
        add_library(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases SHARED
            $<TARGET_OBJECTS:${OP_KERNEL_MODULE_NAME}_common_obj>
            $<TARGET_OBJECTS:${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj>
            )
        target_link_libraries(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases PRIVATE
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
            ${OP_KERNEL_MODULE_NAME}_common_obj
            ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj
        )
    endforeach()
endfunction()

function(add_modules_llt_sources)
    add_modules_ut_sources(${ARGN})
endfunction()

function(add_modules_ut_sources)
    set(options OPTION_RESERVED)
    set(oneValueArgs HOSTNAME MODE DIR)
    set(multiValueArgs MULIT_RESERVED)

    cmake_parse_arguments(MODULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(supportedCategory "matmul" "conv" "control" "index" "rnn" "vfusion" "pooling" "foreach" "loss")
    string(REPLACE "${OPS_NN_DIR}/" "" opCategoryDir ${CMAKE_CURRENT_SOURCE_DIR})
    string(FIND "${opCategoryDir}" "/" firstSlashPos)
    string(SUBSTRING "${opCategoryDir}" 0 ${firstSlashPos} opCategory)
    list(FIND supportedCategory ${opCategory} opCategoryIndex)

    message(STATUS "=== Debug<add_modules_ut_sources>: ${MODULE_HOSTNAME} ${MODULE_MODE} ${MODULE_DIR}")

    if(NOT opCategoryIndex EQUAL -1)
        string(FIND "${MODULE_HOSTNAME}_cases_obj" "tiling" TILING_FOUND_INDEX)
        file(GLOB OPHOST_TILING_SRCS ${MODULE_DIR}/test_*_tiling.cpp)
        if(${TILING_FOUND_INDEX} GREATER_EQUAL 0 AND OPHOST_TILING_SRCS)
            if (NOT TARGET ${MODULE_HOSTNAME}_cases_obj)
                add_optiling_ut_modules(${OP_TILING_MODULE_NAME})
            endif()
            target_sources(${MODULE_HOSTNAME}_cases_obj ${MODULE_MODE} ${OPHOST_TILING_SRCS})
            message(STATUS "=== Debug<add_modules_ut_sources>: ${MODULE_HOSTNAME}_cases_obj ${OPHOST_TILING_SRCS}")
        endif()

        string(FIND "${MODULE_HOSTNAME}_cases_obj" "infershape" INFERSHAPE_FOUND_INDEX)
        file(GLOB OPHOST_INFERSHAPE_SRCS ${MODULE_DIR}/test_*_infershape.cpp)
        if(${INFERSHAPE_FOUND_INDEX} GREATER_EQUAL 0 AND OPHOST_INFERSHAPE_SRCS)
            if (NOT TARGET ${MODULE_HOSTNAME}_cases_obj)
                add_infershape_ut_modules(${OP_INFERSHAPE_MODULE_NAME})
            endif()
            target_sources(${MODULE_HOSTNAME}_cases_obj ${MODULE_MODE} ${OPHOST_INFERSHAPE_SRCS})
            message(STATUS "=== Debug<add_modules_ut_sources>: ${MODULE_HOSTNAME}_cases_obj ${OPHOST_INFERSHAPE_SRCS}")
        endif()

        string(FIND "${MODULE_HOSTNAME}_cases_obj" "op_api" OPAPI_FOUND_INDEX)
        if(${OPAPI_FOUND_INDEX} GREATER_EQUAL 0)
            file(GLOB OPHOST_OPAPI_SRCS ${MODULE_DIR}/test_aclnn_*.cpp)
            if (NOT TARGET ${MODULE_HOSTNAME}_cases_obj)
                add_opapi_ut_modules(${OP_API_MODULE_NAME})
            endif()
            target_sources(${MODULE_HOSTNAME}_cases_obj ${MODULE_MODE} ${OPHOST_OPAPI_SRCS})
            message(STATUS "=== Debug<add_modules_ut_sources>: ${MODULE_HOSTNAME}_cases_obj ${OPHOST_OPAPI_SRCS}")
        endif()
    endif()
endfunction()

if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(fastOpTestSocVersions "" CACHE STRING "fastOp Test SocVersions")
endif()

function(AddOpTestCase opName supportedSocVersion otherCompileOptions)
    set(DEPENDENCY_OPS ${ARGN})

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
    ## kernel src copy
    kernel_src_copy(
        TARGET ${KERNEL_COPY_TARGET}
        OP_LIST ${COMPILED_OPS_UT}
        IMPL_DIR ${COMPILED_OP_DIRS_UT}
        DST_DIR ${ASCEND_KERNEL_SRC_DST}
    )

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

    set(kernel_source_files ${ASCEND_KERNEL_SRC_DST}/${opName})

    foreach(oriSocVersion ${supportedSocVersion})
        ## standardize socVersion
        STRING(REPLACE "ascend" "Ascend" socVersion "${oriSocVersion}")

        file(GLOB_RECURSE EXTRA_TILING_FILES
            "${OPS_NN_DIR}/common/stub/op_tiling/op_cache_tiling.cpp"
            "${OPS_NN_DIR}/common/stub/op_tiling/tbe_tiling_api.cpp"
            "${OPS_NN_DIR}/common/stub/op_tiling/runtime_kb_api.cpp"
        )
        set(tiling_dependency_files
            ${EXTRA_TILING_FILES}
        )

        ## add tiling tmp so: ${opName}_${socVersion}_tiling_tmp.so
        add_library(${opName}_${socVersion}_tiling_tmp SHARED
                ${tiling_dependency_files}
                )
        add_dependencies(${opName}_${socVersion}_tiling_tmp json)
        target_include_directories(${opName}_${socVersion}_tiling_tmp PRIVATE
                ${OPBASE_INC_DIRS}
                ${PROJECT_SOURCE_DIR}/common/inc
                ${JSON_INCLUDE}
                )
        target_sources(${opName}_${socVersion}_tiling_tmp PUBLIC
            $<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>
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

        set(special_ops_list "embedding_bag")
        if(opName IN_LIST special_ops_list)
            # 如果在列表中，使用带_apt后缀的文件
            set(cpp_file "${ASCEND_KERNEL_SRC_DST}/${opName}/${opName}_apt.cpp")
        else()
            # 如果不在列表中，使用普通文件
            set(cpp_file "${ASCEND_KERNEL_SRC_DST}/${opName}/${opName}.cpp")
        endif()

        set(kernelFile ${cpp_file})
        set_source_files_properties(${kernelFile} PROPERTIES GENERATED TRUE)

        ## add object: ${opName}_${socVersion}_cases_obj
        file(GLOB OPKERNEL_CASES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_${opName}*.cpp)
        add_library(${opName}_${socVersion}_cases_obj OBJECT)
        target_sources(${opName}_${socVersion}_cases_obj PRIVATE
            ${kernelFile}
            ${OPKERNEL_CASES_SRC}
        )
        add_dependencies(${opName}_${socVersion}_cases_obj ${gen_tiling_head_tag} ascendc_kernel_ut_src_copy_${opName})
        target_compile_options(${opName}_${socVersion}_cases_obj PRIVATE




        -g ${compileOptions} -DUT_SOC_VERSION="${socVersion}"
                )
        target_include_directories(${opName}_${socVersion}_cases_obj PRIVATE
                ${ASCEND_DIR}/include/base/context_builder
                ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel
                ${PROJECT_SOURCE_DIR}/tests/ut/common
                ${PROJECT_SOURCE_DIR}/conv/conv3d_backprop_input_v2/op_kernel
                ${kernel_source_files}
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
    kernel_src_copy(
        TARGET ${KERNEL_COPY_TARGET}
        OP_LIST ${COMPILED_OPS_UT}
        IMPL_DIR ${COMPILED_OP_DIRS_UT}
        DST_DIR ${ASCEND_KERNEL_SRC_DST}
    )

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

        file(GLOB_RECURSE EXTRA_TILING_FILES
            "${OPS_NN_DIR}/common/stub/op_tiling/op_cache_tiling.cpp"
            "${OPS_NN_DIR}/common/stub/op_tiling/tbe_tiling_api.cpp"
            "${OPS_NN_DIR}/common/stub/op_tiling/runtime_kb_api.cpp"
        )
        set(tiling_dependency_files
            ${EXTRA_TILING_FILES}
        )

        ## add tiling tmp so: ${opName}_${socVersion}_tiling_tmp.so
        add_library(${opName}_${socVersion}_tiling_tmp SHARED
                ${tiling_dependency_files}
                )
        add_dependencies(${opName}_${socVersion}_tiling_tmp json)
        target_include_directories(${opName}_${socVersion}_tiling_tmp PRIVATE
                ${OPBASE_INC_DIRS}
                ${PROJECT_SOURCE_DIR}/common/inc
                ${JSON_INCLUDE}
                )

        target_sources(${opName}_${socVersion}_tiling_tmp PUBLIC
            $<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>
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
        add_dependencies(${opName}_${socVersion}_cases_obj ${gen_tiling_head_tag} ascendc_kernel_ut_src_copy_${opName})
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