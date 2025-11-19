# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 为target link依赖库 useage: add_modules(MODE SUB_LIBS EXTERNAL_LIBS) SUB_LIBS 为内部创建的target, EXTERNAL_LIBS为外部依赖的target
function(add_modules)
  set(oneValueArgs MODE)
  set(multiValueArgs TARGETS SUB_LIBS EXTERNAL_LIBS)

  cmake_parse_arguments(ARGS "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  foreach(target ${ARGS_TARGETS})
    if(TARGET ${target} AND TARGET ${ARGS_SUB_LIBS})
      target_link_libraries(${target} ${ARGS_MODE} ${ARGS_SUB_LIBS})
    endif()
    if(TARGET ${target} AND ARGS_EXTERNAL_LIBS)
      target_link_libraries(${target} ${ARGS_MODE} ${ARGS_EXTERNAL_LIBS})
    endif()
  endforeach()
endfunction()

# 添加infer object
function(add_infer_modules)
  if(NOT TARGET ${OPHOST_NAME}_infer_obj)
    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      npu_op_library(${OPHOST_NAME}_infer_obj GRAPH)
    else()
      add_library(${OPHOST_NAME}_infer_obj OBJECT)
    endif()
    target_include_directories(${OPHOST_NAME}_infer_obj PRIVATE ${OP_PROTO_INCLUDE})
    target_compile_definitions(
      ${OPHOST_NAME}_infer_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="OP_PROTO" OP_SUBMOD_NAME="OPS_NN"
                                       $<$<BOOL:${ENABLE_TEST}>:ASCEND_OPSPROTO_UT> LOG_CPP
      )
    target_compile_options(
      ${OPHOST_NAME}_infer_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                       -fvisibility=hidden
      )
    target_link_libraries(
      ${OPHOST_NAME}_infer_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:ops_base_util_objs>:$<TARGET_OBJECTS:ops_base_util_objs>>
              $<$<TARGET_EXISTS:ops_base_infer_objs>:$<TARGET_OBJECTS:ops_base_infer_objs>>
      )
  endif()
endfunction()

# 添加tiling object
function(add_tiling_modules)
  if(NOT TARGET ${OPHOST_NAME}_tiling_obj)
    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      npu_op_library(${OPHOST_NAME}_tiling_obj TILING)
      add_dependencies(${OPHOST_NAME}_tiling_obj json)
    else()
      add_library(${OPHOST_NAME}_tiling_obj OBJECT)
      add_dependencies(${OPHOST_NAME}_tiling_obj json)
    endif()
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${OP_TILING_INCLUDE})
    target_compile_definitions(
      ${OPHOST_NAME}_tiling_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING" OP_SUBMOD_NAME="OPS_NN"
                                        $<$<BOOL:${ENABLE_TEST}>:ASCEND_OPTILING_UT> LOG_CPP
      )
    target_compile_options(
      ${OPHOST_NAME}_tiling_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                        -fvisibility=hidden -fno-strict-aliasing
      )
    target_link_libraries(
      ${OPHOST_NAME}_tiling_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
              $<$<TARGET_EXISTS:ops_base_util_objs>:$<TARGET_OBJECTS:ops_base_util_objs>>
              $<$<TARGET_EXISTS:ops_base_tiling_objs>:$<TARGET_OBJECTS:ops_base_tiling_objs>>
              tiling_api
      )
  endif()
endfunction()

# 添加opapi object
function(add_opapi_modules)
  if(NOT TARGET ${OPHOST_NAME}_opapi_obj)
    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      npu_op_library(${OPHOST_NAME}_opapi_obj ACLNN)
    else()
      add_library(${OPHOST_NAME}_opapi_obj OBJECT)
    endif()

    if(ENABLE_TEST)
      set(opapi_ut_depends_inc ${UT_PATH}/op_api/stub)
    endif()
    target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE
            ${opapi_ut_depends_inc}
            ${OPAPI_INCLUDE})
    target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_INCLUDE})
    target_compile_options(${OPHOST_NAME}_opapi_obj PRIVATE -Dgoogle=ascend_private -DACLNN_LOG_FMT_CHECK)
    target_compile_definitions(${OPHOST_NAME}_opapi_obj PRIVATE LOG_CPP)
    target_link_libraries(
      ${OPHOST_NAME}_opapi_obj
      PUBLIC $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
      PRIVATE $<BUILD_INTERFACE:adump_headers> $<BUILD_INTERFACE:dlog_headers>
      )
  endif()
endfunction()

function(add_aicpu_kernel_modules)
  message(STATUS "add_aicpu_kernel_modules")
  if(NOT TARGET ${OPHOST_NAME}_aicpu_obj)
    add_library(${OPHOST_NAME}_aicpu_obj OBJECT)
    target_include_directories(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_INCLUDE})
    target_compile_definitions(
      ${OPHOST_NAME}_aicpu_obj PRIVATE _FORTIFY_SOURCE=2 google=ascend_private
                                       $<$<BOOL:${ENABLE_TEST}>:ASCEND_AICPU_UT>
      )
    target_compile_options(
      ${OPHOST_NAME}_aicpu_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                       -fvisibility=hidden ${AICPU_DEFINITIONS}
      )
    target_link_libraries(
      ${OPHOST_NAME}_aicpu_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
      )
  endif()
endfunction()

function(add_aicpu_cust_kernel_modules target_name)
  message(STATUS "add_aicpu_cust_kernel_modules for ${target_name}")
  if(NOT TARGET ${target_name})
    add_library(${target_name} OBJECT)
    target_include_directories(${target_name} PRIVATE ${AICPU_INCLUDE})
    target_compile_definitions(
      ${target_name} PRIVATE
                    _FORTIFY_SOURCE=2 _GLIBCXX_USE_CXX11_ABI=1
                    google=ascend_private
                    $<$<BOOL:${ENABLE_TEST}>:ASCEND_AICPU_UT>
      )
    target_compile_options(
      ${target_name} PRIVATE
                    $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                    -fvisibility=hidden ${AICPU_DEFINITIONS}
      )
    target_link_libraries(
      ${target_name}
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              -Wl,--no-whole-archive
              Eigen3::EigenNn
      )
    if (NOT ${target_name} IN_LIST AICPU_CUST_OBJ_TARGETS)
      set(AICPU_CUST_OBJ_TARGETS ${AICPU_CUST_OBJ_TARGETS} ${target_name} CACHE INTERNAL "All aicpu cust obj targets")
    endif()
  endif()
endfunction()

function(add_graph_plugin_modules)
  if(NOT TARGET ${GRAPH_PLUGIN_NAME}_obj)
    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      npu_op_library(${GRAPH_PLUGIN_NAME}_obj GRAPH)
    else()
      add_library(${GRAPH_PLUGIN_NAME}_obj OBJECT)
    endif()
    target_include_directories(${GRAPH_PLUGIN_NAME}_obj PRIVATE ${OP_PROTO_INCLUDE})
    target_compile_definitions(${GRAPH_PLUGIN_NAME}_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="GRAPH_PLUGIN" LOG_CPP)
    target_compile_options(
      ${GRAPH_PLUGIN_NAME}_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                       -fvisibility=hidden
      )
    target_link_libraries(
      ${GRAPH_PLUGIN_NAME}_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:ops_base_util_objs>:$<TARGET_OBJECTS:ops_base_util_objs>>
              $<$<TARGET_EXISTS:ops_base_infer_objs>:$<TARGET_OBJECTS:ops_base_infer_objs>>
      )
  endif()
endfunction()

macro(add_op_subdirectory)
  file(GLOB CURRENT_DIRS RELATIVE ${OP_DIR} ${OP_DIR}/*)
  list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
  if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
    # 非指定算子，只编译不测试
    set(OP_ONLY_COMPILE on)
  else()
    set(OP_ONLY_COMPILE off)
  endif()
  if((NOT ENABLE_TEST AND NOT BENCHMARK) OR OP_ONLY_COMPILE)
      list(REMOVE_ITEM CURRENT_DIRS tests)
  endif()
  foreach(SUB_DIR ${CURRENT_DIRS})
      if(EXISTS "${OP_DIR}/${SUB_DIR}/CMakeLists.txt")
          add_subdirectory(${OP_DIR}/${SUB_DIR})
      endif()
  endforeach()
endmacro()

# useage: add_category_subdirectory 根据ASCEND_OP_NAME和ASCEND_COMPILE_OPS添加指定算子工程
# ASCEND_OP_NAME 指定的算子 ASCEND_COMPILE_OPS  编译需要的算子
macro(add_category_subdirectory)
  foreach(op_category ${OP_CATEGORY_LIST})
    if(ENABLE_EXPERIMENTAL)
      set(op_category_dir ${CMAKE_CURRENT_SOURCE_DIR}/experimental/${op_category})
    else()
      set(op_category_dir ${CMAKE_CURRENT_SOURCE_DIR}/${op_category})
    endif()
    if (IS_DIRECTORY ${op_category_dir})
      file(GLOB CURRENT_DIRS RELATIVE ${op_category_dir} ${op_category_dir}/*)
      foreach(SUB_DIR ${CURRENT_DIRS})
        set(OP_DIR ${op_category_dir}/${SUB_DIR})
        set(OP_NAME "${SUB_DIR}")
        if(${OP_NAME} STREQUAL "common")
          set(OP_NAME "${op_category}.common")
        endif()
        list(FIND ASCEND_COMPILE_OPS ${OP_NAME} INDEX)
        if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
          # ASCEND_OP_NAME 为空表示全部编译
          continue()
        endif()
        if(EXISTS "${OP_DIR}/CMakeLists.txt")
            add_op_subdirectory()
        else()
            if (EXISTS "${OP_DIR}/op_host/CMakeLists.txt")
                add_subdirectory(${OP_DIR}/op_host)
            endif()
        endif()
      endforeach()
    endif()
  endforeach()
endmacro()

# useage: add_modules_sources(DIR OPTYPE ACLNNTYPE DEPENDENCIES) ACLNNTYPE 支持类型aclnn/aclnn_inner/aclnn_exclude OPTYPE 和 ACLNNTYPE
# DEPENDENCIES 算子依赖
# 需一一对应
function(add_modules_sources)
  set(multiValueArgs OPTYPE ACLNNTYPE DEPENDENCIES)
  set(oneValueArgs DIR)

  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${MODULE_DIR})

  # opapi 默认全部编译
  file(GLOB OPAPI_SRCS ${SOURCE_DIR}/op_api/*.cpp)
  if(OPAPI_SRCS)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRCS})
  endif()

  file(GLOB OPAPI_HEADERS ${SOURCE_DIR}/op_api/aclnn_*.h)
  if(OPAPI_HEADERS)
    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE ${OPAPI_HEADERS})
  endif()

  # 获取算子层级目录名称
  get_filename_component(PARENT_DIR ${SOURCE_DIR} DIRECTORY)
  get_filename_component(OP_NAME ${PARENT_DIR} NAME)
  file(APPEND ${ASCEND_SUB_CONFIG_PATH} "OP_CATEGORY;${op_category};OP_NAME;${OP_NAME};${ARGN}\n")
  # 记录全局的COMPILED_OPS和COMPILED_OP_DIRS，其中COMPILED_OP_DIRS只记录到算子名，例如math/abs，common不记录
  if (NOT ${OP_NAME} STREQUAL "common")
    set(COMPILED_OPS
        ${COMPILED_OPS} ${OP_NAME}
        CACHE STRING "Compiled Ops" FORCE
      )
    set(COMPILED_OP_DIRS
        ${COMPILED_OP_DIRS} ${PARENT_DIR}
        CACHE STRING "Compiled Ops Dirs" FORCE
      )
  endif()

  file(GLOB OPINFER_SRCS ${SOURCE_DIR}/*_infershape*.cpp)
  if(OPINFER_SRCS)
    add_infer_modules()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPINFER_SRCS})
  endif()

  file(GLOB OPTILING_SRCS ${SOURCE_DIR}/*_tiling*.cpp)
  file(GLOB_RECURSE SUB_OPTILING_SRC ${SOURCE_DIR}/op_tiling/*.cpp)
  if (OPTILING_SRCS OR SUB_OPTILING_SRC)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS} ${SUB_OPTILING_SRC})
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${SOURCE_DIR}/../../ ${SOURCE_DIR})
  endif()

  file(GLOB AICPU_SRCS ${SOURCE_DIR}/*_aicpu*.cpp)
  if(AICPU_SRCS)
    add_aicpu_kernel_modules()
    target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_SRCS})
  endif()

  if(MODULE_OPTYPE)
    list(LENGTH MODULE_OPTYPE OpTypeLen)
    list(LENGTH MODULE_ACLNNTYPE AclnnTypeLen)
    if(NOT ${OpTypeLen} EQUAL ${AclnnTypeLen})
      message(FATAL_ERROR "OPTYPE AND ACLNNTYPE Should be One-to-One")
    endif()
    math(EXPR index "${OpTypeLen} - 1")
    foreach(i RANGE ${index})
      list(GET MODULE_OPTYPE ${i} OpType)
      list(GET MODULE_ACLNNTYPE ${i} AclnnType)
      if(${AclnnType} STREQUAL "aclnn"
         OR ${AclnnType} STREQUAL "aclnn_inner"
         OR ${AclnnType} STREQUAL "aclnn_exclude"
        )
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/${OpType}_def*.cpp)
        if(OPDEF_SRCS)
          target_sources(${OPHOST_NAME}_opdef_${AclnnType}_obj INTERFACE ${OPDEF_SRCS})
        endif()
      elseif(${AclnnType} STREQUAL "no_need_alcnn")
        message(STATUS "aicpu or host aicpu no need aclnn.")
      else()
        message(FATAL_ERROR "ACLNN TYPE UNSPPORTED, ONLY SUPPORT aclnn/aclnn_inner/aclnn_exclude")
      endif()
    endforeach()
  else()
    file(GLOB OPDEF_SRCS ${SOURCE_DIR}/*_def*.cpp)
    if(OPDEF_SRCS)
      message(
        FATAL_ERROR
          "Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
          "usage: add_modules_sources(OPTYPE optypes ACLNNTYPE aclnntypes)\n"
          "example: add_modules_sources(OPTYPE add ACLNNTYPE aclnn_exclude)"
        )
    endif()
  endif()
endfunction()

# useage: add_graph_plugin_sources()
macro(add_graph_plugin_sources)
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  # 获取算子层级目录名称，判断是否编译该算子
  get_filename_component(PARENT_DIR ${SOURCE_DIR} DIRECTORY)
  get_filename_component(OP_NAME ${PARENT_DIR} NAME)
  if(DEFINED ASCEND_OP_NAME
     AND NOT "${ASCEND_OP_NAME}" STREQUAL ""
     AND NOT "${ASCEND_OP_NAME}" STREQUAL "all"
     AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL"
    )
    if(NOT ${OP_NAME} IN_LIST ASCEND_OP_NAME)
      return()
    endif()
  endif()

  file(GLOB GRAPH_PLUGIN_SRCS ${SOURCE_DIR}/*_graph_plugin*.cpp)
  if(GRAPH_PLUGIN_SRCS)
    add_graph_plugin_modules()
    target_sources(${GRAPH_PLUGIN_NAME}_obj PRIVATE ${GRAPH_PLUGIN_SRCS})
  endif()

  file(GLOB GRAPH_PLUGIN_PROTO_HEADERS ${SOURCE_DIR}/*_proto*.h)
  if(GRAPH_PLUGIN_PROTO_HEADERS)
    target_sources(${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE ${GRAPH_PLUGIN_PROTO_HEADERS})
  endif()
endmacro()

# ######################################################################################################################
# get operating system info
# ######################################################################################################################
function(get_system_info SYSTEM_INFO)
  if(UNIX)
    execute_process(
      COMMAND
        grep -i ^id= /etc/os-release
      OUTPUT_VARIABLE TEMP
      )
    string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
    set(${SYSTEM_INFO}
        ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}
        PARENT_SCOPE
      )
  elseif(WIN32)
    message(STATUS "System is Windows. Only for pre-build.")
  else()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
  endif()
endfunction()

# ######################################################################################################################
# add compile options, e.g.: -g -O0
# ######################################################################################################################
function(add_ops_compile_options OP_TYPE)
  cmake_parse_arguments(OP_COMPILE "" "OP_TYPE" "COMPUTE_UNIT;OPTIONS" ${ARGN})
  execute_process(
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/ascendc_gen_options.py
      ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS} ${OP_TYPE} ${OP_COMPILE_COMPUTE_UNIT} ${OP_COMPILE_OPTIONS}
    RESULT_VARIABLE EXEC_RESULT
    OUTPUT_VARIABLE EXEC_INFO
    ERROR_VARIABLE EXEC_ERROR
    )
  if(${EXEC_RESULT})
    message("add ops compile options info: ${EXEC_INFO}")
    message("add ops compile options error: ${EXEC_ERROR}")
    message(FATAL_ERROR "add ops compile options failed!")
  endif()
endfunction()

###################################################################################################
# get op_type from *_binary.json
###################################################################################################
function(get_op_type_from_binary_json BINARY_JSON OP_TYPE)
  execute_process(COMMAND grep -w op_type ${BINARY_JSON} OUTPUT_VARIABLE op_type)
  string(REGEX REPLACE "\"op_type\"" "" op_type ${op_type})
  string(REGEX MATCH "\".+\"" op_type ${op_type})
  string(REGEX REPLACE "\"" "" op_type ${op_type})

  set(${OP_TYPE} ${op_type} PARENT_SCOPE)
endfunction()
