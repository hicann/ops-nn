# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
set(PKG_NAME AscendOpsNn)
project(${PKG_NAME} VERSION 1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(UNIX)
  set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)
endif()

set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/modules
  ${CMAKE_MODULE_PATH}
)

include(../../cmake/third_party/nlohmann_json.cmake)
include(../../cmake/third_party/opbase.cmake)
include(../../cmake/dependencies.cmake)
include(../../cmake/torch_extension/python.cmake)
include(../../cmake/torch_extension/torch.cmake)
include(../../cmake/torch_extension/torch_npu.cmake)
include(../../cmake/func.cmake)

message(STATUS "Using Ascend toolkit path: ${ASCEND_DIR}")
set(CMAKE_PREFIX_PATH ${ASCEND_DIR}/)
set(BISHENG "${ASCEND_DIR}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng" CACHE FILEPATH "Path to Bisheng compiler")
message(STATUS "Bisheng compiler path: ${BISHENG}")

# set the default compiler and linker to bisheng
set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})

# set ASCEND_INCLUDE_DIRS
set(ASCEND_INCLUDE_DIRS
    ${ASCEND_DIR}/include
    ${ASCEND_DIR}/compiler/tikcpp/include
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/impl
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/interface
    ${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/impl
    ${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/tiling
    ${ASCEND_DIR}/compiler/ascendc/impl/aicore/basic_api
    ${ASCEND_DIR}/pkg_inc/op_common
    ${ASCEND_DIR}/pkg_inc/base
    ${ASCEND_DIR}/pkg_inc
)

# set NPU architecture, default to ascend910b
if(DEFINED ENV{NPU_ARCH})
    set(NPU_ARCH "$ENV{NPU_ARCH}" CACHE STRING "NPU architecture")
else()
    set(NPU_ARCH "ascend910b" CACHE STRING "NPU architecture")
endif()
# set EXTENSION_MODULE_NAME to `ascend_ops_nn`
set(EXTENSION_MODULE_NAME "ascend_ops_nn" CACHE STRING "Extension module name")

# include directories
set(INCLUDE_DIRECTORIES
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
    ${ASCEND_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# link directories
set(LINK_DIRECTORIES
    ${TORCH_NPU_LIB_PATH}
    ${ASCEND_DIR}/lib64
)

# link libraries
set(LINK_LIBRARIES
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    runtime
)

# compile options
set(COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
    -O3
    -fdiagnostics-color=always
    -w
    -DEXTENSION_MODULE_NAME=${EXTENSION_MODULE_NAME}
)

# define a LIST variable to hold object files
set(PE_OBJECTS_LIST "" CACHE INTERNAL "List of PyTorch extension objects")
message(STATUS "PE_OBJECTS_LIST before adding sources: ${PE_OBJECTS_LIST}")

# Automatically discover all op directories
file(GLOB SRC_SUBDIRS "${CMAKE_CURRENT_SOURCE_DIR}/experimental_tmp/*/*/")

message(STATUS "SRC_SUBDIRS: ${SRC_SUBDIRS}")
foreach(src_dir ${SRC_SUBDIRS})
    if(IS_DIRECTORY ${src_dir})
        message(STATUS "Adding source directory: ${src_dir}")
        add_subdirectory(${src_dir})
    endif()
endforeach()

message(STATUS "PE_OBJECTS_LIST after adding sources: ${PE_OBJECTS_LIST}")

# create shared library
set("EXTENSION_CPP" extension.cpp)
add_library(_C SHARED
    ${EXTENSION_CPP}
    ${PE_OBJECTS_LIST}
)
set_target_properties(_C PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    SUFFIX ".abi3.so"
    OUTPUT_NAME "_C"
)
target_compile_definitions(_C PRIVATE Py_LIMITED_API=0x03080000)
target_compile_options(_C PRIVATE ${COMPILE_OPTIONS})
target_include_directories(_C PRIVATE ${INCLUDE_DIRECTORIES})
target_link_directories(_C PRIVATE ${LINK_DIRECTORIES})
target_link_libraries(_C PRIVATE ${LINK_LIBRARIES})

add_custom_command(TARGET _C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:_C>
    ${CMAKE_CURRENT_SOURCE_DIR}/${EXTENSION_MODULE_NAME}/$<TARGET_FILE_NAME:_C>
    COMMENT "Copying compiled extension $<TARGET_FILE_NAME:_C> to ${CMAKE_CURRENT_SOURCE_DIR}/${EXTENSION_MODULE_NAME}/"
)
