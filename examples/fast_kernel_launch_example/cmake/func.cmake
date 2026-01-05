# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# define functions

# usage: recursive_add_subdirectory()
macro(recursive_add_subdirectory)
    file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
    foreach(SUB_DIR ${CURRENT_DIRS})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/${NPU_ARCH}/CMakeLists.txt")
            add_subdirectory(${SUB_DIR}/${NPU_ARCH})
        endif()
    endforeach()
endmacro()

# usage: add_sources("--npu-arch=dav-3101")
# usage: add_sources("--npu-arch=dav-3101" "file1.cpp;file2.cpp;file3.cpp")
macro(add_sources ARGS)
    # 解析参数
    set(COMPILE_ARGS "${ARGS}")  # 第一个参数为编译参数
    set(CUSTOM_SOURCES "")       # 第二个参数为自定义源文件列表（可选）
    
    # 检查是否有第二个参数
    if(${ARGC} GREATER 1)
        set(CUSTOM_SOURCES "${ARGV1}")
    endif()

    # clear CMAKE_CXX_FLAGS to avoid affecting bisheng compile
    unset(CMAKE_CXX_FLAGS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_C_COMPILER ${BISHENG})
    set(CMAKE_CXX_COMPILER ${BISHENG})
    set(CMAKE_LINKER ${BISHENG})

    message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")

    # get parent dir name as OP_NAME
    get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${PARENT_DIR} NAME)
    message(STATUS "OP_NAME: ${OP_NAME}")

    # get compile flags for current op
    set(COMPILE_FLAGS "${COMPILE_ARGS} -xasc ")
    message(STATUS "COMPILE FLAGS: ${COMPILE_FLAGS}")

    file(GLOB_RECURSE SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

    # 根据是否传入自定义源文件列表决定如何获取源文件
    if(CUSTOM_SOURCES)
        foreach(SRC ${CUSTOM_SOURCES})
            # 确保文件存在
            if(EXISTS ${SRC})
                # 获取相对于当前源目录的相对路径
                file(RELATIVE_PATH REL_SRC ${CMAKE_CURRENT_SOURCE_DIR} ${SRC})
                list(APPEND SOURCE_FILES ${REL_SRC})
            else()
                message(WARNING "Source file not found: ${SRC}")
            endif()
        endforeach()
    endif()
    
    message(STATUS "SOURCE FILES: ${SOURCE_FILES}")
    if(SOURCE_FILES STREQUAL "")
        message(FATAL_ERROR "No source files found")
    endif()

    # set_source_files_properties
    set_source_files_properties(
        ${SOURCE_FILES} PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "${COMPILE_FLAGS}"
    )

    # set target name
    set(TARGET_NAME ${OP_NAME}_obj)
    add_library(${TARGET_NAME} OBJECT ${SOURCE_FILES})
    target_compile_options(${TARGET_NAME} PRIVATE ${COMPILE_OPTIONS})
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIRECTORIES})

    # add target obj to OBJECTS_LIST
    set(NEW_OBJECT_EXPRESSION $<TARGET_OBJECTS:${TARGET_NAME}>)
    set(TEMP_LIST ${OBJECTS_LIST})
    list(APPEND TEMP_LIST ${NEW_OBJECT_EXPRESSION})
    set(OBJECTS_LIST ${TEMP_LIST} CACHE INTERNAL "List of objects" FORCE)
endmacro()
