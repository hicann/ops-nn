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

if(UT_TEST_ALL OR OP_API_UT)
  file(
    GLOB SUBDIRECTORIES
    LIST_DIRECTORIES true
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/*
    )
  foreach(SUBDIR ${SUBDIRECTORIES})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/CMakeLists.txt)
      add_subdirectory(${SUBDIR})
    endif()
  endforeach()
  gen_aclnn_with_opdef()

  ## add opapi so
  add_library(${OPHOST_NAME}_opapi SHARED
          $<$<TARGET_EXISTS:${OPHOST_NAME}_opapi_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>>
          $<$<TARGET_EXISTS:opbuild_gen_aclnn_all>:$<TARGET_OBJECTS:opbuild_gen_aclnn_all>>
          )
  target_sources(${OPHOST_NAME}_opapi PRIVATE empty_runtime.cpp)
  target_link_libraries(${OPHOST_NAME}_opapi PRIVATE
          $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
          opapi_stub
          )

  ## make ophost ut exe
  set(OP_API_UT_COMMON_INC ${CMAKE_CURRENT_SOURCE_DIR}/op_api_ut_common/inc CACHE STRING "opapi ut common include path" FORCE)
  set(OP_API_UT_EXE ${PKG_NAME}_op_api_ut)
  add_executable(${OP_API_UT_EXE}
          ut_main.cpp
          $<$<TARGET_EXISTS:${OP_API_MODULE_NAME}_cases_obj>:$<TARGET_OBJECTS:${OP_API_MODULE_NAME}_cases_obj>>
          )
  target_compile_definitions(${OP_API_UT_EXE} PRIVATE
          LOG_CPP
          )
  target_compile_options(${OP_API_UT_EXE} PUBLIC -fPIE -fno-access-control)
  target_include_directories(${OP_API_UT_EXE} PRIVATE
          ${OP_API_UT_COMMON_INC}
          ${HI_PYTHON_INC}
          )
  target_link_libraries(${OP_API_UT_EXE} PRIVATE
          $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
          -Wl,--whole-archive
          -Wl,--as-needed
              op_api_ut_common
              ${OPHOST_NAME}_opapi
              nnopbase
          -Wl,--no-as-needed
          graph
          gtest
          -Wl,--no-whole-archive
          )

  ## exec ophost ut exe
  if(ENABLE_UT_EXEC)
    if(ENABLE_ASAN)
      add_custom_command(
          TARGET ${OP_API_UT_EXE}
          POST_BUILD
          COMMAND
              ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
              ASAN_OPTIONS=detect_leaks=0
              ./${OP_API_UT_EXE}
              COMMENT "Run ops op_api utest with asan"
      )
    else()
      add_custom_command(
              TARGET ${OP_API_UT_EXE}
              POST_BUILD
              COMMAND
              ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} ./${OP_API_UT_EXE}
              COMMENT "Run ops op_api utest"
      )
    endif()
  endif()
endif()
