# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(ASCEND_OP_NAME "" CACHE STRING "Ascend op names to compile")
set(OP_TYPE "assert")

if(ASCEND_OP_NAME)
  set(_ASCEND_OP_NAME_TMP "${ASCEND_OP_NAME}")
  separate_arguments(_ASCEND_OP_NAME_TMP)

  list(FIND _ASCEND_OP_NAME_TMP "${OP_TYPE}" _index)
  if(_index EQUAL -1)
    message(STATUS "[${OP_TYPE}] skipped, not in ASCEND_OP_NAME list: ${ASCEND_OP_NAME}")
    return()
  else()
    message(STATUS "[${OP_TYPE}] selected, in ASCEND_OP_NAME list: ${ASCEND_OP_NAME}")
  endif()
endif()

if (BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  # aicpu json
  file(GLOB_RECURSE JSON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/*.json)
  set_property(GLOBAL APPEND PROPERTY AICPU_JSON_FILES ${JSON_FILE})

  # aicpu cust kernel
  file(GLOB AICPU_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*_aicpu*.cpp)
  message(STATUS "[assert] Found aicpu sources: ${AICPU_SRC}, ascend dir: ${ASCEND_DIR}, ophsot name: ${OPHOST_NAME}")

  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
  set(CMAKE_CXX_COMPILER ${ASCEND_DIR}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)

  set(OBJ_NAME assert_cust_obj)
  add_aicpu_cust_kernel_modules(${OBJ_NAME})
  target_sources(${OBJ_NAME} PRIVATE ${AICPU_SRC})
else()
  add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE assert ACLNNTYPE no_need_aclnn)
endif()
