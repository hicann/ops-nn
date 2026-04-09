# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(ASCEND_OP_NAME "" CACHE STRING "Ascend op names to compile")
set(OP_TYPE "gather_v2")

skip_aicpu_kernel("${OP_TYPE}" "${ASCEND_OP_NAME}")
if(SKIP_AICPU_FLAG)
  return()
endif()

if (BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  if (NOT (UT_TEST_ALL OR OP_KERNEL_AICPU_UT))
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
    set(CMAKE_CXX_COMPILER ${ASCEND_DIR}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
  endif()

  file(GLOB_RECURSE JSON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/*.json)
  set_property(GLOBAL APPEND PROPERTY AICPU_JSON_FILES ${JSON_FILE})

  file(GLOB AICPU_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*_aicpu*.cpp)
  set(OBJ_NAME gather_v2_cust_obj)
  add_aicpu_cust_kernel_modules(${OBJ_NAME})
  target_sources(${OBJ_NAME} PRIVATE ${AICPU_SRC})
else()
  add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE gather_v2 ACLNNTYPE aclnn_exclude)
endif()

if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
  AddAicpuOpTestCase(gather_v2)
endif()
