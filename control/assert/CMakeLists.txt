# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
list(REMOVE_ITEM CURRENT_DIRS tests)
foreach(SUB_DIR ${CURRENT_DIRS})
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
    add_subdirectory(${SUB_DIR})
  endif()
endforeach()
