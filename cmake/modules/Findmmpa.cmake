# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(mmpa_FOUND)
  message(STATUS "mmpa has been found")
  return()
endif()

include(FindPackageHandleStandardArgs)

set(MMPA_LIB_SEARCH_PATHS
  ${ASCEND_DIR}/${SYSTEM_PREFIX}
)

find_library(MMPA_LIB_DIR
  NAME mmpa
  PATHS ${MMPA_LIB_SEARCH_PATHS}
  PATH_SUFFIXES lib64
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(MMPA_LIB_DIR)
  get_filename_component(MMPA_LIB_DIR ${MMPA_LIB_DIR} REALPATH)
  add_library(mmpa SHARED IMPORTED)
  set_target_properties(mmpa PROPERTIES
    IMPORTED_LOCATION ${MMPA_LIB_DIR}
  )
  message(STATUS "Found mmpa library:${MMPA_LIB_DIR}")
else()
  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "Cannot find library mmpa")
  endif()
endif()