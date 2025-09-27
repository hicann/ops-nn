# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (dlog_FOUND)
  message(STATUS "Package dlog has been found.")
  return()
endif()

set(_cmake_targets_defined "")
set(_cmake_targets_not_defined "")
set(_cmake_expected_targets "")
foreach(_cmake_expected_target IN ITEMS dlog_a dlog_headers)
  list(APPEND _cmake_expected_targets "${_cmake_expected_target}")
  if(TARGET "${_cmake_expected_target}")
    list(APPEND _cmake_targets_defined "${_cmake_expected_target}")
  else()
    list(APPEND _cmake_targets_not_defined "${_cmake_expected_target}")
  endif()
endforeach()
unset(_cmake_expected_target)

if(_cmake_targets_defined STREQUAL _cmake_expected_targets)
  unset(_cmake_targets_defined)
  unset(_cmake_targets_not_defined)
  unset(_cmake_expected_targets)
  unset(CMAKE_IMPORT_FILE_VERSION)
  cmake_policy(POP)
  return()
endif()

if(NOT _cmake_targets_defined STREQUAL "")
  string(REPLACE ";" ", " _cmake_targets_defined_text "${_cmake_targets_defined}")
  string(REPLACE ";" ", " _cmake_targets_not_defined_text "${_cmake_targets_not_defined}")
  message(FATAL_ERROR "Some (but not all) targets in this export set were already defined.\nTargets Defined: ${_cmake_targets_defined_text}\nTargets not yet defined: ${_cmake_targets_not_defined_text}\n")
endif()
unset(_cmake_targets_defined)
unset(_cmake_targets_not_defined)
unset(_cmake_expected_targets)

set(DLOG_HEAD_SEARCH_PATHS
  ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
  ${TOP_DIR}/abl/slog/inc            # compile with ci
)

find_path(_INCLUDE_DIR
  NAMES toolchain/dlog_pub.h
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
  PATHS ${DLOG_HEAD_SEARCH_PATHS})

find_library(dlog_SHARED_LIBRARY
  NAMES libascendalog.so
  PATH_SUFFIXES lib64
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(dlog
  FOUND_VAR
    dlog_FOUND
  REQUIRED_VARS
    _INCLUDE_DIR
)

if(dlog_FOUND)
  set(dlog_INCLUDE_DIR "${_INCLUDE_DIR}")
  include(CMakePrintHelpers)
  message(STATUS "Variables in dlog module:")
  cmake_print_variables(dlog_INCLUDE_DIR)
  cmake_print_variables(dlog_SHARED_LIBRARY)

  if(dlog_SHARED_LIBRARY)
    add_library(dlog SHARED IMPORTED)
    set_target_properties(dlog PROPERTIES
      INTERFACE_COMPILE_DEFINITIONS "LOG_CPP;PROCESS_LOG"
      INTERFACE_LINK_LIBRARIES "dlog_headers"
      IMPORTED_LOCATION "${dlog_SHARED_LIBRARY}"
    )
  endif()

  add_library(dlog_headers INTERFACE IMPORTED)
  set_target_properties(dlog_headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${dlog_INCLUDE_DIR};${dlog_INCLUDE_DIR}/toolchain"
  )

  include(CMakePrintHelpers)
  cmake_print_properties(TARGETS dlog
    PROPERTIES INTERFACE_COMPILE_DEFINITIONS INTERFACE_LINK_LIBRARIES IMPORTED_LOCATION
  )
  cmake_print_properties(TARGETS dlog_headers
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
  )
endif()

# Cleanup temporary variables.
set(_INCLUDE_DIR)
