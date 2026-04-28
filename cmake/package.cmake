# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#### CPACK to package run #####

function(pack_custom)
  add_cann_third_party(makeself-fetch)

  npu_op_package(${PACK_CUSTOM_NAME}
    TYPE RUN
    CONFIG
      ENABLE_SOURCE_PACKAGE True
      ENABLE_BINARY_PACKAGE True
      INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/
      VENDOR_NAME "${VENDOR_PACKAGE_NAME}"
      ENABLE_DEFAULT_PACKAGE_NAME_RULE False
  )
  set(op_package_list)
  if(ENABLE_ASC_BUILD)
    add_custom_kernel_library(ascendc_kernels)
    list(APPEND op_package_list ${ascendc_kernels})
  endif()
  if(TARGET ${OPHOST_NAME}_opapi_obj OR TARGET opbuild_gen_aclnn_all)
    list(APPEND op_package_list cust_opapi)
  endif()
  if(TARGET ${OPHOST_NAME}_infer_obj)
    list(APPEND op_package_list cust_proto)
  endif()
  if(TARGET ${OPHOST_NAME}_tiling_obj)
    list(APPEND op_package_list cust_opmaster)
  endif()

  list(LENGTH op_package_list OP_PACKAGE_LENGTH)
  if(OP_PACKAGE_LENGTH GREATER 0)
    npu_op_package_add(${PACK_CUSTOM_NAME}
      LIBRARY
        ${op_package_list}
    )
  endif()
endfunction()

macro(INSTALL_SCRIPTS src_path)
  install(DIRECTORY ${src_path}/
      DESTINATION share/info/ops_nn/script
      FILE_PERMISSIONS
          OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
          GROUP_READ GROUP_EXECUTE
          WORLD_READ WORLD_EXECUTE
      DIRECTORY_PERMISSIONS
          OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 目录权限
          GROUP_READ GROUP_EXECUTE
          WORLD_READ WORLD_EXECUTE
      REGEX "(setenv|prereq_check)\\.(bash|fish|csh)" EXCLUDE
  )
endmacro()

function(pack_built_in)
  # 打印路径
  message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "CANN_CMAKE_DIR = ${CANN_CMAKE_DIR}")
  message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")

  set(script_prefix ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/ops_nn/scripts)
  INSTALL_SCRIPTS(${script_prefix})

  set(SCRIPTS_FILES
      ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
      ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.sh
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.csh
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.fish
      ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
  )

  install(FILES ${SCRIPTS_FILES}
      DESTINATION share/info/ops_nn/script
  )
  set(COMMON_FILES
      ${CANN_CMAKE_DIR}/scripts/install/install_common_parser.sh
      ${CANN_CMAKE_DIR}/scripts/install/common_func_v2.inc
      ${CANN_CMAKE_DIR}/scripts/install/common_installer.inc
      ${CANN_CMAKE_DIR}/scripts/install/script_operator.inc
      ${CANN_CMAKE_DIR}/scripts/install/version_cfg.inc
  )

  set(PACKAGE_FILES
      ${COMMON_FILES}
      ${CANN_CMAKE_DIR}/scripts/install/multi_version.inc
  )
  set(LATEST_MANGER_FILES
      ${COMMON_FILES}
      ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
      ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
      ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
  )
  set(CONF_FILES
      ${CANN_CMAKE_DIR}/scripts/package/cfg/path.cfg
  )
  install(FILES ${CMAKE_BINARY_DIR}/version.ops-nn.info
      DESTINATION share/info/ops_nn
      RENAME version.info
  )
  install(FILES ${CONF_FILES}
      DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/conf
  )
  install(FILES ${PACKAGE_FILES}
      DESTINATION share/info/ops_nn/script
  )

  string(FIND "${ASCEND_COMPUTE_UNIT}" ";" SEMICOLON_INDEX)
  if (SEMICOLON_INDEX GREATER -1)
      # 截取分号前的字串
      math(EXPR SUBSTRING_LENGTH "${SEMICOLON_INDEX}")
      string(SUBSTRING "${ASCEND_COMPUTE_UNIT}" 0 "${SUBSTRING_LENGTH}" compute_unit)
  else()
      # 没有分号取全部内容
      set(compute_unit "${ASCEND_COMPUTE_UNIT}")
  endif()

  message(STATUS "current compute_unit is: ${compute_unit}")
  set(script_with_soc_prefix ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/ops_nn/scripts)
  INSTALL_SCRIPTS(${script_with_soc_prefix})

  include(${CMAKE_SOURCE_DIR}/cmake/runtimeKB.cmake)
  set_cann_cpack_config(ops-nn ENABLE_DEVICE ${ENABLE_DEVICE} COMPUTE_UNIT ${ASCEND_COMPUTE_UNIT} SHARE_INFO_NAME ops_nn)
endfunction()