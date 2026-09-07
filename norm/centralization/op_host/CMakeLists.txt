# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(SUPPORT_COMPUTE_UNIT "ascend950")
set(SUPPORT_TILING_DIR "arch35")
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OPTYPE centralization ACLNNTYPE aclnn_exclude
                    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR}
                    DISABLE_IN_OPP TRUE DEPENDENCIES norm_common)

if(ENABLE_CUSTOM AND BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG AND NOT ENABLE_TEST)
  file(GLOB CENTRALIZATION_TF_PLUGIN_SRCS
       ${CMAKE_CURRENT_SOURCE_DIR}/../framework/*_tf_plugin.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../framework/tf_plugin/*.cpp)
  if(CENTRALIZATION_TF_PLUGIN_SRCS AND TARGET ${TF_PLUGIN_NAME}_obj AND NOT TARGET ${TF_PLUGIN_NAME})
    add_library(${TF_PLUGIN_NAME} SHARED $<TARGET_OBJECTS:${TF_PLUGIN_NAME}_obj>)
    target_link_libraries(
      ${TF_PLUGIN_NAME}
      PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
              c_sec
              -Wl,--no-as-needed
              register
              -Wl,--as-needed
              -Wl,--whole-archive
              rt2_registry_static
              -Wl,--no-whole-archive
              $<$<CONFIG:Release>:-s>
    )
    install(
      TARGETS ${TF_PLUGIN_NAME}
      LIBRARY DESTINATION packages/vendors/${VENDOR_PACKAGE_NAME}/framework/tensorflow
    )
  endif()
endif()
