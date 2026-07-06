# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

message(STATUS "=== Debug: start ops.conv.deformable_conv2d.graph_plugin.CMakeLists.txt ")

get_filename_component(OP_PARENT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(OP_NAME ${OP_PARENT} NAME)
if(NOT op_category)
    get_filename_component(CATEGORY_DIR ${OP_PARENT} DIRECTORY)
    get_filename_component(op_category ${CATEGORY_DIR} NAME)
endif()
file(APPEND ${ASCEND_SUB_CONFIG_PATH} "OP_CATEGORY;${op_category};OP_NAME;${OP_NAME};DEPENDENCIES;deformable_offsets\n")

add_graph_plugin_sources()
