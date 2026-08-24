# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/

message(STATUS "=== Debug: start ops.norm.bn_infer.op_host.CMakeLists.txt ")

set(SUPPORT_COMPUTE_UNIT "ascend950")
set(SUPPORT_TILING_DIR "arch35")

add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OPTYPE bn_infer ACLNNTYPE aclnn_exclude
                    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR}
                    DISABLE_IN_OPP TRUE DEPENDENCIES norm_common)

if(ENABLE_PACKAGE AND ENABLE_CUSTOM)
    set(_bn_infer_compat_dir "${CMAKE_CURRENT_BINARY_DIR}/bn_infer_compat")
    file(MAKE_DIRECTORY "${_bn_infer_compat_dir}/impl/dynamic")
    file(WRITE "${_bn_infer_compat_dir}/impl/__init__.py" "'''\nBNInfer compatibility package.\n'''\n")
    file(WRITE "${_bn_infer_compat_dir}/impl/dynamic/__init__.py" "'''\nBNInfer dynamic compatibility package.\n'''\n")

    install(FILES
        "${_bn_infer_compat_dir}/impl/__init__.py"
        DESTINATION "packages/vendors/bn_infer_nn/op_impl/ai_core/tbe/bn_infer_nn_impl/impl"
    )
    install(FILES
        "${_bn_infer_compat_dir}/impl/dynamic/__init__.py"
        "${CMAKE_BINARY_DIR}/tbe/dynamic/bn_infer.py"
        DESTINATION "packages/vendors/bn_infer_nn/op_impl/ai_core/tbe/bn_infer_nn_impl/impl/dynamic"
    )
endif()
