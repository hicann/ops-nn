# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

get_filename_component(MASKED_SCATTER_OP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)

add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${MASKED_SCATTER_OP_ROOT} OPTYPE masked_scatter
                    ACLNNTYPE aclnn_exclude)

target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
    ${OPS_NN_DIR}/index/masked_scatter_with_position/op_host/op_api/masked_scatter_with_position.cpp)
target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE
    ${OPS_NN_DIR}/index/masked_scatter_with_position/op_host/op_api)
