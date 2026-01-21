# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE adaptive_max_pool3d_grad ACLNNTYPE aclnn_exclude DEPENDENCIES max_pool3d_grad_with_argmax max_pool_grad_with_argmax_v3 pool_3d_common)
