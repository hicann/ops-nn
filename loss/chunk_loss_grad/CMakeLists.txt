# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# ACLNN is shared; Host tiling and kernel sources are isolated by architecture.
add_modules_sources(DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OPTYPE chunk_loss_grad ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT ascend910b ascend910_93 ascend950
    TILING_DIR arch22 arch22 arch35)
add_kernel_sources(KERNEL_SRC chunk_loss_grad
    COMPUTE_UNITS ascend910b ascend910_93)
add_kernel_sources(KERNEL_SRC arch35/chunk_loss_grad.cpp
    COMPUTE_UNITS ascend950)
# Keep the Blaze and Tensor API headers from the same ops-tensor dependency.
# This operator-local include order leaves the repository's shared CMake intact.
if("ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
    add_ops_compile_options(ChunkLossGrad
        OPTIONS "-gline-tables-only"
        "-iquote${OPTENSOR_SOURCE_PATH}/include/tensor_api/include"
        "-iquote${OPTENSOR_SOURCE_PATH}/include/tensor_api")
endif()
