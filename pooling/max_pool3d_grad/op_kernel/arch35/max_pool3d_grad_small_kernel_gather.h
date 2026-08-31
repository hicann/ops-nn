/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool3d_grad_small_kernel_gather.h
 * \brief
 */

#ifndef MAX_POOL3D_GRAD_SMALL_KERNEL_GATHER_H
#define MAX_POOL3D_GRAD_SMALL_KERNEL_GATHER_H

#include "pool_utils/arch35/index/max_pool_with_argmax_index.h"
#include "pool_utils/arch35/compute/max_pool_negative_value.h"

namespace MaxPool3DSmallKernelNameSpace {

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"

using namespace AscendC;

} // namespace MaxPool3DSmallKernelNameSpace
#endif // MAX_POOL3D_GRAD_SMALL_KERNEL_GATHER_H
