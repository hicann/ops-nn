/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "kernel_operator.h"
#include "arch35/acts_ulq_kernel.h"
#include "arch35/acts_ulq_tiling_struct.h"

using TilingData4 = ActsUlqTilingData<4>;
using TilingData8 = ActsUlqTilingData<8>;

template<int RANK>
__global__ __aicore__ void acts_ulq(
    GM_ADDR x, GM_ADDR clamp_min, GM_ADDR clamp_max,
    GM_ADDR y, GM_ADDR clamp_min_mask, GM_ADDR clamp_max_mask, GM_ADDR x_clamped_loss,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR ins[3]   = {x, clamp_min, clamp_max};
    GM_ADDR outs[4]  = {y, clamp_min_mask, clamp_max_mask, x_clamped_loss};

    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (RANK == 4) {
        GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
        ActsUlqKernel<DTYPE_X, 4> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
        ActsUlqKernel<DTYPE_X, 8> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    }
}
