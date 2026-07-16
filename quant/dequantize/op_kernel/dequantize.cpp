/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "arch35/dequantize_min_combined_kernel.h"
#include "arch35/dequantize_min_first_kernel.h"
#include "arch35/dequantize_scaled_kernel.h"
#include "arch35/dequantize_tiling_struct.h"
#include "arch35/dequantize_struct.h"

using TilingData4 = DequantizeTilingData<4>;
using TilingData8 = DequantizeTilingData<8>;

template <int MODE, int RANK>
__global__ __aicore__ void dequantize(GM_ADDR x, GM_ADDR minRange, GM_ADDR maxRange, GM_ADDR y, GM_ADDR workspace,
                                      GM_ADDR tiling)
{
    GM_ADDR ins[3] = {x, minRange, maxRange};
    GM_ADDR outs[1] = {y};

    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (MODE == DEQUANTIZE_MODE_MIN_COMBINED) {
        if constexpr (RANK == 4) {
            GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
            DequantizeMinCombinedKernel<DTYPE_X, 4> kernel;
            kernel.Init(ins, outs, &td);
            kernel.Process();
        } else {
            GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
            DequantizeMinCombinedKernel<DTYPE_X, 8> kernel;
            kernel.Init(ins, outs, &td);
            kernel.Process();
        }
    } else if constexpr (MODE == DEQUANTIZE_MODE_MIN_FIRST) {
        if constexpr (RANK == 4) {
            GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
            DequantizeMinFirstKernel<DTYPE_X, 4> kernel;
            kernel.Init(ins, outs, &td);
            kernel.Process();
        } else {
            GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
            DequantizeMinFirstKernel<DTYPE_X, 8> kernel;
            kernel.Init(ins, outs, &td);
            kernel.Process();
        }
    } else {
        if constexpr (RANK == 4) {
            GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
            DequantizeScaledKernel<DTYPE_X, 4> kernel;
            kernel.Init(ins, outs, &td);
            kernel.Process();
        } else {
            GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
            DequantizeScaledKernel<DTYPE_X, 8> kernel;
            kernel.Init(ins, outs, &td);
            kernel.Process();
        }
    }
}
