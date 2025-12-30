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
 * \file quant_update_scatter_apt.cpp
 * \brief quant_update_scatter_apt kernel entry file
 */
#include "kernel_operator.h"
#include "arch35/quant_update_scatter_base.h"
#include "arch35/quant_update_scatter_regbase.h"
#include "arch35/quant_update_scatter_large_batch_regbase.h"
#include "arch35/quant_update_scatter_large_ele_little_quant_regbase.h"
#include "arch35/quant_update_scatter_large_ele_large_quant_regbase.h"
#include "arch35/quant_update_scatter_large_batch_little_quant_regbase.h"
#include "arch35/quant_update_scatter_large_batch_large_quant_regbase.h"
#include "arch35/quant_update_scatter_struct.h"

using namespace AscendC;
using namespace QuantUpdateScatter;

template <uint64_t SplitMode, uint64_t ZeroPointsType, uint64_t DivMode, uint64_t CastRoundMode>
__global__ __aicore__ void quant_update_scatter(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR quant_scales, GM_ADDR quant_zero_points, GM_ADDR out,
    GM_ADDR workSpace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using ZeroType = typename TypeFromEnum<ZeroPointsType>::type;

    if constexpr (SplitMode == TPL_MODE_LITTLE_ELE_LITTLE_QUANT) {
        QuantUpdateScatterRegbase<
            DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES, DTYPE_QUANT_SCALES, ZeroType, DivMode, CastRoundMode>
            op;
        op.Init(var, indices, updates, quant_scales, quant_zero_points, out, &tilingData);
        op.Process();
    } else if constexpr (SplitMode == TPL_MODE_LARGE_BATCH) {
        QuantUpdateScatterLargeBatchRegbase<
            DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES, DTYPE_QUANT_SCALES, ZeroType, DivMode, CastRoundMode>
            op;
        op.Init(var, indices, updates, quant_scales, quant_zero_points, out, &tilingData);
        op.Process();
    } else if constexpr (SplitMode == TPL_MODE_LARGE_ELE_LITTLE_QUANT) {
        QuantUpdateScatterLargeEleLittleQuantRegbase<
            DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES, DTYPE_QUANT_SCALES, ZeroType, DivMode, CastRoundMode>
            op;
        op.Init(var, indices, updates, quant_scales, quant_zero_points, out, &tilingData);
        op.Process();
    } else if constexpr (SplitMode == TPL_MODE_LARGE_ELE_LARGE_QUANT) {
        QuantUpdateScatterLargeEleLargeQuantRegbase<
            DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES, DTYPE_QUANT_SCALES, ZeroType, DivMode, CastRoundMode>
            op;
        op.Init(var, indices, updates, quant_scales, quant_zero_points, out, &tilingData);
        op.Process();
    } else if constexpr (SplitMode == TPL_MODE_LARGE_BATCH_LITTLE_QUANT) {
        QuantUpdateScatterLargeBatchLittleQuantRegbase<
            DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES, DTYPE_QUANT_SCALES, ZeroType, DivMode, CastRoundMode>
            op;
        op.Init(var, indices, updates, quant_scales, quant_zero_points, out, &tilingData);
        op.Process();
    } else if constexpr (SplitMode == TPL_MODE_LARGE_BATCH_LARGE_QUANT) {
        QuantUpdateScatterLargeBatchLargeQuantRegbase<
            DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES, DTYPE_QUANT_SCALES, ZeroType, DivMode, CastRoundMode>
            op;
        op.Init(var, indices, updates, quant_scales, quant_zero_points, out, &tilingData);
        op.Process();
    }
}