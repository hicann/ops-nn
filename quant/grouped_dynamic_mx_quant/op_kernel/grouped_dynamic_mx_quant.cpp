/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_dynamic_mx_quant.cpp
 * \brief
 */

#include "arch35/grouped_dynamic_mx_quant_combine.h"
#include "arch35/grouped_dynamic_mx_quant_tilingdata.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace GroupedDynamicMxQuant;
using namespace GroupedDynamicMxQuantOp;

template <uint64_t roundMode>
__aicore__ inline constexpr AscendC::RoundMode getRoundMode()
{
    if (roundMode == TPL_ROUND_MODE_RINT) {
        return AscendC::RoundMode::CAST_RINT;
    } else if (roundMode == TPL_ROUND_MODE_ROUND) {
        return AscendC::RoundMode::CAST_ROUND;
    } else if (roundMode == TPL_ROUND_MODE_FLOOR) {
        return AscendC::RoundMode::CAST_FLOOR;
    } else {
        return AscendC::RoundMode::CAST_RINT;
    }
}

template <uint64_t scaleAlg, uint64_t dstTypeMax, uint64_t dstType, uint64_t roundMode>
__global__ __aicore__ void grouped_dynamic_mx_quant(GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR mxScale,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GroupedDynamicMxQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(GroupedDynamicMxQuantTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    TPipe pipe;
    GroupedDynamicMxQuant::GroupedDynamicMxQuantCombine<DTYPE_X, DTYPE_Y, scaleAlg, dstTypeMax,
                                                        getRoundMode<roundMode>()>
        op(&tilingData, &pipe);
    op.Init(x, groupIndex, y, mxScale);
    op.Process();

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
