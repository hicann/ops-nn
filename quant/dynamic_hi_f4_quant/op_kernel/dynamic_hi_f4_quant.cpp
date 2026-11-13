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
 * \file dynamic_hi_f4_quant.cpp
 * \brief
 */

#include "arch35/dynamic_hi_f4_quant_struct.h"
#include "arch35/dynamic_hi_f4_quant_tilingdata.h"
#include "arch35/dynamic_hi_f4_quant_tail_axis.h"
#include "arch35/dynamic_hi_f4_quant_not_tail_axis.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60 // 控制饱和模式

using namespace DynamicHiF4Quant;
using namespace DynamicHiF4QuantOp;

template <uint64_t quantAxis>
__global__ __aicore__ void dynamic_hi_f4_quant(GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(DynamicHiF4QuantTilingData);

#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    if constexpr (quantAxis == TPL_QUANT_TAIL_AXIS) {
        GET_TILING_DATA_WITH_STRUCT(DynamicHiF4QuantTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicHiF4QuantTailAxis<DTYPE_X, DTYPE_Y, 2> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(DynamicHiF4QuantTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicHiF4QuantNotTailAxis<DTYPE_X, DTYPE_Y, 2> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    }

#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
