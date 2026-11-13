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
 * \file anti_hi_f4_quant.cpp
 * \brief AntiHiF4Quant kernel entry - HiF4 de-quantization
 */

#include "arch35/anti_hi_f4_quant_not_tail.h"
#include "arch35/anti_hi_f4_quant_tail.h"
#include "arch35/anti_hi_f4_quant_tilingdata.h"
#include "arch35/anti_hi_f4_quant_struct.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace AntiHiF4Quant;
using namespace AntiHiF4QuantOp;

template <uint64_t quantMode>
__global__ __aicore__ void anti_hi_f4_quant(GM_ADDR x, GM_ADDR scale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
#if (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif
    REGISTER_TILING_DEFAULT(AntiHiF4QuantNotTailTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 0", AntiHiF4QuantTailTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    if constexpr (quantMode == TPL_QUANT_MODE_PER_CHANNEL) {
        GET_TILING_DATA_WITH_STRUCT(AntiHiF4QuantTailTilingData, tilingData, tiling);
        TPipe pipe;
        AntiHiF4QuantTail<DTYPE_X, DTYPE_Y> op(&tilingData, &pipe);
        op.Init(x, scale, y);
        op.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(AntiHiF4QuantNotTailTilingData, tilingData, tiling);
        TPipe pipe;
        AntiHiF4QuantNotTail<DTYPE_X, DTYPE_Y> op(&tilingData, &pipe);
        op.Init(x, scale, y);
        op.Process();
    }

#if (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
