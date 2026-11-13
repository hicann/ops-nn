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
 * \file grouped_dynamic_hi_f4_quant.cpp
 * \brief
 */

#include "arch35/grouped_dynamic_hi_f4_quant_base.h"
#include "arch35/grouped_dynamic_hi_f4_quant_tilingdata.h"
#include "arch35/grouped_dynamic_hi_f4_quant_struct.h"
#include "arch35/grouped_dynamic_hi_f4_quant_mode0.h"
#include "arch35/grouped_dynamic_hi_f4_quant_mode1.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace GroupedDynamicHiF4Quant;
using namespace GroupedDynamicHiF4QuantOp;

template <uint64_t mode>
__global__ __aicore__ void grouped_dynamic_hi_f4_quant(GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

#if (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif
    REGISTER_TILING_DEFAULT(GroupedDynamicHiF4QuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(GroupedDynamicHiF4QuantTilingData, tilingData, tiling);
    if constexpr (mode == TPL_MODE_0) {
        TPipe pipe;
        GroupedDynamicHiF4QuantMode0<DTYPE_X, DTYPE_Y> op(&tilingData, &pipe);
        op.Init(x, group_index, y, scale);
        op.Process();
    } else {
        TPipe pipe;
        GroupedDynamicHiF4QuantMode1<DTYPE_X, DTYPE_Y> op(&tilingData, &pipe);
        op.Init(x, group_index, y, scale);
        op.Process();
    }

#if (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
