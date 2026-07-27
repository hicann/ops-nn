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
 * \file non_zero_with_value.cpp
 * \brief NonZeroWithValue arch35 kernel 入口 —— 2D/fp32/transpose=true(坐标主序)。
 *        按 TilingKey 分发:null(空张量) / general(唯一非空计算路径)。
 *        general 重算 mask 且按 tile 循环,覆盖单核小张量与超大张量,无需 full_load/big_mask 分路。
 */
#include "arch35/non_zero_with_value_null.h"
#include "arch35/non_zero_with_value_general.h"

using namespace AscendC;

#define TILING_KEY_NULL 1000
#define TILING_KEY_GENERAL 1020

extern "C" __global__ __aicore__ void non_zero_with_value(GM_ADDR x, GM_ADDR value, GM_ADDR index, GM_ADDR count,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    // bool 走 int8 实例化(reinterpret:值字节一致,VF 微 API 支持 int8 不支持 bool;对齐 non_zero 做法)。
    // KValue = bool ? int8_t : DTYPE_VALUE;8 字节(double/int64/uint64)由 general 内部 sizeof 分支走 B64。
    using KValue = typename AscendC::Conditional<AscendC::IsSameType<DTYPE_VALUE, bool>::value, int8_t,
                                                 DTYPE_VALUE>::type;

    if (TILING_KEY_IS(TILING_KEY_NULL)) {
        NonZeroWithValue::NonZeroWithValueNull<KValue, DTYPE_INDEX> op;
        op.Init(x, value, index, count, userWS, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_GENERAL)) {
        NonZeroWithValue::NonZeroWithValueGeneral<KValue, DTYPE_INDEX> op;
        op.Init(x, value, index, count, userWS, &tilingData);
        op.Process();
        return;
    }
}
