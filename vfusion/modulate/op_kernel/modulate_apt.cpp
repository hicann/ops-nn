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
#include "arch35/modulate_struct.h"
#include "arch35/modulate_regbase.h"
#include "arch35/modulate_regbase_tiling_key.h"

template <uint8_t tilingStrategy, bool isScale, bool isShift>
__global__ __aicore__ void modulate(
    GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe tpipe;
    REGISTER_TILING_DEFAULT(ModulateRegbaseTilingData);
    GET_TILING_DATA(tilingData, tiling);
#if (defined(DTYPE_X))
    if constexpr (tilingStrategy == MODULATE_TILING_L) {
        Modulate::ModulateL<DTYPE_X, isScale, isShift> op(tpipe, tilingData);
        op.Init(x, scale, shift, y);
        op.Process();
    } else if constexpr (tilingStrategy == MODULATE_TILING_D) {
        Modulate::ModulateD<DTYPE_X, isScale, isShift> op(tpipe, tilingData);
        op.Init(x, scale, shift, y);
        op.Process();
    }
#endif
}
