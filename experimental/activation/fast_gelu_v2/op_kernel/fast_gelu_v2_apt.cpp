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
 * \file fast_gelu_v2_arch35.cpp
 * \brief FastGeluV2 kernel entry (arch35)
 *
 * This is the global kernel function registered via ASCENDC_TPL_ARGS_DECL.
 * The template parameters D_T_X and BUFFER_MODE are resolved at compile time
 * based on the TilingKey set by the host-side Tiling function.
 *
 * Parameters (in order, fixed by CANN convention):
 *   x         - Input tensor in Global Memory
 *   y         - Output tensor in Global Memory
 *   workspace - System workspace (unused by this operator)
 *   tiling    - Serialized TilingData structure from host
 */

#include "fast_gelu_v2.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void fast_gelu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FastGeluV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluV2TilingData, tilingData, tiling);
    NsFastGeluV2::FastGeluV2<D_T_X, BUFFER_MODE> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
