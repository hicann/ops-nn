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
 * \file gru.cpp
 * \brief
 */

#include "gru_fp32.h"
#include "gru_tiling_data.h"
#include "gru_tiling_key.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include <stdio.h>

template <uint32_t mmSplit>
__global__ __aicore__ void gru(GM_ADDR x, GM_ADDR wi, GM_ADDR wh, GM_ADDR bi, GM_ADDR bh, GM_ADDR batch_sizes,
                               GM_ADDR init_h, GM_ADDR y, GM_ADDR output_h, GM_ADDR r, GM_ADDR z, GM_ADDR n,
                               GM_ADDR n_h, GM_ADDR workspace, GM_ADDR gruTiling)
{
    REGISTER_TILING_DEFAULT(GruTilingData);
    GET_TILING_DATA_WITH_STRUCT(GruTilingData, tilingData, gruTiling);
    const TCubeTiling* __restrict inputMMTiling = &(tilingData.inputMMParam);
    const TCubeTiling* __restrict hiddenMMTiling = &(tilingData.hiddenMMParam);
    if constexpr (mmSplit == static_cast<uint32_t>(GRU_TPL_MM_FP16_SPLIT)) {
        GruFP32<half> gruOp;
        REGIST_MATMUL_OBJ(&gruOp.pipe, GetSysWorkSpacePtr(), gruOp.inputMM, inputMMTiling, gruOp.hiddenMM,
                          hiddenMMTiling);

        gruOp.Init(x, wi, wh, bi, bh, batch_sizes, init_h, y, output_h, r, z, n, n_h, &tilingData, workspace);
        gruOp.Process();
    } else if constexpr (mmSplit == static_cast<uint32_t>(GRU_TPL_MM_FP32_SPLIT)) {
        GruFP32<float> gruOp;
        REGIST_MATMUL_OBJ(&gruOp.pipe, GetSysWorkSpacePtr(), gruOp.inputMM, inputMMTiling, gruOp.hiddenMM,
                          hiddenMMTiling);

        gruOp.Init(x, wi, wh, bi, bh, batch_sizes, init_h, y, output_h, r, z, n, n_h, &tilingData, workspace);
        gruOp.Process();
    }
}