/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v4.cpp
 * \brief
 */

#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    #include "quant_batch_matmul_v4_msd.h"
    #include "quant_batch_matmul_v4_perblock.h"
    #include "quant_batch_matmul_v4_pergroup.h"
#endif

extern "C" __global__ __aicore__ void quant_batch_matmul_v4(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale,
                                                            GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR x1_offset,
                                                            GM_ADDR x2_offset, GM_ADDR y_offset, GM_ADDR x2_table,
                                                            GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    AscendC::SetSysWorkspace(workspace);
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    AscendC::TPipe tPipe;
    #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        if (TILING_KEY_IS(100400UL)){
            GET_TILING_DATA_WITH_STRUCT(QuantBatchMatmulV4MsdTilingData, tilingDataIn, tiling);
            if ASCEND_IS_AIV {
                QuantBatchMatmulV4MsdPre opPre;
                opPre.Init(x1, x1, userWS, &tilingDataIn, &tPipe);
                opPre.Process();
                tPipe.Reset();
                tPipe.Destroy();
                tPipe.Init();
            }
            SyncAll<false>();
            QuantBatchMatmulV4Msd<AscendC::int4b_t, AscendC::int4b_t, float, DTYPE_Y> op;
            op.Init(x1, x2, bias, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, y, userWS, &tilingDataIn, &tPipe);                                                                                             \
            op.Process();
        } else if (TILING_KEY_IS(200410UL)) {
            GET_TILING_DATA_WITH_STRUCT(QuantBatchMatmulV4PerblockTilingData, tilingDataIn, tiling);
            AscendC::QuantBatchMatmulV4Perblock<int8_t, int8_t, float, float, float, bfloat16_t> op;
            op.Init(x1, x2, bias, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, y, userWS, &tilingDataIn, &tPipe);                                                                                             \
            op.Process();
            tPipe.Destroy();
        } else if (TILING_KEY_IS(33UL)) {
            GET_TILING_DATA_WITH_STRUCT(QuantBatchMatmulV3TilingData, tilingDataIn, tiling);
            AscendC::QuantBatchMatmulV4Pergroup<AscendC::int4b_t, AscendC::int4b_t, float, float, DTYPE_Y> op;
            op.Init(x1, x2, bias, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, y, userWS, &tilingDataIn, &tPipe);                                                                                             \
            op.Process();
            tPipe.Destroy();
        }
    #endif
}

