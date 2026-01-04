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
 * \file quant_batch_matmul_v4.cpp
 * \brief
 */

#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#include "quant_batch_matmul_v4_tiling_key.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    #include "quant_batch_matmul_v4_msd.h"
    #include "quant_batch_matmul_v4_perblock.h"
    #include "quant_batch_matmul_v4_pergroup.h"
#endif

template <int TRANS, int QUANT_TYPE, int OPTION_ATTRS, int WEIGHTNZ, int KERNEL_TEMPLATE_TYPE>
__global__ __aicore__ void quant_batch_matmul_v4(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale,
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
    REGISTER_TILING_DEFAULT(QuantBatchMatmulV4MsdTilingData);
    AscendC::TPipe tPipe;
    #if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        if constexpr (TRANS == QUANT_BATCH_MATMUL_V4_NOT_TRANS && QUANT_TYPE == QUANT_BATCH_MATMUL_V4_MX && OPTION_ATTRS == QUANT_BATCH_MATMUL_V4_OPTION_ATTR_NONE &&
            WEIGHTNZ == QUANT_BATCH_MATMUL_V4_NOT_WEIGHT_NZ && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUL_V4_MSD_BASIS) {
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
        } else if constexpr (
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
            TRANS == QUANT_BATCH_MATMUL_V4_B_TRANS && QUANT_TYPE == QUANT_BATCH_MATMUL_V4_MX && OPTION_ATTRS == QUANT_BATCH_MATMUL_V4_OPTION_ATTR_NONE &&
            WEIGHTNZ == QUANT_BATCH_MATMUL_V4_NOT_WEIGHT_NZ && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUL_V4_PERBLOCK_BASIS) {
            GET_TILING_DATA_WITH_STRUCT(QuantBatchMatmulV4PerblockTilingData, tilingDataIn, tiling);
            AscendC::QuantBatchMatmulV4Perblock<int8_t, int8_t, float, float, float, bfloat16_t> op;
            op.Init(x1, x2, bias, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, y, userWS, &tilingDataIn, &tPipe);                                                                                             \
            op.Process();
            tPipe.Destroy();
#endif
        } else if constexpr (
            TRANS == QUANT_BATCH_MATMUL_V4_NOT_TRANS && QUANT_TYPE == QUANT_BATCH_MATMUL_V4_MX && OPTION_ATTRS == QUANT_BATCH_MATMUL_V4_OPTION_ATTR_NONE &&
            WEIGHTNZ == QUANT_BATCH_MATMUL_V4_NOT_WEIGHT_NZ && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUL_V4_PERGROUP_BASIS) {
            GET_TILING_DATA_WITH_STRUCT(QuantBatchMatmulV3TilingData, tilingDataIn, tiling);
            AscendC::QuantBatchMatmulV4Pergroup<AscendC::int4b_t, AscendC::int4b_t, float, float, DTYPE_Y> op;
            op.Init(x1, x2, bias, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, y, userWS, &tilingDataIn, &tPipe);                                                                                             \
            op.Process();
            tPipe.Destroy();
        }
    #endif
}

