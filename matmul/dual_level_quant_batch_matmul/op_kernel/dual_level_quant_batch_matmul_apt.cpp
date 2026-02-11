/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dual_level_quant_batch_matmul_apt.cpp
 * \brief
 */

#include "arch35/dual_level_quant_batch_matmul_basic_block_controller.h"
#include "dual_level_quant_batch_matmul_tiling_data.h"
#include "dual_level_quant_batch_matmul_tiling_key.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_utils.h"

#ifndef DTYPE_BIAS
#define DTYPE_BIAS DTYPE_Y
#endif

using AscendC::fp8_e8m0_t;

using DualLevelQuantBatchMatmul::Arch35::DualLevelQuantBatchMatmulBasicBlockController;

template <
    int SOC_VERSION_TYPE, int SUB_SOC_VERSION_TYPE, int TEMPLATE_CUSTOM, int LEVEL1_QUANT_TYPE, int LEVEL0_QUANT_TYPE,
    bool TRANS_A, bool TRANS_B, bool HAS_BIAS, bool IS_WEIGHT_NZ>
__global__ __aicore__ void dual_level_quant_batch_matmul(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Level0Scale, GM_ADDR x1Level1Scale, GM_ADDR x2Level0Scale, GM_ADDR x2Level1Scale,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    AscendC::InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(DualLevelQuantBatchMatmulBasicTilingData);

    GET_TILING_DATA_WITH_STRUCT(DualLevelQuantBatchMatmulBasicTilingData, tilingDataIn, tiling);
    DualLevelQuantBatchMatmulBasicBlockController<
        DTYPE_X1, DTYPE_X2, DTYPE_X1_LEVEL1_SCALE, DTYPE_X1_LEVEL0_SCALE, DTYPE_X2_LEVEL1_SCALE, DTYPE_X2_LEVEL0_SCALE,
        DTYPE_BIAS, DTYPE_Y, false, true, HAS_BIAS>
        op;
    op.Init(x1, x2, x1Level0Scale, x1Level1Scale, x2Level0Scale, x2Level1Scale, bias, y, &tilingDataIn);
    op.Process();
}
