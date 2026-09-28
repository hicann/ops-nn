/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_augru_grad.cpp
 * \brief DynamicAUGRUGrad kernel入口：AUGRU反向BPTT（融合seq_length掩码生成）
 */

#include "arch35/dynamic_augru_grad.h"

extern "C" __global__ __aicore__ void dynamic_augru_grad(
    GM_ADDR x, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR weightAtt, GM_ADDR y, GM_ADDR initH, GM_ADDR h,
    GM_ADDR dy, GM_ADDR dh, GM_ADDR update, GM_ADDR updateAtt, GM_ADDR reset, GM_ADDR newGate, GM_ADDR hiddenNew,
    GM_ADDR seqLength, GM_ADDR mask, GM_ADDR dwInput, GM_ADDR dwHidden, GM_ADDR dbInput, GM_ADDR dbHidden, GM_ADDR dx,
    GM_ADDR dhPrev, GM_ADDR dwAtt, GM_ADDR workspace, GM_ADDR tiling)
{
#ifdef __CCE_KT_TEST__
    auto tilingData = *reinterpret_cast<DynamicAUGRUGradTilingData*>(tiling);
#else
    REGISTER_TILING_DEFAULT(DynamicAUGRUGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(DynamicAUGRUGradTilingData, tilingData, tiling);
#endif
    NsDynamicAUGRUGrad::DynamicAUGRUGradKernel<DTYPE_X> op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.dgateMM, &tilingData.dgateMMParam, op.dwInputMM,
                      &tilingData.dwInputMMParam, op.dwHiddenMM, &tilingData.dwHiddenMMParam, op.dxMM,
                      &tilingData.dxMMParam);
    op.Init(x, weightInput, weightHidden, weightAtt, y, initH, h, dy, dh, update, updateAtt, reset, newGate, hiddenNew,
            seqLength, mask, dwInput, dwHidden, dbInput, dbHidden, dx, dhPrev, dwAtt, workspace, &tilingData);
    op.Process();
}
