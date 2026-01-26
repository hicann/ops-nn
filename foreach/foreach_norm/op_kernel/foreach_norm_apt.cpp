/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_norm_apt.cpp
 * \brief
 */

#include "arch35/foreach_norm_regbase.h"

using namespace ForeachNorm;

/**
 * modelCode:
 * 0 p=else... the default operator(not used now)
 * 1 p=0 Calculate the count of not-zero element in each tensor. (not used now)
 * 2 p=1 AbsAndNotNeedPower NotNeedSqrt(now is default as we now only consider p=1 || p=2)
 * 3 p=2 MulSelf(we don't need abs this time) Sqrt(not Power(self,1/p))
 * 4 p=+inf Calculate the max Abs(element) in each tensor.
 * 5 p=-inf Calculate the min Abs(element) in each tensor. (not used now)
 */

template <typename T>
__aicore__ inline void ForeachNormImpl(
    GM_ADDR inputs, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling, TPipe* tPipe, uint8_t modelCode)
{
    GET_TILING_DATA_WITH_STRUCT(ForeachReduceTilingDataRegbase, tiling_data_in, tiling);
    const ForeachReduceTilingDataRegbase* __restrict tilingData = &tiling_data_in;
    if (modelCode == POSITIVE_INF_SCALAR_NORM_MODEL_CODE) {
        ForeachNormNDRegbase<T, POSITIVE_INF_SCALAR_NORM_MODEL_CODE, ForeachReduceTilingDataRegbase> op;
        op.Init(inputs, output, workspace, tilingData, tPipe);
        op.Process();
    } else if (modelCode == ONE_SCALAR_NORM_MODEL_CODE) {
        ForeachNormNDRegbase<T, ONE_SCALAR_NORM_MODEL_CODE, ForeachReduceTilingDataRegbase> op;
        op.Init(inputs, output, workspace, tilingData, tPipe);
        op.Process();
    } else {
        ForeachNormNDRegbase<T, TWO_SCALAR_NORM_MODEL_CODE, ForeachReduceTilingDataRegbase> op;
        op.Init(inputs, output, workspace, tilingData, tPipe);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void foreach_norm(
    GM_ADDR inputs, GM_ADDR scalar, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);

    uint8_t modelCode = TWO_SCALAR_NORM_MODEL_CODE;
    GlobalTensor<DTYPE_SCALAR> inScalarGM;
    inScalarGM.SetGlobalBuffer((__gm__ DTYPE_SCALAR*)scalar, 1);
    float scalarVal = inScalarGM.GetValue(0);

    if constexpr (IsSameType<DTYPE_SCALAR, int64_t>::value) {
        if (static_cast<int>(scalarVal) == 1) {
            modelCode = ONE_SCALAR_NORM_MODEL_CODE;
        }
    } else {
        if ((*((uint32_t*)&scalarVal) & 0x7FFFFFFF) >> 23 == 0xFF && scalarVal > 0) {
            modelCode = POSITIVE_INF_SCALAR_NORM_MODEL_CODE;
        } else if (static_cast<int>(scalarVal) == 1) {
            modelCode = ONE_SCALAR_NORM_MODEL_CODE;
        }
    }

    TPipe pipeOp;
    if (TILING_KEY_IS(FOREACH_TILING_KEY_HALF)) {
        ForeachNormImpl<half>(inputs, output, userWS, tiling, &pipeOp, modelCode);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_FLOAT)) {
        ForeachNormImpl<float>(inputs, output, userWS, tiling, &pipeOp, modelCode);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_BF16)) {
        ForeachNormImpl<bfloat16_t>(inputs, output, userWS, tiling, &pipeOp, modelCode);
    }
    pipeOp.Destroy();
}
