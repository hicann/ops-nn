/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_augru.cpp
 * \brief DynamicAUGRU arch35 regbase kernel entry.
 */

#include "arch35/dynamic_augru_base.h"
#include "arch35/dynamic_augru_tiling_key.h"

using namespace AscendC;
using namespace DynamicAUGRU;

template <typename KernelT>
__aicore__ inline void RunDynamicAUGRU(GM_ADDR x, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR weightAtt,
                                       GM_ADDR biasInput, GM_ADDR biasHidden, GM_ADDR sequenceLength, GM_ADDR initH,
                                       GM_ADDR y, GM_ADDR outputH, GM_ADDR update, GM_ADDR updateAtt, GM_ADDR reset,
                                       GM_ADDR newState, GM_ADDR hiddenNew, GM_ADDR workspace,
                                       const DynamicAUGRUTilingData* tilingData)
{
    KernelT op;
    const TCubeTiling* inputMMTiling = &tilingData->inputMMTiling;
    const TCubeTiling* hiddenMMTiling = &tilingData->hiddenMMTiling;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.inputMM, inputMMTiling, op.hiddenMM, hiddenMMTiling);
    op.Init(x, weightInput, weightHidden, weightAtt, biasInput, biasHidden, sequenceLength, initH, y, outputH, update,
            updateAtt, reset, newState, hiddenNew, workspace, tilingData);
    op.Process();
}

template <uint32_t SEQUENCE_MODE>
__global__ __aicore__ void dynamic_augru(GM_ADDR x, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR weightAtt,
                                         GM_ADDR biasInput, GM_ADDR biasHidden, GM_ADDR sequenceLength, GM_ADDR initH,
                                         GM_ADDR y, GM_ADDR outputH, GM_ADDR update, GM_ADDR updateAtt, GM_ADDR reset,
                                         GM_ADDR newState, GM_ADDR hiddenNew, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DynamicAUGRUTilingData);
    GET_TILING_DATA_WITH_STRUCT(DynamicAUGRUTilingData, tilingData, tiling);
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }

    // DTYPE_Y comes from the def output profile, including absent optional states.
    // All shapes use Cube projections and regbase pointwise computation.
    // TilingKey selects only sequence handling; state arithmetic remains FP32.
    RunDynamicAUGRU<DynamicAUGRUBase<DTYPE_Y, SEQUENCE_MODE>>(
        x, weightInput, weightHidden, weightAtt, biasInput, biasHidden, sequenceLength, initH, y, outputH, update,
        updateAtt, reset, newState, hiddenNew, userWorkspace, &tilingData);
}
