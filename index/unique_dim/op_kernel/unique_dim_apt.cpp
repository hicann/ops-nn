/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "./arch35/unique_dim_kernel.h"

template <typename T, bool RETURN_INVERSE, bool RETURN_COUNTS>
__aicore__ inline void RunUniqueDimKernel(GM_ADDR x, GM_ADDR valueOut, GM_ADDR inverseOut, GM_ADDR countsOut,
                                          GM_ADDR shapeOut, GM_ADDR workspace, const UniqueDimTilingData* tilingData,
                                          TPipe* pipe)
{
    UniqueDimKernel<T, RETURN_INVERSE, RETURN_COUNTS> op(pipe);
    op.Init(x, valueOut, inverseOut, countsOut, shapeOut, workspace, tilingData);
    op.Process();
}

template <typename T>
__aicore__ inline void DispatchByFlags(GM_ADDR x, GM_ADDR valueOut, GM_ADDR inverseOut, GM_ADDR countsOut,
                                       GM_ADDR shapeOut, GM_ADDR workspace, const UniqueDimTilingData* tilingData,
                                       TPipe* pipe)
{
    bool returnInverse = (tilingData->returnInverse != 0);
    bool returnCounts = (tilingData->returnCounts != 0);

    if (returnInverse && returnCounts) {
        RunUniqueDimKernel<T, true, true>(x, valueOut, inverseOut, countsOut, shapeOut, workspace, tilingData, pipe);
    } else if (returnInverse && !returnCounts) {
        RunUniqueDimKernel<T, true, false>(x, valueOut, inverseOut, countsOut, shapeOut, workspace, tilingData, pipe);
    } else if (!returnInverse && returnCounts) {
        RunUniqueDimKernel<T, false, true>(x, valueOut, inverseOut, countsOut, shapeOut, workspace, tilingData, pipe);
    } else {
        RunUniqueDimKernel<T, false, false>(x, valueOut, inverseOut, countsOut, shapeOut, workspace, tilingData, pipe);
    }
}

__aicore__ inline void CopyOutEmptyShape(GM_ADDR shapeOut, TPipe* pipe)
{
    GlobalTensor<uint64_t> shapeGm;
    TBuf<TPosition::VECCALC> shapeBuf;

    shapeGm.SetGlobalBuffer((__gm__ uint64_t*)shapeOut);
    pipe->InitBuffer(shapeBuf, SHAPE_LEN * sizeof(uint64_t));

    LocalTensor<uint64_t> shapeTensor = shapeBuf.Get<uint64_t>();
    Duplicate(shapeTensor, (uint64_t)1, SHAPE_LEN);
    {
        event_t evt = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(evt);
        WaitFlag<HardEvent::V_S>(evt);
    }

    FillEmptyShapeValues(shapeTensor);

    DataCopyExtParams dataCopyParams;
    SetShapeCopyParams(dataCopyParams);

    {
        event_t evt = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(evt);
        WaitFlag<HardEvent::S_MTE3>(evt);
    }
    DataCopyPad(shapeGm, shapeTensor, dataCopyParams);
}

extern "C" __global__ __aicore__ void unique_dim(GM_ADDR x, GM_ADDR valueOut, GM_ADDR inverseOut, GM_ADDR countsOut,
                                                 GM_ADDR shapeOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(UniqueDimTilingData, tilingDataIn, tiling);
    const UniqueDimTilingData* __restrict tilingData = &tilingDataIn;

    if (TILING_KEY_IS(0)) {
        // Empty input — write zero shapes
        CopyOutEmptyShape(shapeOut, &pipe);
        return;
    } else if (TILING_KEY_IS(1)) {
        DispatchByFlags<DTYPE_X>(x, valueOut, inverseOut, countsOut, shapeOut, usrWorkspace, tilingData, &pipe);
    }
}
