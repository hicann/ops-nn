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
 * \file unique_consecutive_apt.cpp
 * \brief
 */

#include "./arch35/unique_consecutive_single_core_kernel.h"
#include "./arch35/unique_consecutive_multi_core_kernel.h"

template <template <typename, typename, typename, bool, bool> 
    class KERNEL_CLASS, 
    typename ORIGIN_TYPE, 
    typename INPUT_TYPE, 
    typename DTYPE_COUNT, 
    bool COUNT_OUT, 
    bool ISINT64>
__aicore__ inline void createAndRunKernel(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR count, GM_ADDR shape_out,
                                          GM_ADDR workspace, const UniqueConsecutiveTilingData* tilingData, TPipe* pipe)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if constexpr (sizeof(ORIGIN_TYPE) == 1) {
        KERNEL_CLASS<int8_t, INPUT_TYPE, DTYPE_COUNT, COUNT_OUT, ISINT64> op(pipe);
        op.Init(x, y, idx, count, shape_out, workspace, tilingData);
        op.Process();
    } else if constexpr (sizeof(ORIGIN_TYPE) == 2) {
        KERNEL_CLASS<int16_t, INPUT_TYPE, DTYPE_COUNT, COUNT_OUT, ISINT64> op(pipe);
        op.Init(x, y, idx, count, shape_out, workspace, tilingData);
        op.Process();
    } else if constexpr (sizeof(ORIGIN_TYPE) == 4) {
        KERNEL_CLASS<int32_t, INPUT_TYPE, DTYPE_COUNT, COUNT_OUT, ISINT64> op(pipe);
        op.Init(x, y, idx, count, shape_out, workspace, tilingData);
        op.Process();
    } else {
        KERNEL_CLASS<int64_t, INPUT_TYPE, DTYPE_COUNT, COUNT_OUT, ISINT64> op(pipe);
        op.Init(x, y, idx, count, shape_out, workspace, tilingData);
        op.Process();
    }
}

__aicore__ inline void CopyOutEmptyShape(GM_ADDR shape_out, AscendC::TPipe* pipe)
{
    GlobalTensor<uint64_t> shapeGm;
    TBuf<TPosition::VECCALC> shapeBuf;

    shapeGm.SetGlobalBuffer((__gm__ uint64_t*)shape_out);
    pipe->InitBuffer(shapeBuf, SHAPE_LEN * sizeof(uint64_t));

    LocalTensor<uint64_t> shapeTensor = shapeBuf.Get<uint64_t>();
    Duplicate(shapeTensor, (uint64_t)1, SHAPE_LEN);
    SimpleNativePipeSync<HardEvent::V_S>();

    shapeTensor.SetValue(SHAPE0_SIZE_IDX, UINT64_SHAPE_DIM_ONE);
    shapeTensor.SetValue(SHAPE0_DIM0_IDX, 0);

    shapeTensor.SetValue(SHAPE1_SIZE_IDX, 1);
    shapeTensor.SetValue(SHAPE1_DIM0_IDX, 0);

    shapeTensor.SetValue(SHAPE2_SIZE_IDX, 1);
    shapeTensor.SetValue(SHAPE2_DIM0_IDX, 0);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = SHAPE_LEN * sizeof(uint64_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    SimpleNativePipeSync<HardEvent::S_MTE3>();
    DataCopyPad(shapeGm, shapeTensor, dataCopyParams);
}

extern "C" __global__ __aicore__ void unique_consecutive(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR count,
                                                         GM_ADDR shape_out, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(UniqueConsecutiveTilingData, tilingDataIn, tiling);
    const UniqueConsecutiveTilingData* __restrict tilingData = &tilingDataIn;

    if (TILING_KEY_IS(11)) {
        createAndRunKernel<UniqueConsecutiveSingleCoreKerenl, DTYPE_X, DTYPE_X, DTYPE_COUNT, true, false>(x, y, idx, count, shape_out, workspace,
                                                                             tilingData, &pipe);
    } else if (TILING_KEY_IS(21)) {
        createAndRunKernel<UniqueConsecutiveMutilCoreKerenl, DTYPE_X, DTYPE_X, DTYPE_COUNT, true, false>(x, y, idx, count, shape_out, workspace,
                                                                            tilingData, &pipe);
    } else if (TILING_KEY_IS(22)) {
        createAndRunKernel<UniqueConsecutiveMutilCoreKerenl, DTYPE_X, DTYPE_X, DTYPE_COUNT, true, true>(x, y, idx, count, shape_out, workspace,
                                                                            tilingData, &pipe);
    } else if (TILING_KEY_IS(10)) {
        createAndRunKernel<UniqueConsecutiveSingleCoreKerenl, DTYPE_X, DTYPE_X, DTYPE_COUNT, false, false>(x, y, idx, count, shape_out, workspace,
                                                                              tilingData, &pipe);
    } else if (TILING_KEY_IS(20)) {
        createAndRunKernel<UniqueConsecutiveMutilCoreKerenl, DTYPE_X, DTYPE_X, DTYPE_COUNT, false, false>(x, y, idx, count, shape_out, workspace,
                                                                             tilingData, &pipe);
    } else if (TILING_KEY_IS(666)) {
        // Empty kernel
        CopyOutEmptyShape(shape_out, &pipe);
    }
}