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
 * \file index_perf.h
 * \brief Ascendc index_perf kernel
 */

#ifndef ASCENDC_INDEX_PERF_H_
#define ASCENDC_INDEX_PERF_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace Index {
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t USED_THREAD = 128;
constexpr uint32_t THREAD_LAUNCH = 512;
#else
constexpr uint32_t USED_THREAD = 512;
constexpr uint32_t THREAD_LAUNCH = 2048;
#endif

template <typename T, typename P, typename T2>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_LAUNCH) inline void SimtCompute(
    __gm__ T* outputGm_, __gm__ T* inputXGm_, __gm__ P* indexTensor0, __gm__ P* indexTensor1,
    __gm__ const IndexPerfSimtTilingData* __restrict tiling)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(IndexPerfSimtTilingData, tilingData, tiling);
    T2 outputLength = static_cast<T2>(tilingData->outputLength);
    int64_t inputShape0 = static_cast<int64_t>(tilingData->inputShape0);
    T2 inputShape1 = static_cast<T2>(tilingData->inputShape1);

    for (T2 i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < outputLength;
         i = i + Simt::GetBlockNum() * Simt::GetThreadNum()) {
        int64_t curIdx0 = indexTensor0[i];
        curIdx0 = curIdx0 < 0 ? (curIdx0 + inputShape0) : curIdx0;
        int64_t curIdx1 = indexTensor1[i];
        curIdx1 = curIdx1 < 0 ? (curIdx1 + static_cast<int64_t>(inputShape1)) : curIdx1;
        T2 inputIndex = curIdx0 * inputShape1 + curIdx1;
        outputGm_[i] = inputXGm_[inputIndex];
    }
}

template <typename P>
__aicore__ inline __gm__ P* GetInputTensorAddr(GM_ADDR indices, uint16_t index)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(indices);
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
                                          // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ P*>(*(tensorPtr + index));
}

template <typename T, typename P, typename T2>
__aicore__ inline void Process(
    __gm__ T* output, __gm__ T* inputX, GM_ADDR indices, __gm__ const IndexPerfSimtTilingData* __restrict tiling)
{
    // get dynamci input tensor
    __gm__ P* indexTensor0 = GetInputTensorAddr<P>(indices, 0);
    __gm__ P* indexTensor1 = GetInputTensorAddr<P>(indices, 1);
    AscendC::Simt::VF_CALL<SimtCompute<T, P, T2>>(
        AscendC::Simt::Dim3{USED_THREAD}, output, inputX, indexTensor0, indexTensor1, tiling);
}
} // namespace Index

#endif // ASCENDC_INDEX_PERF_H_