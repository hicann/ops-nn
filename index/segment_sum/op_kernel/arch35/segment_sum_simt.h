/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SEGMENT_SUM_SIMT_H
#define SEGMENT_SUM_SIMT_H

#include "kernel_operator.h"
#include "segment_sum_struct.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"


namespace SegmentSum
{
constexpr int64_t DOUBLE = 2;
constexpr uint32_t MAX_THREAD_NUM = 2048;
using namespace AscendC;

template <typename TX, typename Index>
class SegmentSumSimt
{
public:
    __aicore__ inline SegmentSumSimt(const SegmentSumSimtTilingData* __restrict tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output);
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<TX> xGm, outputGm, outputGmInit;
    AscendC::GlobalTensor<Index> segmentIdsGm;
    TPipe* pipe_;
    const SegmentSumSimtTilingData* tilingData_;
};

template <typename TX, typename Index>
__aicore__ inline void SegmentSumSimt<TX, Index>::Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output)
{
    xGm.SetGlobalBuffer((__gm__ TX*)(x));
    segmentIdsGm.SetGlobalBuffer((__gm__ Index*)(segmentIds));
    outputGm.SetGlobalBuffer((__gm__ TX*)(output));

    uint64_t initCoreReal = GetBlockIdx() == (GetBlockNum() - 1) ? 
                                                tilingData_->initNumTailCore : tilingData_->initNumPerCore;
    uint64_t outputGmOffset = GetBlockIdx() * tilingData_->initNumPerCore;
    outputGmInit.SetGlobalBuffer((__gm__ TX *)(output) + outputGmOffset);
    InitGlobalMemory(outputGmInit, initCoreReal, static_cast<TX>(0));
    SyncAll();
    return;
}

template <typename TX, typename Index>
__simt_vf__ __launch_bounds__(MAX_THREAD_NUM) inline void Compute(
    __gm__ TX* xAddr, __gm__ Index* segmentIdsAddr, __gm__ TX* outputAddr, uint64_t colSize, uint64_t outerDim)
{
    int16_t blockNum = blockDim.y;
    int16_t blockIdx = threadIdx.y;
    for (Index i = get_block_idx() * blockNum + blockIdx; i < outerDim; i += get_block_num() * blockNum) {
        if (blockIdx > 0 && segmentIdsAddr[i] == segmentIdsAddr[i - 1]) {
            continue;
        }
        for (uint64_t j = threadIdx.x; j < colSize; j += blockDim.x) {
            TX res = xAddr[i * colSize + j];
            int16_t endRow = blockNum - 1;
            for (Index k = blockIdx + 1; k < blockNum; k++) {
                if (segmentIdsAddr[i] != segmentIdsAddr[i + k - blockIdx]) {
                    endRow = k - 1;
                    break;
                }
                res += xAddr[(i + k - blockIdx) * colSize + j];
            }
            uint64_t outOffset = segmentIdsAddr[i] * colSize + j;
            if (blockIdx != 0 && endRow != blockNum - 1) {
                outputAddr[outOffset] = res;
            } else {
                asc_atomic_add(outputAddr + outOffset, res);
            } 
        }
        
    }
    return;
}

template <typename TX, typename Index>
__aicore__ inline void SegmentSumSimt<TX, Index>::Process()
{
    __gm__ TX* input = (__gm__ TX*)xGm.GetPhyAddr();
    __gm__ Index* segmentIds = (__gm__ Index*)segmentIdsGm.GetPhyAddr();
    __gm__ TX* output = (__gm__ TX*)outputGm.GetPhyAddr();

    uint32_t threadNum = MAX_THREAD_NUM > tilingData_->innerDim ? tilingData_->innerDim : MAX_THREAD_NUM;
    uint32_t threadBlock = MAX_THREAD_NUM / threadNum;
    asc_vf_call<Compute<TX, Index>>(dim3{threadNum, threadBlock}, // block dim {x, y}
                                    input, segmentIds, output, tilingData_->innerDim, tilingData_->outerDim);
    return;
}
}
#endif