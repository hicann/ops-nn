/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_nd_simt.h
 * \brief
 */
#ifndef ASCENDC_GATHER_ND_GATHER_ND_SIMT_H_
#define ASCENDC_GATHER_ND_GATHER_ND_SIMT_H_

#include "kernel_operator.h"
#include "../inc/platform.h"

namespace GatherNd
{
using namespace AscendC;

constexpr uint64_t TOTAL_DIGITS = 64;
constexpr uint64_t OFFSET = 32;
constexpr int32_t defaultZero[4] = {0, 0, 0, 0};
constexpr uint32_t MAX_INPUT_RANK = 8;
constexpr uint32_t MAX_INPUT_RANK_SIZE = 1024;

#ifndef __CCE_KT_TEST__
typedef int32_t int4 __attribute__((ext_vector_type(4)));
#endif

// to avoid redundant static cast, using 1024 as chose thread num
#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_DIMS = 128;
#else
constexpr uint32_t THREAD_DIMS = 1024;
#endif

template <typename T1>
__aicore__ inline void SetOutOfBoundValue(__ubuf__ T1 *dstTensor, const uint32_t idx)
{
    if constexpr (IsSameType<T1, int4>::value) {
        dstTensor[idx] = (int4){defaultZero[0], defaultZero[1], defaultZero[2], defaultZero[3]};
    } else {
        dstTensor[idx] = 0;
    }
}

template <typename T1, typename T2, typename T3, const bool NIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIMS) inline void SimtDim(
    __ubuf__ T1 *dstTensor, __ubuf__ T2 *src1Tensor, __ubuf__ T3* inputShape, __gm__ T1 *src2Tensor, 
    const T3 curSize, const T3 curBegin, const T3 indicesRank, const T3 lastDimSize, const uint32_t rank,
    const T3 shift, const T3 m)
{
    for (T3 idx = Simt::GetThreadIdx(); idx < curSize; idx += Simt::GetThreadNum()) {
        bool idxOutOfBound = false;
        T3 outputGlobalIdx = idx + curBegin;
        T3 gatherAxisIdx = 0;
        T3 indicesCurrentIdx = 0;
        T2 srcIdx = 0;
        if constexpr (IsSameType<T3, uint32_t>::value) {
            T3 t1 = Simt::MulHi(outputGlobalIdx, m);
            t1 += outputGlobalIdx;
            T3 indicesIdx = t1 >> shift;
            indicesCurrentIdx = indicesIdx - indicesRank;
            gatherAxisIdx = outputGlobalIdx - indicesIdx * lastDimSize;
        } else {
            indicesCurrentIdx = outputGlobalIdx / lastDimSize - indicesRank;
            gatherAxisIdx = outputGlobalIdx % lastDimSize;
        }

        if constexpr (NIS == 1) {
            for (uint8_t i = 0; i < rank; i++) {
                T2 index = src1Tensor[indicesCurrentIdx * rank + i];
                if (index < 0) {
                    index += inputShape[i];
                }
                if (index >= inputShape[i] || index < 0) {
                    idxOutOfBound = true;
                    break;
                }
                srcIdx = srcIdx * inputShape[i] + index;
            }
            if (idxOutOfBound) {
                SetOutOfBoundValue<T1>(dstTensor, idx);
            } else {
                srcIdx = srcIdx * lastDimSize + gatherAxisIdx;
                dstTensor[idx] = src2Tensor[srcIdx];
            }
        } else {
            for (uint8_t i = 0; i < rank; i++) {
                T2 index = src1Tensor[indicesCurrentIdx * rank + i];
                if (index >= inputShape[i] || index < 0) {
                    idxOutOfBound = true;
                    break;
                }
                srcIdx = srcIdx * inputShape[i] + index;
            }
            if (idxOutOfBound) {
                SetOutOfBoundValue<T1>(dstTensor, idx);
            } else {
                srcIdx = srcIdx * lastDimSize + gatherAxisIdx;
                dstTensor[idx] = src2Tensor[srcIdx];
            }
        }
    }
}

template <typename T1, typename T2, typename T3, const bool NIS>
class GatherNdSimt
{
public:
    __aicore__ inline GatherNdSimt(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace,
                                const GatherNdTilingData* __restrict ordTilingData, TPipe* pipeIn);
    __aicore__ inline void CopyIn(LocalTensor<T2>& dstTensor, uint32_t gatherLength);
    __aicore__ inline void CopyOut(LocalTensor<T1>& srcTensor, uint32_t gatherLength);
    __aicore__ inline void Process();

protected:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> vecInQue;
    TQue<QuePosition::VECOUT, 1> vecOutQue;
    TBuf<TPosition::VECCALC> inShapeBuf;

    uint32_t coreNum;
    uint32_t cBlockIdx;
    const GatherNdTilingData* __restrict tilingData;

    // input
    GlobalTensor<T1> xGm;
    GlobalTensor<T2> indicesGm;

    // output
    GlobalTensor<T1> yGm;

    // shape info
    T3 xShape[8];
    T3 indicesShape[2];

    // buffer info
    uint32_t xUbSize;
    uint32_t indicesUbSize;

    // split info
    T3 gatherSize;
    T3 indicesNum;
    T3 outputSize;
    T3 blockFactor;
    T3 ubFactor;
    uint32_t rank;
    uint32_t blockNum;
    T3 curSize;
    T3 curBegin;
    T3 indicesBegin;
};

template <typename T1, typename T2, typename T3, const bool NIS>
__aicore__ inline void GatherNdSimt<T1, T2, T3, NIS>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace,
                                                           const GatherNdTilingData* __restrict ordTilingData,
                                                           TPipe* pipeIn)
{
    cBlockIdx = GetBlockIdx();
    tilingData = ordTilingData;

    // init pipe
    pipe = pipeIn;

    // init gm
    xGm.SetGlobalBuffer((__gm__ T1*)x);
    indicesGm.SetGlobalBuffer((__gm__ T2*)indices);
    yGm.SetGlobalBuffer((__gm__ T1*)y);

    // tiling info
    for (uint32_t i = 0; i < MAX_INPUT_RANK; i++) {
        xShape[i] = (T3)(tilingData->xShape[i]);
    }

    for (uint32_t i = 0; i < 2; i++) {
        indicesShape[i] = (T3)(tilingData->indicesShape[i]);
    }

    xUbSize = tilingData->xUbSize;
    indicesUbSize = tilingData->indicesUbSize;
    gatherSize = (T3)(tilingData->gatherSize);
    indicesNum = (T3)(tilingData->indicesNum);
    outputSize = (T3)(tilingData->outputSize);
    blockFactor = (T3)(tilingData->blockFactor);
    ubFactor = (T3)(tilingData->ubFactor);
    blockNum = tilingData->blockNum;
    rank = tilingData->rank;

    pipe->InitBuffer(vecInQue, 1, indicesUbSize);
    pipe->InitBuffer(vecOutQue, 1, xUbSize);
    pipe->InitBuffer(inShapeBuf, MAX_INPUT_RANK_SIZE);
}

template <typename T1, typename T2, typename T3, const bool NIS>
__aicore__ inline void GatherNdSimt<T1, T2, T3, NIS>::CopyIn(LocalTensor<T2>& dstTensor, uint32_t indicesLength)
{
    DataCopyPad(dstTensor, indicesGm[indicesBegin],
                {static_cast<uint16_t>(1), static_cast<uint32_t>(indicesLength), 0, 0, 0}, {false, 0, 0, 0});
}

template <typename T1, typename T2, typename T3, const bool NIS>
__aicore__ inline void GatherNdSimt<T1, T2, T3, NIS>::CopyOut(LocalTensor<T1>& srcTensor, uint32_t gatherLength)
{
    DataCopyPad(yGm[curBegin], srcTensor, {static_cast<uint16_t>(1), static_cast<uint32_t>(gatherLength), 0, 0, 0});
}

template <typename T1, typename T2, typename T3, const bool NIS>
__aicore__ inline void GatherNdSimt<T1, T2, T3, NIS>::Process()
{
    T3 shift = 0;
    T3 m = 0;
    GetUintDivMagicAndShift(m, shift, gatherSize);

    T3 beginIdx = cBlockIdx * blockFactor * ubFactor;
    T3 endIdx =
        (cBlockIdx + 1) * blockFactor * ubFactor < outputSize ? (cBlockIdx + 1) * blockFactor * ubFactor : outputSize;
    AscendC::LocalTensor<T3> xInShape = inShapeBuf.Get<T3>();
    for (uint32_t i = 0; i < MAX_INPUT_RANK; i++) {
        xInShape.SetValue(i, (T3)(tilingData->xShape[i]));
    }
    DataSyncBarrier<MemDsbT::UB>();
    for (T3 idx = beginIdx; idx < endIdx; idx = idx + ubFactor) {
        curBegin = idx;
        T3 curEnd = (curBegin + ubFactor) < outputSize ? (curBegin + ubFactor) : outputSize;
        curSize = curEnd - curBegin;
        indicesBegin = curBegin / gatherSize * rank;
        T3 indicesIdxEnd = ((curEnd + gatherSize - 1) / gatherSize) * rank;
        T3 indicesLength = (indicesIdxEnd - indicesBegin) * sizeof(T2);
        T3 indicesRank = indicesBegin / rank;

        LocalTensor<T2> indicesBuffer = vecInQue.AllocTensor<T2>();
        CopyIn(indicesBuffer, indicesLength);
        vecInQue.EnQue(indicesBuffer);
        vecInQue.DeQue<T2>();
        LocalTensor<T1> outputBuffer = vecOutQue.AllocTensor<T1>();

        Simt::VF_CALL<SimtDim<T1, T2, T3, NIS>>(Simt::Dim3(static_cast<uint32_t>(THREAD_DIMS)),
                                                (__ubuf__ T1*)outputBuffer.GetPhyAddr(),
                                                (__ubuf__ T2*)indicesBuffer.GetPhyAddr(),
                                                (__ubuf__ T3*)xInShape.GetPhyAddr(),
                                                (__gm__ T1*)xGm.GetPhyAddr(),
                                                curSize, curBegin, indicesRank, gatherSize, rank, shift, m);
        vecInQue.FreeTensor(indicesBuffer);
        vecOutQue.EnQue(outputBuffer);
        vecOutQue.DeQue<T1>();
        T3 curLength = curSize * sizeof(T1);

        CopyOut(outputBuffer, curLength);
        vecOutQue.FreeTensor(outputBuffer);
    }
}
}  // namespace GatherNd

#endif  // ASCENDC_GATHER_ND_GATHER_ND_SIMT_H_