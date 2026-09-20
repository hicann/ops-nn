/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Template J: elementwise single-reduction route.
//
// When numBlocks == 1, every output element is its own reduction.  The
// slice-major templates still perform a complete norm pass followed by a
// complete scale pass, even though the norm is simply abs(x) (or the
// non-zero indicator for p == 0).  This kernel keeps the same FP32
// arithmetic, but computes the scale while the input tile is already in UB
// and writes the result once.
#ifndef _RENORM_ELEMENTWISE_H_
#define _RENORM_ELEMENTWISE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"
#include "renorm_tiling_key.h"

namespace NsRenormElementwise {

using namespace AscendC;

constexpr int32_t NORM_MODE_P_POSITIVE = 0;
constexpr int32_t NORM_MODE_P_ZERO = 1;
constexpr int32_t NORM_MODE_P_INF = 2;
constexpr int32_t NORM_MODE_MAXNORM_ZERO = 3;
constexpr int64_t VECTOR_ALIGN_BYTES = 32;

template <typename D_T_X>
class RenormElementwise {
public:
    __aicore__ inline RenormElementwise() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RenormTilingData* tilingData);
    __aicore__ inline void Process();

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> dataBuf;
    TBuf<QuePosition::VECCALC> workBuf;
    TBuf<QuePosition::VECCALC> scaleBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> zerosBuf;
    TBuf<QuePosition::VECCALC> onesBuf;
    TBuf<QuePosition::VECCALC> maxNormBuf;

    GlobalTensor<D_T_X> inputGM;
    GlobalTensor<D_T_X> outputGM;

    int64_t totalElements_ = 0;
    int64_t elementsPerCore_ = 0;
    int64_t tileLength_ = 0;
    float maxNorm_ = 0.0f;
    float eps_ = 0.0f;
    int32_t normMode_ = NORM_MODE_P_POSITIVE;
};

template <typename D_T_X>
__aicore__ inline void RenormElementwise<D_T_X>::Init(GM_ADDR x, GM_ADDR y, const RenormTilingData* tilingData)
{
    totalElements_ = tilingData->totalElements;
    elementsPerCore_ = tilingData->slicesPerCore;
    tileLength_ = tilingData->tileLength;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;
    normMode_ = tilingData->normMode;

    if (totalElements_ <= 0 || elementsPerCore_ <= 0 || tileLength_ <= 0) {
        return;
    }

    inputGM.SetGlobalBuffer((__gm__ D_T_X*)x, totalElements_);
    outputGM.SetGlobalBuffer((__gm__ D_T_X*)y, totalElements_);

    int64_t alignElems = VECTOR_ALIGN_BYTES / static_cast<int64_t>(sizeof(D_T_X));
    int64_t alignedTile = (tileLength_ + alignElems - 1) / alignElems * alignElems;
    pipe.InitBuffer(dataBuf, alignedTile * sizeof(D_T_X));
    pipe.InitBuffer(workBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(scaleBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(maskBuf, alignedTile);
    pipe.InitBuffer(zerosBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(onesBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(maxNormBuf, alignedTile * sizeof(float));
}

template <typename D_T_X>
__aicore__ inline void RenormElementwise<D_T_X>::Process()
{
    if (totalElements_ <= 0 || elementsPerCore_ <= 0 || tileLength_ <= 0) {
        return;
    }

    int64_t blockIdx = GetBlockIdx();
    int64_t start = blockIdx * elementsPerCore_;
    int64_t end = start + elementsPerCore_;
    if (end > totalElements_) {
        end = totalElements_;
    }
    if (start >= end) {
        return;
    }

    LocalTensor<D_T_X> dataLocal = dataBuf.Get<D_T_X>();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> zerosLocal = zerosBuf.Get<float>();
    LocalTensor<float> onesLocal = onesBuf.Get<float>();
    LocalTensor<float> maxNormLocal = maxNormBuf.Get<float>();

    int64_t alignElems = VECTOR_ALIGN_BYTES / static_cast<int64_t>(sizeof(D_T_X));
    int64_t maxAlignedTile = (tileLength_ + alignElems - 1) / alignElems * alignElems;
    Duplicate(zerosLocal, 0.0f, static_cast<int32_t>(maxAlignedTile));
    Duplicate(onesLocal, 1.0f, static_cast<int32_t>(maxAlignedTile));
    Duplicate(maxNormLocal, maxNorm_, static_cast<int32_t>(maxAlignedTile));
    PipeBarrier<PIPE_V>();

    for (int64_t off = start; off < end; off += tileLength_) {
        int64_t cur = (tileLength_ < (end - off)) ? tileLength_ : (end - off);
        int64_t aligned = (cur + alignElems - 1) / alignElems * alignElems;
        uint8_t rightPad = static_cast<uint8_t>(aligned - cur);

        DataCopyExtParams loadParams;
        loadParams.blockCount = 1;
        loadParams.blockLen = static_cast<uint32_t>(cur * sizeof(D_T_X));
        loadParams.srcStride = 0;
        loadParams.dstStride = 0;
        DataCopyPadExtParams<D_T_X> loadPad = {true, 0, rightPad, 0};
        DataCopyPad(dataLocal, inputGM[off], loadParams, loadPad);
        TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(loadDone);
        WaitFlag<HardEvent::MTE2_V>(loadDone);

        if constexpr (sizeof(D_T_X) == sizeof(float)) {
            DataCopy(workLocal, dataLocal, static_cast<int32_t>(aligned));
        } else {
            Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(aligned));
        }
        PipeBarrier<PIPE_V>();

        if (normMode_ == NORM_MODE_MAXNORM_ZERO) {
            Duplicate(dataLocal, static_cast<D_T_X>(0), static_cast<int32_t>(aligned));
        } else {
            // Keep the original signed FP32 values for the final store.  The
            // norm path below overwrites workLocal with abs(x) (or the
            // p==0 indicator), so using it for the output would lose signs.
            DataCopy(scaleLocal, workLocal, static_cast<int32_t>(aligned));
            PipeBarrier<PIPE_V>();
            Abs(workLocal, workLocal, static_cast<int32_t>(aligned));
            PipeBarrier<PIPE_V>();

            if (normMode_ == NORM_MODE_P_ZERO) {
                Compare(maskLocal, workLocal, zerosLocal, CMPMODE::GT, static_cast<int32_t>(aligned));
                PipeBarrier<PIPE_V>();
                Select(workLocal, maskLocal, onesLocal, zerosLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                       static_cast<int32_t>(aligned));
                PipeBarrier<PIPE_V>();
            }

            Compare(maskLocal, workLocal, maxNormLocal, CMPMODE::GT, static_cast<int32_t>(aligned));
            PipeBarrier<PIPE_V>();
            Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(aligned));
            PipeBarrier<PIPE_V>();
            Reciprocal(workLocal, workLocal, static_cast<int32_t>(aligned));
            PipeBarrier<PIPE_V>();
            Muls(workLocal, workLocal, maxNorm_, static_cast<int32_t>(aligned));
            PipeBarrier<PIPE_V>();
            Select(workLocal, maskLocal, workLocal, onesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   static_cast<int32_t>(aligned));
            PipeBarrier<PIPE_V>();

            // scaleLocal holds the original signed values; workLocal holds
            // the computed per-element scale.
            Muls(scaleLocal, scaleLocal, workLocal, static_cast<int32_t>(aligned));
            PipeBarrier<PIPE_V>();
            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(dataLocal, scaleLocal, static_cast<int32_t>(aligned));
            } else {
                Cast(dataLocal, scaleLocal, RoundMode::CAST_RINT, static_cast<int32_t>(aligned));
            }
        }

        TEventID storeReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(storeReady);
        WaitFlag<HardEvent::V_MTE3>(storeReady);
        DataCopyExtParams storeParams;
        storeParams.blockCount = 1;
        storeParams.blockLen = static_cast<uint32_t>(cur * sizeof(D_T_X));
        storeParams.srcStride = 0;
        storeParams.dstStride = 0;
        DataCopyPad(outputGM[off], dataLocal, storeParams);
        TEventID storeDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(storeDone);
        WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
    }
}

} // namespace NsRenormElementwise

#endif // _RENORM_ELEMENTWISE_H_
