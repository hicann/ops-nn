/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_BM_CR_TILED_H_
#define _RENORM_BM_CR_TILED_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"
#include "common/renorm_common.h"

namespace NsRenormBmCrTiled {

using namespace AscendC;

constexpr int32_t NORM_MODE_P_POSITIVE = 0;
constexpr int64_t FP32_ALIGN = 8;
constexpr int64_t ATOMIC_ALIGN = 16;

// Block-major cross-core reduction for a large logical block whose complete
// [slice, inner] matrix does not fit in UB.  Each core owns complete outer
// blocks and moves several adjacent slices in one aligned DMA transaction.
template <typename D_T_X, bool GENERIC_POSITIVE = false>
class RenormBmCrTiled {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RenormTilingData* tilingData,
                                TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    TPipe* pipe = nullptr;
    TBuf<QuePosition::VECCALC> dataBuf;
    TBuf<QuePosition::VECCALC> workBuf;
    TBuf<QuePosition::VECCALC> patternBuf;
    TBuf<QuePosition::VECCALC> partialBuf;
    TBuf<QuePosition::VECCALC> scaleBuf;
    TBuf<QuePosition::VECCALC> maskBuf;

    GlobalTensor<D_T_X> inputGM;
    GlobalTensor<D_T_X> outputGM;
    GlobalTensor<float> workspaceGM;

    int64_t totalElements_ = 0;
    int64_t sliceCount_ = 0;
    int64_t blockSize_ = 0;
    int64_t numBlocks_ = 0;
    int64_t sliceTile_ = 0;
    int64_t tileElements_ = 0;
    int64_t wsStride_ = 0;
    int64_t coreNum_ = 0;
    float p_ = 0.0f;
    float maxNorm_ = 0.0f;
    float eps_ = 0.0f;
    int32_t normMode_ = 0;
};

template <typename D_T_X, bool GENERIC_POSITIVE>
__aicore__ inline void RenormBmCrTiled<D_T_X, GENERIC_POSITIVE>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                                      const RenormTilingData* tilingData, TPipe* pipeIn)
{
    pipe = pipeIn;
    totalElements_ = tilingData->totalElements;
    sliceCount_ = tilingData->sliceCount;
    blockSize_ = tilingData->blockSize;
    numBlocks_ = tilingData->numBlocks;
    sliceTile_ = tilingData->sliceTileLength;
    tileElements_ = tilingData->tileLength;
    p_ = tilingData->p;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;
    normMode_ = tilingData->normMode;
    coreNum_ = GetBlockNum();

    if (totalElements_ == 0 || sliceCount_ == 0 || blockSize_ == 0 || numBlocks_ == 0 || sliceTile_ == 0 ||
        tileElements_ == 0) {
        return;
    }

    inputGM.SetGlobalBuffer((__gm__ D_T_X*)x, totalElements_);
    outputGM.SetGlobalBuffer((__gm__ D_T_X*)y, totalElements_);
    SetSysWorkspace(workspace);
    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    workspaceGM.SetGlobalBuffer((__gm__ float*)userWs, tilingData->workspaceSize / sizeof(float));

    wsStride_ = (sliceCount_ + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;
    int64_t alignedTile = (tileElements_ + 63) / 64 * 64;
    pipe->InitBuffer(dataBuf, alignedTile * sizeof(D_T_X));
    pipe->InitBuffer(workBuf, alignedTile * sizeof(float));
    pipe->InitBuffer(patternBuf, alignedTile * sizeof(float));
    pipe->InitBuffer(partialBuf, wsStride_ * sizeof(float));
    pipe->InitBuffer(scaleBuf, wsStride_ * sizeof(float));
    pipe->InitBuffer(maskBuf, wsStride_);
}

template <typename D_T_X, bool GENERIC_POSITIVE>
__aicore__ inline void RenormBmCrTiled<D_T_X, GENERIC_POSITIVE>::Process()
{
    if (totalElements_ == 0 || sliceCount_ == 0 || blockSize_ == 0 || numBlocks_ == 0 || sliceTile_ == 0 ||
        tileElements_ == 0 || normMode_ != NORM_MODE_P_POSITIVE) {
        return;
    }

    const int64_t blockIdx = GetBlockIdx();
    const int64_t blockLen = sliceCount_ * blockSize_;
    LocalTensor<D_T_X> dataLocal = dataBuf.Get<D_T_X>();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<uint8_t> patternLocal = patternBuf.Get<uint8_t>();
    LocalTensor<float> scaleTensor = patternBuf.Get<float>();
    LocalTensor<float> partialLocal = partialBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();

    DataCopyExtParams wsParams;
    wsParams.blockCount = 1;
    wsParams.blockLen = static_cast<uint32_t>(wsStride_ * sizeof(float));
    wsParams.srcStride = 0;
    wsParams.dstStride = 0;

    if (blockIdx == 0) {
        Duplicate(partialLocal, 0.0f, static_cast<int32_t>(wsStride_));
        PipeBarrier<PIPE_V>();
        TEventID clearReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(clearReady);
        WaitFlag<HardEvent::V_MTE3>(clearReady);
        DataCopyPad(workspaceGM[0], partialLocal, wsParams);
        TEventID clearDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(clearDone);
        WaitFlag<HardEvent::MTE3_MTE2>(clearDone);
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(workspaceGM);
    }
    SyncAll();

    if constexpr (GENERIC_POSITIVE) {
        Duplicate(partialLocal, 0.0f, static_cast<int32_t>(wsStride_));
        PipeBarrier<PIPE_V>();
    }

    DataCopyPadExtParams<D_T_X> loadPad = {false, 0, 0, 0};
    for (int64_t b = blockIdx; b < numBlocks_; b += coreNum_) {
        for (int64_t sliceStart = 0; sliceStart < sliceCount_; sliceStart += sliceTile_) {
            int64_t currentSlices = sliceTile_;
            if (currentSlices > sliceCount_ - sliceStart) {
                currentSlices = sliceCount_ - sliceStart;
            }
            int64_t currentElements = currentSlices * blockSize_;
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(currentElements * sizeof(D_T_X));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            int64_t gmOffset = b * blockLen + sliceStart * blockSize_;
            DataCopyPad(dataLocal, inputGM[gmOffset], copyParams, loadPad);
            TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(loadDone);
            WaitFlag<HardEvent::MTE2_V>(loadDone);

            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(workLocal, dataLocal, static_cast<int32_t>(currentElements));
            } else {
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(currentElements));
            }
            PipeBarrier<PIPE_V>();
            if constexpr (GENERIC_POSITIVE) {
                Abs(workLocal, workLocal, static_cast<int32_t>(currentElements));
                PipeBarrier<PIPE_V>();
                if (p_ == 2.0f) {
                    Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(currentElements));
                    PipeBarrier<PIPE_V>();
                } else if (p_ != 1.0f) {
                    Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(currentElements));
                    PipeBarrier<PIPE_V>();
                    Log(workLocal, workLocal, static_cast<int32_t>(currentElements));
                    PipeBarrier<PIPE_V>();
                    Muls(workLocal, workLocal, p_, static_cast<int32_t>(currentElements));
                    PipeBarrier<PIPE_V>();
                    Exp(workLocal, workLocal, static_cast<int32_t>(currentElements));
                    PipeBarrier<PIPE_V>();
                }
            } else {
                Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(currentElements));
                PipeBarrier<PIPE_V>();
            }

            uint32_t reduceShape[2] = {static_cast<uint32_t>(currentSlices), static_cast<uint32_t>(blockSize_)};
            LocalTensor<float> tileReduceLocal = GENERIC_POSITIVE ? scaleLocal : partialLocal;
            ReduceSum<float, Pattern::Reduce::AR, false>(tileReduceLocal, workLocal, patternLocal, reduceShape, false);
            PipeBarrier<PIPE_V>();

            if constexpr (GENERIC_POSITIVE) {
                Add(partialLocal[sliceStart], partialLocal[sliceStart], tileReduceLocal,
                    static_cast<int32_t>(currentSlices));
                PipeBarrier<PIPE_V>();
                TEventID tileReusable = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                SetFlag<HardEvent::V_MTE2>(tileReusable);
                WaitFlag<HardEvent::V_MTE2>(tileReusable);
            } else {
                DataCopyExtParams atomicParams;
                atomicParams.blockCount = 1;
                atomicParams.blockLen = static_cast<uint32_t>(currentSlices * sizeof(float));
                atomicParams.srcStride = 0;
                atomicParams.dstStride = 0;
                TEventID partialReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                SetFlag<HardEvent::V_MTE3>(partialReady);
                WaitFlag<HardEvent::V_MTE3>(partialReady);
                SetAtomicAdd<float>();
                DataCopyPad(workspaceGM[sliceStart], partialLocal, atomicParams);
                SetAtomicNone();
                TEventID partialStored = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
                SetFlag<HardEvent::MTE3_V>(partialStored);
                WaitFlag<HardEvent::MTE3_V>(partialStored);
            }
        }
    }
    if constexpr (GENERIC_POSITIVE) {
        TEventID partialReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(partialReady);
        WaitFlag<HardEvent::V_MTE3>(partialReady);
        SetAtomicAdd<float>();
        DataCopyPad(workspaceGM[0], partialLocal, wsParams);
        SetAtomicNone();
        TEventID partialStored = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(partialStored);
        WaitFlag<HardEvent::MTE3_MTE2>(partialStored);
    }
    SyncAll();

    DataCopyPadExtParams<float> wsReadPad = {false, 0, 0, 0.0f};
    DataCopyPad(partialLocal, workspaceGM[0], wsParams, wsReadPad);
    TEventID normLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
    SetFlag<HardEvent::MTE2_V>(normLoaded);
    WaitFlag<HardEvent::MTE2_V>(normLoaded);

    int64_t alignedSlices = (sliceCount_ + FP32_ALIGN - 1) / FP32_ALIGN * FP32_ALIGN;
    if constexpr (GENERIC_POSITIVE) {
        if (p_ == 2.0f) {
            Sqrt(partialLocal, partialLocal, static_cast<int32_t>(alignedSlices));
            PipeBarrier<PIPE_V>();
        } else if (p_ != 1.0f) {
            Maxs(partialLocal, partialLocal, eps_, static_cast<int32_t>(alignedSlices));
            PipeBarrier<PIPE_V>();
            Log(partialLocal, partialLocal, static_cast<int32_t>(alignedSlices));
            PipeBarrier<PIPE_V>();
            Muls(partialLocal, partialLocal, 1.0f / p_, static_cast<int32_t>(alignedSlices));
            PipeBarrier<PIPE_V>();
            Exp(partialLocal, partialLocal, static_cast<int32_t>(alignedSlices));
            PipeBarrier<PIPE_V>();
        }
    } else {
        Sqrt(partialLocal, partialLocal, static_cast<int32_t>(alignedSlices));
        PipeBarrier<PIPE_V>();
    }
    Duplicate(workLocal, 1.0f, static_cast<int32_t>(alignedSlices));
    Duplicate(scaleTensor, maxNorm_, static_cast<int32_t>(alignedSlices));
    PipeBarrier<PIPE_V>();
    Maxs(scaleLocal, partialLocal, eps_, static_cast<int32_t>(alignedSlices));
    PipeBarrier<PIPE_V>();
    Reciprocal(scaleLocal, scaleLocal, static_cast<int32_t>(alignedSlices));
    PipeBarrier<PIPE_V>();
    Muls(scaleLocal, scaleLocal, maxNorm_, static_cast<int32_t>(alignedSlices));
    PipeBarrier<PIPE_V>();
    Compare(maskLocal, partialLocal, scaleTensor, CMPMODE::GT, static_cast<int32_t>(alignedSlices));
    PipeBarrier<PIPE_V>();
    Select(scaleLocal, maskLocal, scaleLocal, workLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
           static_cast<int32_t>(alignedSlices));
    PipeBarrier<PIPE_V>();

    for (int64_t b = blockIdx; b < numBlocks_; b += coreNum_) {
        for (int64_t sliceStart = 0; sliceStart < sliceCount_; sliceStart += sliceTile_) {
            int64_t currentSlices = sliceTile_;
            if (currentSlices > sliceCount_ - sliceStart) {
                currentSlices = sliceCount_ - sliceStart;
            }
            int64_t currentElements = currentSlices * blockSize_;
            uint32_t dstShape[2] = {static_cast<uint32_t>(currentSlices), static_cast<uint32_t>(blockSize_)};
            uint32_t srcShape[2] = {static_cast<uint32_t>(currentSlices), 1};
            BroadCast<float, 2, 1>(scaleTensor, scaleLocal[sliceStart], dstShape, srcShape);
            PipeBarrier<PIPE_V>();

            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(currentElements * sizeof(D_T_X));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            int64_t gmOffset = b * blockLen + sliceStart * blockSize_;
            DataCopyPad(dataLocal, inputGM[gmOffset], copyParams, loadPad);
            TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(loadDone);
            WaitFlag<HardEvent::MTE2_V>(loadDone);

            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(workLocal, dataLocal, static_cast<int32_t>(currentElements));
            } else {
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(currentElements));
            }
            PipeBarrier<PIPE_V>();
            Mul(workLocal, workLocal, scaleTensor, static_cast<int32_t>(currentElements));
            PipeBarrier<PIPE_V>();
            NsRenorm::CastBackToDtype<D_T_X>(dataLocal, workLocal, currentElements);
            PipeBarrier<PIPE_V>();
            TEventID storeReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(storeReady);
            WaitFlag<HardEvent::V_MTE3>(storeReady);
            DataCopyPad(outputGM[gmOffset], dataLocal, copyParams);
            TEventID storeDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(storeDone);
            WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
        }
    }
}

} // namespace NsRenormBmCrTiled

#endif // _RENORM_BM_CR_TILED_H_
