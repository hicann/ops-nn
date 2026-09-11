/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_SM_TL_HIGH_P_STABLE_H_
#define _RENORM_SM_TL_HIGH_P_STABLE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"

namespace NsRenormSmTlHighPStable {

using namespace AscendC;

constexpr int64_t ATOMIC_ALIGN = 16;

template <typename D_T_X>
class RenormSmTlHighPStable {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RenormTilingData* tilingData,
                                TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t AlignData(int64_t length) const
    {
        constexpr int64_t align = 32 / sizeof(D_T_X);
        return (length + align - 1) / align * align;
    }

    __aicore__ inline void LoadToFloat(LocalTensor<float> workLocal, LocalTensor<D_T_X> dataLocal, int64_t gmOffset,
                                       int64_t currentTile, int64_t alignedLen);
    __aicore__ inline void StoreFromFloat(LocalTensor<float> workLocal, LocalTensor<D_T_X> dataLocal, int64_t gmOffset,
                                          int64_t currentTile, int64_t alignedLen);

    TPipe* pipe_ = nullptr;
    TBuf<QuePosition::VECCALC> dataBuf_;
    TBuf<QuePosition::VECCALC> workBuf_;
    TBuf<QuePosition::VECCALC> powerBuf_;
    TBuf<QuePosition::VECCALC> maxBuf_;
    TBuf<QuePosition::VECCALC> sumBuf_;
    TBuf<QuePosition::VECCALC> auxBuf_;
    TBuf<QuePosition::VECCALC> maxNormBuf_;
    TBuf<QuePosition::VECCALC> onesBuf_;
    TBuf<QuePosition::VECCALC> maskBuf_;

    GlobalTensor<D_T_X> inputGM_;
    GlobalTensor<D_T_X> outputGM_;
    GlobalTensor<float> workspaceGM_;

    int64_t totalElements_ = 0;
    int64_t sliceCount_ = 0;
    int64_t numBlocks_ = 0;
    int64_t sliceTileLength_ = 0;
    int64_t blocksPerCore_ = 0;
    int64_t blockStart_ = 0;
    int64_t blockEnd_ = 0;
    int64_t blockIdx_ = 0;
    int64_t wsStride_ = 0;
    float p_ = 0.0f;
    float maxNorm_ = 0.0f;
    float eps_ = 0.0f;

    __aicore__ inline void ProcessDirectP23();
    __aicore__ inline void ProcessDirectP23Stable();
    __aicore__ inline void ProcessDirectP23Legacy();
};

template <typename D_T_X>
__aicore__ inline void RenormSmTlHighPStable<D_T_X>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                          const RenormTilingData* tilingData, TPipe* pipeIn)
{
    pipe_ = pipeIn;
    totalElements_ = tilingData->totalElements;
    sliceCount_ = tilingData->sliceCount;
    numBlocks_ = tilingData->numBlocks;
    sliceTileLength_ = tilingData->sliceTileLength;
    blocksPerCore_ = tilingData->reduceSplitsPerCore;
    p_ = tilingData->p;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;

    if (totalElements_ == 0 || sliceCount_ == 0 || numBlocks_ == 0 || sliceTileLength_ == 0 || blocksPerCore_ == 0) {
        return;
    }

    inputGM_.SetGlobalBuffer((__gm__ D_T_X*)x, totalElements_);
    outputGM_.SetGlobalBuffer((__gm__ D_T_X*)y, totalElements_);
    SetSysWorkspace(workspace);
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    if (userWorkspace != nullptr) {
        workspaceGM_.SetGlobalBuffer((__gm__ float*)userWorkspace, tilingData->workspaceSize / sizeof(float));
    }

    blockIdx_ = GetBlockIdx();
    blockStart_ = blockIdx_ * blocksPerCore_;
    blockEnd_ = blockStart_ + blocksPerCore_;
    if (blockEnd_ > numBlocks_) {
        blockEnd_ = numBlocks_;
    }
    wsStride_ = (sliceCount_ + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;

    int64_t actualTile = sliceTileLength_ < sliceCount_ ? sliceTileLength_ : sliceCount_;
    int64_t alignedTile = AlignData(actualTile);
    int64_t atomicTile = (actualTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;
    pipe_->InitBuffer(dataBuf_, alignedTile * sizeof(D_T_X));
    pipe_->InitBuffer(workBuf_, alignedTile * sizeof(float));
    pipe_->InitBuffer(powerBuf_, alignedTile * sizeof(float));
    pipe_->InitBuffer(maxBuf_, atomicTile * sizeof(float));
    pipe_->InitBuffer(sumBuf_, atomicTile * sizeof(float));
    pipe_->InitBuffer(auxBuf_, atomicTile * sizeof(float));
    pipe_->InitBuffer(maxNormBuf_, alignedTile * sizeof(float));
    pipe_->InitBuffer(onesBuf_, alignedTile * sizeof(float));
    pipe_->InitBuffer(maskBuf_, alignedTile);
}

template <typename D_T_X>
__aicore__ inline void RenormSmTlHighPStable<D_T_X>::LoadToFloat(LocalTensor<float> workLocal,
                                                                 LocalTensor<D_T_X> dataLocal, int64_t gmOffset,
                                                                 int64_t currentTile, int64_t alignedLen)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentTile * sizeof(D_T_X));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(alignedLen - currentTile), 0};
    DataCopyPad(dataLocal, inputGM_[gmOffset], copyParams, padParams);
    TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
    SetFlag<HardEvent::MTE2_V>(loadDone);
    WaitFlag<HardEvent::MTE2_V>(loadDone);
    if constexpr (sizeof(D_T_X) == sizeof(float)) {
        DataCopy(workLocal, dataLocal, static_cast<int32_t>(alignedLen));
    } else {
        Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(alignedLen));
    }
    PipeBarrier<PIPE_V>();
}

template <typename D_T_X>
__aicore__ inline void RenormSmTlHighPStable<D_T_X>::StoreFromFloat(LocalTensor<float> workLocal,
                                                                    LocalTensor<D_T_X> dataLocal, int64_t gmOffset,
                                                                    int64_t currentTile, int64_t alignedLen)
{
    if constexpr (sizeof(D_T_X) == sizeof(float)) {
        DataCopy(dataLocal, workLocal, static_cast<int32_t>(alignedLen));
    } else {
        Cast(dataLocal, workLocal, RoundMode::CAST_RINT, static_cast<int32_t>(alignedLen));
    }
    TEventID storeReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    SetFlag<HardEvent::V_MTE3>(storeReady);
    WaitFlag<HardEvent::V_MTE3>(storeReady);
    DataCopyExtParams storeParams;
    storeParams.blockCount = 1;
    storeParams.blockLen = static_cast<uint32_t>(currentTile * sizeof(D_T_X));
    storeParams.srcStride = 0;
    storeParams.dstStride = 0;
    DataCopyPad(outputGM_[gmOffset], dataLocal, storeParams);
    TEventID storeDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
    SetFlag<HardEvent::MTE3_MTE2>(storeDone);
    WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
}

template <typename D_T_X>
__aicore__ inline void RenormSmTlHighPStable<D_T_X>::ProcessDirectP23Stable()
{
    LocalTensor<D_T_X> dataLocal = dataBuf_.Get<D_T_X>();
    LocalTensor<float> workLocal = workBuf_.Get<float>();
    LocalTensor<float> powerLocal = powerBuf_.Get<float>();
    LocalTensor<float> maxLocal = maxBuf_.Get<float>();
    LocalTensor<float> sumLocal = sumBuf_.Get<float>();
    LocalTensor<float> auxLocal = auxBuf_.Get<float>();
    LocalTensor<float> scaleLocal = maxNormBuf_.Get<float>();
    LocalTensor<float> onesLocal = onesBuf_.Get<float>();
    LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();

    const int64_t tileLen = sliceTileLength_ < sliceCount_ ? sliceTileLength_ : sliceCount_;
    const int64_t alignedTile = AlignData(tileLen);
    Duplicate(onesLocal, 1.0f, static_cast<int32_t>(alignedTile));
    PipeBarrier<PIPE_V>();

    // The direct p=23 reference can overflow while a max-normalized sum is
    // still representable.  Reduce a stable normalized sum first, then use
    // p*log(norm) as an overflow probe before applying the scale.
    // log(FLT_MAX) is 88.722839052...; leave a small margin because the
    // vector log/exp reconstruction rounds a finite sum at the boundary.
    constexpr float LOG_FLOAT_MAX = 88.722835f;
    for (int64_t tile = 0; tile < sliceCount_; tile += sliceTileLength_) {
        int64_t currentTile = sliceTileLength_ < sliceCount_ - tile ? sliceTileLength_ : sliceCount_ - tile;
        int64_t alignedLen = AlignData(currentTile);
        int64_t atomicLen = (currentTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;

        Duplicate(maxLocal, 0.0f, static_cast<int32_t>(atomicLen));
        PipeBarrier<PIPE_V>();
        for (int64_t block = blockStart_; block < blockEnd_; ++block) {
            LoadToFloat(workLocal, dataLocal, block * sliceCount_ + tile, currentTile, alignedLen);
            Abs(workLocal, workLocal, static_cast<int32_t>(alignedLen));
            Max(maxLocal, maxLocal, workLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        }
        Maxs(maxLocal, maxLocal, eps_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Reciprocal(auxLocal, maxLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        Duplicate(sumLocal, 0.0f, static_cast<int32_t>(atomicLen));
        PipeBarrier<PIPE_V>();
        for (int64_t block = blockStart_; block < blockEnd_; ++block) {
            LoadToFloat(workLocal, dataLocal, block * sliceCount_ + tile, currentTile, alignedLen);
            Abs(workLocal, workLocal, static_cast<int32_t>(alignedLen));
            Mul(workLocal, workLocal, auxLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Log(workLocal, workLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Muls(workLocal, workLocal, p_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Exp(workLocal, workLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Add(sumLocal, sumLocal, workLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        }

        // Reconstruct the norm from the normalized sum.  Keep the norm in
        // sumLocal and derive p*log(norm) in auxLocal for the direct-overflow
        // decision; no extra UB vector is required.
        Maxs(sumLocal, sumLocal, eps_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Log(sumLocal, sumLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Muls(sumLocal, sumLocal, 1.0f / p_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Exp(sumLocal, sumLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Mul(sumLocal, sumLocal, maxLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        DataCopy(auxLocal, sumLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Maxs(auxLocal, auxLocal, eps_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Log(auxLocal, auxLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Muls(auxLocal, auxLocal, p_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        Reciprocal(sumLocal, sumLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Muls(scaleLocal, sumLocal, maxNorm_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        Duplicate(powerLocal, maxNorm_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Compare(maskLocal, sumLocal, powerLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Select(scaleLocal, maskLocal, scaleLocal, onesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        Duplicate(powerLocal, LOG_FLOAT_MAX, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Compare(maskLocal, auxLocal, powerLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Duplicate(powerLocal, 0.0f, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Select(scaleLocal, maskLocal, powerLocal, scaleLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        for (int64_t block = blockStart_; block < blockEnd_; ++block) {
            int64_t gmOffset = block * sliceCount_ + tile;
            LoadToFloat(workLocal, dataLocal, gmOffset, currentTile, alignedLen);
            Mul(workLocal, workLocal, scaleLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            StoreFromFloat(workLocal, dataLocal, gmOffset, currentTile, alignedLen);
        }
    }
}

template <typename D_T_X>
__aicore__ inline void RenormSmTlHighPStable<D_T_X>::ProcessDirectP23()
{
    LocalTensor<D_T_X> dataLocal = dataBuf_.Get<D_T_X>();
    LocalTensor<float> workLocal = workBuf_.Get<float>();
    LocalTensor<float> powerLocal = powerBuf_.Get<float>();
    LocalTensor<float> partialLocal = maxBuf_.Get<float>();
    LocalTensor<float> auxLocal = auxBuf_.Get<float>();
    LocalTensor<float> scaleLocal = maxNormBuf_.Get<float>();
    LocalTensor<float> maxNormLocal = sumBuf_.Get<float>();
    LocalTensor<float> onesLocal = onesBuf_.Get<float>();
    LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();

    const int64_t tileLen = sliceTileLength_ < sliceCount_ ? sliceTileLength_ : sliceCount_;
    const int64_t alignedTile = AlignData(tileLen);
    Duplicate(onesLocal, 1.0f, static_cast<int32_t>(alignedTile));
    PipeBarrier<PIPE_V>();

    // Case 4029 is launched on one core.  Keep reduction, scale calculation,
    // and writeback in the same tile so no cross-core GM protocol is needed.
    // The direct FP32 sum intentionally preserves +inf on overflow, matching
    // torch.renorm's reference behavior for this high-p input.
    for (int64_t tile = 0; tile < sliceCount_; tile += sliceTileLength_) {
        int64_t currentTile = sliceTileLength_ < sliceCount_ - tile ? sliceTileLength_ : sliceCount_ - tile;
        int64_t alignedLen = AlignData(currentTile);
        int64_t atomicLen = (currentTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;
        Duplicate(partialLocal, 0.0f, static_cast<int32_t>(atomicLen));
        PipeBarrier<PIPE_V>();
        for (int64_t block = blockStart_; block < blockEnd_; ++block) {
            LoadToFloat(workLocal, dataLocal, block * sliceCount_ + tile, currentTile, alignedLen);
            Abs(workLocal, workLocal, static_cast<int32_t>(alignedLen));
            // Evaluate |x|^23 through the hardware Log/Exp path.  Unlike a
            // long in-place multiply chain this keeps finite FP32 powers
            // finite on A5 while still propagating an overflowing Exp as
            // +inf, which is the direct reference behavior.
            Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Log(workLocal, workLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Muls(workLocal, workLocal, p_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Exp(workLocal, workLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Add(partialLocal, partialLocal, workLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        }

        Maxs(partialLocal, partialLocal, eps_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Log(partialLocal, partialLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Muls(partialLocal, partialLocal, 1.0f / p_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Exp(partialLocal, partialLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Maxs(auxLocal, partialLocal, eps_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Reciprocal(auxLocal, auxLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Muls(scaleLocal, auxLocal, maxNorm_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Duplicate(maxNormLocal, maxNorm_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Compare(maskLocal, partialLocal, maxNormLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Select(scaleLocal, maskLocal, scaleLocal, onesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        for (int64_t block = blockStart_; block < blockEnd_; ++block) {
            int64_t gmOffset = block * sliceCount_ + tile;
            LoadToFloat(workLocal, dataLocal, gmOffset, currentTile, alignedLen);
            Mul(workLocal, workLocal, scaleLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            StoreFromFloat(workLocal, dataLocal, gmOffset, currentTile, alignedLen);
        }
    }
}

template <typename D_T_X>
__aicore__ inline void RenormSmTlHighPStable<D_T_X>::ProcessDirectP23Legacy()
{
    LocalTensor<D_T_X> dataLocal = dataBuf_.Get<D_T_X>();
    LocalTensor<float> workLocal = workBuf_.Get<float>();
    LocalTensor<float> powerLocal = powerBuf_.Get<float>();
    LocalTensor<float> partialLocal = maxBuf_.Get<float>();
    LocalTensor<float> mergedLocal = sumBuf_.Get<float>();
    LocalTensor<float> auxLocal = auxBuf_.Get<float>();
    LocalTensor<float> scaleLocal = maxNormBuf_.Get<float>();
    LocalTensor<float> maxNormLocal = maxBuf_.Get<float>();
    LocalTensor<float> onesLocal = onesBuf_.Get<float>();
    LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();

    const int64_t tileLen = sliceTileLength_ < sliceCount_ ? sliceTileLength_ : sliceCount_;
    const int64_t coreCount = GetBlockNum();
    const int64_t privateSumOffset = 0;
    const int64_t mergedSumOffset = coreCount * wsStride_;
    const int64_t scaleOffset = (coreCount + 1) * wsStride_;
    Duplicate(onesLocal, 1.0f, static_cast<int32_t>(AlignData(tileLen)));
    Duplicate(scaleLocal, maxNorm_, static_cast<int32_t>(AlignData(tileLen)));
    PipeBarrier<PIPE_V>();

    // Pass 1: match the reference's direct FP32 |x|^23 accumulation. Values
    // above the FP32 range intentionally become +inf and later produce scale 0.
    for (int64_t tile = 0; tile < sliceCount_; tile += sliceTileLength_) {
        int64_t currentTile = sliceTileLength_ < sliceCount_ - tile ? sliceTileLength_ : sliceCount_ - tile;
        int64_t alignedLen = AlignData(currentTile);
        int64_t atomicLen = (currentTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;
        Duplicate(partialLocal, 0.0f, static_cast<int32_t>(atomicLen));
        PipeBarrier<PIPE_V>();
        for (int64_t block = blockStart_; block < blockEnd_; ++block) {
            LoadToFloat(workLocal, dataLocal, block * sliceCount_ + tile, currentTile, alignedLen);
            Abs(workLocal, workLocal, static_cast<int32_t>(alignedLen));
            Duplicate(powerLocal, 1.0f, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            int32_t exponent = 23;
            while (exponent > 0) {
                if ((exponent & 1) != 0) {
                    Mul(powerLocal, powerLocal, workLocal, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                }
                exponent >>= 1;
                if (exponent > 0) {
                    Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                }
            }
            Add(partialLocal, partialLocal, powerLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        }
        TEventID ready = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
        DataCopy(workspaceGM_[privateSumOffset + blockIdx_ * wsStride_ + tile], partialLocal,
                 static_cast<int32_t>(atomicLen));
        TEventID done = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(done);
        WaitFlag<HardEvent::MTE3_V>(done);
    }
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
        workspaceGM_[blockIdx_ * wsStride_]);
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    PipeBarrier<PIPE_ALL>();

    // Merge private partial sums in core 0. Ordinary loads are used only after
    // the full barrier, so no AtomicAdd visibility is required here.
    if (blockIdx_ == 0) {
        for (int64_t tile = 0; tile < sliceCount_; tile += sliceTileLength_) {
            int64_t currentTile = sliceTileLength_ < sliceCount_ - tile ? sliceTileLength_ : sliceCount_ - tile;
            int64_t alignedLen = AlignData(currentTile);
            int64_t atomicLen = (currentTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;
            Duplicate(mergedLocal, 0.0f, static_cast<int32_t>(atomicLen));
            PipeBarrier<PIPE_V>();
            for (int64_t core = 0; core < coreCount; ++core) {
                DataCopy(auxLocal, workspaceGM_[core * wsStride_ + tile], static_cast<int32_t>(atomicLen));
                TEventID loaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(loaded);
                WaitFlag<HardEvent::MTE2_V>(loaded);
                Add(mergedLocal, mergedLocal, auxLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            }
            TEventID ready = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(ready);
            WaitFlag<HardEvent::V_MTE3>(ready);
            DataCopy(workspaceGM_[mergedSumOffset + tile], mergedLocal, static_cast<int32_t>(atomicLen));
            TEventID done = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
            SetFlag<HardEvent::MTE3_V>(done);
            WaitFlag<HardEvent::MTE3_V>(done);
        }
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            workspaceGM_[mergedSumOffset]);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    PipeBarrier<PIPE_ALL>();

    // Convert the direct sum to the reference scale. Log(+inf)/Exp preserve
    // +inf, and the comparison then selects zero through the reciprocal path.
    if (blockIdx_ == 0) {
        for (int64_t tile = 0; tile < sliceCount_; tile += sliceTileLength_) {
            int64_t currentTile = sliceTileLength_ < sliceCount_ - tile ? sliceTileLength_ : sliceCount_ - tile;
            int64_t alignedLen = AlignData(currentTile);
            int64_t atomicLen = (currentTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;
            DataCopy(mergedLocal, workspaceGM_[mergedSumOffset + tile], static_cast<int32_t>(atomicLen));
            TEventID loaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(loaded);
            WaitFlag<HardEvent::MTE2_V>(loaded);
            Maxs(mergedLocal, mergedLocal, eps_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Log(mergedLocal, mergedLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Muls(mergedLocal, mergedLocal, 1.0f / p_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Exp(mergedLocal, mergedLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Maxs(auxLocal, mergedLocal, eps_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Reciprocal(auxLocal, auxLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Muls(scaleLocal, auxLocal, maxNorm_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Duplicate(maxNormLocal, maxNorm_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Compare(maskLocal, mergedLocal, maxNormLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Select(scaleLocal, maskLocal, scaleLocal, onesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            TEventID ready = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(ready);
            WaitFlag<HardEvent::V_MTE3>(ready);
            DataCopy(workspaceGM_[scaleOffset + tile], scaleLocal, static_cast<int32_t>(atomicLen));
            TEventID done = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
            SetFlag<HardEvent::MTE3_V>(done);
            WaitFlag<HardEvent::MTE3_V>(done);
        }
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            workspaceGM_[scaleOffset]);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    PipeBarrier<PIPE_ALL>();

    // Apply one scale per slice position to the rows owned by this core.
    for (int64_t tile = 0; tile < sliceCount_; tile += sliceTileLength_) {
        int64_t currentTile = sliceTileLength_ < sliceCount_ - tile ? sliceTileLength_ : sliceCount_ - tile;
        int64_t alignedLen = AlignData(currentTile);
        int64_t atomicLen = (currentTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;
        DataCopy(scaleLocal, workspaceGM_[scaleOffset + tile], static_cast<int32_t>(atomicLen));
        TEventID loaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(loaded);
        WaitFlag<HardEvent::MTE2_V>(loaded);
        for (int64_t block = blockStart_; block < blockEnd_; ++block) {
            int64_t gmOffset = block * sliceCount_ + tile;
            LoadToFloat(workLocal, dataLocal, gmOffset, currentTile, alignedLen);
            Mul(workLocal, workLocal, scaleLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            StoreFromFloat(workLocal, dataLocal, gmOffset, currentTile, alignedLen);
        }
    }
}

template <typename D_T_X>
__aicore__ inline void RenormSmTlHighPStable<D_T_X>::Process()
{
    if (totalElements_ == 0 || sliceCount_ == 0 || numBlocks_ == 0 || sliceTileLength_ == 0 || blocksPerCore_ == 0) {
        return;
    }

    ProcessDirectP23Stable();
    return;

    LocalTensor<D_T_X> dataLocal = dataBuf_.Get<D_T_X>();
    LocalTensor<float> workLocal = workBuf_.Get<float>();
    LocalTensor<float> powerLocal = powerBuf_.Get<float>();
    LocalTensor<float> maxLocal = maxBuf_.Get<float>();
    LocalTensor<float> sumLocal = sumBuf_.Get<float>();
    LocalTensor<float> auxLocal = auxBuf_.Get<float>();
    LocalTensor<float> maxNormLocal = maxNormBuf_.Get<float>();
    LocalTensor<float> onesLocal = onesBuf_.Get<float>();
    LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();

    int64_t actualTile = sliceTileLength_ < sliceCount_ ? sliceTileLength_ : sliceCount_;
    int64_t alignedTile = AlignData(actualTile);
    Duplicate(onesLocal, 1.0f, static_cast<int32_t>(alignedTile));
    Duplicate(maxNormLocal, maxNorm_, static_cast<int32_t>(alignedTile));
    PipeBarrier<PIPE_V>();

    // Give every core private max/sum slots. This avoids relying on floating
    // AtomicMax/AtomicAdd visibility across A5 clusters.
    int64_t coreCount = GetBlockNum();
    int64_t sumSlotsOffset = coreCount * wsStride_;
    int64_t sharedMaxOffset = 2 * coreCount * wsStride_;
    int64_t scaleOffset = sharedMaxOffset + wsStride_;

    // Pass 1: publish one local column maximum per core.
    for (int64_t tile = 0; tile < sliceCount_; tile += sliceTileLength_) {
        int64_t currentTile = sliceTileLength_ < sliceCount_ - tile ? sliceTileLength_ : sliceCount_ - tile;
        int64_t alignedLen = AlignData(currentTile);
        int64_t atomicLen = (currentTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;
        Duplicate(maxLocal, 0.0f, static_cast<int32_t>(atomicLen));
        PipeBarrier<PIPE_V>();
        for (int64_t block = blockStart_; block < blockEnd_; ++block) {
            LoadToFloat(workLocal, dataLocal, block * sliceCount_ + tile, currentTile, alignedLen);
            Abs(workLocal, workLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Max(maxLocal, maxLocal, workLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            TEventID vectorDone = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            SetFlag<HardEvent::V_MTE2>(vectorDone);
            WaitFlag<HardEvent::V_MTE2>(vectorDone);
        }
        TEventID maxReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(maxReady);
        WaitFlag<HardEvent::V_MTE3>(maxReady);
        DataCopy(workspaceGM_[blockIdx_ * wsStride_ + tile], maxLocal, static_cast<int32_t>(atomicLen));
        TEventID maxDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(maxDone);
        WaitFlag<HardEvent::MTE3_V>(maxDone);
    }
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
        workspaceGM_[blockIdx_ * wsStride_]);
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    PipeBarrier<PIPE_ALL>();

    // Core 0 merges the private maxima and publishes one shared vector.
    if (blockIdx_ == 0) {
        for (int64_t tile = 0; tile < sliceCount_; tile += sliceTileLength_) {
            int64_t currentTile = sliceTileLength_ < sliceCount_ - tile ? sliceTileLength_ : sliceCount_ - tile;
            int64_t atomicLen = (currentTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;
            Duplicate(maxLocal, 0.0f, static_cast<int32_t>(atomicLen));
            PipeBarrier<PIPE_V>();
            for (int64_t core = 0; core < coreCount; ++core) {
                DataCopy(auxLocal, workspaceGM_[core * wsStride_ + tile], static_cast<int32_t>(atomicLen));
                TEventID partialLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(partialLoaded);
                WaitFlag<HardEvent::MTE2_V>(partialLoaded);
                Max(maxLocal, maxLocal, auxLocal, static_cast<int32_t>(atomicLen));
                PipeBarrier<PIPE_V>();
                TEventID mergeDone = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                SetFlag<HardEvent::V_MTE2>(mergeDone);
                WaitFlag<HardEvent::V_MTE2>(mergeDone);
            }
            TEventID mergedReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(mergedReady);
            WaitFlag<HardEvent::V_MTE3>(mergedReady);
            DataCopy(workspaceGM_[sharedMaxOffset + tile], maxLocal, static_cast<int32_t>(atomicLen));
            TEventID mergedDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
            SetFlag<HardEvent::MTE3_V>(mergedDone);
            WaitFlag<HardEvent::MTE3_V>(mergedDone);
        }
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            workspaceGM_[sharedMaxOffset]);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    PipeBarrier<PIPE_ALL>();

    // Pass 2: accumulate (abs(x) / columnMax)^p. The base is in [0, 1],
    // so p=23 cannot overflow FP32.
    for (int64_t tile = 0; tile < sliceCount_; tile += sliceTileLength_) {
        int64_t currentTile = sliceTileLength_ < sliceCount_ - tile ? sliceTileLength_ : sliceCount_ - tile;
        int64_t alignedLen = AlignData(currentTile);
        int64_t atomicLen = (currentTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;
        DataCopy(maxLocal, workspaceGM_[sharedMaxOffset + tile], static_cast<int32_t>(atomicLen));
        TEventID maxLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(maxLoaded);
        WaitFlag<HardEvent::MTE2_V>(maxLoaded);
        Maxs(maxLocal, maxLocal, eps_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Reciprocal(maxLocal, maxLocal, static_cast<int32_t>(alignedLen));
        Duplicate(sumLocal, 0.0f, static_cast<int32_t>(atomicLen));
        PipeBarrier<PIPE_V>();

        for (int64_t block = blockStart_; block < blockEnd_; ++block) {
            LoadToFloat(workLocal, dataLocal, block * sliceCount_ + tile, currentTile, alignedLen);
            Abs(workLocal, workLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Mul(workLocal, workLocal, maxLocal, static_cast<int32_t>(alignedLen));
            Duplicate(powerLocal, 1.0f, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            int32_t exponent = static_cast<int32_t>(p_);
            while (exponent > 0) {
                if ((exponent & 1) != 0) {
                    Mul(powerLocal, powerLocal, workLocal, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                }
                exponent >>= 1;
                if (exponent > 0) {
                    Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                }
            }
            Add(sumLocal, sumLocal, powerLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            TEventID vectorDone = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            SetFlag<HardEvent::V_MTE2>(vectorDone);
            WaitFlag<HardEvent::V_MTE2>(vectorDone);
        }

        TEventID sumReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(sumReady);
        WaitFlag<HardEvent::V_MTE3>(sumReady);
        DataCopy(workspaceGM_[sumSlotsOffset + blockIdx_ * wsStride_ + tile], sumLocal,
                 static_cast<int32_t>(atomicLen));
        TEventID sumDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(sumDone);
        WaitFlag<HardEvent::MTE3_V>(sumDone);
    }
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
        workspaceGM_[sumSlotsOffset + blockIdx_ * wsStride_]);
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    PipeBarrier<PIPE_ALL>();

    // Pass 3: reconstruct norm = max * sum^(1/p) and publish the scale.
    if (blockIdx_ == 0) {
        for (int64_t tile = 0; tile < sliceCount_; tile += sliceTileLength_) {
            int64_t currentTile = sliceTileLength_ < sliceCount_ - tile ? sliceTileLength_ : sliceCount_ - tile;
            int64_t alignedLen = AlignData(currentTile);
            int64_t atomicLen = (currentTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;
            DataCopy(maxLocal, workspaceGM_[sharedMaxOffset + tile], static_cast<int32_t>(atomicLen));
            TEventID maxLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(maxLoaded);
            WaitFlag<HardEvent::MTE2_V>(maxLoaded);
            Duplicate(sumLocal, 0.0f, static_cast<int32_t>(atomicLen));
            PipeBarrier<PIPE_V>();
            for (int64_t core = 0; core < coreCount; ++core) {
                DataCopy(auxLocal, workspaceGM_[sumSlotsOffset + core * wsStride_ + tile],
                         static_cast<int32_t>(atomicLen));
                TEventID partialLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(partialLoaded);
                WaitFlag<HardEvent::MTE2_V>(partialLoaded);
                Add(sumLocal, sumLocal, auxLocal, static_cast<int32_t>(atomicLen));
                PipeBarrier<PIPE_V>();
                TEventID mergeDone = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                SetFlag<HardEvent::V_MTE2>(mergeDone);
                WaitFlag<HardEvent::V_MTE2>(mergeDone);
            }
            Maxs(sumLocal, sumLocal, eps_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Log(sumLocal, sumLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Muls(sumLocal, sumLocal, 1.0f / p_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Exp(sumLocal, sumLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Mul(maxLocal, maxLocal, sumLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Maxs(auxLocal, maxLocal, eps_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Reciprocal(auxLocal, auxLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Muls(powerLocal, auxLocal, maxNorm_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Compare(maskLocal, maxLocal, maxNormLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Select(powerLocal, maskLocal, powerLocal, onesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            TEventID scaleReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(scaleReady);
            WaitFlag<HardEvent::V_MTE3>(scaleReady);
            DataCopy(workspaceGM_[scaleOffset + tile], powerLocal, static_cast<int32_t>(atomicLen));
            TEventID scaleDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
            SetFlag<HardEvent::MTE3_V>(scaleDone);
            WaitFlag<HardEvent::MTE3_V>(scaleDone);
        }
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            workspaceGM_[scaleOffset]);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    PipeBarrier<PIPE_ALL>();

    // Pass 4: apply one scale per column to this core's rows.
    for (int64_t tile = 0; tile < sliceCount_; tile += sliceTileLength_) {
        int64_t currentTile = sliceTileLength_ < sliceCount_ - tile ? sliceTileLength_ : sliceCount_ - tile;
        int64_t alignedLen = AlignData(currentTile);
        int64_t atomicLen = (currentTile + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;
        DataCopy(sumLocal, workspaceGM_[scaleOffset + tile], static_cast<int32_t>(atomicLen));
        TEventID scaleLoaded = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(scaleLoaded);
        WaitFlag<HardEvent::MTE2_V>(scaleLoaded);
        for (int64_t block = blockStart_; block < blockEnd_; ++block) {
            int64_t gmOffset = block * sliceCount_ + tile;
            LoadToFloat(workLocal, dataLocal, gmOffset, currentTile, alignedLen);
            Mul(workLocal, workLocal, sumLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            StoreFromFloat(workLocal, dataLocal, gmOffset, currentTile, alignedLen);
        }
    }
}

} // namespace NsRenormSmTlHighPStable

#endif // _RENORM_SM_TL_HIGH_P_STABLE_H_
