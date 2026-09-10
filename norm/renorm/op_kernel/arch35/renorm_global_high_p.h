/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_GLOBAL_HIGH_P_H_
#define _RENORM_GLOBAL_HIGH_P_H_
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"
#include "renorm_tiling_key.h"
#include "common/renorm_common.h"
namespace NsRenormGlobalHighP {
using namespace AscendC;
constexpr int32_t NORM_MODE_P_POSITIVE = 0;
constexpr int32_t NORM_MODE_P_ZERO = 1;
constexpr int32_t NORM_MODE_P_INF = 2;
constexpr int32_t NORM_MODE_MAXNORM_ZERO = 3;
constexpr int64_t CMP_ALIGN = 8;
template <typename D_T_X, bool LARGE_GM_OFFSET = false>
class RenormGlobalHighP {
public:
    __aicore__ inline RenormGlobalHighP() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RenormTilingData* tilingData,
                                TPipe* pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessMultiCore();

private:
    TPipe* pipe_ = nullptr;
    TBuf<QuePosition::VECCALC> dataBuf0;
    TBuf<QuePosition::VECCALC> dataBuf1;
    TBuf<QuePosition::VECCALC> workBuf;
    TBuf<QuePosition::VECCALC> normBuf;
    TBuf<QuePosition::VECCALC> zerosBuf;
    TBuf<QuePosition::VECCALC> onesBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> reduceBuf;
    TBuf<QuePosition::VECCALC> patternTmpBuf;
    TBuf<QuePosition::VECCALC> scaleBuf;
    TBuf<QuePosition::VECCALC> maxNormBuf;
    GlobalTensor<D_T_X> inputGM;
    GlobalTensor<D_T_X> outputGM;
    GlobalTensor<float> workspaceGM;
    int64_t totalElements_ = 0;
    int64_t perCore_ = 0;
    int64_t tileLength_ = 0;
    int64_t blockIdx_ = 0;
    int64_t coreStart_ = 0;
    int64_t coreEnd_ = 0;
    float p_ = 0.0f;
    float maxNorm_ = 0.0f;
    float eps_ = 0.0f;
    int32_t normMode_ = 0;
    bool multiCore_ = false;
};
template <typename D_T_X, bool LARGE_GM_OFFSET>
__aicore__ inline void RenormGlobalHighP<D_T_X, LARGE_GM_OFFSET>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                                       const RenormTilingData* tilingData,
                                                                       TPipe* pipeIn)
{
    pipe_ = pipeIn;
    totalElements_ = tilingData->totalElements;
    perCore_ = tilingData->slicesPerCore;
    tileLength_ = tilingData->tileLength;
    p_ = tilingData->p;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;
    normMode_ = tilingData->normMode;
    multiCore_ = GetBlockNum() > 1;
    if (totalElements_ == 0 || perCore_ == 0 || tileLength_ == 0) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    coreStart_ = blockIdx_ * perCore_;
    coreEnd_ = coreStart_ + perCore_;
    if (coreEnd_ > totalElements_) {
        coreEnd_ = totalElements_;
    }
    if constexpr (LARGE_GM_OFFSET) {
        // Keep every core's view below the A5 DMA address limit.  The
        // logical operation is still over totalElements_; only the local GM
        // base and index range are rebased.
        int64_t localElements = coreEnd_ - coreStart_;
        inputGM.SetGlobalBuffer((__gm__ D_T_X*)x + coreStart_, localElements);
        outputGM.SetGlobalBuffer((__gm__ D_T_X*)y + coreStart_, localElements);
        coreEnd_ = localElements;
        coreStart_ = 0;
    } else {
        inputGM.SetGlobalBuffer((__gm__ D_T_X*)x, totalElements_);
        outputGM.SetGlobalBuffer((__gm__ D_T_X*)y, totalElements_);
    }
    if (multiCore_) {
        SetSysWorkspace(workspace);
        GM_ADDR userWorkspace = GetUserWorkspace(workspace);
        if (userWorkspace == nullptr) {
            return;
        }
        workspaceGM.SetGlobalBuffer((__gm__ float*)userWorkspace, tilingData->workspaceSize / sizeof(float));
    }
    int64_t typeSize = sizeof(D_T_X);
    int64_t alignElems = 32 / typeSize;
    int64_t alignedTile = (tileLength_ + alignElems - 1) / alignElems * alignElems;
    if (alignedTile < CMP_ALIGN) {
        alignedTile = CMP_ALIGN;
    }
    if (multiCore_) {
        // Multi-core reduction only needs one input tile, one FP32 tile, a
        // reduction scratch tile, and two 32B scalar workspace buffers.
        pipe_->InitBuffer(dataBuf0, static_cast<uint32_t>(alignedTile * typeSize));
        pipe_->InitBuffer(workBuf, static_cast<uint32_t>(alignedTile * sizeof(float)));
        pipe_->InitBuffer(reduceBuf, 32);
        pipe_->InitBuffer(patternTmpBuf, static_cast<uint32_t>(alignedTile * sizeof(float)));
        pipe_->InitBuffer(zerosBuf, 32);
        pipe_->InitBuffer(onesBuf, 32);
        pipe_->InitBuffer(scaleBuf, 32);
        return;
    }
    pipe_->InitBuffer(dataBuf0, static_cast<uint32_t>(alignedTile * typeSize));
    pipe_->InitBuffer(dataBuf1, static_cast<uint32_t>(alignedTile * typeSize));
    pipe_->InitBuffer(workBuf, static_cast<uint32_t>(alignedTile * sizeof(float)));
    pipe_->InitBuffer(normBuf, static_cast<uint32_t>(alignedTile * sizeof(float)));
    pipe_->InitBuffer(zerosBuf, static_cast<uint32_t>(alignedTile * sizeof(float)));
    pipe_->InitBuffer(onesBuf, static_cast<uint32_t>(alignedTile * sizeof(float)));
    pipe_->InitBuffer(maskBuf, static_cast<uint32_t>(alignedTile));
    pipe_->InitBuffer(reduceBuf, 32);
    pipe_->InitBuffer(patternTmpBuf, static_cast<uint32_t>(alignedTile * sizeof(float)));
    pipe_->InitBuffer(scaleBuf, static_cast<uint32_t>(alignedTile * sizeof(float)));
    pipe_->InitBuffer(maxNormBuf, static_cast<uint32_t>(alignedTile * sizeof(float)));
}
template <typename D_T_X, bool LARGE_GM_OFFSET>
__aicore__ inline void RenormGlobalHighP<D_T_X, LARGE_GM_OFFSET>::ProcessMultiCore()
{
    LocalTensor<D_T_X> dataLocal = dataBuf0.Get<D_T_X>();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<float> reduceLocal = reduceBuf.Get<float>();
    LocalTensor<uint8_t> patternTmpLocal = patternTmpBuf.Get<uint8_t>();
    LocalTensor<float> zerosLocal = zerosBuf.Get<float>();
    LocalTensor<float> onesLocal = onesBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    int64_t typeSize = sizeof(D_T_X);
    int64_t alignElems = 32 / typeSize;
    int64_t alignedTile = (tileLength_ + alignElems - 1) / alignElems * alignElems;
    if (alignedTile < CMP_ALIGN) {
        alignedTile = CMP_ALIGN;
    }
    Duplicate(zerosLocal, 0.0f, 8);
    Duplicate(onesLocal, 1.0f, 8);
    PipeBarrier<PIPE_V>();
    int64_t coreStart = coreStart_;
    int64_t coreEnd = coreEnd_;
    if constexpr (!LARGE_GM_OFFSET) {
        coreStart = blockIdx_ * perCore_;
        coreEnd = coreStart + perCore_;
        if (coreEnd > totalElements_) {
            coreEnd = totalElements_;
        }
    }
    if (coreStart >= coreEnd) {
        return;
    }

    // Initialize the reduction slot before any core starts its local scan.
    if (blockIdx_ == 0) {
        Duplicate(reduceLocal, 0.0f, 8);
        PipeBarrier<PIPE_V>();
        TEventID initReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(initReady);
        WaitFlag<HardEvent::V_MTE3>(initReady);
        DataCopy(workspaceGM[0], reduceLocal, 8);
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(workspaceGM[0]);
        TEventID initDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(initDone);
        WaitFlag<HardEvent::MTE3_V>(initDone);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    PipeBarrier<PIPE_ALL>();

    float localReduce = 0.0f;
    for (int64_t off = coreStart; off < coreEnd; off += tileLength_) {
        int64_t cur = (tileLength_ < (coreEnd - off)) ? tileLength_ : (coreEnd - off);
        int64_t al = (cur + alignElems - 1) / alignElems * alignElems;
        uint32_t shape[2] = {1, static_cast<uint32_t>(al)};
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(cur * typeSize);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(al - cur), 0};
        bool gmAligned = ((off * typeSize) & 31) == 0;
        if (cur == al && gmAligned) {
            DataCopy(dataLocal, inputGM[off], static_cast<int32_t>(al));
        } else {
            DataCopyPad(dataLocal, inputGM[off], copyParams, padParams);
        }
        TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(loadDone);
        WaitFlag<HardEvent::MTE2_V>(loadDone);
        if constexpr (sizeof(D_T_X) == sizeof(float)) {
            DataCopy(workLocal, dataLocal, static_cast<int32_t>(al));
        } else {
            Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(al));
        }
        PipeBarrier<PIPE_V>();
        Abs(workLocal, workLocal, static_cast<int32_t>(al));
        PipeBarrier<PIPE_V>();
        if (normMode_ == NORM_MODE_P_INF) {
            ReduceMax<float, Pattern::Reduce::AR, false>(reduceLocal, workLocal, patternTmpLocal, shape, false);
        } else if (normMode_ == NORM_MODE_P_ZERO) {
            Compare(patternTmpLocal, workLocal, zerosLocal, CMPMODE::GT, static_cast<int32_t>(al));
            PipeBarrier<PIPE_V>();
            Select(workLocal, patternTmpLocal, onesLocal, zerosLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   static_cast<int32_t>(al));
            PipeBarrier<PIPE_V>();
            ReduceSum<float, Pattern::Reduce::AR, false>(reduceLocal, workLocal, patternTmpLocal, shape, false);
        } else {
            if (p_ == 2.0f) {
                Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(al));
                PipeBarrier<PIPE_V>();
            } else if (p_ != 1.0f) {
                Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(al));
                PipeBarrier<PIPE_V>();
                Log(workLocal, workLocal, static_cast<int32_t>(al));
                PipeBarrier<PIPE_V>();
                Muls(workLocal, workLocal, p_, static_cast<int32_t>(al));
                PipeBarrier<PIPE_V>();
                Exp(workLocal, workLocal, static_cast<int32_t>(al));
                PipeBarrier<PIPE_V>();
            }
            ReduceSum<float, Pattern::Reduce::AR, false>(reduceLocal, workLocal, patternTmpLocal, shape, false);
        }
        PipeBarrier<PIPE_V>();
        TEventID reduceToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(reduceToScalar);
        WaitFlag<HardEvent::V_S>(reduceToScalar);
        float tileReduce = reduceLocal.GetValue(0);
        TEventID scalarToVector = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(scalarToVector);
        WaitFlag<HardEvent::S_V>(scalarToVector);
        if (normMode_ == NORM_MODE_P_INF) {
            if (tileReduce > localReduce) {
                localReduce = tileReduce;
            }
        } else {
            localReduce += tileReduce;
        }
    }

    Duplicate(reduceLocal, localReduce, 8);
    PipeBarrier<PIPE_V>();
    TEventID atomicReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    SetFlag<HardEvent::V_MTE3>(atomicReady);
    WaitFlag<HardEvent::V_MTE3>(atomicReady);
    if (normMode_ == NORM_MODE_P_INF) {
        SetAtomicMax<float>();
    } else {
        SetAtomicAdd<float>();
    }
    DataCopy(workspaceGM[0], reduceLocal, 8);
    SetAtomicNone();
    TEventID atomicDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
    SetFlag<HardEvent::MTE3_V>(atomicDone);
    WaitFlag<HardEvent::MTE3_V>(atomicDone);
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    PipeBarrier<PIPE_ALL>();
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(workspaceGM[0]);
    PipeBarrier<PIPE_ALL>();

    if (blockIdx_ == 0) {
        DataCopy(scaleLocal, workspaceGM[0], 8);
        TEventID normReady = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(normReady);
        WaitFlag<HardEvent::MTE2_V>(normReady);
        TEventID normToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(normToScalar);
        WaitFlag<HardEvent::V_S>(normToScalar);
        float globalReduce = scaleLocal.GetValue(0);
        TEventID scalarDone = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(scalarDone);
        WaitFlag<HardEvent::S_V>(scalarDone);
        float normValue;
        if (normMode_ == NORM_MODE_P_INF) {
            normValue = globalReduce;
        } else if (normMode_ == NORM_MODE_P_POSITIVE && p_ != 1.0f) {
            normValue = NsRenorm::ScalarPow(scaleLocal, globalReduce, 1.0f / p_, eps_);
        } else {
            normValue = globalReduce;
        }
        float scaleValue = NsRenorm::ComputeScale(normValue, maxNorm_, eps_);
        Duplicate(scaleLocal, scaleValue, 8);
        PipeBarrier<PIPE_V>();
        TEventID scaleReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(scaleReady);
        WaitFlag<HardEvent::V_MTE3>(scaleReady);
        DataCopy(workspaceGM[16], scaleLocal, 8);
        TEventID scaleDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(scaleDone);
        WaitFlag<HardEvent::MTE3_V>(scaleDone);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(workspaceGM[16]);
    PipeBarrier<PIPE_ALL>();

    DataCopy(scaleLocal, workspaceGM[16], 8);
    TEventID scaleLoadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
    SetFlag<HardEvent::MTE2_V>(scaleLoadDone);
    WaitFlag<HardEvent::MTE2_V>(scaleLoadDone);
    TEventID scaleToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(scaleToScalar);
    WaitFlag<HardEvent::V_S>(scaleToScalar);
    float scaleValue = scaleLocal.GetValue(0);
    TEventID scaleScalarDone = GetTPipePtr()->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(scaleScalarDone);
    WaitFlag<HardEvent::S_V>(scaleScalarDone);

    for (int64_t off = coreStart; off < coreEnd; off += tileLength_) {
        int64_t cur = (tileLength_ < (coreEnd - off)) ? tileLength_ : (coreEnd - off);
        int64_t al = (cur + alignElems - 1) / alignElems * alignElems;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(cur * typeSize);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(al - cur), 0};
        if (scaleValue == 0.0f) {
            Duplicate(dataLocal, static_cast<D_T_X>(0), static_cast<int32_t>(al));
            PipeBarrier<PIPE_V>();
        } else {
            bool gmAligned = ((off * typeSize) & 31) == 0;
            if (cur == al && gmAligned) {
                DataCopy(dataLocal, inputGM[off], static_cast<int32_t>(al));
            } else {
                DataCopyPad(dataLocal, inputGM[off], copyParams, padParams);
            }
            TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(loadDone);
            WaitFlag<HardEvent::MTE2_V>(loadDone);
        }
        if (scaleValue != 0.0f && scaleValue != 1.0f) {
            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(workLocal, dataLocal, static_cast<int32_t>(al));
            } else {
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(al));
            }
            PipeBarrier<PIPE_V>();
            Muls(workLocal, workLocal, scaleValue, static_cast<int32_t>(al));
            PipeBarrier<PIPE_V>();
            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(dataLocal, workLocal, static_cast<int32_t>(al));
            } else {
                Cast(dataLocal, workLocal, RoundMode::CAST_RINT, static_cast<int32_t>(al));
            }
            PipeBarrier<PIPE_V>();
        }
        TEventID storeReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(storeReady);
        WaitFlag<HardEvent::V_MTE3>(storeReady);
        bool gmAligned = ((off * typeSize) & 31) == 0;
        if (cur == al && gmAligned) {
            DataCopy(outputGM[off], dataLocal, static_cast<int32_t>(al));
        } else {
            DataCopyPad(outputGM[off], dataLocal, copyParams);
        }
        TEventID storeDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(storeDone);
        WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
    }
}

template <typename D_T_X, bool LARGE_GM_OFFSET>
__aicore__ inline void RenormGlobalHighP<D_T_X, LARGE_GM_OFFSET>::Process()
{
    if (totalElements_ == 0 || perCore_ == 0 || tileLength_ == 0) {
        return;
    }
    if (multiCore_) {
        ProcessMultiCore();
        return;
    }
    if (coreStart_ >= totalElements_) {
        return;
    }
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<float> normLocal = normBuf.Get<float>();
    LocalTensor<float> zerosLocal = zerosBuf.Get<float>();
    LocalTensor<float> onesLocal = onesBuf.Get<float>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> reduceLocal = reduceBuf.Get<float>();
    LocalTensor<uint8_t> patternTmpLocal = patternTmpBuf.Get<uint8_t>();
    int64_t typeSize = sizeof(D_T_X);
    int64_t alignElems = 32 / typeSize;
    int64_t alignedTile = (tileLength_ + alignElems - 1) / alignElems * alignElems;
    if (alignedTile < CMP_ALIGN) {
        alignedTile = CMP_ALIGN;
    }
    Duplicate(zerosLocal, 0.0f, static_cast<int32_t>(alignedTile));
    Duplicate(onesLocal, 1.0f, static_cast<int32_t>(alignedTile));
    PipeBarrier<PIPE_V>();
    if (normMode_ == NORM_MODE_MAXNORM_ZERO) {
        TEventID vm3Id = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        TEventID m3m2Id = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        for (int64_t off = coreStart_; off < coreEnd_; off += tileLength_) {
            int64_t cur = (tileLength_ < (coreEnd_ - off)) ? tileLength_ : (coreEnd_ - off);
            int64_t al = (cur + alignElems - 1) / alignElems * alignElems;
            LocalTensor<D_T_X> dataLocal = dataBuf0.Get<D_T_X>();
            Duplicate(dataLocal, static_cast<D_T_X>(0), static_cast<int32_t>(al));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vm3Id);
            WaitFlag<HardEvent::V_MTE3>(vm3Id);
            DataCopyExtParams sp;
            sp.blockCount = 1;
            sp.blockLen = static_cast<uint32_t>(cur * typeSize);
            sp.srcStride = 0;
            sp.dstStride = 0;
            DataCopyPad(outputGM[off], dataLocal, sp);
            SetFlag<HardEvent::MTE3_MTE2>(m3m2Id);
            WaitFlag<HardEvent::MTE3_MTE2>(m3m2Id);
        }
        return;
    }
    Duplicate(normLocal, 0.0f, static_cast<int32_t>(alignedTile));
    PipeBarrier<PIPE_V>();
    uint32_t srcShape[2] = {1, static_cast<uint32_t>(alignedTile)};
    int toggle = 0;
    for (int64_t off = coreStart_; off < coreEnd_; off += tileLength_) {
        int64_t cur = (tileLength_ < (coreEnd_ - off)) ? tileLength_ : (coreEnd_ - off);
        int64_t al = (cur + alignElems - 1) / alignElems * alignElems;
        uint8_t rightPad = static_cast<uint8_t>(al - cur);
        LocalTensor<D_T_X> dataLocal = (toggle == 0) ? dataBuf0.Get<D_T_X>() : dataBuf1.Get<D_T_X>();
        DataCopyExtParams cp;
        cp.blockCount = 1;
        cp.blockLen = static_cast<uint32_t>(cur * typeSize);
        cp.srcStride = 0;
        cp.dstStride = 0;
        DataCopyPadExtParams<D_T_X> pp = {true, 0, rightPad, 0};
        DataCopyPad(dataLocal, inputGM[off], cp, pp);
        {
            TEventID e = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(e);
            WaitFlag<HardEvent::MTE2_V>(e);
        }
        if constexpr (sizeof(D_T_X) == sizeof(float)) {
            DataCopy(workLocal, dataLocal, static_cast<int32_t>(al));
        } else {
            Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(al));
        }
        PipeBarrier<PIPE_V>();
        Abs(workLocal, workLocal, static_cast<int32_t>(al));
        PipeBarrier<PIPE_V>();
        if (normMode_ == NORM_MODE_P_ZERO) {
            Compare(maskLocal, workLocal, zerosLocal, CMPMODE::GT, static_cast<int32_t>(al));
            PipeBarrier<PIPE_V>();
            Select(workLocal, maskLocal, onesLocal, zerosLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   static_cast<int32_t>(al));
            PipeBarrier<PIPE_V>();
        } else if (normMode_ == NORM_MODE_P_POSITIVE) {
            if (p_ == 1.0f) {
            } else if (p_ == 2.0f) {
                Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(al));
                PipeBarrier<PIPE_V>();
            } else {
                Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(al));
                PipeBarrier<PIPE_V>();
                Log(workLocal, workLocal, static_cast<int32_t>(al));
                PipeBarrier<PIPE_V>();
                Muls(workLocal, workLocal, p_, static_cast<int32_t>(al));
                PipeBarrier<PIPE_V>();
                Exp(workLocal, workLocal, static_cast<int32_t>(al));
                PipeBarrier<PIPE_V>();
            }
        }
        if (normMode_ == NORM_MODE_P_INF) {
            Max(normLocal, normLocal, workLocal, static_cast<int32_t>(al));
        } else {
            Add(normLocal, normLocal, workLocal, static_cast<int32_t>(al));
        }
        PipeBarrier<PIPE_V>();
        {
            TEventID e = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            SetFlag<HardEvent::V_MTE2>(e);
            WaitFlag<HardEvent::V_MTE2>(e);
        }
        toggle = 1 - toggle;
    }
    srcShape[1] = static_cast<uint32_t>(alignedTile);
    if (normMode_ == NORM_MODE_P_INF) {
        ReduceMax<float, Pattern::Reduce::AR, false>(reduceLocal, normLocal, patternTmpLocal, srcShape, false);
    } else {
        ReduceSum<float, Pattern::Reduce::AR, false>(reduceLocal, normLocal, patternTmpLocal, srcShape, false);
    }
    PipeBarrier<PIPE_V>();
    {
        TEventID e = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(e);
        WaitFlag<HardEvent::V_S>(e);
    }
    float partial = reduceLocal.GetValue(0);
    {
        TEventID e = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(e);
        WaitFlag<HardEvent::S_V>(e);
    }
    float normRoot = partial;
    if (normMode_ == NORM_MODE_P_POSITIVE) {
        LocalTensor<float> powTmp = patternTmpBuf.Get<float>();
        if (p_ == 2.0f) {
            normRoot = NsRenorm::ScalarSqrt(powTmp, partial);
        } else if (p_ != 1.0f) {
            normRoot = NsRenorm::ScalarPow(powTmp, partial, 1.0f / p_, eps_);
        }
    }
    float scaleVal = NsRenorm::ComputeScale(normRoot, maxNorm_, eps_);
    TEventID vm3Id = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    TEventID m3m2Id = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
    int toggle2 = 0;
    for (int64_t off = coreStart_; off < coreEnd_; off += tileLength_) {
        int64_t cur = (tileLength_ < (coreEnd_ - off)) ? tileLength_ : (coreEnd_ - off);
        int64_t al = (cur + alignElems - 1) / alignElems * alignElems;
        uint8_t rightPad = static_cast<uint8_t>(al - cur);
        LocalTensor<D_T_X> dataLocal = (toggle2 == 0) ? dataBuf0.Get<D_T_X>() : dataBuf1.Get<D_T_X>();
        DataCopyExtParams cp;
        cp.blockCount = 1;
        cp.blockLen = static_cast<uint32_t>(cur * typeSize);
        cp.srcStride = 0;
        cp.dstStride = 0;
        DataCopyPadExtParams<D_T_X> pp = {true, 0, rightPad, 0};
        DataCopyPad(dataLocal, inputGM[off], cp, pp);
        {
            TEventID e = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(e);
            WaitFlag<HardEvent::MTE2_V>(e);
        }
        if constexpr (sizeof(D_T_X) == sizeof(float)) {
            DataCopy(workLocal, dataLocal, static_cast<int32_t>(al));
        } else {
            Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(al));
        }
        PipeBarrier<PIPE_V>();
        Muls(workLocal, workLocal, scaleVal, static_cast<int32_t>(al));
        PipeBarrier<PIPE_V>();
        if constexpr (sizeof(D_T_X) == sizeof(float)) {
            DataCopy(dataLocal, workLocal, static_cast<int32_t>(al));
        } else {
            Cast(dataLocal, workLocal, RoundMode::CAST_RINT, static_cast<int32_t>(al));
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(vm3Id);
        WaitFlag<HardEvent::V_MTE3>(vm3Id);
        DataCopyExtParams sp;
        sp.blockCount = 1;
        sp.blockLen = static_cast<uint32_t>(cur * typeSize);
        sp.srcStride = 0;
        sp.dstStride = 0;
        DataCopyPad(outputGM[off], dataLocal, sp);
        SetFlag<HardEvent::MTE3_MTE2>(m3m2Id);
        WaitFlag<HardEvent::MTE3_MTE2>(m3m2Id);
        toggle2 = 1 - toggle2;
    }
}
} // namespace NsRenormGlobalHighP
#endif // _RENORM_GLOBAL_HIGH_P_H_
