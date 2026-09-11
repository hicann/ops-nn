/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_GLOBAL_H_
#define _RENORM_GLOBAL_H_
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"
#include "renorm_tiling_key.h"
#include "common/renorm_common.h"
namespace NsRenormGlobal {
using namespace AscendC;
constexpr int32_t NORM_MODE_P_POSITIVE = 0;
constexpr int32_t NORM_MODE_P_ZERO = 1;
constexpr int32_t NORM_MODE_P_INF = 2;
constexpr int32_t NORM_MODE_MAXNORM_ZERO = 3;
constexpr int64_t CMP_ALIGN = 8;
constexpr int64_t ATOMIC_ALIGN_ELEMENTS = 16;
template <typename D_T_X, bool OPTIMIZED_SINGLE_SLICE = false, bool SAFE_PER_CORE_MAX = false>
class RenormGlobal {
public:
    __aicore__ inline RenormGlobal() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RenormTilingData* tilingData,
                                TPipe* pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessMultiCoreInf();

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
template <typename D_T_X, bool OPTIMIZED_SINGLE_SLICE, bool SAFE_PER_CORE_MAX>
__aicore__ inline void RenormGlobal<D_T_X, OPTIMIZED_SINGLE_SLICE, SAFE_PER_CORE_MAX>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RenormTilingData* tilingData, TPipe* pipeIn)
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
    inputGM.SetGlobalBuffer((__gm__ D_T_X*)x, totalElements_);
    outputGM.SetGlobalBuffer((__gm__ D_T_X*)y, totalElements_);
    if (multiCore_) {
        SetSysWorkspace(workspace);
        GM_ADDR userWorkspace = GetUserWorkspace(workspace);
        if (userWorkspace == nullptr) {
            return;
        }
        workspaceGM.SetGlobalBuffer((__gm__ float*)userWorkspace, tilingData->workspaceSize / sizeof(float));
    }
    blockIdx_ = GetBlockIdx();
    coreStart_ = blockIdx_ * perCore_;
    coreEnd_ = coreStart_ + perCore_;
    if (coreEnd_ > totalElements_) {
        coreEnd_ = totalElements_;
    }
    int64_t typeSize = sizeof(D_T_X);
    int64_t alignElems = 32 / typeSize;
    int64_t alignedTile = (tileLength_ + alignElems - 1) / alignElems * alignElems;
    if (alignedTile < CMP_ALIGN) {
        alignedTile = CMP_ALIGN;
    }
    if (multiCore_) {
        // Multi-core p=inf only needs one input tile, one FP32 tile, a
        // reduction scratch tile, and two 32B scalar workspace buffers.
        pipe_->InitBuffer(dataBuf0, static_cast<uint32_t>(alignedTile * typeSize));
        if constexpr (OPTIMIZED_SINGLE_SLICE) {
            // Native FP16 max reduction avoids the FP32 cast buffer on the
            // isolated A2-shaped route. The result is converted only once
            // per tile for the cross-core scalar max.
            pipe_->InitBuffer(dataBuf1, 32);
        }
        pipe_->InitBuffer(workBuf, static_cast<uint32_t>(alignedTile * sizeof(float)));
        pipe_->InitBuffer(reduceBuf, 32);
        pipe_->InitBuffer(patternTmpBuf, static_cast<uint32_t>(alignedTile * sizeof(float)));
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
template <typename D_T_X, bool OPTIMIZED_SINGLE_SLICE, bool SAFE_PER_CORE_MAX>
__aicore__ inline void RenormGlobal<D_T_X, OPTIMIZED_SINGLE_SLICE, SAFE_PER_CORE_MAX>::ProcessMultiCoreInf()
{
    LocalTensor<D_T_X> dataLocal = dataBuf0.Get<D_T_X>();
    LocalTensor<D_T_X> reduceTypeLocal;
    if constexpr (OPTIMIZED_SINGLE_SLICE) {
        reduceTypeLocal = dataBuf1.Get<D_T_X>();
    }
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<float> reduceLocal = reduceBuf.Get<float>();
    LocalTensor<uint8_t> patternTmpLocal = patternTmpBuf.Get<uint8_t>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    int64_t typeSize = sizeof(D_T_X);
    int64_t alignElems = 32 / typeSize;
    int64_t alignedTile = (tileLength_ + alignElems - 1) / alignElems * alignElems;
    if (alignedTile < CMP_ALIGN) {
        alignedTile = CMP_ALIGN;
    }
    int64_t coreStart = blockIdx_ * perCore_;
    int64_t coreEnd = coreStart + perCore_;
    if (coreEnd > totalElements_) {
        coreEnd = totalElements_;
    }
    if (coreStart >= coreEnd) {
        return;
    }

    if constexpr (!SAFE_PER_CORE_MAX) {
        // Initialize the atomic max slot before any core starts its local scan.
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
    }

    float localMax = 0.0f;
    uint32_t shape[2] = {1, static_cast<uint32_t>(alignedTile)};
    for (int64_t off = coreStart; off < coreEnd; off += tileLength_) {
        int64_t cur = (tileLength_ < (coreEnd - off)) ? tileLength_ : (coreEnd - off);
        int64_t al = (cur + alignElems - 1) / alignElems * alignElems;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(cur * typeSize);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(al - cur), 0};
        if constexpr (OPTIMIZED_SINGLE_SLICE) {
            if (al == cur && (cur * typeSize) % 32 == 0) {
                DataCopy(dataLocal, inputGM[off], static_cast<int32_t>(al));
            } else {
                DataCopyPad(dataLocal, inputGM[off], copyParams, padParams);
            }
        } else {
            DataCopyPad(dataLocal, inputGM[off], copyParams, padParams);
        }
        TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(loadDone);
        WaitFlag<HardEvent::MTE2_V>(loadDone);
        if constexpr (OPTIMIZED_SINGLE_SLICE) {
            Abs(dataLocal, dataLocal, static_cast<int32_t>(al));
            PipeBarrier<PIPE_V>();
            ReduceMax<D_T_X>(reduceTypeLocal, dataLocal, dataLocal, static_cast<uint32_t>(al));
            PipeBarrier<PIPE_V>();
            TEventID reduceToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(reduceToScalar);
            WaitFlag<HardEvent::V_S>(reduceToScalar);
            float tileMax = static_cast<float>(reduceTypeLocal.GetValue(0));
            TEventID scalarToVector = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);
            if (tileMax > localMax) {
                localMax = tileMax;
            }
        } else if constexpr (SAFE_PER_CORE_MAX) {
            // The pattern AR reduction is fast, but its scalar result loses
            // the tail maximum on the affected A5 p=inf layouts. This
            // isolated template uses the scalar ReduceMax form already used
            // by Template A to obtain each core's exact partial maximum.
            Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(al));
            PipeBarrier<PIPE_V>();
            Abs(workLocal, workLocal, static_cast<int32_t>(al));
            PipeBarrier<PIPE_V>();
            ReduceMax<float>(reduceLocal, workLocal, workLocal, static_cast<uint32_t>(al));
            PipeBarrier<PIPE_V>();
            TEventID reduceToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(reduceToScalar);
            WaitFlag<HardEvent::V_S>(reduceToScalar);
            float tileMax = reduceLocal.GetValue(0);
            TEventID scalarToVector = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);
            if (tileMax > localMax) {
                localMax = tileMax;
            }
        } else {
            Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(al));
            PipeBarrier<PIPE_V>();
            Abs(workLocal, workLocal, static_cast<int32_t>(al));
            PipeBarrier<PIPE_V>();
            ReduceMax<float, Pattern::Reduce::AR, false>(reduceLocal, workLocal, patternTmpLocal, shape, false);
            PipeBarrier<PIPE_V>();
            TEventID reduceToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(reduceToScalar);
            WaitFlag<HardEvent::V_S>(reduceToScalar);
            float tileMax = reduceLocal.GetValue(0);
            TEventID scalarToVector = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);
            if (tileMax > localMax) {
                localMax = tileMax;
            }
        }
    }

    float scaleValue = 1.0f;
    int64_t scaleWorkspaceOffset = ATOMIC_ALIGN_ELEMENTS;
    if constexpr (SAFE_PER_CORE_MAX) {
        // AtomicMax on the shared slot is not reliable for this A5 path when
        // several cores update the same cache line. Keep each local result on
        // an independent line, then reduce the tiny vector on every core.
        int64_t localWorkspaceOffset = blockIdx_ * ATOMIC_ALIGN_ELEMENTS;
        Duplicate(reduceLocal, localMax, 8);
        PipeBarrier<PIPE_V>();
        TEventID localMaxReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(localMaxReady);
        WaitFlag<HardEvent::V_MTE3>(localMaxReady);
        DataCopy(workspaceGM[localWorkspaceOffset], reduceLocal, 8);
        TEventID localMaxStored = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(localMaxStored);
        WaitFlag<HardEvent::MTE3_V>(localMaxStored);
        // SyncAll orders cores but does not write back another core's L1.
        // Each partial occupies one 64B slot, so clean the exact line before
        // the cross-core merge reads it from GM.
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            workspaceGM[localWorkspaceOffset]);
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        PipeBarrier<PIPE_ALL>();

        // Keep the final merge on one core. On A5, vector loads of a
        // cross-core-written workspace can still retain stale lanes even
        // after SyncAll. The scalar loop is only O(core count) and leaves
        // the full input scan parallel.
        scaleWorkspaceOffset = GetBlockNum() * ATOMIC_ALIGN_ELEMENTS;
        if (blockIdx_ == 0) {
            float globalMax = 0.0f;
            for (int64_t coreIdx = 0; coreIdx < GetBlockNum(); ++coreIdx) {
                int64_t partialOffset = coreIdx * ATOMIC_ALIGN_ELEMENTS;
                DataCopy(reduceLocal, workspaceGM[partialOffset], 8);
                TEventID partialReady = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(partialReady);
                WaitFlag<HardEvent::MTE2_V>(partialReady);
                TEventID partialToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(partialToScalar);
                WaitFlag<HardEvent::V_S>(partialToScalar);
                float partialMax = reduceLocal.GetValue(0);
                TEventID scalarDone = GetTPipePtr()->FetchEventID(HardEvent::S_V);
                SetFlag<HardEvent::S_V>(scalarDone);
                WaitFlag<HardEvent::S_V>(scalarDone);
                if (partialMax > globalMax) {
                    globalMax = partialMax;
                }
                if (coreIdx + 1 < GetBlockNum()) {
                    TEventID reduceReusable = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                    SetFlag<HardEvent::V_MTE2>(reduceReusable);
                    WaitFlag<HardEvent::V_MTE2>(reduceReusable);
                }
            }
            scaleValue = NsRenorm::ComputeScale(globalMax, maxNorm_, eps_);
            Duplicate(reduceLocal, scaleValue, 8);
            PipeBarrier<PIPE_V>();
            TEventID scaleReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(scaleReady);
            WaitFlag<HardEvent::V_MTE3>(scaleReady);
            DataCopy(workspaceGM[scaleWorkspaceOffset], reduceLocal, 8);
            TEventID scaleStored = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
            SetFlag<HardEvent::MTE3_V>(scaleStored);
            WaitFlag<HardEvent::MTE3_V>(scaleStored);
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                workspaceGM[scaleWorkspaceOffset]);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        PipeBarrier<PIPE_ALL>();
        DataCopy(scaleLocal, workspaceGM[scaleWorkspaceOffset], 8);
        TEventID scaleLoadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(scaleLoadDone);
        WaitFlag<HardEvent::MTE2_V>(scaleLoadDone);
        TEventID scaleToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(scaleToScalar);
        WaitFlag<HardEvent::V_S>(scaleToScalar);
        scaleValue = scaleLocal.GetValue(0);
        TEventID scaleScalarDone = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(scaleScalarDone);
        WaitFlag<HardEvent::S_V>(scaleScalarDone);
    } else {
        Duplicate(reduceLocal, localMax, 8);
        PipeBarrier<PIPE_V>();
        TEventID atomicReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(atomicReady);
        WaitFlag<HardEvent::V_MTE3>(atomicReady);
        SetAtomicMax<float>();
        DataCopy(workspaceGM[0], reduceLocal, 8);
        SetAtomicNone();
        TEventID atomicDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(atomicDone);
        WaitFlag<HardEvent::MTE3_V>(atomicDone);
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        PipeBarrier<PIPE_ALL>();
        // The core which initialized the slot can retain the old zero cache line
        // after other cores update it atomically. Invalidate before core 0 reads
        // the global max; this only affects the multi-core global path.
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
            float globalMax = scaleLocal.GetValue(0);
            TEventID scalarDone = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(scalarDone);
            WaitFlag<HardEvent::S_V>(scalarDone);
            float scaleValue = NsRenorm::ComputeScale(globalMax, maxNorm_, eps_);
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
    }
    if constexpr (!SAFE_PER_CORE_MAX) {
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            workspaceGM[scaleWorkspaceOffset]);
        PipeBarrier<PIPE_ALL>();

        DataCopy(scaleLocal, workspaceGM[scaleWorkspaceOffset], 8);
        TEventID scaleLoadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(scaleLoadDone);
        WaitFlag<HardEvent::MTE2_V>(scaleLoadDone);
        TEventID scaleToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(scaleToScalar);
        WaitFlag<HardEvent::V_S>(scaleToScalar);
        scaleValue = scaleLocal.GetValue(0);
        TEventID scaleScalarDone = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(scaleScalarDone);
        WaitFlag<HardEvent::S_V>(scaleScalarDone);
    }

    for (int64_t off = coreStart; off < coreEnd; off += tileLength_) {
        int64_t cur = (tileLength_ < (coreEnd - off)) ? tileLength_ : (coreEnd - off);
        int64_t al = (cur + alignElems - 1) / alignElems * alignElems;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(cur * typeSize);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(al - cur), 0};
        if constexpr (OPTIMIZED_SINGLE_SLICE) {
            if (al == cur && (cur * typeSize) % 32 == 0) {
                DataCopy(dataLocal, inputGM[off], static_cast<int32_t>(al));
            } else {
                DataCopyPad(dataLocal, inputGM[off], copyParams, padParams);
            }
        } else {
            DataCopyPad(dataLocal, inputGM[off], copyParams, padParams);
        }
        TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(loadDone);
        WaitFlag<HardEvent::MTE2_V>(loadDone);
        if (scaleValue != 1.0f) {
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
        if constexpr (OPTIMIZED_SINGLE_SLICE) {
            if (al == cur && (cur * typeSize) % 32 == 0) {
                DataCopy(outputGM[off], dataLocal, static_cast<int32_t>(al));
            } else {
                DataCopyPad(outputGM[off], dataLocal, copyParams);
            }
        } else {
            DataCopyPad(outputGM[off], dataLocal, copyParams);
        }
        TEventID storeDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(storeDone);
        WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
    }
}

template <typename D_T_X, bool OPTIMIZED_SINGLE_SLICE, bool SAFE_PER_CORE_MAX>
__aicore__ inline void RenormGlobal<D_T_X, OPTIMIZED_SINGLE_SLICE, SAFE_PER_CORE_MAX>::Process()
{
    if (totalElements_ == 0 || perCore_ == 0 || tileLength_ == 0) {
        return;
    }
    if (normMode_ == NORM_MODE_MAXNORM_ZERO) {
        if (coreStart_ >= coreEnd_) {
            return;
        }
        int64_t alignElems = 32 / sizeof(D_T_X);
        int64_t zeroChunk = (tileLength_ / alignElems) * alignElems;
        if (zeroChunk < alignElems) {
            zeroChunk = alignElems;
        }
        LocalTensor<D_T_X> dataLocal = dataBuf0.Get<D_T_X>();
        Duplicate(dataLocal, static_cast<D_T_X>(0), static_cast<int32_t>(zeroChunk));
        PipeBarrier<PIPE_V>();
        TEventID zeroReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(zeroReady);
        WaitFlag<HardEvent::V_MTE3>(zeroReady);
        for (int64_t off = coreStart_; off < coreEnd_; off += zeroChunk) {
            int64_t cur = (zeroChunk < (coreEnd_ - off)) ? zeroChunk : (coreEnd_ - off);
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(cur * sizeof(D_T_X));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;

            DataCopyPad(outputGM[off], dataLocal, copyParams);
        }
        TEventID zeroStored = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(zeroStored);
        WaitFlag<HardEvent::MTE3_MTE2>(zeroStored);
        return;
    }
    if (multiCore_) {
        ProcessMultiCoreInf();
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
    Duplicate(normLocal, 0.0f, static_cast<int32_t>(alignedTile));
    PipeBarrier<PIPE_V>();
    uint32_t srcShape[2] = {1, static_cast<uint32_t>(alignedTile)};
    float scalarMax = 0.0f;
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
            if (totalElements_ == 1) {
                // A one-element slice has ||x||_p == |x| for every p > 0.
                // Keep the value in FP32 and avoid Log/Exp overflow.
            } else if (p_ == 1.0f) {
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
            if constexpr (SAFE_PER_CORE_MAX) {
                // Pattern ReduceMax drops the tail maximum for the isolated
                // unaligned FP32 row. Keep this exact-template path scalar.
                ReduceMax<float>(reduceLocal, workLocal, workLocal, static_cast<uint32_t>(al));
                PipeBarrier<PIPE_V>();
                TEventID reduceReady = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(reduceReady);
                WaitFlag<HardEvent::V_S>(reduceReady);
                float tileMax = reduceLocal.GetValue(0);
                TEventID scalarDone = GetTPipePtr()->FetchEventID(HardEvent::S_V);
                SetFlag<HardEvent::S_V>(scalarDone);
                WaitFlag<HardEvent::S_V>(scalarDone);
                if (tileMax > scalarMax) {
                    scalarMax = tileMax;
                }
            } else {
                Max(normLocal, normLocal, workLocal, static_cast<int32_t>(al));
            }
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
    float partial = scalarMax;
    if constexpr (!SAFE_PER_CORE_MAX) {
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
        partial = reduceLocal.GetValue(0);
        {
            TEventID e = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(e);
            WaitFlag<HardEvent::S_V>(e);
        }
    } else if (normMode_ != NORM_MODE_P_INF) {
        srcShape[1] = static_cast<uint32_t>(alignedTile);
        ReduceSum<float, Pattern::Reduce::AR, false>(reduceLocal, normLocal, patternTmpLocal, srcShape, false);
        PipeBarrier<PIPE_V>();
        {
            TEventID e = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(e);
            WaitFlag<HardEvent::V_S>(e);
        }
        partial = reduceLocal.GetValue(0);
        {
            TEventID e = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(e);
            WaitFlag<HardEvent::S_V>(e);
        }
    }
    float normRoot = partial;
    if (normMode_ == NORM_MODE_P_POSITIVE) {
        LocalTensor<float> powTmp = patternTmpBuf.Get<float>();
        if (totalElements_ == 1) {
            // The reduction already contains the exact one-element norm.
        } else if (p_ == 2.0f) {
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
} // namespace NsRenormGlobal
#endif // _RENORM_GLOBAL_H_
