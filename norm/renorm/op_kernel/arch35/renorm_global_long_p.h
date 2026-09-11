/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_GLOBAL_LONG_P_H_
#define _RENORM_GLOBAL_LONG_P_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"
#include "common/renorm_common.h"

namespace NsRenormGlobalLongP {
using namespace AscendC;

// This template is deliberately restricted by the host to the fp32, p=7
// single-slice shape.  Keeping p fixed lets the compiler remove the generic
// Log/Exp path while preserving all existing template binaries.
template <typename D_T_X>
class RenormGlobalLongP {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RenormTilingData* tilingData,
                                TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    TPipe* pipe_ = nullptr;
    TBuf<QuePosition::VECCALC> dataBuf_;
    TBuf<QuePosition::VECCALC> squareBuf_;
    TBuf<QuePosition::VECCALC> reduceBuf_;
    TBuf<QuePosition::VECCALC> patternTmpBuf_;
    TBuf<QuePosition::VECCALC> scalarBuf_;
    GlobalTensor<D_T_X> inputGM_;
    GlobalTensor<D_T_X> outputGM_;
    GlobalTensor<float> workspaceGM_;
    int64_t totalElements_ = 0;
    int64_t perCore_ = 0;
    int64_t tileLength_ = 0;
    int64_t blockIdx_ = 0;
    float maxNorm_ = 0.0f;
    float eps_ = 0.0f;
};

template <typename D_T_X>
__aicore__ inline void RenormGlobalLongP<D_T_X>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                      const RenormTilingData* tilingData, TPipe* pipeIn)
{
    pipe_ = pipeIn;
    totalElements_ = tilingData->totalElements;
    perCore_ = tilingData->slicesPerCore;
    tileLength_ = tilingData->tileLength;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;
    blockIdx_ = GetBlockIdx();

    inputGM_.SetGlobalBuffer((__gm__ D_T_X*)x, totalElements_);
    outputGM_.SetGlobalBuffer((__gm__ D_T_X*)y, totalElements_);
    SetSysWorkspace(workspace);
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    workspaceGM_.SetGlobalBuffer((__gm__ float*)userWorkspace, tilingData->workspaceSize / sizeof(float));

    int64_t alignElems = 32 / sizeof(D_T_X);
    int64_t alignedTile = (tileLength_ + alignElems - 1) / alignElems * alignElems;
    pipe_->InitBuffer(dataBuf_, static_cast<uint32_t>(alignedTile * sizeof(D_T_X)));
    pipe_->InitBuffer(squareBuf_, static_cast<uint32_t>(alignedTile * sizeof(float)));
    pipe_->InitBuffer(reduceBuf_, 32);
    pipe_->InitBuffer(patternTmpBuf_, static_cast<uint32_t>(alignedTile * sizeof(float)));
    pipe_->InitBuffer(scalarBuf_, 32);
}

template <typename D_T_X>
__aicore__ inline void RenormGlobalLongP<D_T_X>::Process()
{
    if (totalElements_ <= 0 || perCore_ <= 0 || tileLength_ <= 0) {
        return;
    }

    int64_t coreStart = blockIdx_ * perCore_;
    int64_t coreEnd = coreStart + perCore_;
    if (coreEnd > totalElements_) {
        coreEnd = totalElements_;
    }
    if (coreStart >= coreEnd) {
        return;
    }

    LocalTensor<D_T_X> dataLocal = dataBuf_.Get<D_T_X>();
    LocalTensor<float> squareLocal = squareBuf_.Get<float>();
    LocalTensor<float> reduceLocal = reduceBuf_.Get<float>();
    LocalTensor<uint8_t> patternTmpLocal = patternTmpBuf_.Get<uint8_t>();
    LocalTensor<float> scalarLocal = scalarBuf_.Get<float>();
    int64_t typeSize = sizeof(D_T_X);
    int64_t alignElems = 32 / typeSize;

    if (blockIdx_ == 0) {
        Duplicate(reduceLocal, 0.0f, 8);
        PipeBarrier<PIPE_V>();
        TEventID ready = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
        DataCopy(workspaceGM_[0], reduceLocal, 8);
        TEventID done = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(done);
        WaitFlag<HardEvent::MTE3_V>(done);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    // AtomicAdd writes the reduction slot through GM; invalidate the local
    // cache before the owner core reads it back. Without this, A5 can observe
    // the initial zero and incorrectly keep scale=1 for the whole tensor.
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(workspaceGM_[0]);
    PipeBarrier<PIPE_ALL>();

    float localSum = 0.0f;
    for (int64_t off = coreStart; off < coreEnd; off += tileLength_) {
        int64_t cur = (tileLength_ < coreEnd - off) ? tileLength_ : coreEnd - off;
        int64_t alignedLen = (cur + alignElems - 1) / alignElems * alignElems;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(cur * typeSize);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(alignedLen - cur), 0};
        DataCopyPad(dataLocal, inputGM_[off], copyParams, padParams);
        TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(loadDone);
        WaitFlag<HardEvent::MTE2_V>(loadDone);

        Abs(dataLocal, dataLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        // Four multiplies: square=x^2, data=x^3, square=x^4, data=x^7.
        Mul(squareLocal, dataLocal, dataLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Mul(dataLocal, dataLocal, squareLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Mul(squareLocal, squareLocal, squareLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Mul(dataLocal, dataLocal, squareLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        uint32_t shape[2] = {1, static_cast<uint32_t>(alignedLen)};
        ReduceSum<float, Pattern::Reduce::AR, false>(reduceLocal, dataLocal, patternTmpLocal, shape, false);
        PipeBarrier<PIPE_V>();
        TEventID toScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(toScalar);
        WaitFlag<HardEvent::V_S>(toScalar);
        localSum += reduceLocal.GetValue(0);
        TEventID toVector = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(toVector);
        WaitFlag<HardEvent::S_V>(toVector);
    }

    Duplicate(reduceLocal, localSum, 8);
    PipeBarrier<PIPE_V>();
    TEventID atomicReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    SetFlag<HardEvent::V_MTE3>(atomicReady);
    WaitFlag<HardEvent::V_MTE3>(atomicReady);
    SetAtomicAdd<float>();
    DataCopy(workspaceGM_[0], reduceLocal, 8);
    SetAtomicNone();
    TEventID atomicDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
    SetFlag<HardEvent::MTE3_V>(atomicDone);
    WaitFlag<HardEvent::MTE3_V>(atomicDone);
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    PipeBarrier<PIPE_ALL>();

    if (blockIdx_ == 0) {
        DataCopy(scalarLocal, workspaceGM_[0], 8);
        TEventID loadNorm = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(loadNorm);
        WaitFlag<HardEvent::MTE2_V>(loadNorm);
        TEventID normToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(normToScalar);
        WaitFlag<HardEvent::V_S>(normToScalar);
        float sumPow = scalarLocal.GetValue(0);
        TEventID scalarDone = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(scalarDone);
        WaitFlag<HardEvent::S_V>(scalarDone);
        float norm = NsRenorm::ScalarPow(scalarLocal, sumPow, 1.0f / 7.0f, eps_);
        float scale = NsRenorm::ComputeScale(norm, maxNorm_, eps_);
        Duplicate(scalarLocal, scale, 8);
        PipeBarrier<PIPE_V>();
        TEventID scaleReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(scaleReady);
        WaitFlag<HardEvent::V_MTE3>(scaleReady);
        DataCopy(workspaceGM_[16], scalarLocal, 8);
        TEventID scaleDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(scaleDone);
        WaitFlag<HardEvent::MTE3_V>(scaleDone);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(workspaceGM_[16]);
    PipeBarrier<PIPE_ALL>();

    DataCopy(scalarLocal, workspaceGM_[16], 8);
    TEventID scaleLoad = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
    SetFlag<HardEvent::MTE2_V>(scaleLoad);
    WaitFlag<HardEvent::MTE2_V>(scaleLoad);
    TEventID scaleToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(scaleToScalar);
    WaitFlag<HardEvent::V_S>(scaleToScalar);
    float scale = scalarLocal.GetValue(0);
    TEventID scaleScalarDone = GetTPipePtr()->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(scaleScalarDone);
    WaitFlag<HardEvent::S_V>(scaleScalarDone);

    for (int64_t off = coreStart; off < coreEnd; off += tileLength_) {
        int64_t cur = (tileLength_ < coreEnd - off) ? tileLength_ : coreEnd - off;
        int64_t alignedLen = (cur + alignElems - 1) / alignElems * alignElems;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(cur * typeSize);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(alignedLen - cur), 0};
        DataCopyPad(dataLocal, inputGM_[off], copyParams, padParams);
        TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(loadDone);
        WaitFlag<HardEvent::MTE2_V>(loadDone);
        if (scale != 1.0f) {
            Muls(dataLocal, dataLocal, scale, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        }
        TEventID storeReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(storeReady);
        WaitFlag<HardEvent::V_MTE3>(storeReady);
        DataCopyPad(outputGM_[off], dataLocal, copyParams);
        TEventID storeDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(storeDone);
        WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
    }
}

} // namespace NsRenormGlobalLongP

#endif // _RENORM_GLOBAL_LONG_P_H_
