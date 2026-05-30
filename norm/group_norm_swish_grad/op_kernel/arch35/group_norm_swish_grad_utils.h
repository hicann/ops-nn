/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file group_norm_swish_grad_utils.h
 * \brief
 */

#ifndef GROUP_NORM_SWISH_GRAD_UTILS_H
#define GROUP_NORM_SWISH_GRAD_UTILS_H

// Common GM/UB IO helpers and scalar math helpers.

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::InitOutputGm()
{
    if constexpr (!isDeterministic && IsSameType<T, float>::value) {
        if (this->curBlockIdx == 0) {
            InitOutput<float>(dgammaGm, c, 0.0f);
            InitOutput<float>(dbetaGm, c, 0.0f);
        }
    } else if constexpr (!isDeterministic && !IsSameType<T, float>::value) {
        if (this->curBlockIdx == 0) {
            InitOutput<float>(dgammaWorkspace, c, 0.0f);
            InitOutput<float>(dbetaWorkspace, c, 0.0f);
        }
    } else if constexpr (isDeterministic && IsSameType<T, float>::value) {
        if (this->curBlockIdx == 0) {
            InitOutput<float>(dgammaWorkspace, workspaceSize, 0.0f);
            InitOutput<float>(dbetaWorkspace, workspaceSize, 0.0f);
            InitOutput<float>(dgammaGm, c, 0.0f);
            InitOutput<float>(dbetaGm, c, 0.0f);
        }
    } else if constexpr (isDeterministic && !IsSameType<T, float>::value) {
        if (this->curBlockIdx == 0) {
            InitOutput<float>(dgammaWorkspace, workspaceSize, 0.0f);
            InitOutput<float>(dbetaWorkspace, workspaceSize, 0.0f);
        }
    }
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::InitUnifiedBuffer(GroupNormSwishGradTilingData* tilingData)
{
    if (tilingKey == groupFitUbTilingKey) {
        InitGroupFitUbBuffer();
    } else if (tilingKey == channelFitUbTilingKey) {
        InitChannelFitUbBuffer();
    } else if (tilingKey == channelSplitUbTilingKey) {
        InitChannelSplitUbBuffer(tilingData);
    }
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::CopyMeanRstdToUb(
    LocalTensor<float>& meanRstdLocal, int32_t taskIdx)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(meanRstdLocal[meanValueOffset], meanGm[taskIdx], copyParams, padParams);
        DataCopyPad(meanRstdLocal[rstdValueOffset], rstdGm[taskIdx], copyParams, padParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    } else {
        uint32_t rstdTOffset = b16EleNumPerBlock;
        LocalTensor<T> meanTemp = inQueueMeanRstdT.AllocTensor<T>();
        LocalTensor<T> rstdTemp = meanTemp[rstdTOffset];
        DataCopyPad(meanTemp, meanGm[taskIdx], copyParams, padParams);
        DataCopyPad(rstdTemp, rstdGm[taskIdx], copyParams, padParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(meanRstdLocal[meanValueOffset], meanTemp, RoundMode::CAST_NONE, 1);
        PipeBarrier<PIPE_V>();
        Cast(meanRstdLocal[rstdValueOffset], rstdTemp, RoundMode::CAST_NONE, 1);
        PipeBarrier<PIPE_V>();
        inQueueMeanRstdT.FreeTensor(meanTemp);
    }
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::CustomDataCopyIn(
    const LocalTensor<float>& inLocal, TQue<QuePosition::VECIN, 1>& inQue, const GlobalTensor<T>& gm,
    const uint64_t offset, const uint32_t count)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(inLocal, gm[offset], copyParams, padParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    } else {
        LocalTensor<T> temp = inQue.AllocTensor<T>();
        DataCopyPad(temp, gm[offset], copyParams, padParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(inLocal, temp, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        inQue.FreeTensor(temp);
    }
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::CustomDataCopyOut(
    LocalTensor<float>& outLocal, const uint64_t gmOffset, TQue<QuePosition::VECOUT, 1>& outQue,
    const uint32_t count)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(dxGm[gmOffset], outLocal, copyParams);
    } else {
        LocalTensor<T> temp = outQue.AllocTensor<T>();
        Cast(temp, outLocal, RoundMode::CAST_ROUND, count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(dxGm[gmOffset], temp, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        outQue.FreeTensor(temp);
    }
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::CustomDataCopyOut(
    LocalTensor<float>& outLocal, const uint32_t ubOffset, const uint64_t gmOffset,
    TQue<QuePosition::VECOUT, 1>& outQue, const uint32_t count)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(dxGm[gmOffset], outLocal[ubOffset], copyParams);
    } else {
        LocalTensor<T> temp = outQue.AllocTensor<T>();
        Cast(temp, outLocal[ubOffset], RoundMode::CAST_ROUND, count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(dxGm[gmOffset], temp, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        outQue.FreeTensor(temp);
    }
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::CustomDataCopyOut(
    LocalTensor<float>& outLocal, GlobalTensor<T>& gmOut, const uint64_t gmOffset,
    TQue<QuePosition::VECOUT, 1>& outQue,
    const uint32_t count)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(gmOut[gmOffset], outLocal, copyParams);
    } else {
        LocalTensor<T> temp = outQue.AllocTensor<T>();
        Cast(temp, outLocal, RoundMode::CAST_ROUND, count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(gmOut[gmOffset], temp, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        outQue.FreeTensor(temp);
    }
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::Fp32DgammaDbeta2Gm(
    uint32_t channelIdx, GlobalTensor<float>& dgammaOut, const LocalTensor<float>& dgammaUb,
    GlobalTensor<float>& dbetaOut, const LocalTensor<float>& dbetaUb)
{
    bool hasAtomicCopy = false;
    SetAtomicAdd<float>();
    DataCopyParams copyParams{1, (uint16_t)(this->cG * sizeof(float)), 0, 0};
    if (dbetaIsRequire) {
        event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        DataCopyPad(dbetaOut[channelIdx], dbetaUb, copyParams);
        hasAtomicCopy = true;
    }
    if (dgammaIsRequire) {
        event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        DataCopyPad(dgammaOut[channelIdx], dgammaUb, copyParams);
        hasAtomicCopy = true;
    }
    if (hasAtomicCopy) {
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }
    SetAtomicNone();
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::NonFp32DgammaDbeta2Gm(
    uint32_t channelIdx, GlobalTensor<T>& dgammaOut, const LocalTensor<float>& dgammaUb,
    GlobalTensor<T>& dbetaOut, const LocalTensor<float>& dbetaUb,
    TQue<QuePosition::VECOUT, 1>& dgammaOutQue, TQue<QuePosition::VECOUT, 1>& dbetaOutQue)
{
    DataCopyParams copyParams{1, (uint16_t)(this->cG * sizeof(T)), 0, 0};
    if (dbetaIsRequire) {
        LocalTensor<T> dbetaTemp = dbetaOutQue.AllocTensor<T>();
        PipeBarrier<PIPE_V>();
        Cast(dbetaTemp, dbetaUb, RoundMode::CAST_ROUND, this->cG);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(dbetaOut[channelIdx], dbetaTemp, copyParams);
        dbetaOutQue.FreeTensor(dbetaTemp);
    }
    if (dgammaIsRequire) {
        LocalTensor<T> dgammaTemp = dgammaOutQue.AllocTensor<T>();
        PipeBarrier<PIPE_V>();
        Cast(dgammaTemp, dgammaUb, RoundMode::CAST_ROUND, this->cG);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(dgammaOut[channelIdx], dgammaTemp, copyParams);
        dgammaOutQue.FreeTensor(dgammaTemp);
    }
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::ModeSelection(
    int32_t taskIdx, const LocalTensor<float>& dgammaUb, const LocalTensor<float>& dbetaUb)
{
    uint32_t outChannelIdx;
    constexpr uint32_t splitCount = 2;
    if constexpr (!isDeterministic && IsSameType<T, float>::value) {
        outChannelIdx = (taskIdx % this->g) * this->cG;
        Fp32DgammaDbeta2Gm(outChannelIdx, dgammaGm, dgammaUb, dbetaGm, dbetaUb);
    } else if constexpr (!isDeterministic && !IsSameType<T, float>::value) {
        outChannelIdx = (taskIdx % this->g) * this->cG;
        if (n == 1) {
            NonFp32DgammaDbeta2Gm(
                outChannelIdx, dgammaGm, dgammaUb, dbetaGm, dbetaUb, outQueueDgammaT, outQueueDbetaT);
        } else {
            Fp32DgammaDbeta2Gm(outChannelIdx, dgammaWorkspace, dgammaUb, dbetaWorkspace, dbetaUb);
        }
    } else if constexpr (isDeterministic) {
        outChannelIdx = ((taskIdx / this->g) / splitCount * this->g + (taskIdx % this->g)) * this->cG;
        Fp32DgammaDbeta2Gm(outChannelIdx, dgammaWorkspace, dgammaUb, dbetaWorkspace, dbetaUb);
    }
}

template <typename T, bool isDeterministic>
__aicore__ inline uint32_t GroupNormSwishGrad<T, isDeterministic>::DivCeil(uint32_t a, uint32_t b)
{
    if (b != 0) {
        return (a - 1) / b + 1;
    } else {
        return 0;
    }
}

template <typename T, bool isDeterministic>
__aicore__ inline uint32_t GroupNormSwishGrad<T, isDeterministic>::Ceil(uint32_t a, uint32_t b)
{
    if (b != 0) {
        return ((a - 1) / b + 1) * b;
    } else {
        return 0;
    }
}

template <typename T, bool isDeterministic>
__aicore__ inline uint32_t GroupNormSwishGrad<T, isDeterministic>::Floor(uint32_t a, uint32_t b)
{
    if (b != 0) {
        return (a / b * b);
    } else {
        return 0;
    }
}

// AICORE wrappers that prepare raw UB pointers for VF kernels.

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::MulsAddsTemplate(
    const LocalTensor<float>& srcLocal, const LocalTensor<float>& meanRstdLocal, int32_t calCount)
{
    __ubuf__ float* src0Ptr = (__ubuf__ float*)srcLocal.GetPhyAddr();
    __ubuf__ float* meanRstdPtr = (__ubuf__ float*)meanRstdLocal.GetPhyAddr();
    PipeBarrier<PIPE_V>();
    MulsAddsVf(src0Ptr, meanRstdPtr, calCount);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::SwishGradMulXReduceTemplate(
    const LocalTensor<float>& temp1Local, const LocalTensor<float>& temp2Local,
    const LocalTensor<float>& dyNewSumLocal, const LocalTensor<float>& xMulDySumLocal,
    const LocalTensor<float>& xLocal, const LocalTensor<float>& dyLocal, const LocalTensor<float>& gammaLocal,
    const LocalTensor<float>& betaLocal, uint32_t cGIdx, float swishScaleValue, uint32_t calCount)
{
    __ubuf__ float* temp1Ptr = (__ubuf__ float*)temp1Local.GetPhyAddr();
    __ubuf__ float* temp2Ptr = (__ubuf__ float*)temp2Local.GetPhyAddr();
    __ubuf__ float* dyNewSumPtr = (__ubuf__ float*)dyNewSumLocal.GetPhyAddr();
    __ubuf__ float* xMulDySumPtr = (__ubuf__ float*)xMulDySumLocal.GetPhyAddr();
    __ubuf__ float* xPtr = (__ubuf__ float*)xLocal.GetPhyAddr();
    __ubuf__ float* dyPtr = (__ubuf__ float*)dyLocal.GetPhyAddr();
    __ubuf__ float* gammaPtr = (__ubuf__ float*)gammaLocal.GetPhyAddr();
    __ubuf__ float* betaPtr = (__ubuf__ float*)betaLocal.GetPhyAddr();
    PipeBarrier<PIPE_V>();
    SwishGradMulXReduceVf(temp1Ptr, temp2Ptr, dyNewSumPtr, xMulDySumPtr, xPtr, dyPtr, gammaPtr, betaPtr, cGIdx,
        swishScaleValue, calCount);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::SwishDxTemplate(
    const LocalTensor<float>& dxLocal, const LocalTensor<float>& xLocal, const LocalTensor<float>& dyLocal,
    const LocalTensor<float>& gammaLocal, const LocalTensor<float>& betaLocal, const LocalTensor<float>& c1Local,
    const LocalTensor<float>& c2Local, const LocalTensor<float>& meanRstdLocal, uint32_t cGIdx,
    float swishScaleValue,
    uint32_t calCount)
{
    __ubuf__ float* dxPtr = (__ubuf__ float*)dxLocal.GetPhyAddr();
    __ubuf__ float* xPtr = (__ubuf__ float*)xLocal.GetPhyAddr();
    __ubuf__ float* dyPtr = (__ubuf__ float*)dyLocal.GetPhyAddr();
    __ubuf__ float* gammaPtr = (__ubuf__ float*)gammaLocal.GetPhyAddr();
    __ubuf__ float* betaPtr = (__ubuf__ float*)betaLocal.GetPhyAddr();
    __ubuf__ float* c1Ptr = (__ubuf__ float*)c1Local.GetPhyAddr();
    __ubuf__ float* c2Ptr = (__ubuf__ float*)c2Local.GetPhyAddr();
    __ubuf__ float* meanRstdPtr = (__ubuf__ float*)meanRstdLocal.GetPhyAddr();
    PipeBarrier<PIPE_V>();
    SwishDxVf(dxPtr, xPtr, dyPtr, gammaPtr, betaPtr, c1Ptr, c2Ptr, meanRstdPtr, cGIdx, swishScaleValue, calCount);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::MulReduceSumTemplate(const LocalTensor<float>& dstLocal1,
    const LocalTensor<float>& dstLocal2, const LocalTensor<float>& srcLocal, uint32_t calCount)
{
    __ubuf__ float* src0Ptr = (__ubuf__ float*)dstLocal1.GetPhyAddr();
    __ubuf__ float* src1Ptr = (__ubuf__ float*)dstLocal2.GetPhyAddr();
    __ubuf__ float* src2Ptr = (__ubuf__ float*)srcLocal.GetPhyAddr();
    float normFactor = 1.0f / static_cast<float>(this->eleNumPerGroup);
    PipeBarrier<PIPE_V>();
    DualMulReduceSumVf(src0Ptr, src1Ptr, src2Ptr, normFactor, calCount);
    PipeBarrier<PIPE_V>();
}

// Stage-2 workspace reduction and cast helpers.

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::InitBufferForStage2Cast()
{
    pipe.InitBuffer(inQueueDgammaChannel, 1, castEleNum * sizeof(float));
    pipe.InitBuffer(inQueueDbetaChannel, 1, castEleNum * sizeof(float));
    pipe.InitBuffer(outQueueDgammaChannelT, 1, castEleNum * sizeof(T));
    pipe.InitBuffer(outQueueDbetaChannelT, 1, castEleNum * sizeof(T));
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::InitBufferForStage2Reduce()
{
    pipe.InitBuffer(calQueueDgammaReduce, 1, stage2KahanBufferCount * castEleNum * sizeof(float));
    pipe.InitBuffer(calQueueDbetaReduce, 1, stage2KahanBufferCount * castEleNum * sizeof(float));
    pipe.InitBuffer(inQueueDgammaChannel, 1, coreBatchParts * castEleNum * sizeof(float));
    pipe.InitBuffer(inQueueDbetaChannel, 1, coreBatchParts * castEleNum * sizeof(float));
}

// Deterministic non-float stage-2 first reduces FP32 workspace slices in UB,
// then casts the final channel values before GM out.
template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::InitBufferForStage2ReduceCast()
{
    pipe.InitBuffer(calQueueDgammaReduce, 1, stage2KahanBufferCount * castEleNum * sizeof(float));
    pipe.InitBuffer(calQueueDbetaReduce, 1, stage2KahanBufferCount * castEleNum * sizeof(float));
    pipe.InitBuffer(inQueueDgammaChannel, 1, coreBatchParts * castEleNum * sizeof(float));
    pipe.InitBuffer(inQueueDbetaChannel, 1, coreBatchParts * castEleNum * sizeof(float));
    pipe.InitBuffer(outQueueDgammaChannelT, 1, castEleNum * sizeof(T));
    pipe.InitBuffer(outQueueDbetaChannelT, 1, castEleNum * sizeof(T));
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::CastDgammaDbetaWsp2Gm(uint64_t channelIdx, uint32_t count)
{
    DataCopyExtParams copyParamsFp16{1, static_cast<uint32_t>(count * sizeof(half)), 0, 0, 0};
    DataCopyExtParams copyParamsFp32{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    if (dgammaIsRequire) {
        LocalTensor<float> dgammaLocal = inQueueDgammaChannel.AllocTensor<float>();
        DataCopyPad(dgammaLocal, dgammaWorkspace[channelIdx], copyParamsFp32, padParams);
        LocalTensor<T> dgammaTemp = outQueueDgammaChannelT.AllocTensor<T>();
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(dgammaTemp, dgammaLocal, RoundMode::CAST_ROUND, count);
        inQueueDgammaChannel.FreeTensor(dgammaLocal);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(dgammaGm[channelIdx], dgammaTemp, copyParamsFp16);
        outQueueDgammaChannelT.FreeTensor(dgammaTemp);
    }
    if (dbetaIsRequire) {
        LocalTensor<float> dbetaLocal = inQueueDbetaChannel.AllocTensor<float>();
        DataCopyPad(dbetaLocal, dbetaWorkspace[channelIdx], copyParamsFp32, padParams);
        LocalTensor<T> dbetaTemp = outQueueDbetaChannelT.AllocTensor<T>();
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(dbetaTemp, dbetaLocal, RoundMode::CAST_ROUND, count);
        inQueueDbetaChannel.FreeTensor(dbetaLocal);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(dbetaGm[channelIdx], dbetaTemp, copyParamsFp16);
        outQueueDbetaChannelT.FreeTensor(dbetaTemp);
    }
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::ReduceAxisNWsp2Ub(
    TQue<QuePosition::VECIN, 1>& vecInQue, const GlobalTensor<float>& workspace, uint64_t workspaceOffset,
    uint32_t repeatTime, uint32_t count, const LocalTensor<float>& sumLocal,
    const LocalTensor<float>& compensationLocal)
{
    LocalTensor<float> vecInLocal = vecInQue.AllocTensor<float>();
    DataCopyExtParams copyParamsIn{
        static_cast<uint16_t>(repeatTime), static_cast<uint32_t>(count * sizeof(float)),
        static_cast<uint32_t>((c - count) * sizeof(float)),
        static_cast<uint32_t>((castEleNum - count) / floatPerBlock), 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(vecInLocal, workspace[workspaceOffset], copyParamsIn, padParams);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    // One DataCopyPad loads several workspace rows into UB. Reduce the rows by
    // repeatedly adding the back half into the front half. One Add covers all
    // pairs in the current level, and odd rows stay in the front half.
    uint32_t activeRowCount = repeatTime;
    while (activeRowCount > 1) {
        uint32_t leftRowCount = (activeRowCount + 1) >> 1;
        uint32_t rightRowCount = activeRowCount - leftRowCount;
        Add(vecInLocal, vecInLocal, vecInLocal[leftRowCount * castEleNum], rightRowCount * castEleNum);
        PipeBarrier<PIPE_V>();
        activeRowCount = leftRowCount;
    }
    KahanAccumulateVf((__ubuf__ float*)sumLocal.GetPhyAddr(), (__ubuf__ float*)compensationLocal.GetPhyAddr(),
        (__ubuf__ float*)vecInLocal.GetPhyAddr(), count);
    vecInQue.FreeTensor(vecInLocal);
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::ReduceDgammaDbetaWsp2Gm(
    uint64_t startOffset, uint64_t channelIdx, uint32_t count, uint32_t reduceAxisNum)
{
    uint32_t repeatTime = 0;
    LocalTensor<float> dbetaReduceLocal = calQueueDbetaReduce.AllocTensor<float>();
    LocalTensor<float> dgammaReduceLocal = calQueueDgammaReduce.AllocTensor<float>();
    LocalTensor<float> dbetaSumLocal = dbetaReduceLocal;
    LocalTensor<float> dgammaSumLocal = dgammaReduceLocal;
    LocalTensor<float> dbetaCompLocal = dbetaReduceLocal[castEleNum];
    LocalTensor<float> dgammaCompLocal = dgammaReduceLocal[castEleNum];
    Duplicate<float>(dbetaSumLocal, 0.0f, castEleNum);
    Duplicate<float>(dgammaSumLocal, 0.0f, castEleNum);
    Duplicate<float>(dbetaCompLocal, 0.0f, castEleNum);
    Duplicate<float>(dgammaCompLocal, 0.0f, castEleNum);
    PipeBarrier<PIPE_V>();
    for (uint32_t loopIdx = 0; loopIdx < DivCeil(reduceAxisNum, coreBatchParts); loopIdx++) {
        uint64_t loopOffset = static_cast<uint64_t>(loopIdx) * coreBatchParts * c;
        if (loopIdx < DivCeil(reduceAxisNum, coreBatchParts) - 1) {
            repeatTime = coreBatchParts;
        } else if (loopIdx == DivCeil(reduceAxisNum, coreBatchParts) - 1) {
            repeatTime = reduceAxisNum % coreBatchParts == 0 ? coreBatchParts : reduceAxisNum % coreBatchParts;
        }
        if (dgammaIsRequire) {
            ReduceAxisNWsp2Ub(
                inQueueDgammaChannel, dgammaWorkspace, startOffset + loopOffset + channelIdx, repeatTime,
                count, dgammaSumLocal, dgammaCompLocal);
        }
        if (dbetaIsRequire) {
            ReduceAxisNWsp2Ub(
                inQueueDbetaChannel, dbetaWorkspace, startOffset + loopOffset + channelIdx, repeatTime,
                count, dbetaSumLocal, dbetaCompLocal);
        }
    }
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyExtParams copyParamsOut{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    bool hasAtomicCopy = false;
    SetAtomicAdd<float>();
    if (dgammaIsRequire) {
        DataCopyPad(dgammaGm[channelIdx], dgammaSumLocal, copyParamsOut);
        hasAtomicCopy = true;
    }
    if (dbetaIsRequire) {
        DataCopyPad(dbetaGm[channelIdx], dbetaSumLocal, copyParamsOut);
        hasAtomicCopy = true;
    }
    if (hasAtomicCopy) {
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }
    SetAtomicNone();
    calQueueDgammaReduce.FreeTensor(dgammaReduceLocal);
    calQueueDbetaReduce.FreeTensor(dbetaReduceLocal);
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::ReduceCastDgammaDbetaWsp2Gm(
    uint64_t channelIdx, uint32_t count, uint32_t reduceAxisNum)
{
    uint32_t repeatTime = 0;
    LocalTensor<float> dbetaReduceLocal = calQueueDbetaReduce.AllocTensor<float>();
    LocalTensor<float> dgammaReduceLocal = calQueueDgammaReduce.AllocTensor<float>();
    LocalTensor<float> dbetaSumLocal = dbetaReduceLocal;
    LocalTensor<float> dgammaSumLocal = dgammaReduceLocal;
    LocalTensor<float> dbetaCompLocal = dbetaReduceLocal[castEleNum];
    LocalTensor<float> dgammaCompLocal = dgammaReduceLocal[castEleNum];
    Duplicate<float>(dbetaSumLocal, 0.0f, castEleNum);
    Duplicate<float>(dgammaSumLocal, 0.0f, castEleNum);
    Duplicate<float>(dbetaCompLocal, 0.0f, castEleNum);
    Duplicate<float>(dgammaCompLocal, 0.0f, castEleNum);
    PipeBarrier<PIPE_V>();
    for (uint32_t loopIdx = 0; loopIdx < DivCeil(reduceAxisNum, coreBatchParts); loopIdx++) {
        uint64_t loopOffset = static_cast<uint64_t>(loopIdx) * coreBatchParts * c;
        if (loopIdx < DivCeil(reduceAxisNum, coreBatchParts) - 1) {
            repeatTime = coreBatchParts;
        } else if (loopIdx == DivCeil(reduceAxisNum, coreBatchParts) - 1) {
            repeatTime = reduceAxisNum % coreBatchParts == 0 ? coreBatchParts : reduceAxisNum % coreBatchParts;
        }
        if (dgammaIsRequire) {
            ReduceAxisNWsp2Ub(
                inQueueDgammaChannel, dgammaWorkspace, loopOffset + channelIdx, repeatTime, count, dgammaSumLocal,
                dgammaCompLocal);
        }
        if (dbetaIsRequire) {
            ReduceAxisNWsp2Ub(
                inQueueDbetaChannel, dbetaWorkspace, loopOffset + channelIdx, repeatTime, count, dbetaSumLocal,
                dbetaCompLocal);
        }
    }
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams copyParamsOut{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    if (dgammaIsRequire) {
        CustomDataCopyOut(dgammaSumLocal, dgammaGm, channelIdx, outQueueDgammaChannelT, count);
    }
    calQueueDgammaReduce.FreeTensor(dgammaReduceLocal);
    if (dbetaIsRequire) {
        CustomDataCopyOut(dbetaSumLocal, dbetaGm, channelIdx, outQueueDbetaChannelT, count);
    }
    calQueueDbetaReduce.FreeTensor(dbetaReduceLocal);
}

// Basic VF kernels shared by the stage-1 paths.

template <typename T, bool isDeterministic>
__simd_vf__ inline void GroupNormSwishGrad<T, isDeterministic>::MulsAddsVf(__ubuf__ float* src0Ptr,
                                                                           __ubuf__ float* meanRstdPtr,
                                                                           uint32_t calCount)
{
    AscendC::MicroAPI::MaskReg maskFull0;
    AscendC::MicroAPI::MaskReg maskFull1;
    AscendC::MicroAPI::RegTensor<float> vMean, vRstd;
    AscendC::MicroAPI::RegTensor<float> vSrc0, vSrc1;
    AscendC::MicroAPI::RegTensor<float> vTmp0, vTmp1;
    AscendC::MicroAPI::RegTensor<float> vOut0, vOut1;

    uint32_t fullRepeatTimes = (calCount + oneRepeatSizeB32 - 1) / oneRepeatSizeB32;
    uint16_t pairRepeatTimes = static_cast<uint16_t>((fullRepeatTimes + 1) / 2);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
        vMean, meanRstdPtr + meanValueOffset);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
        vRstd, meanRstdPtr + rstdValueOffset);
    for (uint16_t i = 0; i < pairRepeatTimes; i++) {
        maskFull0 = AscendC::MicroAPI::UpdateMask<float>(calCount);
        maskFull1 = AscendC::MicroAPI::UpdateMask<float>(calCount);
        uint32_t offsetA = i * 2 * oneRepeatSizeB32;
        uint32_t offsetB = offsetA + oneRepeatSizeB32;
        AscendC::MicroAPI::LoadAlign(vSrc0, src0Ptr + offsetA);
        AscendC::MicroAPI::LoadAlign(vSrc1, src0Ptr + offsetB);
        AscendC::MicroAPI::Sub(vTmp0, vSrc0, vMean, maskFull0);
        AscendC::MicroAPI::Sub(vTmp1, vSrc1, vMean, maskFull1);
        AscendC::MicroAPI::Mul(vOut0, vTmp0, vRstd, maskFull0);
        AscendC::MicroAPI::Mul(vOut1, vTmp1, vRstd, maskFull1);
        AscendC::MicroAPI::StoreAlign(src0Ptr + offsetA, vOut0, maskFull0);
        AscendC::MicroAPI::StoreAlign(src0Ptr + offsetB, vOut1, maskFull1);
    }
}

template <typename T, bool isDeterministic>
__simd_vf__ inline void GroupNormSwishGrad<T, isDeterministic>::SwishDxVf(__ubuf__ float* dxPtr,
                                                                          __ubuf__ float* xPtr,
                                                                          __ubuf__ float* dyPtr,
                                                                          __ubuf__ float* gammaPtr,
                                                                          __ubuf__ float* betaPtr,
                                                                          __ubuf__ float* c1Ptr,
                                                                          __ubuf__ float* c2Ptr,
                                                                          __ubuf__ float* meanRstdPtr,
                                                                          uint32_t cGIdx,
                                                                          float swishScaleValue,
                                                                          uint32_t calCount)
{
    AscendC::MicroAPI::MaskReg maskFull0;
    AscendC::MicroAPI::MaskReg maskFull1;
    AscendC::MicroAPI::RegTensor<float> vGamma, vBeta, vC1, vC2, vRstd;
    AscendC::MicroAPI::RegTensor<float> vX0, vDy0, vYMul0, vY0, vDenScaled0, vDenExp0, vDen0;
    AscendC::MicroAPI::RegTensor<float> vDswish0, vDyNew0, vCorrectionMul0, vCorrection0, vDx0;
    AscendC::MicroAPI::RegTensor<float> vX1, vDy1, vYMul1, vY1, vDenScaled1, vDenExp1, vDen1;
    AscendC::MicroAPI::RegTensor<float> vDswish1, vDyNew1, vCorrectionMul1, vCorrection1, vDx1;
    uint32_t fullRepeatTimes = (calCount + oneRepeatSizeB32 - 1) / oneRepeatSizeB32;
    uint16_t pairRepeatTimes = static_cast<uint16_t>((fullRepeatTimes + 1) / 2);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vGamma, gammaPtr + cGIdx);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vBeta, betaPtr + cGIdx);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vC1, c1Ptr);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vC2, c2Ptr);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
        vRstd, meanRstdPtr + rstdValueOffset);
    for (uint16_t i = 0; i < pairRepeatTimes; i++) {
        uint32_t offset0 = i * 2 * oneRepeatSizeB32;
        uint32_t offset1 = offset0 + oneRepeatSizeB32;
        maskFull0 = AscendC::MicroAPI::UpdateMask<float>(calCount);
        maskFull1 = AscendC::MicroAPI::UpdateMask<float>(calCount);
        AscendC::MicroAPI::LoadAlign(vX0, xPtr + offset0);
        AscendC::MicroAPI::LoadAlign(vDy0, dyPtr + offset0);
        AscendC::MicroAPI::LoadAlign(vX1, xPtr + offset1);
        AscendC::MicroAPI::LoadAlign(vDy1, dyPtr + offset1);
        AscendC::MicroAPI::Mul(vYMul0, vX0, vGamma, maskFull0);
        AscendC::MicroAPI::Mul(vYMul1, vX1, vGamma, maskFull1);
        AscendC::MicroAPI::Add(vY0, vYMul0, vBeta, maskFull0);
        AscendC::MicroAPI::Add(vY1, vYMul1, vBeta, maskFull1);
        AscendC::MicroAPI::Muls(vYMul0, vY0, swishScaleValue, maskFull0);
        AscendC::MicroAPI::Muls(vYMul1, vY1, swishScaleValue, maskFull1);
        AscendC::MicroAPI::Muls(vDenScaled0, vYMul0, float(-1.0), maskFull0);
        AscendC::MicroAPI::Muls(vDenScaled1, vYMul1, float(-1.0), maskFull1);
        AscendC::MicroAPI::Exp(vDenExp0, vDenScaled0, maskFull0);
        AscendC::MicroAPI::Exp(vDenExp1, vDenScaled1, maskFull1);
        AscendC::MicroAPI::Adds(vDen0, vDenExp0, float(1.0), maskFull0);
        AscendC::MicroAPI::Adds(vDen1, vDenExp1, float(1.0), maskFull1);
        AscendC::MicroAPI::Div(vDenScaled0, vYMul0, vDen0, maskFull0);
        AscendC::MicroAPI::Div(vDenScaled1, vYMul1, vDen1, maskFull1);
        AscendC::MicroAPI::Sub(vDenExp0, vYMul0, vDenScaled0, maskFull0);
        AscendC::MicroAPI::Sub(vDenExp1, vYMul1, vDenScaled1, maskFull1);
        AscendC::MicroAPI::Adds(vYMul0, vDenExp0, float(1.0), maskFull0);
        AscendC::MicroAPI::Adds(vYMul1, vDenExp1, float(1.0), maskFull1);
        AscendC::MicroAPI::Div(vDswish0, vYMul0, vDen0, maskFull0);
        AscendC::MicroAPI::Div(vDswish1, vYMul1, vDen1, maskFull1);
        AscendC::MicroAPI::Mul(vDyNew0, vDswish0, vDy0, maskFull0);
        AscendC::MicroAPI::Mul(vDyNew1, vDswish1, vDy1, maskFull1);
        AscendC::MicroAPI::Mul(vDswish0, vDyNew0, vGamma, maskFull0);
        AscendC::MicroAPI::Mul(vDswish1, vDyNew1, vGamma, maskFull1);
        AscendC::MicroAPI::Mul(vDyNew0, vDswish0, vRstd, maskFull0);
        AscendC::MicroAPI::Mul(vDyNew1, vDswish1, vRstd, maskFull1);
        AscendC::MicroAPI::Mul(vCorrectionMul0, vX0, vC2, maskFull0);
        AscendC::MicroAPI::Mul(vCorrectionMul1, vX1, vC2, maskFull1);
        AscendC::MicroAPI::Add(vCorrection0, vCorrectionMul0, vC1, maskFull0);
        AscendC::MicroAPI::Add(vCorrection1, vCorrectionMul1, vC1, maskFull1);
        AscendC::MicroAPI::Mul(vCorrectionMul0, vCorrection0, vRstd, maskFull0);
        AscendC::MicroAPI::Mul(vCorrectionMul1, vCorrection1, vRstd, maskFull1);
        AscendC::MicroAPI::Sub(vDx0, vDyNew0, vCorrectionMul0, maskFull0);
        AscendC::MicroAPI::Sub(vDx1, vDyNew1, vCorrectionMul1, maskFull1);
        AscendC::MicroAPI::StoreAlign(dxPtr + offset0, vDx0, maskFull0);
        AscendC::MicroAPI::StoreAlign(dxPtr + offset1, vDx1, maskFull1);
    }
}

// Fused VF reduction kernels shared by the stage-1 paths.

template <typename T, bool isDeterministic>
__simd_vf__ inline void GroupNormSwishGrad<T, isDeterministic>::SwishGradMulXReduceVf(__ubuf__ float* temp1Ptr,
                                                                                      __ubuf__ float* temp2Ptr,
                                                                                      __ubuf__ float* dyNewSumPtr,
                                                                                      __ubuf__ float* xMulDySumPtr,
                                                                                      __ubuf__ float* xPtr,
                                                                                      __ubuf__ float* dyPtr,
                                                                                      __ubuf__ float* gammaPtr,
                                                                                      __ubuf__ float* betaPtr,
                                                                                      uint32_t cGIdx,
                                                                                      float swishScaleValue,
                                                                                      uint32_t calCount)
{
    AscendC::MicroAPI::MaskReg maskFull0;
    AscendC::MicroAPI::MaskReg maskFull1;
    AscendC::MicroAPI::MaskReg maskScalar;
    AscendC::MicroAPI::UnalignRegForStore uStoreDyNewSum;
    AscendC::MicroAPI::UnalignRegForStore uStoreXMulDySum;
    AscendC::MicroAPI::UnalignRegForStore uStoreTemp1;
    AscendC::MicroAPI::UnalignRegForStore uStoreTemp2;
    AscendC::MicroAPI::RegTensor<float> vGamma, vBeta;
    AscendC::MicroAPI::RegTensor<float> vX0, vDy0, vYMul0, vY0, vDenScaled0, vDenExp0, vDen0;
    AscendC::MicroAPI::RegTensor<float> vDswish0, vDyNew0, vXMulDy0, vTemp1Acc;
    AscendC::MicroAPI::RegTensor<float> vX1, vDy1, vYMul1, vY1, vDenScaled1, vDenExp1, vDen1;
    AscendC::MicroAPI::RegTensor<float> vDswish1, vDyNew1, vXMulDy1, vTemp2Acc;
    uint32_t storeCount = 1;
    maskScalar = AscendC::MicroAPI::UpdateMask<float>(storeCount);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vGamma, gammaPtr + cGIdx);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vBeta, betaPtr + cGIdx);

    uint32_t fullRepeatTimes = (calCount + oneRepeatSizeB32 - 1) / oneRepeatSizeB32;
    uint16_t pairRepeatTimes = static_cast<uint16_t>((fullRepeatTimes + 1) / 2);
    __ubuf__ float* dyNewSumStorePtr = dyNewSumPtr;
    __ubuf__ float* xMulDySumStorePtr = xMulDySumPtr;
    // First pass: compute two VLs per iteration and store one scalar sum per VL.
    // The loop stays branch-free for the VF compiler; masks carry the tail.
    for (uint16_t i = 0; i < pairRepeatTimes; i++) {
        maskFull0 = AscendC::MicroAPI::UpdateMask<float>(calCount);
        maskFull1 = AscendC::MicroAPI::UpdateMask<float>(calCount);
        uint32_t offset0 = i * 2 * oneRepeatSizeB32;
        uint32_t offset1 = offset0 + oneRepeatSizeB32;
        AscendC::MicroAPI::LoadAlign(vX0, xPtr + offset0);
        AscendC::MicroAPI::LoadAlign(vDy0, dyPtr + offset0);
        AscendC::MicroAPI::LoadAlign(vX1, xPtr + offset1);
        AscendC::MicroAPI::LoadAlign(vDy1, dyPtr + offset1);
        AscendC::MicroAPI::Mul(vYMul0, vX0, vGamma, maskFull0);
        AscendC::MicroAPI::Mul(vYMul1, vX1, vGamma, maskFull1);
        AscendC::MicroAPI::Add(vY0, vYMul0, vBeta, maskFull0);
        AscendC::MicroAPI::Add(vY1, vYMul1, vBeta, maskFull1);
        AscendC::MicroAPI::Muls(vYMul0, vY0, swishScaleValue, maskFull0);
        AscendC::MicroAPI::Muls(vYMul1, vY1, swishScaleValue, maskFull1);
        AscendC::MicroAPI::Muls(vDenScaled0, vYMul0, float(-1.0), maskFull0);
        AscendC::MicroAPI::Muls(vDenScaled1, vYMul1, float(-1.0), maskFull1);
        AscendC::MicroAPI::Exp(vDenExp0, vDenScaled0, maskFull0);
        AscendC::MicroAPI::Exp(vDenExp1, vDenScaled1, maskFull1);
        AscendC::MicroAPI::Adds(vDen0, vDenExp0, float(1.0), maskFull0);
        AscendC::MicroAPI::Adds(vDen1, vDenExp1, float(1.0), maskFull1);
        AscendC::MicroAPI::Div(vDenScaled0, vYMul0, vDen0, maskFull0);
        AscendC::MicroAPI::Div(vDenScaled1, vYMul1, vDen1, maskFull1);
        AscendC::MicroAPI::Sub(vDenExp0, vYMul0, vDenScaled0, maskFull0);
        AscendC::MicroAPI::Sub(vDenExp1, vYMul1, vDenScaled1, maskFull1);
        AscendC::MicroAPI::Adds(vYMul0, vDenExp0, float(1.0), maskFull0);
        AscendC::MicroAPI::Adds(vYMul1, vDenExp1, float(1.0), maskFull1);
        AscendC::MicroAPI::Div(vDswish0, vYMul0, vDen0, maskFull0);
        AscendC::MicroAPI::Div(vDswish1, vYMul1, vDen1, maskFull1);
        AscendC::MicroAPI::Mul(vDyNew0, vDswish0, vDy0, maskFull0);
        AscendC::MicroAPI::Mul(vDyNew1, vDswish1, vDy1, maskFull1);
        AscendC::MicroAPI::Mul(vXMulDy0, vX0, vDyNew0, maskFull0);
        AscendC::MicroAPI::Mul(vXMulDy1, vX1, vDyNew1, maskFull1);
        AscendC::MicroAPI::ReduceSum(vYMul0, vDyNew0, maskFull0);
        AscendC::MicroAPI::ReduceSum(vYMul1, vDyNew1, maskFull1);
        AscendC::MicroAPI::ReduceSum(vDenScaled0, vXMulDy0, maskFull0);
        AscendC::MicroAPI::ReduceSum(vDenScaled1, vXMulDy1, maskFull1);
        AscendC::MicroAPI::StoreUnAlign(dyNewSumStorePtr, vYMul0, uStoreDyNewSum, static_cast<uint32_t>(1));
        AscendC::MicroAPI::StoreUnAlign(dyNewSumStorePtr, vYMul1, uStoreDyNewSum, static_cast<uint32_t>(1));
        AscendC::MicroAPI::StoreUnAlign(xMulDySumStorePtr, vDenScaled0, uStoreXMulDySum, static_cast<uint32_t>(1));
        AscendC::MicroAPI::StoreUnAlign(xMulDySumStorePtr, vDenScaled1, uStoreXMulDySum, static_cast<uint32_t>(1));
    }
    AscendC::MicroAPI::StoreUnAlignPost(dyNewSumStorePtr, uStoreDyNewSum, static_cast<int32_t>(0));
    AscendC::MicroAPI::StoreUnAlignPost(xMulDySumStorePtr, uStoreXMulDySum, static_cast<int32_t>(0));
    uint32_t sumReduceCount = fullRepeatTimes;
    uint32_t sumRepeatTimes = (fullRepeatTimes + oneRepeatSizeB32 - 1) / oneRepeatSizeB32;
    dyNewSumStorePtr = dyNewSumPtr;
    xMulDySumStorePtr = xMulDySumPtr;
    // StoreUnAlign writes are read back as vectors below, so fence VEC_STORE to
    // VEC_LOAD before each reduction pass over the scratch scalars.
    for (uint16_t i = 0; i < sumRepeatTimes; i++) {
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        maskFull0 = AscendC::MicroAPI::UpdateMask<float>(sumReduceCount);
        uint32_t offset = i * oneRepeatSizeB32;
        AscendC::MicroAPI::LoadAlign(vDyNew0, dyNewSumPtr + offset);
        AscendC::MicroAPI::LoadAlign(vXMulDy0, xMulDySumPtr + offset);
        AscendC::MicroAPI::ReduceSum(vYMul0, vDyNew0, maskFull0);
        AscendC::MicroAPI::ReduceSum(vDenScaled0, vXMulDy0, maskFull0);
        AscendC::MicroAPI::StoreUnAlign(dyNewSumStorePtr, vYMul0, uStoreDyNewSum, static_cast<uint32_t>(1));
        AscendC::MicroAPI::StoreUnAlign(xMulDySumStorePtr, vDenScaled0, uStoreXMulDySum, static_cast<uint32_t>(1));
    }
    AscendC::MicroAPI::StoreUnAlignPost(dyNewSumStorePtr, uStoreDyNewSum, static_cast<int32_t>(0));
    AscendC::MicroAPI::StoreUnAlignPost(xMulDySumStorePtr, uStoreXMulDySum, static_cast<int32_t>(0));

    uint32_t finalReduceCount0 = sumRepeatTimes;
    uint32_t finalReduceCount1 = sumRepeatTimes;
    uint32_t finalRepeatTimes = (sumRepeatTimes + oneRepeatSizeB32 - 1) / oneRepeatSizeB32;
    dyNewSumStorePtr = dyNewSumPtr;
    xMulDySumStorePtr = xMulDySumPtr;
    for (uint16_t i = 0; i < finalRepeatTimes; i++) {
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        maskFull0 = AscendC::MicroAPI::UpdateMask<float>(finalReduceCount0);
        maskFull1 = AscendC::MicroAPI::UpdateMask<float>(finalReduceCount1);
        uint32_t offset = i * oneRepeatSizeB32;
        AscendC::MicroAPI::LoadAlign(vDyNew0, dyNewSumPtr + offset);
        AscendC::MicroAPI::LoadAlign(vXMulDy0, xMulDySumPtr + offset);
        AscendC::MicroAPI::ReduceSum(vYMul0, vDyNew0, maskFull0);
        AscendC::MicroAPI::ReduceSum(vDenScaled0, vXMulDy0, maskFull1);
        AscendC::MicroAPI::StoreUnAlign(dyNewSumStorePtr, vYMul0, uStoreDyNewSum, static_cast<uint32_t>(1));
        AscendC::MicroAPI::StoreUnAlign(xMulDySumStorePtr, vDenScaled0, uStoreXMulDySum, static_cast<uint32_t>(1));
    }
    AscendC::MicroAPI::StoreUnAlignPost(dyNewSumStorePtr, uStoreDyNewSum, static_cast<int32_t>(0));
    AscendC::MicroAPI::StoreUnAlignPost(xMulDySumStorePtr, uStoreXMulDySum, static_cast<int32_t>(0));

    uint32_t lastReduceCount0 = finalRepeatTimes;
    uint32_t lastReduceCount1 = finalRepeatTimes;
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    maskFull0 = AscendC::MicroAPI::UpdateMask<float>(lastReduceCount0);
    maskFull1 = AscendC::MicroAPI::UpdateMask<float>(lastReduceCount1);
    AscendC::MicroAPI::LoadAlign(vDyNew0, dyNewSumPtr);
    AscendC::MicroAPI::LoadAlign(vXMulDy0, xMulDySumPtr);
    AscendC::MicroAPI::ReduceSum(vYMul0, vDyNew0, maskFull0);
    AscendC::MicroAPI::ReduceSum(vDenScaled0, vXMulDy0, maskFull1);
    __ubuf__ float* temp1LoadPtr = temp1Ptr + cGIdx;
    __ubuf__ float* temp2LoadPtr = temp2Ptr + cGIdx;
    __ubuf__ float* temp1StorePtr = temp1Ptr + cGIdx;
    __ubuf__ float* temp2StorePtr = temp2Ptr + cGIdx;
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vTemp1Acc, temp1LoadPtr);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vTemp2Acc, temp2LoadPtr);
    AscendC::MicroAPI::Add(vYMul1, vYMul0, vTemp1Acc, maskScalar);
    AscendC::MicroAPI::Add(vDenScaled1, vDenScaled0, vTemp2Acc, maskScalar);
    AscendC::MicroAPI::StoreUnAlign(temp1StorePtr, vYMul1, uStoreTemp1, static_cast<uint32_t>(1));
    AscendC::MicroAPI::StoreUnAlignPost(temp1StorePtr, uStoreTemp1, static_cast<int32_t>(0));
    AscendC::MicroAPI::StoreUnAlign(temp2StorePtr, vDenScaled1, uStoreTemp2, static_cast<uint32_t>(1));
    AscendC::MicroAPI::StoreUnAlignPost(temp2StorePtr, uStoreTemp2, static_cast<int32_t>(0));
}

template <typename T, bool isDeterministic>
__simd_vf__ inline void GroupNormSwishGrad<T, isDeterministic>::DualMulReduceSumVf(__ubuf__ float* dstLocal0,
                                                                                   __ubuf__ float* dstLocal1,
                                                                                   __ubuf__ float* dstLocal2,
                                                                                   float normFactor,
                                                                                   uint32_t calCount)
{
    AscendC::MicroAPI::MaskReg maskTail;
    AscendC::MicroAPI::MaskReg maskScalar;
    AscendC::MicroAPI::MaskReg reduceMask;
    AscendC::MicroAPI::UnalignRegForStore uStoreDst0;
    AscendC::MicroAPI::UnalignRegForStore uStoreDst1;
    AscendC::MicroAPI::RegTensor<float> vDst0, vDst1, vDst2;
    AscendC::MicroAPI::RegTensor<float> vTmp1, vTmp2;
    AscendC::MicroAPI::RegTensor<float> vSum1, vSum2;
    uint32_t storeCount = 1;
    __ubuf__ float* oriDstLocal0 = dstLocal0;
    __ubuf__ float* oriDstLocal1 = dstLocal1;
    __ubuf__ float* oriDstLocal2 = dstLocal2;
    maskScalar = AscendC::MicroAPI::UpdateMask<float>(storeCount);
    uint32_t repeatTimes = (calCount + oneRepeatSizeB32 - 1) / oneRepeatSizeB32;
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    // Produce per-VL partial sums for both reduction streams in one pass.
    for (uint16_t i = 0; i < repeatTimes; i++) {
        uint32_t offset = i * oneRepeatSizeB32;
        maskTail = AscendC::MicroAPI::UpdateMask<float>(calCount);
        AscendC::MicroAPI::LoadAlign(vDst0, oriDstLocal0 + offset);
        AscendC::MicroAPI::LoadAlign(vDst1, oriDstLocal1 + offset);
        AscendC::MicroAPI::LoadAlign(vDst2, oriDstLocal2 + offset);
        AscendC::MicroAPI::Mul(vTmp1, vDst0, vDst2, maskTail);
        AscendC::MicroAPI::Mul(vTmp2, vDst1, vDst2, maskTail);
        AscendC::MicroAPI::ReduceSum(vSum1, vTmp1, maskTail);
        AscendC::MicroAPI::ReduceSum(vSum2, vTmp2, maskTail);
        AscendC::MicroAPI::StoreUnAlign(dstLocal0, vSum1, uStoreDst0, static_cast<uint32_t>(1));
        AscendC::MicroAPI::StoreUnAlign(dstLocal1, vSum2, uStoreDst1, static_cast<uint32_t>(1));
    }
    AscendC::MicroAPI::StoreUnAlignPost(dstLocal0, uStoreDst0, static_cast<int32_t>(0));
    AscendC::MicroAPI::StoreUnAlignPost(dstLocal1, uStoreDst1, static_cast<int32_t>(0));
    reduceMask = AscendC::MicroAPI::UpdateMask<float>(repeatTimes);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    AscendC::MicroAPI::LoadAlign(vDst1, oriDstLocal0);
    AscendC::MicroAPI::LoadAlign(vDst2, oriDstLocal1);
    AscendC::MicroAPI::ReduceSum(vSum1, vDst1, reduceMask);
    AscendC::MicroAPI::ReduceSum(vSum2, vDst2, reduceMask);
    AscendC::MicroAPI::Muls(vTmp1, vSum1, normFactor, maskScalar);
    AscendC::MicroAPI::Muls(vTmp2, vSum2, normFactor, maskScalar);
    AscendC::MicroAPI::StoreUnAlign(oriDstLocal0, vTmp1, uStoreDst0, static_cast<uint32_t>(1));
    AscendC::MicroAPI::StoreUnAlign(oriDstLocal1, vTmp2, uStoreDst1, static_cast<uint32_t>(1));
    AscendC::MicroAPI::StoreUnAlignPost(oriDstLocal0, uStoreDst0, static_cast<int32_t>(0));
    AscendC::MicroAPI::StoreUnAlignPost(oriDstLocal1, uStoreDst1, static_cast<int32_t>(0));
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
}

template <typename T, bool isDeterministic>
__simd_vf__ inline void GroupNormSwishGrad<T, isDeterministic>::KahanAccumulateVf(
    __ubuf__ float* sumPtr, __ubuf__ float* compensationPtr, __ubuf__ float* chunkPtr, uint32_t calCount)
{
    AscendC::MicroAPI::MaskReg maskTail0;
    AscendC::MicroAPI::MaskReg maskTail1;
    AscendC::MicroAPI::RegTensor<float> vChunk0, vSum0, vCompensation0;
    AscendC::MicroAPI::RegTensor<float> vY0, vTemp0, vCompensationDelta0, vNewCompensation0;
    AscendC::MicroAPI::RegTensor<float> vChunk1, vSum1, vCompensation1;
    AscendC::MicroAPI::RegTensor<float> vY1, vTemp1, vCompensationDelta1, vNewCompensation1;

    uint32_t updateCount = calCount;
    uint32_t fullRepeatTimes = (calCount + oneRepeatSizeB32 - 1) / oneRepeatSizeB32;
    uint16_t pairRepeatTimes = static_cast<uint16_t>((fullRepeatTimes + 1) >> 1);
    // Kahan: y = chunk - c; t = sum + y; c = (t - sum) - y; sum = t.
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    for (uint16_t i = 0; i < pairRepeatTimes; i++) {
        uint32_t offset0 = (static_cast<uint32_t>(i) << 1) * oneRepeatSizeB32;
        uint32_t offset1 = offset0 + oneRepeatSizeB32;
        maskTail0 = AscendC::MicroAPI::UpdateMask<float>(updateCount);
        maskTail1 = AscendC::MicroAPI::UpdateMask<float>(updateCount);
        AscendC::MicroAPI::LoadAlign(vChunk0, chunkPtr + offset0);
        AscendC::MicroAPI::LoadAlign(vSum0, sumPtr + offset0);
        AscendC::MicroAPI::LoadAlign(vCompensation0, compensationPtr + offset0);
        AscendC::MicroAPI::LoadAlign(vChunk1, chunkPtr + offset1);
        AscendC::MicroAPI::LoadAlign(vSum1, sumPtr + offset1);
        AscendC::MicroAPI::LoadAlign(vCompensation1, compensationPtr + offset1);
        AscendC::MicroAPI::Sub(vY0, vChunk0, vCompensation0, maskTail0);
        AscendC::MicroAPI::Sub(vY1, vChunk1, vCompensation1, maskTail1);
        AscendC::MicroAPI::Add(vTemp0, vSum0, vY0, maskTail0);
        AscendC::MicroAPI::Add(vTemp1, vSum1, vY1, maskTail1);
        AscendC::MicroAPI::Sub(vCompensationDelta0, vTemp0, vSum0, maskTail0);
        AscendC::MicroAPI::Sub(vCompensationDelta1, vTemp1, vSum1, maskTail1);
        AscendC::MicroAPI::Sub(vNewCompensation0, vCompensationDelta0, vY0, maskTail0);
        AscendC::MicroAPI::Sub(vNewCompensation1, vCompensationDelta1, vY1, maskTail1);
        AscendC::MicroAPI::StoreAlign(compensationPtr + offset0, vNewCompensation0, maskTail0);
        AscendC::MicroAPI::StoreAlign(compensationPtr + offset1, vNewCompensation1, maskTail1);
        AscendC::MicroAPI::StoreAlign(sumPtr + offset0, vTemp0, maskTail0);
        AscendC::MicroAPI::StoreAlign(sumPtr + offset1, vTemp1, maskTail1);
    }
}

#endif // GROUP_NORM_SWISH_GRAD_UTILS_H
