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
 * \file group_norm_swish_grad.h
 * \brief
 */

#ifndef GROUP_NORM_SWISH_GRAD_H
#define GROUP_NORM_SWISH_GRAD_H

#include "kernel_operator.h"
using namespace AscendC;

template <typename T, bool isDeterministic>
class GroupNormSwishGrad
{
public:
    __aicore__ inline GroupNormSwishGrad(
        GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR dx, GM_ADDR dgamma,
        GM_ADDR dbeta, GM_ADDR workspace, GroupNormSwishGradTilingData* tilingData);

    __aicore__ inline void Process();

private:
    // UB can hold one complete group (C/G * HxW).
    __aicore__ inline void CopyInGroupFitUb(int32_t taskIdx);

    __aicore__ inline void ComputeGroupFitUb(int32_t taskIdx);

    __aicore__ inline void InitGroupFitUbBuffer();

    static __simd_vf__ inline void SwishGradMulXReduceUnAlignVf(__ubuf__ float* temp1Ptr,
                                                                 __ubuf__ float* temp2Ptr,
                                                                 __ubuf__ float* dyNewSumPtr,
                                                                 __ubuf__ float* xMulDySumPtr,
                                                                 __ubuf__ float* xPtr,
                                                                 __ubuf__ float* dyPtr,
                                                                 __ubuf__ float* gammaPtr,
                                                                 __ubuf__ float* betaPtr,
                                                                 uint32_t cGIdx,
                                                                 float swishScaleValue,
                                                                 uint32_t calCount);

    __aicore__ inline void SwishGradMulXReduceUnAlignTemplate(const LocalTensor<float>& temp1Local,
        const LocalTensor<float>& temp2Local, const LocalTensor<float>& dyNewSumLocal,
        const LocalTensor<float>& xMulDySumLocal, const LocalTensor<float>& xLocal, const LocalTensor<float>& dyLocal,
        const LocalTensor<float>& gammaLocal, const LocalTensor<float>& betaLocal, uint32_t cGIdx,
        float swishScaleValue,
        uint32_t calCount);

    static __simd_vf__ inline void SwishDxUnAlignVf(__ubuf__ float* dxPtr,
                                                    __ubuf__ float* xPtr,
                                                    __ubuf__ float* dyPtr,
                                                    __ubuf__ float* gammaPtr,
                                                    __ubuf__ float* betaPtr,
                                                    __ubuf__ float* c1Ptr,
                                                    __ubuf__ float* c2Ptr,
                                                    __ubuf__ float* meanRstdPtr,
                                                    uint32_t cGIdx,
                                                    float swishScaleValue,
                                                    uint32_t calCount);

    __aicore__ inline void SwishDxUnAlignTemplate(const LocalTensor<float>& dxLocal, const LocalTensor<float>& xLocal,
        const LocalTensor<float>& dyLocal, const LocalTensor<float>& gammaLocal, const LocalTensor<float>& betaLocal,
        const LocalTensor<float>& c1Local, const LocalTensor<float>& c2Local,
        const LocalTensor<float>& meanRstdLocal, uint32_t cGIdx, float swishScaleValue, uint32_t calCount);

    // UB can hold one complete channel HxW, but not the whole group.
    __aicore__ inline void ComputeChannelFitUb(int32_t taskIdx);

    __aicore__ inline void InitChannelFitUbBuffer();

    // One channel HxW is split into multiple UB-sized segments.
    __aicore__ inline void CopyInChannelSplitUb(
        uint64_t offset, uint32_t processNum, const LocalTensor<float>& meanRstdLocal);

    __aicore__ inline void ComputeDgammaDbetaChannelSplitUb(int32_t iterCGIdx,
        const LocalTensor<float>& temp1Local, const LocalTensor<float>& temp2Local,
        const LocalTensor<float>& gammaLocal, const LocalTensor<float>& betaLocal, uint64_t offset,
        uint32_t processNum);

    __aicore__ inline void ComputeChannelSplitUb(int32_t taskIdx);

    __aicore__ inline void InitChannelSplitUbBuffer(GroupNormSwishGradTilingData* tilingData);

    // Common initialization, IO, and scalar helpers.
    __aicore__ inline void InitOutputGm();

    __aicore__ inline void InitUnifiedBuffer(GroupNormSwishGradTilingData* tilingData);

    __aicore__ inline void CopyMeanRstdToUb(LocalTensor<float>& meanRstdLocal, int32_t taskIdx);

    __aicore__ inline void CustomDataCopyIn(
        const LocalTensor<float>& inLocal, TQue<QuePosition::VECIN, 1>& inQue, const GlobalTensor<T>& gm,
        const uint64_t offset, const uint32_t count);

    __aicore__ inline void CustomDataCopyOut(
        LocalTensor<float>& outLocal, const uint64_t gmOffset, TQue<QuePosition::VECOUT, 1>& outQue,
        const uint32_t count);

    __aicore__ inline void CustomDataCopyOut(
        LocalTensor<float>& outLocal, const uint32_t ubOffset, const uint64_t gmOffset,
        TQue<QuePosition::VECOUT, 1>& outQue, const uint32_t count);

    __aicore__ inline void CustomDataCopyOut(
        LocalTensor<float>& outLocal, GlobalTensor<T>& gmOut, const uint64_t gmOffset,
        TQue<QuePosition::VECOUT, 1>& outQue,
        const uint32_t count);

    __aicore__ inline void Fp32DgammaDbeta2Gm(
        uint32_t channelIdx, GlobalTensor<float>& dgammaOut, const LocalTensor<float>& dgammaUb,
        GlobalTensor<float>& dbetaOut, const LocalTensor<float>& dbetaUb);

    __aicore__ inline void NonFp32DgammaDbeta2Gm(
        uint32_t channelIdx, GlobalTensor<T>& dgammaOut, const LocalTensor<float>& dgammaUb,
        GlobalTensor<T>& dbetaOut, const LocalTensor<float>& dbetaUb,
        TQue<QuePosition::VECOUT, 1>& dgammaOutQue, TQue<QuePosition::VECOUT, 1>& dbetaOutQue);

    __aicore__ inline void ModeSelection(
        int32_t taskIdx, const LocalTensor<float>& dgammaUb, const LocalTensor<float>& dbetaUb);

    __aicore__ inline uint32_t DivCeil(uint32_t a, uint32_t b);

    __aicore__ inline uint32_t Ceil(uint32_t a, uint32_t b);

    __aicore__ inline uint32_t Floor(uint32_t a, uint32_t b);

    // AICORE wrappers around VF kernels.
    static __simd_vf__ inline void MulsAddsVf(__ubuf__ float* src0Ptr,
                                              __ubuf__ float* meanRstdPtr,
                                              uint32_t calCount);

    __aicore__ inline void MulsAddsTemplate(
        const LocalTensor<float>& srcLocal, const LocalTensor<float>& meanRstdLocal, int32_t calCount);

    static __simd_vf__ inline void SwishGradMulXReduceVf(__ubuf__ float* temp1Ptr,
                                                          __ubuf__ float* temp2Ptr,
                                                          __ubuf__ float* dyNewSumPtr,
                                                          __ubuf__ float* xMulDySumPtr,
                                                          __ubuf__ float* xPtr,
                                                          __ubuf__ float* dyPtr,
                                                          __ubuf__ float* gammaPtr,
                                                          __ubuf__ float* betaPtr,
                                                          uint32_t cGIdx,
                                                          float swishScaleValue,
                                                          uint32_t calCount);

    __aicore__ inline void SwishGradMulXReduceTemplate(const LocalTensor<float>& temp1Local,
        const LocalTensor<float>& temp2Local, const LocalTensor<float>& dyNewSumLocal,
        const LocalTensor<float>& xMulDySumLocal, const LocalTensor<float>& xLocal, const LocalTensor<float>& dyLocal,
        const LocalTensor<float>& gammaLocal, const LocalTensor<float>& betaLocal, uint32_t cGIdx,
        float swishScaleValue,
        uint32_t calCount);

    static __simd_vf__ inline void SwishDxVf(__ubuf__ float* dxPtr,
                                             __ubuf__ float* xPtr,
                                             __ubuf__ float* dyPtr,
                                             __ubuf__ float* gammaPtr,
                                             __ubuf__ float* betaPtr,
                                             __ubuf__ float* c1Ptr,
                                             __ubuf__ float* c2Ptr,
                                             __ubuf__ float* meanRstdPtr,
                                             uint32_t cGIdx,
                                             float swishScaleValue,
                                             uint32_t calCount);

    __aicore__ inline void SwishDxTemplate(const LocalTensor<float>& dxLocal, const LocalTensor<float>& xLocal,
        const LocalTensor<float>& dyLocal, const LocalTensor<float>& gammaLocal, const LocalTensor<float>& betaLocal,
        const LocalTensor<float>& c1Local, const LocalTensor<float>& c2Local,
        const LocalTensor<float>& meanRstdLocal, uint32_t cGIdx, float swishScaleValue, uint32_t calCount);

    static __simd_vf__ inline void DualMulReduceSumVf(__ubuf__ float* src0Ptr,
                                             __ubuf__ float* src1Ptr,
                                             __ubuf__ float* src2Ptr,
                                             float normFactor,
                                             uint32_t calCount);

    static __simd_vf__ inline void KahanAccumulateVf(__ubuf__ float* sumPtr,
                                                     __ubuf__ float* compensationPtr,
                                                     __ubuf__ float* chunkPtr,
                                                     uint32_t calCount);

    __aicore__ inline void MulReduceSumTemplate(const LocalTensor<float>& dstLocal1,
        const LocalTensor<float>& dstLocal2, const LocalTensor<float>& srcLocal, uint32_t calCount);

    __aicore__ inline void InitBufferForStage2Cast();

    __aicore__ inline void InitBufferForStage2Reduce();

    // Deterministic non-float stage-2 reduction uses both reduce queues and cast buffers.
    __aicore__ inline void InitBufferForStage2ReduceCast();

    __aicore__ inline void CastDgammaDbetaWsp2Gm(uint64_t channelIdx, uint32_t count);

    __aicore__ inline void ReduceAxisNWsp2Ub(
        TQue<QuePosition::VECIN, 1>& vecInQue, const GlobalTensor<float>& workspace, uint64_t workspaceOffset,
        uint32_t repeatTime, uint32_t count, const LocalTensor<float>& sumLocal,
        const LocalTensor<float>& compensationLocal);

    __aicore__ inline void ReduceDgammaDbetaWsp2Gm(
        uint64_t startOffset, uint64_t channelIdx, uint32_t count, uint32_t reduceAxisNum);

    __aicore__ inline void ReduceCastDgammaDbetaWsp2Gm(
        uint64_t channelIdx, uint32_t count, uint32_t reduceAxisNum);

private:
    // Pipe object
    TPipe pipe;
    // Que object
    TQue<QuePosition::VECIN, 1> inQueueDy, inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGamma, inQueueBeta;
    TQue<QuePosition::VECIN, 1> inQueueDgammaChannel, inQueueDbetaChannel;
    TQue<QuePosition::VECIN, 1> calQueueDgammaReduce, calQueueDbetaReduce;
    TQue<QuePosition::VECIN, 1> inQueueDyT, inQueueXT, inQueueGammaT, inQueueBetaT, inQueueMeanRstdT;
    TQue<QuePosition::VECOUT, 1> outQueueDxT;
    TQue<QuePosition::VECOUT, 1> outQueueDgammaT, outQueueDbetaT;
    TQue<QuePosition::VECOUT, 1> outQueueDgammaChannelT, outQueueDbetaChannelT;
    TBuf<TPosition::VECCALC> inTbufMeanRstd;
    TBuf<TPosition::VECCALC> outTbufDyNew, outTbufDswish;
    // Que for output
    TQue<QuePosition::VECOUT, 1> outQueueDgamma, outQueueDbeta, tempQueueHxw;
    // Global Memory
    GlobalTensor<T> dyGm, meanGm, rstdGm, xGm, gammaGm, betaGm;
    GlobalTensor<float> dgammaWorkspace, dbetaWorkspace;
    GlobalTensor<T> dxGm, dgammaGm, dbetaGm;
    uint32_t tilingKey;
    bool isGammaEqual = true;
    static constexpr uint32_t oneRepeatSizeB32 = AscendC::GetVecLen() / sizeof(float);
    static constexpr uint32_t b32EleNumPerBlock = 8;
    static constexpr uint32_t meanValueOffset = 0;
    static constexpr uint32_t rstdValueOffset = b32EleNumPerBlock;
    // Mean/rstd are scalar values, but VF BRC loads require separate 32B-aligned UB slots:
    // slot0[0] stores mean and slot1[0] stores rstd.
    static constexpr uint32_t meanRstdAlignedBlockCount = 2;
    static constexpr uint32_t meanRstdAlignedFloatEleCount = meanRstdAlignedBlockCount * b32EleNumPerBlock;
    // One 32B block contains 16 half/bfloat16_t elements; use two blocks to avoid temp reuse hazards.
    static constexpr uint32_t b16EleNumPerBlock = 16;
    static constexpr uint32_t meanRstdTempEleCount = meanRstdAlignedBlockCount * b16EleNumPerBlock;
    uint32_t n;
    uint32_t c;
    uint32_t g;
    uint32_t hxw;
    uint32_t nxg;
    uint32_t cG;
    uint64_t allEleNum;
    // number of calculations on each core
    uint64_t eleNumPerGroup;
    // number of tiles on each core
    uint64_t eleNumPerChannel;
    uint32_t taskIdx;
    // number of calculations in each tile
    uint32_t tileLength;
    uint32_t taskNumPerCore;
    uint32_t taskNumPerTailCore;
    uint32_t tailCore;
    uint32_t curCoreTaskNum;
    uint32_t workspaceSize;
    uint32_t stage2CoreUsed;
    uint32_t castEleNum;
    uint32_t tailCastNum;
    int32_t curBlockIdx;
    int32_t startTaskId;
    uint32_t channelTileCapacityEle;
    uint32_t channelTileIterationNum;
    uint32_t channelTileTailNum;
    uint32_t coreBatchParts = 0;
    bool dxIsRequire;
    bool dgammaIsRequire;
    bool dbetaIsRequire;
    float swishScale;
    uint32_t tPerBlock;
    const uint32_t floatPerBlock = b32EleNumPerBlock;
    const uint16_t dstBlkStride = 1;
    const uint16_t srcBlkStride = 1;
    const uint8_t dstRepStride = 8;
    const uint8_t srcRepStride = 8;
    const uint32_t blockBytes = 32;
    const uint32_t maxRepeatTimes = 255;
    const uint32_t comRepeatTimes = 64;
    // Stage-1 UB tiling keys. workspaceStage2TilingKey is reserved for stage-2 buffer selection.
    const int32_t groupFitUbTilingKey = 0; // Whole group fits in UB.
    const int32_t channelFitUbTilingKey = 1; // One channel fits in UB.
    const int32_t workspaceStage2TilingKey = 2; // Stage-2 reduction/cast buffer variant.
    const int32_t channelSplitUbTilingKey = 3; // One channel is split by HxW segments.
    const int doubleBuffer = 2;
    const int32_t vectorOnceBytes = 256;
    // sum + compensation + guard. The guard keeps the second issue lane in-bounds when its mask is empty.
    static constexpr uint32_t stage2KahanBufferCount = 3;
};

template <typename T, bool isDeterministic>
__aicore__ inline GroupNormSwishGrad<T, isDeterministic>::GroupNormSwishGrad(
    GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR dx, GM_ADDR dgamma,
    GM_ADDR dbeta, GM_ADDR workspace, GroupNormSwishGradTilingData* tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    this->curBlockIdx = GetBlockIdx();
    this->tilingKey = tilingData->Tiling_key;
    this->n = tilingData->N;
    this->c = tilingData->C;
    this->g = tilingData->G;
    this->hxw = tilingData->HXW;
    this->nxg = tilingData->NXG;
    this->cG = tilingData->C_G;
    this->taskNumPerCore = tilingData->task_num_per_core;
    this->taskNumPerTailCore = tilingData->task_num_per_tail_core;
    this->tailCore = tilingData->tail_core;
    this->workspaceSize = tilingData->workSpaceSize;
    this->allEleNum = static_cast<uint64_t>(n) * c * hxw;
    this->eleNumPerGroup = static_cast<uint64_t>(cG) * hxw;
    this->eleNumPerChannel = hxw;
    this->dgammaIsRequire = static_cast<bool>(tilingData->dgamma_is_require);
    this->dbetaIsRequire = static_cast<bool>(tilingData->dbeta_is_require);
    this->swishScale = tilingData->swish_scale;
    this->stage2CoreUsed = tilingData->stage2CoreUsed;
    this->castEleNum = tilingData->castEleNum;
    this->tailCastNum = tilingData->tailCastNum;
    this->coreBatchParts = tilingData->coreBatchParts;
    this->curCoreTaskNum = this->taskNumPerCore;
    this->startTaskId = this->taskNumPerCore * this->curBlockIdx;

    if (this->curBlockIdx >= this->tailCore) {
        this->curCoreTaskNum = this->taskNumPerTailCore;
        this->startTaskId = this->taskNumPerTailCore * this->curBlockIdx + this->tailCore;
    }
    this->tPerBlock = blockBytes / sizeof(T);
    dyGm.SetGlobalBuffer((__gm__ T*)dy, this->allEleNum);
    meanGm.SetGlobalBuffer((__gm__ T*)mean, this->nxg);
    rstdGm.SetGlobalBuffer((__gm__ T*)rstd, this->nxg);
    xGm.SetGlobalBuffer((__gm__ T*)x, this->allEleNum);
    gammaGm.SetGlobalBuffer((__gm__ T*)gamma, this->c);
    betaGm.SetGlobalBuffer((__gm__ T*)beta, this->c);
    dxGm.SetGlobalBuffer((__gm__ T*)dx, this->allEleNum);
    dgammaGm.SetGlobalBuffer((__gm__ T*)dgamma, this->c);
    dbetaGm.SetGlobalBuffer((__gm__ T*)dbeta, this->c);
    dgammaWorkspace.SetGlobalBuffer((__gm__ float*)workspace, this->workspaceSize);
    dbetaWorkspace.SetGlobalBuffer((__gm__ float*)workspace + this->workspaceSize, this->workspaceSize);
    // Prepare dgamma/dbeta outputs or deterministic workspace before stage 1.
    InitOutputGm();
    // Allocate UB buffers for the selected stage-1 tiling mode.
    InitUnifiedBuffer(tilingData);
#ifndef __CCE_KT_TEST__
    // Synchronize after global output/workspace initialization.
    SyncAll();
#endif
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::Process()
{
    // Stage 1 dispatch. workspaceStage2TilingKey is reserved for stage-2
    // buffer selection, so stage 1 enters one of the UB-capacity paths here.
    if (tilingKey == groupFitUbTilingKey) {
        for (int32_t taskIdx = 0; taskIdx < this->curCoreTaskNum; taskIdx++) {
            CopyInGroupFitUb(taskIdx + this->startTaskId);
            ComputeGroupFitUb(taskIdx + this->startTaskId);
        }
    } else if (tilingKey == channelFitUbTilingKey) {
        for (int32_t taskIdx = 0; taskIdx < this->curCoreTaskNum; taskIdx++) {
            ComputeChannelFitUb(taskIdx + this->startTaskId);
        }
    } else if (tilingKey == channelSplitUbTilingKey) {
        for (int32_t taskIdx = 0; taskIdx < this->curCoreTaskNum; taskIdx++) {
            ComputeChannelSplitUb(taskIdx + this->startTaskId);
        }
    }
    pipe.Reset();
    constexpr uint32_t splitCount = 2;
    if constexpr (!isDeterministic && !IsSameType<T, float>::value) {
#ifndef __CCE_KT_TEST__
        // Wait for all stage-1 producers before casting workspace back to GM.
        SyncAll();
#endif
        if (n != 1 && curBlockIdx < stage2CoreUsed) {
            InitBufferForStage2Cast();
            // Cast workspace values to the output GM type.
            if (curBlockIdx < stage2CoreUsed - 1) {
                CastDgammaDbetaWsp2Gm(static_cast<uint64_t>(curBlockIdx) * castEleNum, castEleNum);
            } else if (curBlockIdx == stage2CoreUsed - 1) {
                CastDgammaDbetaWsp2Gm(static_cast<uint64_t>(curBlockIdx) * castEleNum, tailCastNum);
            }
        }
    } else if constexpr (isDeterministic && IsSameType<T, float>::value) {
#ifndef __CCE_KT_TEST__
        // Wait for all stage-1 producers before reducing workspace.
        SyncAll();
#endif
        InitBufferForStage2Reduce();
        if (curBlockIdx < stage2CoreUsed - 1) {
            ReduceDgammaDbetaWsp2Gm(
                0, static_cast<uint64_t>(curBlockIdx) * castEleNum, castEleNum, DivCeil(DivCeil(n, splitCount), splitCount));
        } else if (stage2CoreUsed <= curBlockIdx && curBlockIdx < splitCount * stage2CoreUsed - 1) {
            ReduceDgammaDbetaWsp2Gm(
                static_cast<uint64_t>(DivCeil(DivCeil(n, splitCount), splitCount)) * c, static_cast<uint64_t>(curBlockIdx % stage2CoreUsed) * castEleNum,
                castEleNum, (DivCeil(n, splitCount) - DivCeil(DivCeil(n, splitCount), splitCount)));
        } else if (curBlockIdx == stage2CoreUsed - 1) {
            ReduceDgammaDbetaWsp2Gm(
                0, static_cast<uint64_t>(curBlockIdx) * castEleNum, tailCastNum, DivCeil(DivCeil(n, splitCount), splitCount));
        } else if (curBlockIdx == splitCount * stage2CoreUsed - 1) {
            ReduceDgammaDbetaWsp2Gm(
                static_cast<uint64_t>(DivCeil(DivCeil(n, splitCount), splitCount)) * c, static_cast<uint64_t>(curBlockIdx % stage2CoreUsed) * castEleNum,
                tailCastNum, (DivCeil(n, splitCount) - DivCeil(DivCeil(n, splitCount), splitCount)));
        }
    } else if constexpr (isDeterministic && !IsSameType<T, float>::value) {
#ifndef __CCE_KT_TEST__
        // Wait for all stage-1 producers before reduce-and-cast.
        SyncAll();
#endif
        InitBufferForStage2ReduceCast();
        if (curBlockIdx < stage2CoreUsed - 1) {
            ReduceCastDgammaDbetaWsp2Gm(static_cast<uint64_t>(curBlockIdx) * castEleNum, castEleNum, DivCeil(n, splitCount));
        } else if (curBlockIdx == stage2CoreUsed - 1) {
            ReduceCastDgammaDbetaWsp2Gm(static_cast<uint64_t>(curBlockIdx) * castEleNum, tailCastNum, DivCeil(n, splitCount));
        }
    }
}

#endif // GROUP_NORM_SWISH_GRAD_H
