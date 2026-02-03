/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file weight_quant_batch_matmul_experimental_basic_vec_compute.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_EXPERIMENT_VEC_COMPUTE_H
#define WEIGHT_QUANT_BATCH_MATMUL_EXPERIMENT_VEC_COMPUTE_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#include "weight_quant_batch_matmul_experiment_tool.h"

using AscendC::AIV;
using AscendC::BinaryRepeatParams;
using AscendC::BlockReduceMax;
using AscendC::GetSubBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::MaskMode;
using AscendC::PipeBarrier;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::SetMaskNorm;
using AscendC::SetVectorMask;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::UnaryRepeatParams;
using AscendC::WaitFlag;

namespace WeightQuantBatchMatmulExperimental {

#define WQBMM_EXP_VEC_COMPUTE_TEMPLATE_PARAM template <typename xType, typename wType, typename CUBE_FIXP_DTYPE>

#define WQBMM_EXP_VEC_COMPUTE_CLASS WqbmmExpVecCompute<xType, wType, CUBE_FIXP_DTYPE>

WQBMM_EXP_VEC_COMPUTE_TEMPLATE_PARAM
class WqbmmExpVecCompute {
public:
    __aicore__ inline WqbmmExpVecCompute(){};
    __aicore__ inline void UnfoldA(
        uint64_t curMSize, uint64_t mPreSum, const A16W4MsdConstParam& constParams, const GlobalTensor<xType>& xGlobal,
        const GlobalTensor<float>& aMaxWorkspaceGm, const GlobalTensor<wType>& aUnfoldGlobal);
    __aicore__ inline void Init(GM_ADDR antiquantScale, GM_ADDR y, TPipe* tPipe);
    __aicore__ inline void MergeY(
        uint64_t mGmOffset, uint64_t nGmOffset, uint64_t kGmOffset, bool isLastK,
        const GlobalTensor<float>& aMaxWorkspaceGm, const GlobalTensor<CUBE_FIXP_DTYPE>& cS32GlobalGm,
        const A16W4MsdConstParam& constParams);
    __aicore__ inline void End();

private:
    static constexpr uint64_t CV_RATIO = 2;
    static constexpr uint64_t Y_F32_UB_OFFSET = 10 * 256;
    static constexpr uint64_t A_MAX_FP32_MERGE_UB_OFFSET = 512;
    static constexpr uint64_t SCALE_B16_MERGE_UB_OFFSET = 512;
    static constexpr uint64_t Y_F32_SUM_UB_SIZE = 10 * 256;
    uint64_t mte2BufIdx_ = 0;
    GlobalTensor<xType> yGm_;
    GlobalTensor<xType> scaleGm_;

    event_t eventIdsVToMte2_[2];

    TBuf<> ubBuffer_;
    // 前处理buf分配
    LocalTensor<float> aF32TmpTensor_;
    LocalTensor<xType> aOriginTensor_;
    LocalTensor<half> aFp16Tensor_;
    LocalTensor<wType> aUnfoldLocalTensor_;
    LocalTensor<float> aMaxTensor_;
    LocalTensor<float> aSumTensor_;
    LocalTensor<float> aF32Tensor_;

    // 后处理buf分配
    LocalTensor<CUBE_FIXP_DTYPE> yS32Ub_;
    LocalTensor<float> aMaxUb_;
    LocalTensor<float> yF32ComputeUb_;
    LocalTensor<float> yF32CastUb_;
    LocalTensor<float> yF32UbSum_;
    LocalTensor<xType> yOutputUb_;
    LocalTensor<xType> scaleF16Ub_;
    LocalTensor<float> scaleF32Ub_;

    BinaryRepeatParams commonRepeatParams_;
    UnaryRepeatParams commonUnaryRepeatParams_;
    // msd算法的展开系数
    float multiFactors_[UNFLOD_TIMES] = {7.49f, 14.98f, 14.98f};

    __aicore__ inline void ComputeMaxA(
        uint64_t mPreSum, const A16W4MsdConstParam& constParams, const GlobalTensor<float>& aMaxWorkspaceGm);
    __aicore__ inline void UnfoldAMatrix(
        uint64_t mPreSum, const A16W4MsdConstParam& constParams, const GlobalTensor<wType>& aUnfoldGlobal);
    __aicore__ inline void CopyAFp16GmToUb(
        uint64_t mPreSum, const A16W4MsdConstParam& constParams, const GlobalTensor<xType>& xGlobal);
    __aicore__ inline void TransToWType(
        uint64_t unfoldATimes, uint64_t defaultOffset, uint64_t mainRepeatK, float multiFactors,
        const BinaryRepeatParams& f32BinaryRepeatParams, const UnaryRepeatParams& f32UnaryRepeatParams,
        const UnaryRepeatParams& f32ToF16RepeatParams, const A16W4MsdConstParam& constParams);
    __aicore__ inline void TransToWTypeFirstStep(
        uint64_t defaultOffset, uint64_t mainRepeatK, const BinaryRepeatParams& f32BinaryRepeatParams,
        const A16W4MsdConstParam& constParams);
    __aicore__ inline void CopyUbToGm(
        uint64_t curMGmOffset, uint64_t nGmOffset, uint64_t curMSize, bool isLastK,
        const A16W4MsdConstParam& constParams);
    __aicore__ inline void ComputeMergeY(
        uint64_t curMSize, const LocalTensor<float>& aMaxUb, const LocalTensor<CUBE_FIXP_DTYPE>& yS32Ub,
        const LocalTensor<xType>& scaleF16Ub, const A16W4MsdConstParam& constParams);
    __aicore__ inline void CopyMergeYInputGmToUb(
        uint64_t curMGmOffset, uint64_t curMSize, uint64_t kGmOffset, uint64_t nGmOffset,
        const LocalTensor<float>& aMaxUb, const LocalTensor<CUBE_FIXP_DTYPE>& yS32Ub,
        const LocalTensor<xType>& scaleF16Ub, const GlobalTensor<float>& aMaxWorkspaceGm,
        const GlobalTensor<CUBE_FIXP_DTYPE>& cS32GlobalGm, const A16W4MsdConstParam& constParams);
};

WQBMM_EXP_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_VEC_COMPUTE_CLASS::Init(GM_ADDR antiquantScale, GM_ADDR y, TPipe* tPipe)
{
    yGm_.SetGlobalBuffer((__gm__ xType*)y);
    scaleGm_.SetGlobalBuffer((__gm__ xType*)antiquantScale);

    tPipe->InitBuffer(ubBuffer_, 192 * 1024); // UB可用空间192Kb

    // 基础场景
    // 前处理空间定义
    aF32TmpTensor_ = ubBuffer_.Get<float>();                        // 3k 0-3k分配给float的aTmp矩阵
    aOriginTensor_ = ubBuffer_.Get<xType>()[3 * GetKBUnit<half>()]; // 3k 3-6k分配给xType的a矩阵
    aFp16Tensor_ = ubBuffer_.Get<half>()[6 * GetKBUnit<half>()];    // 3k 6-9k分配给half的a矩阵 空间复用
    // 3k 9-12k分配给unfold的a矩阵
    aUnfoldLocalTensor_ = ubBuffer_.Get<wType>()[9 * GetKBUnit<wType>()];
    aMaxTensor_ = ubBuffer_.Get<float>()[12 * GetKBUnit<float>()]; // 3k 12-15k分配给aMax的a矩阵
    aSumTensor_ = ubBuffer_.Get<float>()[15 * GetKBUnit<float>()]; // 3k 15-18k分配给aSum的a矩阵
    aF32Tensor_ = ubBuffer_.Get<float>()[18 * GetKBUnit<float>()]; // 3k 18-21k分配给float的a矩阵

    // 后处理空间定义
    yS32Ub_ = ubBuffer_.Get<CUBE_FIXP_DTYPE>()[21 * GetKBUnit<CUBE_FIXP_DTYPE>()]; // 21k - 41k分配给Y_Int矩阵
    aMaxUb_ = ubBuffer_.Get<float>()[41 * GetKBUnit<float>()];                     // 31k - 34k分配给aMax后处理

    yF32ComputeUb_ = ubBuffer_.Get<float>()[44 * GetKBUnit<float>()]; // 34k
    yF32CastUb_ = ubBuffer_.Get<float>()[54 * GetKBUnit<float>()];    // 34k
    yF32UbSum_ = ubBuffer_.Get<float>()[64 * GetKBUnit<float>()];     // 当前空间预留10k，支持M<=5
    yOutputUb_ = ubBuffer_.Get<xType>()[74 * GetKBUnit<xType>()];
    scaleF16Ub_ = ubBuffer_.Get<xType>()[79 * GetKBUnit<xType>()];
    scaleF32Ub_ = ubBuffer_.Get<float>()[84 * GetKBUnit<float>()];

    commonRepeatParams_.dstBlkStride = 1;
    commonRepeatParams_.src0BlkStride = 1;
    commonRepeatParams_.src1BlkStride = 1;
    commonRepeatParams_.dstRepStride = FP32_MASK_BLK_NUM;
    commonRepeatParams_.src0RepStride = FP32_MASK_BLK_NUM;
    commonRepeatParams_.src1RepStride = FP32_MASK_BLK_NUM;

    commonUnaryRepeatParams_.dstBlkStride = 1;
    commonUnaryRepeatParams_.srcBlkStride = 1;
    commonUnaryRepeatParams_.dstRepStride = FP32_MASK_BLK_NUM;
    commonUnaryRepeatParams_.srcRepStride = FP32_MASK_BLK_NUM;

    eventIdsVToMte2_[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdsVToMte2_[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());

    SetFlag<HardEvent::V_MTE2>(eventIdsVToMte2_[0]);
    SetFlag<HardEvent::V_MTE2>(eventIdsVToMte2_[1]);
}
WQBMM_EXP_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_VEC_COMPUTE_CLASS::End()
{
    WaitFlag<HardEvent::V_MTE2>(eventIdsVToMte2_[0]);
    WaitFlag<HardEvent::V_MTE2>(eventIdsVToMte2_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdsVToMte2_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdsVToMte2_[1]);
}

WQBMM_EXP_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_VEC_COMPUTE_CLASS::UnfoldA(
    uint64_t curMSize, uint64_t mPreSum, const A16W4MsdConstParam& constParams, const GlobalTensor<xType>& xGlobal,
    const GlobalTensor<float>& aMaxWorkspaceGm, const GlobalTensor<wType>& aUnfoldGlobal)
{
    CopyAFp16GmToUb(mPreSum, constParams, xGlobal);

    PipeBarrier<PIPE_V>();

    event_t eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);

    ComputeMaxA(mPreSum, constParams, aMaxWorkspaceGm);
    PipeBarrier<PIPE_V>();

    UnfoldAMatrix(mPreSum, constParams, aUnfoldGlobal);
}

WQBMM_EXP_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_VEC_COMPUTE_CLASS::CopyAFp16GmToUb(
    uint64_t mPreSum, const A16W4MsdConstParam& constParams, const GlobalTensor<xType>& xGlobal)
{
    DataCopyPad2D(aOriginTensor_, xGlobal, 1, constParams.kUbSize, constParams.kUbSize, constParams.kGmSize);
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    Cast(aF32Tensor_, aOriginTensor_, RoundMode::CAST_NONE, constParams.kUbSize);

    // 避免较大场景同步问题
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
}

WQBMM_EXP_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_VEC_COMPUTE_CLASS::ComputeMaxA(
    uint64_t mPreSum, const A16W4MsdConstParam& constParams, const GlobalTensor<float>& aMaxWorkspaceGm)
{
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(FP32_MAX_MASK_SIZE);
    uint64_t numRepeatK = constParams.kUbSize / FP32_MAX_MASK_SIZE;
    AscendC::Abs<float, false>(aF32TmpTensor_, aF32Tensor_, FP32_MAX_MASK_SIZE, numRepeatK, commonUnaryRepeatParams_);
    PipeBarrier<PIPE_V>();
    uint64_t processGroupNum = constParams.kUbSize / constParams.groupSize;

    BinaryRepeatParams param;
    param.dstBlkStride = 1;
    param.src0BlkStride = 1;
    param.src1BlkStride = 1;
    param.dstRepStride = FP32_MASK_BLK_NUM;
    param.src0RepStride = FP32_MASK_BLK_NUM * 2;
    param.src1RepStride = FP32_MASK_BLK_NUM * 2;
    AscendC::Max<float, false>(
        aF32TmpTensor_, aF32TmpTensor_, aF32TmpTensor_[FP32_MAX_MASK_SIZE], FP32_MAX_MASK_SIZE, processGroupNum, param);
    PipeBarrier<PIPE_V>();

    BlockReduceMax(aF32TmpTensor_, aF32TmpTensor_, processGroupNum, FP32_MAX_MASK_SIZE, 1, 1, FP32_MASK_BLK_NUM);

    PipeBarrier<PIPE_V>();
    BlockReduceMax(
        aF32TmpTensor_, aF32TmpTensor_, CeilDiv(processGroupNum, FP32_BLOCK_SIZE), FP32_MAX_MASK_SIZE, 1, 1,
        VEC_REPEAT_MAX_STRIDE);

    PipeBarrier<PIPE_V>();
    Brcb(aMaxTensor_, aF32TmpTensor_, CeilDiv(processGroupNum, FP32_BLOCK_SIZE), {1, 8});

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    uint64_t gmOffset = mPreSum * FP32_BLOCK_SIZE;
    DataCopyPad2D(aMaxWorkspaceGm[gmOffset], aMaxTensor_, 1, FP32_BLOCK_SIZE, processGroupNum * FP32_BLOCK_SIZE);
}

WQBMM_EXP_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_VEC_COMPUTE_CLASS::UnfoldAMatrix(
    uint64_t mPreSum, const A16W4MsdConstParam& constParams, const GlobalTensor<wType>& aUnfoldGlobal)
{
    // 避免bank冲突，tensor使用时每次额外往前偏移一个repeat的数据量
    uint64_t defaultOffset = UNFLOD_TIMES * FP32_MAX_MASK_SIZE;
    uint64_t mainRepeatK = constParams.groupSize / FP32_MAX_MASK_SIZE;
    BinaryRepeatParams f32BinaryRepeatParams;
    UnaryRepeatParams f32UnaryRepeatParams;
    UnaryRepeatParams f32ToF16RepeatParams;
    f32BinaryRepeatParams.dstBlkStride = 1;
    f32BinaryRepeatParams.src0BlkStride = 1;
    f32BinaryRepeatParams.src1BlkStride = 0;
    f32BinaryRepeatParams.dstRepStride = FP32_MASK_BLK_NUM;
    f32BinaryRepeatParams.src0RepStride = FP32_MASK_BLK_NUM;
    f32BinaryRepeatParams.src1RepStride = 0;

    f32UnaryRepeatParams.dstBlkStride = 1;
    f32UnaryRepeatParams.srcBlkStride = 1;
    f32UnaryRepeatParams.dstRepStride = FP32_MASK_BLK_NUM;
    f32UnaryRepeatParams.srcRepStride = FP32_MASK_BLK_NUM;

    f32ToF16RepeatParams.dstBlkStride = 1;
    f32ToF16RepeatParams.srcBlkStride = 1;
    f32ToF16RepeatParams.dstRepStride = FP32_MAX_MASK_SIZE / FP16_BLOCK_SIZE;
    f32ToF16RepeatParams.srcRepStride = FP32_MASK_BLK_NUM;

    float multiFactors[3] = {multiFactors_[0], multiFactors_[1], multiFactors_[2]};
    for (uint64_t unfoldATimes = 0; unfoldATimes < UNFLOD_TIMES; unfoldATimes++, defaultOffset -= FP32_MAX_MASK_SIZE) {
        TransToWType(
            unfoldATimes, defaultOffset, mainRepeatK, multiFactors[unfoldATimes], f32BinaryRepeatParams,
            f32UnaryRepeatParams, f32ToF16RepeatParams, constParams);

        event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        uint64_t gmOffset = (mPreSum * UNFLOD_TIMES + unfoldATimes) * constParams.kUbSize;
        DataCopyPad2D(
            aUnfoldGlobal[gmOffset], aUnfoldLocalTensor_[unfoldATimes * constParams.kUbSize], 1, constParams.kUbSize,
            constParams.kUbSize, constParams.kUbSize);
    }
}

WQBMM_EXP_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_VEC_COMPUTE_CLASS::TransToWType(
    uint64_t unfoldATimes, uint64_t defaultOffset, uint64_t mainRepeatK, float multiFactors,
    const BinaryRepeatParams& f32BinaryRepeatParams, const UnaryRepeatParams& f32UnaryRepeatParams,
    const UnaryRepeatParams& f32ToF16RepeatParams, const A16W4MsdConstParam& constParams)
{
    if (likely(unfoldATimes > 0)) {
        Sub(aF32TmpTensor_[defaultOffset], aF32Tensor_, aF32TmpTensor_[defaultOffset + FP32_MAX_MASK_SIZE],
            FP32_MAX_MASK_SIZE, mainRepeatK, commonRepeatParams_);
    } else {
        TransToWTypeFirstStep(defaultOffset, mainRepeatK, f32BinaryRepeatParams, constParams);
    }
    PipeBarrier<PIPE_V>();
    Muls<float, false>(
        aF32Tensor_, aF32TmpTensor_[defaultOffset], multiFactors, FP32_MAX_MASK_SIZE, mainRepeatK,
        f32UnaryRepeatParams);
    PipeBarrier<PIPE_V>();

    AscendC::Cast<float, float, false>(
        aF32TmpTensor_[defaultOffset], aF32Tensor_, RoundMode::CAST_ROUND, FP32_MAX_MASK_SIZE, mainRepeatK,
        f32UnaryRepeatParams);
    PipeBarrier<PIPE_V>();

    AscendC::Cast<half, float, false>(
        aFp16Tensor_, aF32TmpTensor_[defaultOffset], RoundMode::CAST_NONE, FP32_MAX_MASK_SIZE, mainRepeatK,
        f32ToF16RepeatParams);

    PipeBarrier<PIPE_V>();
    Cast(
        aUnfoldLocalTensor_[unfoldATimes * constParams.kUbSize], aFp16Tensor_, RoundMode::CAST_NONE,
        constParams.kUbSize);
}

WQBMM_EXP_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_VEC_COMPUTE_CLASS::TransToWTypeFirstStep(
    uint64_t defaultOffset, uint64_t mainRepeatK, const BinaryRepeatParams& f32BinaryRepeatParams,
    const A16W4MsdConstParam& constParams)
{
    uint64_t processGroupNum = constParams.kUbSize / constParams.groupSize;
    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = constParams.groupSize / FP32_MASK_BLK_NUM;
    repeatParams.src0RepStride = repeatParams.dstRepStride;
    repeatParams.src1RepStride = 1;
    AscendC::Div(
        aF32TmpTensor_[defaultOffset], aF32Tensor_, aMaxTensor_, FP32_MAX_MASK_SIZE, processGroupNum, repeatParams);

    AscendC::Div(
        aF32TmpTensor_[defaultOffset + 64], aF32Tensor_[64], aMaxTensor_, FP32_MAX_MASK_SIZE, processGroupNum,
        repeatParams);
}

WQBMM_EXP_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_VEC_COMPUTE_CLASS::MergeY(
    uint64_t mGmOffset, uint64_t nGmOffset, uint64_t kGmOffset, bool isLastK,
    const GlobalTensor<float>& aMaxWorkspaceGm, const GlobalTensor<CUBE_FIXP_DTYPE>& cS32GlobalGm,
    const A16W4MsdConstParam& constParams)
{
    event_t eventIdsVToMte2[2] = {eventIdsVToMte2_[0], eventIdsVToMte2_[1]};
    // 按m分核
    uint64_t curMGmOffset = GetSubBlockIdx() == 0 ? 0 : constParams.mL1Size / UNFLOD_TIMES / CV_RATIO;
    uint64_t curMSize = GetSubBlockIdx() == 0 ? constParams.mL1Size / UNFLOD_TIMES / CV_RATIO :
                                                constParams.mL1Size / UNFLOD_TIMES - curMGmOffset;
    if (curMSize <= 0) {
        return;
    }
    if (kGmOffset == 0) {
        PipeBarrier<PIPE_V>();
        Duplicate(yF32UbSum_, (float)0.0, Y_F32_SUM_UB_SIZE);
    }

    WaitFlag<HardEvent::V_MTE2>(eventIdsVToMte2[mte2BufIdx_ % DOUBLE_BUFFER_NUM]);
    LocalTensor<float> aMaxUb = aMaxUb_[(mte2BufIdx_ % DOUBLE_BUFFER_NUM) * A_MAX_FP32_MERGE_UB_OFFSET];
    LocalTensor<CUBE_FIXP_DTYPE> yS32Ub = yS32Ub_[(mte2BufIdx_ % DOUBLE_BUFFER_NUM) * Y_F32_UB_OFFSET];
    LocalTensor<xType> scaleF16Ub = scaleF16Ub_[(mte2BufIdx_ % DOUBLE_BUFFER_NUM) * SCALE_B16_MERGE_UB_OFFSET];
    CopyMergeYInputGmToUb(
        curMGmOffset, curMSize, kGmOffset, nGmOffset, aMaxUb, yS32Ub, scaleF16Ub, aMaxWorkspaceGm, cS32GlobalGm,
        constParams);

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    ComputeMergeY(curMSize, aMaxUb, yS32Ub, scaleF16Ub, constParams);

    SetFlag<HardEvent::V_MTE2>(eventIdsVToMte2[mte2BufIdx_ % DOUBLE_BUFFER_NUM]);
    mte2BufIdx_++;
    CopyUbToGm(mGmOffset + curMGmOffset, nGmOffset, curMSize, isLastK, constParams);
}

WQBMM_EXP_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_VEC_COMPUTE_CLASS::CopyMergeYInputGmToUb(
    uint64_t curMGmOffset, uint64_t curMSize, uint64_t kGmOffset, uint64_t nGmOffset, const LocalTensor<float>& aMaxUb,
    const LocalTensor<CUBE_FIXP_DTYPE>& yS32Ub, const LocalTensor<xType>& scaleF16Ub,
    const GlobalTensor<float>& aMaxWorkspaceGm, const GlobalTensor<CUBE_FIXP_DTYPE>& cS32GlobalGm,
    const A16W4MsdConstParam& constParams)
{
    DataCopyPad2D(
        aMaxUb, aMaxWorkspaceGm[curMGmOffset * FP32_BLOCK_SIZE], 1, curMSize * FP32_BLOCK_SIZE,
        curMSize * FP32_BLOCK_SIZE, curMSize * FP32_BLOCK_SIZE);

    DataCopyPad2D(
        yS32Ub, cS32GlobalGm, UNFLOD_TIMES * curMSize, constParams.nL1Size, constParams.nL1Size, constParams.nGmSize);

    DataCopyPad2D(
        scaleF16Ub, scaleGm_[(kGmOffset / constParams.groupSize) * constParams.nGmSize + nGmOffset], 1,
        constParams.nL1Size, constParams.nL1Size, constParams.nGmSize);
}

WQBMM_EXP_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_VEC_COMPUTE_CLASS::ComputeMergeY(
    uint64_t curMSize, const LocalTensor<float>& aMaxUb, const LocalTensor<CUBE_FIXP_DTYPE>& yS32Ub,
    const LocalTensor<xType>& scaleF16Ub, const A16W4MsdConstParam& constParams)
{
    Cast(scaleF32Ub_, scaleF16Ub, RoundMode::CAST_NONE, constParams.nL1Size);
    Cast(yF32CastUb_, yS32Ub, RoundMode::CAST_NONE, curMSize * constParams.nL1Size * UNFLOD_TIMES);
    Duplicate(yF32ComputeUb_, 0.0f, curMSize * constParams.nL1Size);
    PipeBarrier<PIPE_V>();
    float divFactors[3] = {
        1.0f / multiFactors_[0], 1.0f / (multiFactors_[0] * multiFactors_[1]),
        1.0f / (multiFactors_[0] * multiFactors_[1] * multiFactors_[2])};
    for (uint64_t unfoldATimes = 0; unfoldATimes < UNFLOD_TIMES; unfoldATimes++) {
        for (uint64_t idxM = 0; idxM < curMSize; idxM++) {
            uint64_t cOffset = (idxM * UNFLOD_TIMES + unfoldATimes) * constParams.nL1Size;
            AscendC::Axpy(
                yF32ComputeUb_[idxM * constParams.nL1Size], yF32CastUb_[cOffset], divFactors[unfoldATimes],
                FP32_MAX_MASK_SIZE, constParams.nL1Size / FP32_MAX_MASK_SIZE, commonUnaryRepeatParams_);
        }
        PipeBarrier<PIPE_V>();
    }
    for (uint64_t mIdx = 0; mIdx < curMSize; mIdx++) {
        Mul(yF32ComputeUb_[mIdx * constParams.nL1Size], yF32ComputeUb_[mIdx * constParams.nL1Size], scaleF32Ub_,
            constParams.nL1Size);
    }
    PipeBarrier<PIPE_V>();
    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = FP32_MASK_BLK_NUM;
    repeatParams.src0RepStride = FP32_MASK_BLK_NUM;
    repeatParams.src1RepStride = 0;
    for (uint64_t idxM = 0; idxM < curMSize; idxM++) {
        uint64_t maxOffset = idxM * FP32_BLOCK_SIZE;
        AscendC::MulAddDst(
            yF32UbSum_[idxM * constParams.nL1Size], yF32ComputeUb_[idxM * constParams.nL1Size], aMaxUb[maxOffset],
            FP32_MAX_MASK_SIZE, constParams.nL1Size / FP32_MAX_MASK_SIZE, repeatParams);
    }
    PipeBarrier<PIPE_V>();
}

WQBMM_EXP_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_EXP_VEC_COMPUTE_CLASS::CopyUbToGm(
    uint64_t mGmOffset, uint64_t nGmOffset, uint64_t curMSize, bool isLastK, const A16W4MsdConstParam& constParams)
{
    if (isLastK) {
        Cast(yOutputUb_, yF32UbSum_, RoundMode::CAST_RINT, curMSize * constParams.nL1Size);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad2D(
            yGm_[mGmOffset * constParams.nGmSize + nGmOffset], yOutputUb_, curMSize, constParams.nL1Size,
            constParams.nL1Size, constParams.nGmSize);
    }
}
} // namespace WeightQuantBatchMatmulExperimental
#endif