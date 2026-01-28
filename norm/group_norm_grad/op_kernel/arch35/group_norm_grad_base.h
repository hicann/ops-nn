/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file group_norm_grad_base.h
 * \brief
 */
#ifndef GROUP_NORM_GRAD_BASE_H
#define GROUP_NORM_GRAD_BASE_H

#pragma once

#include "kernel_operator.h"
#include "group_norm_grad_common.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace GroupNormGrad {
using namespace AscendC;
using namespace NormCommon;
using namespace NormCommon::NormCommonRegbase;

constexpr int DOUBLE_BUFFER = 2;

/**
 * Get the size of vector registers in bytes
 */
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}
/**
 * Get the block size of unified buffer in bytes
 */
__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

template <typename T, typename U>
class GroupNormGradBase {
public:
    __aicore__ inline GroupNormGradBase(){};
    __aicore__ inline ~GroupNormGradBase(){};

protected:
    __aicore__ inline void InitCommon(
        GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta,
        GM_ADDR workspace, const GroupNormGradRegBaseTilingData* __restrict tilingData);
    __aicore__ inline void InitOutPutOrWorkSpace();
    __aicore__ inline void Stage2Process();

protected:
    __aicore__ inline void ParseTilingData(const GroupNormGradRegBaseTilingData* tilingData);
    __aicore__ inline void CustomReduceSum(
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const uint32_t idx);
    __aicore__ inline bool IsNeedUpdateLevel1Cache(const int64_t basicBlockIdx);
    __aicore__ inline bool IsNeedUpdateLevel2Cache(const int64_t basicBlockIdx);
    __aicore__ inline void UpdateCache(
        const int64_t basicBlockIdx, const LocalTensor<float>& cacheDbeta, const LocalTensor<float>& cacheDgamma);
    __aicore__ inline void LoadMeanRstd(int32_t taskIdx);
    __aicore__ inline void LoadDataToUb(
        TQue<TPosition::VECIN, 1>& inQue, TBuf<TPosition::VECCALC>& tbuf, const GlobalTensor<U>& gm,
        const uint32_t offset, const uint32_t count);
    __aicore__ inline void CopyInDyAndX(
        const LocalTensor<T>& dyTensor, const LocalTensor<T>& xTensor, const int64_t offset, const uint32_t burstLen);
    __aicore__ inline void StoreDxToGm(
        TQue<TPosition::VECOUT, 1>& outQue, const uint32_t gmOffset, const uint32_t count);
    __aicore__ inline void Fp32StoreDgamma(
        uint32_t channelIdx, const GlobalTensor<float>& dgammaOut, const LocalTensor<float>& dsTensor,
        const LocalTensor<float>& dbetaTensor, const float mean, const float rstd);
    __aicore__ inline void NonFp32StoreDgamma(
        uint32_t channelIdx, const GlobalTensor<U>& dgammaOut, const LocalTensor<float>& dsTensor,
        const LocalTensor<float>& dbetaTensor, const float mean, const float rstd);
    __aicore__ inline void Fp32DgammaDbeta2GM(
        uint32_t channelIdx, GlobalTensor<float>& dgammaOut, const LocalTensor<float>& dsTensor,
        GlobalTensor<float>& dbetaOut, const LocalTensor<float>& dbetaTensor, const float mean, const float rstd);
    __aicore__ inline void NonFp32DgammaDbeta2GM(
        uint32_t channelIdx, const LocalTensor<float>& dsTensor, const LocalTensor<float>& dbetaTensor,
        const float mean, const float rstd);
    __aicore__ inline void StoreDgammaDbeta(
        const int32_t taskIdx, const LocalTensor<float>& dsTensor, const LocalTensor<float>& dbetaTensor,
        const float mean, const float rstd);
    __aicore__ inline void ComputeSum1Sum2(
        const LocalTensor<float>& dbetaTensor, const LocalTensor<float>& dsTensor,
        const LocalTensor<float>& gammaTensor, float& sum1, float& sum2);
    __aicore__ inline void VFComputeSum1Sum2(
        const LocalTensor<float>& dbetaTensor, const LocalTensor<float>& dsTensor,
        const LocalTensor<float>& gammaTensor, float& sum1, float& sum2);
    __aicore__ inline void VFComputeBinaryFoldSum1Sum2(
        const LocalTensor<float>& dbetaTensor, const LocalTensor<float>& dsTensor,
        const LocalTensor<float>& gammaTensor, float& sum1, float& sum2);
    __aicore__ inline void InitStage2Mode2Buffer();
    __aicore__ inline void InitStage2Mode1Buffer();
    __aicore__ inline void stage2Mode1Process(uint32_t cOffset, uint32_t currentCNum);
    __aicore__ inline void ProcessStage2Mode2(const GlobalTensor<float>& workspace, const GlobalTensor<U>& gmOut);
    __aicore__ inline void VFReduceNC(
        const LocalTensor<float>& inTensor, const LocalTensor<U>& outTensor, const uint32_t currCNum,
        const uint32_t cNumAlign, const uint32_t currNNum);
    __aicore__ inline void reduceNMode1Wsp2Ub(
        TQue<QuePosition::VECIN, 1>& vecInQue, const GlobalTensor<float>& workspace, uint32_t gmOffset,
        uint32_t currentCNum);
    __aicore__ inline void stage2Mode1B32Compute(
        TQue<QuePosition::VECIN, 1>& inQue, TQue<QuePosition::VECOUT, 1>& calQue, TBuf<TPosition::VECCALC>& outTbuf,
        TBuf<TPosition::VECCALC>& tempbuf, const GlobalTensor<float>& workspace, GlobalTensor<U>& gmOut,
        uint32_t cOffset, uint32_t currentCNum);
    __aicore__ inline void stage2Mode1B16Compute(
        TQue<QuePosition::VECIN, 1>& inQue, TQue<QuePosition::VECOUT, 1>& calQue, TBuf<TPosition::VECCALC>& outTbuf,
        TBuf<TPosition::VECCALC>& tempbuf, const GlobalTensor<float>& workspace, GlobalTensor<U>& gmOut,
        uint32_t cOffset, uint32_t currentCNum);
    __aicore__ inline void reduceNMode1LessThan2(
        __local_mem__ float* inUbAddr, __local_mem__ float* calUbAddr, uint32_t currentCNum);
    __aicore__ inline void reduceNMode1LessThan4(
        __local_mem__ float* inUbAddr, __local_mem__ float* calUbAddr, uint32_t currentCNum);
    __aicore__ inline void reduceNMode1LessThan8(
        __local_mem__ float* inUbAddr, __local_mem__ float* calUbAddr, uint32_t currentCNum);
    __aicore__ inline void reduceNMode1MoreThan8(
        __local_mem__ float* inUbAddr, __local_mem__ float* tempUbAddr, __local_mem__ float* calUbAddr,
        uint32_t currentCNum);
    __aicore__ inline void TwoRowAddWithTail(
        RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg, uint32_t offset1, uint32_t offset2,
        uint32_t offset3, uint32_t offset4, RegTensor<float>& rem, RegTensor<float>& nextRow,
        RegTensor<float>& remNextRow);
    __aicore__ inline void TwoRowAdd(
        RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg, uint32_t offset1, uint32_t offset2,
        RegTensor<float>& nextRow);

protected:
    // Pipe object
    TPipe* pipe;
    // Que object
    TQue<QuePosition::VECIN, 1> inQueDy_;
    TQue<QuePosition::VECIN, 1> inQueX_;
    TQue<QuePosition::VECIN, 1> inQueGamma_;
    TQue<QuePosition::VECIN, 1> inQueMean_;
    TQue<QuePosition::VECIN, 1> inQueRstd_;
    TQue<QuePosition::VECIN, 1> inQueDgammaChannel_;
    TQue<QuePosition::VECIN, 1> inQueDbetaChannel_;
    TQue<QuePosition::VECIN, 1> inQueDs_;
    TQue<QuePosition::VECIN, 1> inQueDb_;
    TBuf<TPosition::VECCALC> tBufGamma_;
    TBuf<TPosition::VECCALC> tBufDbeta_;
    TBuf<TPosition::VECCALC> outTbufDgammaChannel_;
    TBuf<TPosition::VECCALC> outTbufDbetaChannel_;
    TBuf<TPosition::VECCALC> tempMeanBuf_;
    TBuf<TPosition::VECCALC> tempRstdBuf_;
    TBuf<TPosition::VECCALC> cacheDbetaBuf_;
    TBuf<TPosition::VECCALC> cacheDgammaBuf_;
    TBuf<TPosition::VECCALC> tempDgammaBuf_;
    TBuf<TPosition::VECCALC> tempDbetaBuf_;
    // Que for output
    TQue<QuePosition::VECOUT, 1> outQueDgamma_;
    TQue<QuePosition::VECOUT, 1> outQueDs_;
    TQue<QuePosition::VECOUT, 1> outQueDbeta_;
    // 二分折叠计算ds, dbeta时，复用outQueDx做临时空间使用，使用空间大小为(HxW / 64) * sizeof(float)
    TQue<QuePosition::VECOUT, 1> outQueDx_;
    TQue<QuePosition::VECOUT, 1> calQueDgammaReduce_;
    TQue<QuePosition::VECOUT, 1> calQueDbetaReduce_;
    // Global Memory
    GlobalTensor<T> xGm_;
    GlobalTensor<T> dyGm_;
    GlobalTensor<T> dxGm_;
    GlobalTensor<float> dgammaWorkSpace_;
    GlobalTensor<float> dbetaWorkspace_;
    GlobalTensor<U> meanGm_;
    GlobalTensor<U> rstdGm_;
    GlobalTensor<U> gammaGm_;
    GlobalTensor<U> dgammaGm_;
    GlobalTensor<U> dbetaGm_;
    int64_t N_;
    int64_t C_;
    int64_t G_;
    int64_t HXW_;
    int64_t NXG_;
    int64_t NXC_;
    uint32_t C_G_; // tiling中有约束最大值
    int64_t allEleNum_;
    // number of calculations on each core
    int64_t eleNumPerG_;
    // number of tiles on each core
    int64_t eleNumPerC_;
    uint32_t clrBlockSize_ = 0;
    uint32_t clrBlockNum_ = 0;
    uint32_t clrTailBlockSize_ = 0;
    uint32_t taskNumPerCore_;
    uint32_t taskNumPerTailCore_;
    uint32_t tailCore_;
    uint32_t stage1CoreUsed_;
    uint32_t mode0UbCapGNum_ = 0;
    uint32_t mode1UbCapCNum_ = 0;
    uint32_t curCoreTaskNum_;
    uint32_t workSpaceSize_;
    int32_t curBlockIdx_;
    int32_t startTaskId_;
    float meanScalar_;
    float rstdScalar_;
    uint32_t mode2UbCapEle_;
    uint32_t mode2UbIterNum_;
    uint32_t mode2UbTailNum_;
    uint32_t mode2MainLoopCnt_;
    uint32_t mode2FoldLoopCnt_;
    uint32_t mode2OneLoopSize_;
    uint32_t mode2FoldTail_;
    uint32_t elemTPerBlock_;
    uint32_t elemUPerBlock_;
    uint32_t binaryAddQuotient_ = 0;
    uint32_t binaryAddK_ = 0;
    uint32_t binaryAddLastNum_ = 0;
    uint32_t binaryCGQuotient_ = 0;
    uint32_t binaryCGK_ = 0;
    uint32_t binaryCGLastNum_ = 0;
    uint32_t cFactor_;
    uint32_t cBlockFactor_;
    uint32_t cTailBlockFactor_;
    uint32_t stage2CoreUsed_;
    uint32_t stage2BinaryAddQuotient_ = 0;
    uint32_t stage2BinaryAddK_ = 0;
    int64_t reduceNCnt_;
    uint32_t stage2Mode_;
    uint32_t cNumMode2PerCore_;
    uint32_t tailcNumMode2PerCore_;
    int64_t stage2FactorN_;
    int64_t stage2LoopCnt_;
    bool dxIsRequire_;
    bool dgammaIsRequire_;
    bool dbetaIsRequire_;
    const uint32_t PF32_PER_BLOCK = GetUbBlockSize() / sizeof(float);
    const uint32_t BLOCK_BYTES = GetUbBlockSize();
    const uint32_t CACHE_BUFF_SIZE = 256;
    const uint32_t MAX_CACHE_DEPTH = 3;
    const uint32_t CACHE_LEVEL0_IDX = 0;
    const uint32_t CACHE_LEVEL1_IDX = 256;
    const uint32_t CACHE_LEVEL2_IDX = 512;
    const uint32_t SPLIT_COUNT = 2;
    constexpr static uint32_t VecLen_ = GetVRegSize() / sizeof(float);
    const int32_t STAGE2_MODE_1 = 1;
    const int32_t STAGE2_MODE_2 = 2;
    const int64_t SCALE_COEF_TWO = 2;
    const int64_t SCALE_COEF_FOUR = 4;
    const int64_t SCALE_COEF_EIGHT = 8;
    const uint16_t ROW_ZERO = 0;
    const uint16_t ROW_ONE = 1;
    const uint16_t ROW_TWO = 2;
    const uint16_t ROW_THREE = 3;
    const uint16_t ROW_FOUR = 4;
    const uint16_t ROW_FIVE = 5;
    const uint16_t ROW_SIX = 6;
    const uint16_t ROW_SEVEN = 7;
    const uint32_t ROW_TWO_OFFSET = 2;
    const uint32_t ROW_THREE_OFFSET = 3;
    const uint32_t ROW_FOUR_OFFSET = 4;
    const uint32_t ROW_FIVE_OFFSET = 5;
    const uint32_t ROW_SIX_OFFSET = 6;
    const uint32_t ROW_SEVEN_OFFSET = 7;
};

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::InitCommon(
    GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta,
    GM_ADDR workspace, const GroupNormGradRegBaseTilingData* __restrict tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    this->curBlockIdx_ = GetBlockIdx();
    ParseTilingData(tilingData);

    if (this->curBlockIdx_ >= this->tailCore_) {
        this->curCoreTaskNum_ = this->taskNumPerTailCore_;
        this->startTaskId_ = this->taskNumPerTailCore_ * this->curBlockIdx_ + this->tailCore_;
    }
    this->elemTPerBlock_ = BLOCK_BYTES / sizeof(T);
    this->elemUPerBlock_ = BLOCK_BYTES / sizeof(U);
    dyGm_.SetGlobalBuffer((__gm__ T*)dy, this->allEleNum_);
    xGm_.SetGlobalBuffer((__gm__ T*)x, this->allEleNum_);
    dxGm_.SetGlobalBuffer((__gm__ T*)dx, this->allEleNum_);
    meanGm_.SetGlobalBuffer((__gm__ U*)mean, this->NXG_);
    rstdGm_.SetGlobalBuffer((__gm__ U*)rstd, this->NXG_);
    gammaGm_.SetGlobalBuffer((__gm__ U*)gamma, this->C_);
    dgammaGm_.SetGlobalBuffer((__gm__ U*)dgamma, this->C_);
    dbetaGm_.SetGlobalBuffer((__gm__ U*)dbeta, this->C_);
    dgammaWorkSpace_.SetGlobalBuffer((__gm__ float*)workspace, this->workSpaceSize_);
    dbetaWorkspace_.SetGlobalBuffer((__gm__ float*)workspace + this->workSpaceSize_, this->workSpaceSize_);
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::ParseTilingData(const GroupNormGradRegBaseTilingData* tilingData)
{
    this->N_ = tilingData->gNGBaseParams.N;
    this->C_ = tilingData->gNGBaseParams.C;
    this->G_ = tilingData->gNGBaseParams.G;
    this->HXW_ = tilingData->gNGBaseParams.HXW;
    this->NXG_ = this->N_ * this->G_;
    this->NXC_ = this->N_ * this->C_;
    this->C_G_ = this->C_ / this->G_;
    this->clrBlockSize_ = tilingData->gNGBaseParams.clrBlockSize;
    this->clrBlockNum_ = tilingData->gNGBaseParams.clrBlockNum;
    this->clrTailBlockSize_ = tilingData->gNGBaseParams.clrTailBlockSize;
    this->taskNumPerCore_ = tilingData->gNGBaseParams.taskNumPerCore;
    this->taskNumPerTailCore_ = tilingData->gNGBaseParams.taskNumPerTailCore;
    this->tailCore_ = tilingData->gNGBaseParams.tailCore;
    this->stage1CoreUsed_ = tilingData->gNGBaseParams.stage1CoreUsed;
    this->mode0UbCapGNum_ = tilingData->gNGBaseParams.mode0UbCapGNum;
    this->mode1UbCapCNum_ = tilingData->gNGBaseParams.mode1UbCapCNum;
    this->workSpaceSize_ = tilingData->gNGBaseParams.workSpaceSize;
    this->allEleNum_ = this->N_ * this->C_ * this->HXW_;
    this->eleNumPerG_ = this->C_G_ * this->HXW_;
    this->eleNumPerC_ = this->HXW_;
    this->dxIsRequire_ = tilingData->gNGBaseParams.dxIsRequire;
    this->dgammaIsRequire_ = tilingData->gNGBaseParams.dgammaIsRequire;
    this->dbetaIsRequire_ = tilingData->gNGBaseParams.dbetaIsRequire;
    this->binaryAddQuotient_ = tilingData->gNGReduceParams.binaryAddQuotient;
    this->binaryAddK_ = tilingData->gNGReduceParams.binaryAddK;
    this->binaryAddLastNum_ = tilingData->gNGReduceParams.binaryAddLastNum;
    this->binaryCGQuotient_ = tilingData->gNGReduceParams.binaryCGQuotient;
    this->binaryCGK_ = tilingData->gNGReduceParams.binaryCGK;
    this->binaryCGLastNum_ = tilingData->gNGReduceParams.binaryCGLastNum;
    this->mode2UbCapEle_ = tilingData->gNGBaseParams.mode2UbCapEle;
    this->mode2UbIterNum_ = tilingData->gNGBaseParams.mode2UbIterNum;
    this->mode2UbTailNum_ = tilingData->gNGBaseParams.mode2UbTailNum;
    this->mode2MainLoopCnt_ = tilingData->gNGBaseParams.mode2MainLoopCnt;
    this->mode2FoldLoopCnt_ = tilingData->gNGBaseParams.mode2FoldLoopCnt;
    this->mode2OneLoopSize_ = tilingData->gNGBaseParams.mode2OneLoopSize;
    this->mode2FoldTail_ = tilingData->gNGBaseParams.mode2FoldTail;
    this->cNumMode2PerCore_ = tilingData->gNGKernel2Params.cNumMode2PerCore;
    this->stage2FactorN_ = tilingData->gNGKernel2Params.stage2FactorN;
    this->stage2LoopCnt_ = tilingData->gNGKernel2Params.stage2LoopCnt;
    this->tailcNumMode2PerCore_ = tilingData->gNGKernel2Params.tailcNumMode2PerCore;
    this->cFactor_ = tilingData->gNGKernel2Params.cFactor;
    this->cBlockFactor_ = tilingData->gNGKernel2Params.cBlockFactor;
    this->cTailBlockFactor_ = tilingData->gNGKernel2Params.cTailBlockFactor;
    this->stage2BinaryAddQuotient_ = tilingData->gNGKernel2Params.stage2BinaryAddQuotient;
    this->stage2BinaryAddK_ = tilingData->gNGKernel2Params.stage2BinaryAddK;
    this->stage2CoreUsed_ = tilingData->gNGKernel2Params.stage2CoreUsed;
    this->reduceNCnt_ = tilingData->gNGKernel2Params.reduceNCnt;
    this->stage2Mode_ = tilingData->gNGKernel2Params.stage2Mode;
    this->curCoreTaskNum_ = this->taskNumPerCore_;
    this->startTaskId_ = this->taskNumPerCore_ * this->curBlockIdx_;
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::InitOutPutOrWorkSpace()
{
    // init output or dgamma_workspace
    auto clrBlockOffset = this->clrBlockSize_ * this->curBlockIdx_;
    uint32_t currClrBlockSize = 0;
    if (this->curBlockIdx_ == this->clrBlockNum_ - 1) {
        currClrBlockSize = this->clrTailBlockSize_;
    } else {
        currClrBlockSize = this->clrBlockSize_;
    }
    if (this->curBlockIdx_ < this->clrBlockNum_) {
        if (this->N_ <= 2) {
            InitOutput<U>(dgammaGm_[clrBlockOffset], currClrBlockSize, 0.0f);
            InitOutput<U>(dbetaGm_[clrBlockOffset], currClrBlockSize, 0.0f);
        } else {
            InitOutput<float>(dgammaWorkSpace_[clrBlockOffset], currClrBlockSize, 0.0f);
            InitOutput<float>(dbetaWorkspace_[clrBlockOffset], currClrBlockSize, 0.0f);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::Stage2Process()
{
    if (this->N_ > 2) {
        // wait core
        this->pipe->Reset();
        SyncAll();
        if (this->curBlockIdx_ < this->stage2CoreUsed_) {
            if (stage2Mode_ == STAGE2_MODE_1) {
                InitStage2Mode1Buffer();
                uint32_t stage2CurCoreCNum =
                    (this->curBlockIdx_ == (this->stage2CoreUsed_ - 1)) ? this->cTailBlockFactor_ : this->cBlockFactor_;
                uint32_t quotient = (stage2CurCoreCNum + this->cFactor_ - 1) / this->cFactor_;
                for (uint32_t ubLoopIdx = 0; ubLoopIdx < quotient; ubLoopIdx++) {
                    uint32_t cOffset = ubLoopIdx * this->cFactor_ + this->curBlockIdx_ * this->cBlockFactor_;
                    uint32_t currentC = (ubLoopIdx == (quotient - 1)) ?
                                            (stage2CurCoreCNum - (quotient - 1) * this->cFactor_) :
                                            this->cFactor_;
                    stage2Mode1Process(cOffset, currentC);
                }
            } else if (stage2Mode_ == STAGE2_MODE_2) {
                InitStage2Mode2Buffer();
                if (dgammaIsRequire_) {
                    ProcessStage2Mode2(dgammaWorkSpace_, dgammaGm_);
                }
                if (dbetaIsRequire_) {
                    ProcessStage2Mode2(dbetaWorkspace_, dbetaGm_);
                }
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::CustomReduceSum(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const uint32_t idx)
{
    __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
    __ubuf__ float* src = (__ubuf__ float*)srcTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> x2;
        RegTensor<float> x3;
        RegTensor<float> x4;
        RegTensor<float> sum1;
        RegTensor<float> sum2;
        RegTensor<float> sum12;
        RegTensor<float> vlSum;

        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        DataCopy(x1, src + 0 * VecLen_);
        DataCopy(x2, src + 1 * VecLen_);
        DataCopy(x3, src + 2 * VecLen_);
        DataCopy(x4, src + 3 * VecLen_);
        Add(sum1, x1, x3, pregAll);
        Add(sum2, x2, x4, pregAll);
        Add(sum12, sum1, sum2, pregAll);
        ReduceSum(vlSum, sum12, pregAll);
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dst + idx, vlSum, pregMerge);
    }
}

template <typename T, typename U>
__aicore__ inline bool GroupNormGradBase<T, U>::IsNeedUpdateLevel1Cache(const int64_t basicBlockIdx)
{
    return ((basicBlockIdx + 1) & 0xff) == 0;
}

template <typename T, typename U>
__aicore__ inline bool GroupNormGradBase<T, U>::IsNeedUpdateLevel2Cache(const int64_t basicBlockIdx)
{
    return ((basicBlockIdx + 1) & 0xffff) == 0;
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::UpdateCache(
    const int64_t basicBlockIdx, const LocalTensor<float>& cacheDbeta, const LocalTensor<float>& cacheDgamma)
{
    if (IsNeedUpdateLevel1Cache(basicBlockIdx)) {
        CustomReduceSum(cacheDbeta[CACHE_LEVEL1_IDX], cacheDbeta[CACHE_LEVEL0_IDX], (basicBlockIdx & 0Xff00) >> 8);
        CustomReduceSum(cacheDgamma[CACHE_LEVEL1_IDX], cacheDgamma[CACHE_LEVEL0_IDX], (basicBlockIdx & 0Xff00) >> 8);
    }
    if (IsNeedUpdateLevel2Cache(basicBlockIdx)) {
        CustomReduceSum(cacheDbeta[CACHE_LEVEL2_IDX], cacheDbeta[CACHE_LEVEL1_IDX], basicBlockIdx >> 16);
        CustomReduceSum(cacheDgamma[CACHE_LEVEL2_IDX], cacheDgamma[CACHE_LEVEL1_IDX], basicBlockIdx >> 16);
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::LoadMeanRstd(int32_t taskIdx)
{
    if constexpr (IsSameType<U, bfloat16_t>::value) {
        this->meanScalar_ = ToFloat(this->meanGm_.GetValue(taskIdx));
        this->rstdScalar_ = ToFloat(this->rstdGm_.GetValue(taskIdx));
    } else if constexpr (IsSameType<U, half>::value) {
        this->meanScalar_ = static_cast<float>(this->meanGm_.GetValue(taskIdx));
        this->rstdScalar_ = static_cast<float>(this->rstdGm_.GetValue(taskIdx));
    } else {
        this->meanScalar_ = this->meanGm_.GetValue(taskIdx);
        this->rstdScalar_ = this->rstdGm_.GetValue(taskIdx);
    }
    TEventID eventIDSToV0 = GetTPipePtr()->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(eventIDSToV0);
    WaitFlag<HardEvent::S_V>(eventIDSToV0);
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::LoadDataToUb(
    TQue<TPosition::VECIN, 1>& inQue, TBuf<TPosition::VECCALC>& tbuf, const GlobalTensor<U>& gm,
    const uint32_t gmOffset, const uint32_t count)
{
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = count * sizeof(U);

    DataCopyPadExtParams<U> padParams;
    padParams.isPad = false;

    LocalTensor<float> localIn = inQue.AllocTensor<float>();
    if constexpr (IsSameType<U, float>::value) {
        DataCopyPad<U, PaddingMode::Normal>(localIn, gm[gmOffset], params, padParams);
        inQue.EnQue(localIn);
    } else {
        LocalTensor<U> temp = tbuf.Get<U>();
        DataCopyPad<U, PaddingMode::Normal>(temp, gm[gmOffset], params, padParams);
        TEventID eventIDMTE2ToV0 = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        VFCastT2Float<U>((__ubuf__ float*)localIn.GetPhyAddr(), (__ubuf__ U*)temp.GetPhyAddr(), count, this->VecLen_);
        inQue.EnQue(localIn);
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::CopyInDyAndX(
    const LocalTensor<T>& dyTensor, const LocalTensor<T>& xTensor, const int64_t offset, const uint32_t burstLen)
{
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = burstLen * sizeof(T);

    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;

    DataCopyPad<T, PaddingMode::Normal>(xTensor, this->xGm_[offset], params, padParams);
    DataCopyPad<T, PaddingMode::Normal>(dyTensor, this->dyGm_[offset], params, padParams);
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::StoreDxToGm(
    TQue<TPosition::VECOUT, 1>& outQue, const uint32_t gmOffset, const uint32_t count)
{
    LocalTensor<T> out = outQue.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(dxGm_[gmOffset], out, copyParams);
    outQue.FreeTensor(out);
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::Fp32StoreDgamma(
    uint32_t channelIdx, const GlobalTensor<float>& dgammaOut, const LocalTensor<float>& dsTensor,
    const LocalTensor<float>& dbetaTensor, const float mean, const float rstd)
{
    auto rstdScalar = rstd;
    auto meanScalar = mean;
    LocalTensor<float> dgammaTensor = outQueDgamma_.AllocTensor<float>();
    __ubuf__ float* ubDbeta = (__ubuf__ float*)dbetaTensor.GetPhyAddr();
    __ubuf__ float* ubDs = (__ubuf__ float*)dsTensor.GetPhyAddr();
    __ubuf__ float* ubDgamma = (__ubuf__ float*)dgammaTensor.GetPhyAddr();
    uint32_t calCount = this->C_G_;
    uint16_t repeatTimes = CeilDiv(calCount, this->VecLen_);

    __VEC_SCOPE__
    {
        RegTensor<float> vregDbeta;
        RegTensor<float> vregDs;
        MaskReg preg;
        uint32_t sreg = (uint32_t)calCount;
        uint32_t sregvl = (uint32_t)this->VecLen_;
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = UpdateMask<float>(sreg);
            DataCopy(vregDs, ubDs + i * sregvl);
            DataCopy(vregDbeta, ubDbeta + i * sregvl);
            Muls(vregDbeta, vregDbeta, meanScalar, preg);
            Sub(vregDs, vregDs, vregDbeta, preg);
            Muls(vregDs, vregDs, rstdScalar, preg);
            DataCopy(ubDgamma + i * sregvl, vregDs, preg);
        }
    }
    TEventID eventIDVToMTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->C_G_ * sizeof(float)), 0, 0, 0};
    SetAtomicAdd<float>();
    DataCopyPad(dgammaOut[channelIdx], dgammaTensor, copyParams);
    SetAtomicNone();
    outQueDgamma_.FreeTensor(dgammaTensor);
}

/*
    Reduce((ds[N] - db[N] * mean) * rstd)
*/
template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::NonFp32StoreDgamma(
    uint32_t channelIdx, const GlobalTensor<U>& dgammaOut, const LocalTensor<float>& dsTensor,
    const LocalTensor<float>& dbetaTensor, const float mean, const float rstd)
{
    auto rstdScalar = rstd;
    auto meanScalar = mean;
    LocalTensor<U> dgammaTensor = outQueDgamma_.AllocTensor<U>();
    __ubuf__ float* ubDbeta = (__ubuf__ float*)dbetaTensor.GetPhyAddr();
    __ubuf__ float* ubDs = (__ubuf__ float*)dsTensor.GetPhyAddr();
    __ubuf__ U* ubDgamma = (__ubuf__ U*)dgammaTensor.GetPhyAddr();
    uint32_t calCount = this->C_G_;
    uint16_t repeatTimes = CeilDiv(calCount, this->VecLen_);

    __VEC_SCOPE__
    {
        RegTensor<float> vregDbeta;
        RegTensor<float> vregDs;
        MaskReg preg;
        uint32_t sreg = (uint32_t)calCount;
        uint32_t sregvl = (uint32_t)this->VecLen_;
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = UpdateMask<float>(sreg);
            DataCopy(vregDs, ubDs + i * sregvl);
            DataCopy(vregDbeta, ubDbeta + i * sregvl);
            Muls(vregDbeta, vregDbeta, meanScalar, preg);
            Sub(vregDs, vregDs, vregDbeta, preg);
            Muls(vregDs, vregDs, rstdScalar, preg);
            StoreOneTensorForDtypeT<U>(ubDgamma, vregDs, preg, i * sregvl);
        }
    }
    TEventID eventIDVToMTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->C_G_ * sizeof(U)), 0, 0, 0};
    SetAtomicAdd<U>();
    DataCopyPad(dgammaOut[channelIdx], dgammaTensor, copyParams);
    SetAtomicNone();
    outQueDgamma_.FreeTensor(dgammaTensor);
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::Fp32DgammaDbeta2GM(
    uint32_t channelIdx, GlobalTensor<float>& dgammaOut, const LocalTensor<float>& dsTensor,
    GlobalTensor<float>& dbetaOut, const LocalTensor<float>& dbetaTensor, const float mean, const float rstd)
{
    if (dbetaIsRequire_) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->C_G_ * sizeof(float)), 0, 0, 0};
        SetAtomicAdd<float>();
        DataCopyPad(dbetaOut[channelIdx], dbetaTensor, copyParams);
        SetAtomicNone();
    }
    if (dgammaIsRequire_) {
        Fp32StoreDgamma(channelIdx, dgammaOut, dsTensor, dbetaTensor, mean, rstd);
        TEventID eventIDMTE3ToV = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::NonFp32DgammaDbeta2GM(
    uint32_t channelIdx, const LocalTensor<float>& dsTensor, const LocalTensor<float>& dbetaTensor, const float mean,
    const float rstd)
{
    if (dbetaIsRequire_) {
        LocalTensor<U> dbeta_temp = tBufDbeta_.Get<U>();
        VFCastFloat2T<U>(
            (__ubuf__ U*)dbeta_temp.GetPhyAddr(), (__ubuf__ float*)dbetaTensor.GetPhyAddr(), this->C_G_, this->VecLen_);
        TEventID eventIDVToMTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->C_G_ * sizeof(U)), 0, 0, 0};
        SetAtomicAdd<U>();
        DataCopyPad(dbetaGm_[channelIdx], dbeta_temp, copyParams);
        SetAtomicNone();
        TEventID eventIDMTE3ToV = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    }
    if (dgammaIsRequire_) {
        NonFp32StoreDgamma(channelIdx, dgammaGm_, dsTensor, dbetaTensor, mean, rstd);
        TEventID eventIDMTE3ToV = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::StoreDgammaDbeta(
    const int32_t taskIdx, const LocalTensor<float>& dsTensor, const LocalTensor<float>& dbetaTensor, const float mean,
    const float rstd)
{
    uint32_t cIdx;
    // When N == 1 or N = 2, directly written to the GM, workSpace is not required.
    if (this->N_ <= 2) {
        cIdx = (taskIdx % this->G_) * this->C_G_;
        if constexpr (IsSameType<U, float>::value) {
            Fp32DgammaDbeta2GM(cIdx, dgammaGm_, dsTensor, dbetaGm_, dbetaTensor, mean, rstd);
        } else {
            NonFp32DgammaDbeta2GM(cIdx, dsTensor, dbetaTensor, mean, rstd);
        }
    } else {
        // written to the workspace based on the C_G granularity
        cIdx = ((taskIdx / this->G_) / SPLIT_COUNT * this->G_ + (taskIdx % this->G_)) * this->C_G_;
        Fp32DgammaDbeta2GM(cIdx, dgammaWorkSpace_, dsTensor, dbetaWorkspace_, dbetaTensor, mean, rstd);
    }
}

/*
  sum1 = reduceSum(dbeta * gamma)
  sum2 = reduceSum(dgamma * gamma)
*/
template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::VFComputeBinaryFoldSum1Sum2(
    const LocalTensor<float>& dbetaTensor, const LocalTensor<float>& dsTensor, const LocalTensor<float>& gammaTensor,
    float& sum1, float& sum2)
{
    __ubuf__ float* ubDbeta = (__ubuf__ float*)dbetaTensor.GetPhyAddr();
    __ubuf__ float* ubDgamma = (__ubuf__ float*)dsTensor.GetPhyAddr();
    __ubuf__ float* ubGamma = (__ubuf__ float*)gammaTensor.GetPhyAddr();
    uint32_t binaryQuotientOffset = this->binaryCGQuotient_;
    uint32_t binaryCGRemainder = this->C_G_ - this->binaryCGQuotient_;
    uint16_t remainderLoop = CeilDiv(binaryCGRemainder, VecLen_);
    uint16_t remainderGeneral = remainderLoop == 0 ? 0 : remainderLoop - 1;
    // the binary fold, the head half o the 64-aligned
    uint16_t quotientLoop = CeilDiv(this->binaryCGQuotient_, VecLen_);
    uint16_t binaryCGKLoop = this->binaryCGK_;
    uint16_t binaryCGLoop = ((this->binaryCGQuotient_ / VecLen_) / VecLen_);
    uint32_t binaryCGLastNum = this->binaryCGLastNum_;
    uint32_t backHalfOffset = this->C_G_ / DOUBLE_BUFFER / DOUBLE_BUFFER;
    backHalfOffset = CeilAlign(backHalfOffset, this->PF32_PER_BLOCK);
    // 复用outQueDgamma 做为二分折叠计算sum1,sum2时的临时空间
    LocalTensor<float> binaryDbetaTensor = outQueDgamma_.AllocTensor<float>();
    LocalTensor<float> binarydsTensor = binaryDbetaTensor[backHalfOffset];
    LocalTensor<float> outDbetaTensor = tempMeanBuf_.Get<float>();
    LocalTensor<float> outdsTensor = tempRstdBuf_.Get<float>();
    __ubuf__ float* ubBinaryDbeta = (__ubuf__ float*)binaryDbetaTensor.GetPhyAddr();
    __ubuf__ float* ubBinaryDgamma = (__ubuf__ float*)binarydsTensor.GetPhyAddr();
    __ubuf__ float* outDbeta = (__ubuf__ float*)outDbetaTensor.GetPhyAddr();
    __ubuf__ float* outDgamma = (__ubuf__ float*)outdsTensor.GetPhyAddr();
    uint32_t sregvl = this->VecLen_;

    __VEC_SCOPE__
    {
        RegTensor<float> vregSumDgamma;
        RegTensor<float> vregSumDbeta;
        RegTensor<float> vregDbeta;
        RegTensor<float> vregDbetaQ;
        RegTensor<float> vregDbetaR;
        RegTensor<float> vregDgamma;
        RegTensor<float> vregDgammaQ;
        RegTensor<float> vregDgammaR;
        RegTensor<float> vregGamma;
        RegTensor<float> vregGammaQ;
        RegTensor<float> vregGammaR;
        RegTensor<float> tempDgamma;
        RegTensor<float> tempDbeta;
        uint32_t sreg0 = binaryCGRemainder;
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        // step1: reduce quotient remaider overlap part, reduce to 1
        for (uint16_t i = 0; i < remainderGeneral; i++) {
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            LoadTwoTensorForDtypeT<float>(
                ubDbeta, ubDbeta, vregDbetaQ, vregDbetaR, pregMain, pregLoop, i * sregvl,
                i * sregvl + binaryQuotientOffset);
            LoadTwoTensorForDtypeT<float>(
                ubDgamma, ubDgamma, vregDgammaQ, vregDgammaR, pregMain, pregLoop, i * sregvl,
                i * sregvl + binaryQuotientOffset);
            LoadTwoTensorForDtypeT<float>(
                ubGamma, ubGamma, vregGammaQ, vregGammaR, pregMain, pregLoop, i * sregvl,
                i * sregvl + binaryQuotientOffset);
            Mul(vregDbetaQ, vregGammaQ, vregDbetaQ, pregMain);
            Mul(vregDbetaR, vregGammaR, vregDbetaR, pregLoop);
            Mul(vregDgammaQ, vregGammaQ, vregDgammaQ, pregMain);
            Mul(vregDgammaR, vregGammaR, vregDgammaR, pregLoop);
            Add(vregDbetaQ, vregDbetaQ, vregDbetaR, pregLoop);
            Add(vregDgammaQ, vregDgammaQ, vregDgammaR, pregLoop);
            ReduceSum(vregSumDgamma, vregDgammaQ, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubBinaryDgamma + i, vregSumDgamma, pregMerge);
            ReduceSum(vregSumDbeta, vregDbetaQ, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubBinaryDbeta + i, vregSumDbeta, pregMerge);
        }
        // step2: the tail (last 64 or less than 64) blocks reduce to 1.
        {
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            LoadTwoTensorForDtypeT<float>(
                ubDbeta, ubDbeta, vregDbetaQ, vregDbetaR, pregMain, pregLoop, remainderGeneral * sregvl,
                remainderGeneral * sregvl + binaryQuotientOffset);
            LoadTwoTensorForDtypeT<float>(
                ubDgamma, ubDgamma, vregDgammaQ, vregDgammaR, pregMain, pregLoop, remainderGeneral * sregvl,
                remainderGeneral * sregvl + binaryQuotientOffset);
            LoadTwoTensorForDtypeT<float>(
                ubGamma, ubGamma, vregGammaQ, vregGammaR, pregMain, pregLoop, remainderGeneral * sregvl,
                remainderGeneral * sregvl + binaryQuotientOffset);
            Mul(vregDbetaQ, vregGammaQ, vregDbetaQ, pregMain);
            Mul(vregDbetaR, vregGammaR, vregDbetaR, pregLoop);
            Mul(vregDgammaQ, vregGammaQ, vregDgammaQ, pregMain);
            Mul(vregDgammaR, vregGammaR, vregDgammaR, pregLoop);
            Add(tempDbeta, vregDbetaQ, vregDbetaR, pregLoop);
            Add(tempDgamma, vregDgammaQ, vregDgammaR, pregLoop);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vregDbetaQ, tempDbeta, pregLoop);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vregDgammaQ, tempDgamma, pregLoop);
            ReduceSum(vregSumDgamma, vregDgammaQ, pregMain);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                ubBinaryDgamma + remainderGeneral, vregSumDgamma, pregMerge);
            ReduceSum(vregSumDbeta, vregDbetaQ, pregMain);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                ubBinaryDbeta + remainderGeneral, vregSumDbeta, pregMerge);
        }
        // step3: non-overlapping portions of the first half reduce by 64, this part always 64 align
        for (uint16_t i = 0; i < static_cast<uint16_t>(quotientLoop - remainderLoop); i++) {
            LoadOneTensorForDtypeT<float>(ubDbeta, vregDbeta, pregMain, ((i + remainderLoop) * sregvl));
            LoadOneTensorForDtypeT<float>(ubDgamma, vregDgamma, pregMain, ((i + remainderLoop) * sregvl));
            LoadOneTensorForDtypeT<float>(ubGamma, vregGamma, pregMain, ((i + remainderLoop) * sregvl));
            Mul(vregDbeta, vregGamma, vregDbeta, pregMain);
            Mul(vregDgamma, vregGamma, vregDgamma, pregMain);
            ReduceSum(vregSumDgamma, vregDgamma, pregMain);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                ubBinaryDgamma + remainderLoop + i, vregSumDgamma, pregMerge);
            ReduceSum(vregSumDbeta, vregDbeta, pregMain);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                ubBinaryDbeta + remainderLoop + i, vregSumDbeta, pregMerge);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        // step4: binary folding reduce calculation
        pregMain = CreateMask<float, MaskPattern::ALL>();
        uint16_t curBinaryCGLoop = binaryCGLoop;
        for (uint16_t i = 0; i < binaryCGKLoop; i++) {
            curBinaryCGLoop = curBinaryCGLoop / 2;
            for (uint16_t j = 0; j < curBinaryCGLoop; j++) {
                DataCopy(vregDgammaQ, ((__ubuf__ float*)ubBinaryDgamma + j * sregvl));
                DataCopy(vregDgammaR, ((__ubuf__ float*)ubBinaryDgamma + (j + curBinaryCGLoop) * sregvl));
                Add(vregDgammaQ, vregDgammaQ, vregDgammaR, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubBinaryDgamma + j * sregvl, vregDgammaQ, pregMain);
                DataCopy(vregDbetaQ, ((__ubuf__ float*)ubBinaryDbeta + j * sregvl));
                DataCopy(vregDbetaR, ((__ubuf__ float*)ubBinaryDbeta + (j + curBinaryCGLoop) * sregvl));
                Add(vregDbetaQ, vregDbetaQ, vregDbetaR, pregMain);
                DataCopy(ubBinaryDbeta + j * sregvl, vregDbetaQ, pregMain);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        // step5: vcadd reduce to 1
        {
            uint32_t sreg2 = binaryCGLastNum;
            MaskReg pregLoop = UpdateMask<float>(sreg2);
            DataCopy(vregDgamma, ((__ubuf__ float*)ubBinaryDgamma));
            ReduceSum(vregDgamma, vregDgamma, pregLoop);
            DataCopy(outDgamma, vregDgamma, pregMerge);
            DataCopy(vregDbeta, ((__ubuf__ float*)ubBinaryDbeta));
            ReduceSum(vregDbeta, vregDbeta, pregLoop);
            DataCopy(outDbeta, vregDbeta, pregMerge);
        }
    }
    outQueDgamma_.FreeTensor(binaryDbetaTensor);
    TEventID eventIDVtoS1 = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventIDVtoS1);
    WaitFlag<HardEvent::V_S>(eventIDVtoS1);
    sum1 = outDgamma[0];
    sum2 = outDbeta[0];
}

/*
    sum1 = ReduceSum(dbeta * gamma) / D * HxW
    sum2 = ReduceSum(dgamma * gamma) / D * HxW
*/
template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::VFComputeSum1Sum2(
    const LocalTensor<float>& dbetaTensor, const LocalTensor<float>& dsTensor, const LocalTensor<float>& gammaTensor,
    float& sum1, float& sum2)
{
    __ubuf__ float* ubDbeta = (__ubuf__ float*)dbetaTensor.GetPhyAddr();
    __ubuf__ float* ubDs = (__ubuf__ float*)dsTensor.GetPhyAddr();
    __ubuf__ float* ubGamma = (__ubuf__ float*)gammaTensor.GetPhyAddr();
    LocalTensor<float> outDbetaTensor = tempMeanBuf_.Get<float>();
    LocalTensor<float> outDsTensor = tempRstdBuf_.Get<float>();
    __ubuf__ float* outDbeta = (__ubuf__ float*)outDbetaTensor.GetPhyAddr();
    __ubuf__ float* outDs = (__ubuf__ float*)outDsTensor.GetPhyAddr();
    uint16_t repeatTimes = CeilDiv(this->C_G_, VecLen_);
    uint32_t C_G_Cnt = this->C_G_;

    __VEC_SCOPE__
    {
        uint32_t sreg = C_G_Cnt;
        uint32_t sregvl = (uint32_t)VecLen_;
        RegTensor<float> vregSumDs;
        RegTensor<float> vregSumDbeta;
        RegTensor<float> vregGamma;
        RegTensor<float> vregDs;
        RegTensor<float> vregDbeta;
        RegTensor<float> tempDs;
        RegTensor<float> tempDbeta;
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        Duplicate(vregSumDbeta, 0, pregAll);
        Duplicate(vregSumDs, 0, pregAll);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            MaskReg preg = UpdateMask<float>(sreg);
            DataCopy(vregGamma, ubGamma + i * sregvl);
            DataCopy(vregDbeta, ubDbeta + i * sregvl);
            DataCopy(vregDs, ubDs + i * sregvl);
            Mul(vregDbeta, vregGamma, vregDbeta, preg);
            Mul(vregDs, vregGamma, vregDs, preg);
            Add(tempDbeta, vregSumDbeta, vregDbeta, preg);
            Add(tempDs, vregSumDs, vregDs, preg);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vregSumDbeta, tempDbeta, preg);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vregSumDs, tempDs, preg);
        }
        ReduceSum(vregSumDbeta, vregSumDbeta, pregAll);
        ReduceSum(vregSumDs, vregSumDs, pregAll);
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(outDbeta, vregSumDbeta, pregMerge);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(outDs, vregSumDs, pregMerge);
    }
    TEventID eventIDVtoS1 = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventIDVtoS1);
    WaitFlag<HardEvent::V_S>(eventIDVtoS1);
    sum1 = outDs[0];
    sum2 = outDbeta[0];
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::ComputeSum1Sum2(
    const LocalTensor<float>& dbetaTensor, const LocalTensor<float>& dsTensor, const LocalTensor<float>& gammaTensor,
    float& sum1, float& sum2)
{
    if (this->C_G_ == 1) {
        TEventID eventIDSToV0 = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(eventIDSToV0);
        WaitFlag<HardEvent::S_V>(eventIDSToV0);
        float dbeta = dbetaTensor.GetValue(0);
        float ds = dsTensor.GetValue(0);
        float gamma = gammaTensor.GetValue(0);
        sum1 = ds * gamma;
        sum2 = dbeta * gamma;
        TEventID eventIDVtoS1 = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventIDVtoS1);
        WaitFlag<HardEvent::V_S>(eventIDVtoS1);
    } else if (this->C_G_ <= this->VecLen_) {
        VFComputeSum1Sum2(dbetaTensor, dsTensor, gammaTensor, sum1, sum2);
    } else {
        VFComputeBinaryFoldSum1Sum2(dbetaTensor, dsTensor, gammaTensor, sum1, sum2);
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::InitStage2Mode2Buffer()
{
    pipe->InitBuffer(inQueDgammaChannel_, DOUBLE_BUFFER, stage2FactorN_ * cFactor_ * sizeof(float));
    pipe->InitBuffer(outQueDgamma_, DOUBLE_BUFFER, cFactor_ * sizeof(U));
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::InitStage2Mode1Buffer()
{
    uint32_t cFactorAlign = (((this->cFactor_ * sizeof(U) + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES) / sizeof(U);
    uint32_t tempSize = CeilDiv(reduceNCnt_, SCALE_COEF_EIGHT) * cFactorAlign * sizeof(float);
    pipe->InitBuffer(calQueDgammaReduce_, DOUBLE_BUFFER, cFactorAlign * sizeof(U));
    pipe->InitBuffer(calQueDbetaReduce_, DOUBLE_BUFFER, cFactorAlign * sizeof(U));
    pipe->InitBuffer(inQueDgammaChannel_, DOUBLE_BUFFER, (reduceNCnt_ + 1) * cFactorAlign * sizeof(float));
    pipe->InitBuffer(inQueDbetaChannel_, DOUBLE_BUFFER, (reduceNCnt_ + 1) * cFactorAlign * sizeof(float));
    if constexpr (IsSameType<U, half>::value || IsSameType<U, bfloat16_t>::value) {
        pipe->InitBuffer(outTbufDgammaChannel_, cFactorAlign * sizeof(float));
        pipe->InitBuffer(outTbufDbetaChannel_, cFactorAlign * sizeof(float));
    }
    if (reduceNCnt_ > SCALE_COEF_EIGHT) {
        pipe->InitBuffer(tempDgammaBuf_, tempSize);
        pipe->InitBuffer(tempDbetaBuf_, tempSize);
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::stage2Mode1Process(uint32_t cOffset, uint32_t currentCNum)
{
    if (dgammaIsRequire_) {
        if constexpr (IsSameType<U, half>::value || IsSameType<U, bfloat16_t>::value) {
            stage2Mode1B16Compute(
                inQueDgammaChannel_, calQueDgammaReduce_, outTbufDgammaChannel_, tempDgammaBuf_, dgammaWorkSpace_,
                dgammaGm_, cOffset, currentCNum);
        } else {
            stage2Mode1B32Compute(
                inQueDgammaChannel_, calQueDgammaReduce_, outTbufDgammaChannel_, tempDgammaBuf_, dgammaWorkSpace_,
                dgammaGm_, cOffset, currentCNum);
        }
    }
    if (dbetaIsRequire_) {
        if constexpr (IsSameType<U, half>::value || IsSameType<U, bfloat16_t>::value) {
            stage2Mode1B16Compute(
                inQueDbetaChannel_, calQueDbetaReduce_, outTbufDbetaChannel_, tempDbetaBuf_, dbetaWorkspace_, dbetaGm_,
                cOffset, currentCNum);
        } else {
            stage2Mode1B32Compute(
                inQueDbetaChannel_, calQueDbetaReduce_, outTbufDbetaChannel_, tempDbetaBuf_, dbetaWorkspace_, dbetaGm_,
                cOffset, currentCNum);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::reduceNMode1Wsp2Ub(
    TQue<QuePosition::VECIN, 1>& vecInQue, const GlobalTensor<float>& workspace, uint32_t gmOffset,
    uint32_t currentCNum)
{
    LocalTensor<float> vecInTensor = vecInQue.AllocTensor<float>();
    uint32_t currentANumAlign =
        (((currentCNum * sizeof(float) + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES) / sizeof(float);
    DataCopyPadExtParams<float> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = (currentCNum != currentANumAlign);
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = (currentANumAlign - currentCNum);
    dataCopyPadExtParams.paddingValue = 0;
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = this->reduceNCnt_;
    copyInParams.blockLen = currentCNum * sizeof(float);
    copyInParams.srcStride = (this->C_ - currentCNum) * sizeof(float);
    copyInParams.dstStride = 0;
    DataCopyPad(vecInTensor, workspace[gmOffset], copyInParams, dataCopyPadExtParams);
    vecInQue.EnQue(vecInTensor);
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::stage2Mode1B32Compute(
    TQue<QuePosition::VECIN, 1>& inQue, TQue<QuePosition::VECOUT, 1>& calQue, TBuf<TPosition::VECCALC>& outTbuf,
    TBuf<TPosition::VECCALC>& tempTbuf, const GlobalTensor<float>& workspace, GlobalTensor<U>& gmOut, uint32_t cOffset,
    uint32_t currentCNum)
{
    reduceNMode1Wsp2Ub(inQue, workspace, cOffset, currentCNum);
    LocalTensor<float> inUb = inQue.template DeQue<float>();
    LocalTensor<float> calUb = calQue.AllocTensor<float>();
    __local_mem__ float* inUbAddr = (__local_mem__ float*)inUb.GetPhyAddr();
    __local_mem__ float* calUbAddr = (__local_mem__ float*)calUb.GetPhyAddr();
    if (reduceNCnt_ <= SCALE_COEF_TWO) {
        reduceNMode1LessThan2(inUbAddr, calUbAddr, currentCNum);
    } else if (reduceNCnt_ <= SCALE_COEF_FOUR) {
        reduceNMode1LessThan4(inUbAddr, calUbAddr, currentCNum);
    } else if (reduceNCnt_ <= SCALE_COEF_EIGHT) {
        reduceNMode1LessThan8(inUbAddr, calUbAddr, currentCNum);
    } else {
        LocalTensor<float> tempUb = tempTbuf.AllocTensor<float>();
        __local_mem__ float* tempUbAddr = (__local_mem__ float*)tempUb.GetPhyAddr();
        reduceNMode1MoreThan8(inUbAddr, tempUbAddr, calUbAddr, currentCNum);
    }
    inQue.FreeTensor(inUb);
    calQue.EnQue(calUb);
    LocalTensor<float> calOutUb = calQue.template DeQue<float>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentCNum * sizeof(float)), 0, 0, 0};
    DataCopyPad(gmOut[cOffset], calOutUb, copyParams);
    calQue.FreeTensor(calOutUb);
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::stage2Mode1B16Compute(
    TQue<QuePosition::VECIN, 1>& inQue, TQue<QuePosition::VECOUT, 1>& calQue, TBuf<TPosition::VECCALC>& outTbuf,
    TBuf<TPosition::VECCALC>& tempTbuf, const GlobalTensor<float>& workspace, GlobalTensor<U>& gmOut, uint32_t cOffset,
    uint32_t currentCNum)
{
    reduceNMode1Wsp2Ub(inQue, workspace, cOffset, currentCNum);
    LocalTensor<float> inUb = inQue.template DeQue<float>();
    LocalTensor<U> calUb = calQue.AllocTensor<U>();
    __local_mem__ float* inUbAddr = (__local_mem__ float*)inUb.GetPhyAddr();
    __local_mem__ U* calUbAddr = (__local_mem__ U*)calUb.GetPhyAddr();
    LocalTensor<float> outTbufUb = outTbuf.Get<float>();
    __local_mem__ float* outTbufUbAddr = (__local_mem__ float*)outTbufUb.GetPhyAddr();
    if (reduceNCnt_ <= SCALE_COEF_TWO) {
        reduceNMode1LessThan2(inUbAddr, outTbufUbAddr, currentCNum);
    } else if (reduceNCnt_ <= SCALE_COEF_FOUR) {
        reduceNMode1LessThan4(inUbAddr, outTbufUbAddr, currentCNum);
    } else if (reduceNCnt_ <= SCALE_COEF_EIGHT) {
        reduceNMode1LessThan8(inUbAddr, outTbufUbAddr, currentCNum);
    } else {
        LocalTensor<float> tempUb = tempTbuf.AllocTensor<float>();
        __local_mem__ float* tempUbAddr = (__local_mem__ float*)tempUb.GetPhyAddr();
        reduceNMode1MoreThan8(inUbAddr, tempUbAddr, outTbufUbAddr, currentCNum);
    }
    VFCastFloat2T<U>((__local_mem__ U*)calUbAddr, (__local_mem__ float*)outTbufUbAddr, currentCNum, this->VecLen_);
    inQue.FreeTensor(inUb);
    calQue.EnQue(calUb);
    LocalTensor<U> calOutUb = calQue.template DeQue<U>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentCNum * sizeof(U)), 0, 0, 0};
    DataCopyPad(gmOut[cOffset], calOutUb, copyParams);
    calQue.FreeTensor(calOutUb);
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::VFReduceNC(
    const LocalTensor<float>& inTensor, const LocalTensor<U>& outTensor, const uint32_t currCNum,
    const uint32_t cNumAlign, const uint32_t currNNum)
{
    __local_mem__ float* ubIn = (__local_mem__ float*)inTensor.GetPhyAddr();
    __local_mem__ U* ubOut = (__local_mem__ U*)outTensor.GetPhyAddr();
    uint16_t repeatTimes = CeilDiv(currCNum, VecLen_);
    uint32_t sregvl = this->VecLen_;
    __local_mem__ float* currUbIn;

    __VEC_SCOPE__
    {
        RegTensor<float> vregOut;
        uint32_t sreg = currCNum;
        MaskReg preg;
        MaskReg pregLoop;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            RegTensor<float> vregIn;
            preg = UpdateMask<float>(sreg);
            pregLoop = CreateMask<float, MaskPattern::ALL>();
            LoadOneTensorForDtypeT<U>(ubOut, vregOut, preg, i * sregvl);
            for (uint16_t idxN = 0; idxN < static_cast<uint16_t>(currNNum); idxN++) {
                currUbIn = ubIn + (idxN * cNumAlign);
                LoadOneTensorForDtypeT<float>(currUbIn, vregIn, preg, i * sregvl);
                Add(vregOut, vregOut, vregIn, preg);
            }
            StoreOneTensorForDtypeT<U>(ubOut, vregOut, preg, i * sregvl);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::ProcessStage2Mode2(
    const GlobalTensor<float>& workspace, const GlobalTensor<U>& gmOut)
{
    uint32_t currProcessCNum = (curBlockIdx_ == stage2CoreUsed_ - 1) ? tailcNumMode2PerCore_ : cNumMode2PerCore_;
    uint32_t currLoopCnt = (curBlockIdx_ == stage2CoreUsed_ - 1) ? stage2LoopCnt_ : CeilDiv(currProcessCNum, cFactor_);
    uint32_t loopCNum = cFactor_ > cNumMode2PerCore_ ? cNumMode2PerCore_ : cFactor_;
    uint32_t currCNum = loopCNum;
    for (uint32_t loopNum = 0; loopNum < currLoopCnt; loopNum++) {
        if (loopNum == currLoopCnt - 1 && currProcessCNum % cFactor_) {
            currCNum = currProcessCNum % cFactor_;
        }
        uint32_t strideTime = CeilDiv(reduceNCnt_, stage2FactorN_);
        uint32_t currNNum = stage2FactorN_;
        LocalTensor<U> outTensor = outQueDgamma_.AllocTensor<U>();
        Duplicate<U>(outTensor, 0.0f, currCNum);
        int64_t baseOffset = curBlockIdx_ * cNumMode2PerCore_ + loopNum * cFactor_;
        for (uint32_t strideNum = 0; strideNum < strideTime; strideNum++) {
            if (strideNum == strideTime - 1 && reduceNCnt_ % stage2FactorN_) {
                currNNum = reduceNCnt_ % stage2FactorN_;
            }
            LocalTensor<float> inTensor = inQueDgammaChannel_.AllocTensor<float>();
            int64_t inOffset = baseOffset + strideNum * stage2FactorN_ * C_;
            uint32_t cNumAlign = CeilAlign(currCNum, PF32_PER_BLOCK);
            DataCopyExtParams copyParams;
            copyParams.blockCount = currNNum;
            copyParams.blockLen = currCNum * sizeof(float);
            copyParams.srcStride = (C_ - currCNum) * sizeof(float);
            copyParams.dstStride = 0;
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            DataCopyPad(inTensor, workspace[inOffset], copyParams, padParams);
            inQueDgammaChannel_.EnQue(inTensor);
            inTensor = inQueDgammaChannel_.DeQue<float>();
            VFReduceNC(inTensor, outTensor, currCNum, cNumAlign, currNNum);
            inQueDgammaChannel_.FreeTensor(inTensor);
        }
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = currCNum * sizeof(U);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        outQueDgamma_.EnQue(outTensor);
        outTensor = outQueDgamma_.DeQue<U>();
        DataCopyPad(gmOut[baseOffset], outTensor, copyOutParams);
        outQueDgamma_.FreeTensor(outTensor);
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::reduceNMode1LessThan2(
    __local_mem__ float* inUbAddr, __local_mem__ float* calUbAddr, uint32_t currentCNum)
{
    uint32_t rStride = (((currentCNum * sizeof(float) + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES) / sizeof(float);
    uint16_t rLoopCount = static_cast<uint16_t>(reduceNCnt_);
    uint16_t cLoopCount = CeilDiv(currentCNum, VecLen_);

    __VEC_SCOPE__
    {
        RegTensor<float> inld;
        RegTensor<float> sum;

        MaskReg pregLoop;
        uint32_t sreg0 = currentCNum;
        for (uint16_t k = 0; k < cLoopCount; k++) {
            pregLoop = UpdateMask<float>(sreg0);
            Duplicate(sum, 0.0, pregLoop);
            for (uint16_t i = 0; i < rLoopCount; i++) {
                DataCopy(inld, ((__local_mem__ float*)inUbAddr + i * rStride + k * VecLen_));
                Add(sum, sum, inld, pregLoop);
            }
            DataCopy(((__local_mem__ float*)calUbAddr + k * VecLen_), sum, pregLoop);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::reduceNMode1LessThan4(
    __local_mem__ float* inUbAddr, __local_mem__ float* calUbAddr, uint32_t currentCNum)
{
    uint32_t currentANumAlign =
        (((currentCNum * sizeof(float) + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES) / sizeof(float);
    uint32_t remainderOffset = SCALE_COEF_TWO * currentANumAlign;
    uint32_t aLength = currentANumAlign;
    uint32_t validNumInXUb = reduceNCnt_ * currentANumAlign;
    uint16_t remainderTailCount = reduceNCnt_ - SCALE_COEF_TWO;
    uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderOffset;
    uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderOffset + aLength;

    uint16_t aLoopCount = CeilDiv(currentCNum, VecLen_);
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> nextRow;
        RegTensor<float> rem;
        RegTensor<float> remNextRow;

        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> zero;
        Duplicate(zero, 0.0, pregMain);

        MaskReg pregLoop;
        uint32_t sreg0 = currentCNum;
        for (uint16_t k = 0; k < aLoopCount; k++) {
            pregLoop = UpdateMask<float>(sreg0);
            uint32_t aLoopOffset = k * VecLen_;
            DataCopy(((__local_mem__ float*)inUbAddr + validNumInXUb + aLoopOffset), zero, pregLoop);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            TwoRowAddWithTail(
                x1, inUbAddr, pregLoop, aLoopOffset, remainderTailOffset0 + aLoopOffset, aLength + aLoopOffset,
                remainderTailOffset1 + aLoopOffset, rem, nextRow, remNextRow);
            DataCopy(((__local_mem__ float*)calUbAddr + aLoopOffset), x1, pregLoop);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::reduceNMode1LessThan8(
    __local_mem__ float* inUbAddr, __local_mem__ float* calUbAddr, uint32_t currentCNum)
{
    uint32_t currentANumAlign =
        (((currentCNum * sizeof(float) + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES) / sizeof(float);
    uint32_t remainderOffset = SCALE_COEF_FOUR * currentANumAlign;
    uint32_t aLength = currentANumAlign;
    uint32_t validNumInXUb = reduceNCnt_ * currentANumAlign;
    uint16_t remainderTailCount = reduceNCnt_ - SCALE_COEF_FOUR;
    uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderOffset;
    uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderOffset + aLength;
    uint32_t remainderTailOffset2 =
        (ROW_TWO > remainderTailCount) ? validNumInXUb : remainderOffset + ROW_TWO_OFFSET * aLength;
    uint32_t remainderTailOffset3 =
        (ROW_THREE > remainderTailCount) ? validNumInXUb : remainderOffset + ROW_THREE_OFFSET * aLength;

    uint16_t aLoopCount = CeilDiv(currentCNum, VecLen_);
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> x2;
        RegTensor<float> nextRow;
        RegTensor<float> rem;
        RegTensor<float> remNextRow;

        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> zero;
        Duplicate(zero, 0.0, pregMain);

        MaskReg pregLoop;
        uint32_t sreg0 = currentCNum;
        for (uint16_t k = 0; k < aLoopCount; k++) {
            pregLoop = UpdateMask<float>(sreg0);
            uint32_t aLoopOffset = k * VecLen_;
            DataCopy(((__local_mem__ float*)inUbAddr + validNumInXUb + aLoopOffset), zero, pregLoop);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            TwoRowAddWithTail(
                x1, inUbAddr, pregLoop, aLoopOffset, remainderTailOffset0 + aLoopOffset, aLength + aLoopOffset,
                remainderTailOffset1 + aLoopOffset, rem, nextRow, remNextRow);
            TwoRowAddWithTail(
                x2, inUbAddr, pregLoop, ROW_TWO_OFFSET * aLength + aLoopOffset, remainderTailOffset2 + aLoopOffset,
                ROW_THREE_OFFSET * aLength + aLoopOffset, remainderTailOffset3 + aLoopOffset, rem, nextRow, remNextRow);
            Add(x1, x1, x2, pregLoop);
            DataCopy(((__local_mem__ float*)calUbAddr + aLoopOffset), x1, pregLoop);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::reduceNMode1MoreThan8(
    __local_mem__ float* inUbAddr, __local_mem__ float* tempUbAddr, __local_mem__ float* calUbAddr,
    uint32_t currentCNum)
{
    uint32_t currentANumAlign =
        (((currentCNum * sizeof(float) + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES) / sizeof(float);
    uint16_t remainderLoopCount =
        (reduceNCnt_ - this->stage2BinaryAddQuotient_ + SCALE_COEF_EIGHT - 1) / SCALE_COEF_EIGHT;
    uint16_t quotientLoopCount = (this->stage2BinaryAddQuotient_ / SCALE_COEF_EIGHT) - remainderLoopCount;
    uint32_t remainderOffset = this->stage2BinaryAddQuotient_ * currentANumAlign;

    uint32_t baseLineOffset = SCALE_COEF_EIGHT * currentANumAlign;
    uint32_t aLength = currentANumAlign;
    uint16_t binaryAddKLoop = this->stage2BinaryAddK_;
    uint16_t binaryAddInnerLoop = this->stage2BinaryAddQuotient_ / SCALE_COEF_EIGHT;
    uint32_t validNumInXUb = this->reduceNCnt_ * currentANumAlign;

    uint16_t remainderTailCount =
        (this->reduceNCnt_ - this->stage2BinaryAddQuotient_) - (remainderLoopCount - 1) * SCALE_COEF_EIGHT;
    uint32_t quotientTailOffset = (remainderLoopCount - 1) * baseLineOffset;
    uint32_t remainderTailOffset = quotientTailOffset + remainderOffset;
    uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderTailOffset;
    uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderTailOffset + aLength;
    uint32_t remainderTailOffset2 =
        (ROW_TWO > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_TWO_OFFSET * aLength;
    uint32_t remainderTailOffset3 =
        (ROW_THREE > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_THREE_OFFSET * aLength;
    uint32_t remainderTailOffset4 =
        (ROW_FOUR > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_FOUR_OFFSET * aLength;
    uint32_t remainderTailOffset5 =
        (ROW_FIVE > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_FIVE_OFFSET * aLength;
    uint32_t remainderTailOffset6 =
        (ROW_SIX > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_SIX_OFFSET * aLength;
    uint32_t remainderTailOffset7 =
        (ROW_SEVEN > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_SEVEN_OFFSET * aLength;

    uint32_t rowTwoOffset = ROW_TWO_OFFSET;
    uint32_t rowThreeOffset = ROW_THREE_OFFSET;
    uint32_t rowFourOffset = ROW_FOUR_OFFSET;
    uint32_t rowFiveOffset = ROW_FIVE_OFFSET;
    uint32_t rowSixOffset = ROW_SIX_OFFSET;
    uint32_t rowSevenOffset = ROW_SEVEN_OFFSET;

    uint16_t aLoopCount = CeilDiv(currentCNum, VecLen_);
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> x2;
        RegTensor<float> x3;
        RegTensor<float> x4;

        RegTensor<float> nextRow;
        RegTensor<float> rem;
        RegTensor<float> remNextRow;

        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> zero;
        Duplicate(zero, 0.0, pregMain);

        MaskReg pregLoop;
        uint32_t sreg0 = currentCNum;
        for (uint16_t k = 0; k < aLoopCount; k++) {
            pregLoop = UpdateMask<float>(sreg0);
            uint32_t aLoopOffset = k * VecLen_;
            DataCopy(((__local_mem__ float*)inUbAddr + validNumInXUb + aLoopOffset), zero, pregLoop);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            // 前半部分与后半部分中，都为8行的部分
            for (uint16_t i = 0; i < static_cast<uint16_t>(remainderLoopCount - 1); i++) {
                uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                TwoRowAddWithTail(
                    x1, inUbAddr, pregLoop, quotOffset, remOffset, quotOffset + aLength, remOffset + aLength, rem,
                    nextRow, remNextRow);
                TwoRowAddWithTail(
                    x2, inUbAddr, pregLoop, quotOffset + rowTwoOffset * aLength, remOffset + rowTwoOffset * aLength,
                    quotOffset + rowThreeOffset * aLength, remOffset + rowThreeOffset * aLength, rem, nextRow,
                    remNextRow);
                Add(x1, x1, x2, pregLoop);
                TwoRowAddWithTail(
                    x3, inUbAddr, pregLoop, quotOffset + rowFourOffset * aLength, remOffset + rowFourOffset * aLength,
                    quotOffset + rowFiveOffset * aLength, remOffset + rowFiveOffset * aLength, rem, nextRow,
                    remNextRow);
                TwoRowAddWithTail(
                    x4, inUbAddr, pregLoop, quotOffset + rowSixOffset * aLength, remOffset + rowSixOffset * aLength,
                    quotOffset + rowSevenOffset * aLength, remOffset + rowSevenOffset * aLength, rem, nextRow,
                    remNextRow);
                Add(x3, x3, x4, pregLoop);
                Add(x1, x1, x3, pregLoop);
                DataCopy(((__local_mem__ float*)tempUbAddr + i * aLength + aLoopOffset), x1, pregLoop);
            }
            // 前半部分为8行，后半部分可能不足8行
            {
                TwoRowAddWithTail(
                    x1, inUbAddr, pregLoop, quotientTailOffset + aLoopOffset, remainderTailOffset0 + aLoopOffset,
                    quotientTailOffset + aLength + aLoopOffset, remainderTailOffset1 + aLoopOffset, rem, nextRow,
                    remNextRow);
                TwoRowAddWithTail(
                    x2, inUbAddr, pregLoop, quotientTailOffset + rowTwoOffset * aLength + aLoopOffset,
                    remainderTailOffset2 + aLoopOffset, quotientTailOffset + rowThreeOffset * aLength + aLoopOffset,
                    remainderTailOffset3 + aLoopOffset, rem, nextRow, remNextRow);
                Add(x1, x1, x2, pregLoop);
                TwoRowAddWithTail(
                    x3, inUbAddr, pregLoop, quotientTailOffset + rowFourOffset * aLength + aLoopOffset,
                    remainderTailOffset4 + aLoopOffset, quotientTailOffset + rowFiveOffset * aLength + aLoopOffset,
                    remainderTailOffset5 + aLoopOffset, rem, nextRow, remNextRow);
                TwoRowAddWithTail(
                    x4, inUbAddr, pregLoop, quotientTailOffset + rowSixOffset * aLength + aLoopOffset,
                    remainderTailOffset6 + aLoopOffset, quotientTailOffset + rowSevenOffset * aLength + aLoopOffset,
                    remainderTailOffset7 + aLoopOffset, rem, nextRow, remNextRow);
                Add(x3, x3, x4, pregLoop);
                Add(x1, x1, x3, pregLoop);
                DataCopy(
                    ((__local_mem__ float*)tempUbAddr + (remainderLoopCount - 1) * aLength + aLoopOffset), x1,
                    pregLoop);
            }
            // 剩余的前半部分，一次for循环，处理8行
            for (uint16_t i = 0; i < quotientLoopCount; i++) {
                uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                TwoRowAdd(x1, inUbAddr, pregLoop, baseOffset, baseOffset + aLength, nextRow);
                TwoRowAdd(
                    x2, inUbAddr, pregLoop, baseOffset + rowTwoOffset * aLength, baseOffset + rowThreeOffset * aLength,
                    nextRow);
                Add(x1, x1, x2, pregLoop);
                TwoRowAdd(
                    x3, inUbAddr, pregLoop, baseOffset + rowFourOffset * aLength, baseOffset + rowFiveOffset * aLength,
                    nextRow);
                TwoRowAdd(
                    x4, inUbAddr, pregLoop, baseOffset + rowSixOffset * aLength, baseOffset + rowSevenOffset * aLength,
                    nextRow);
                Add(x3, x3, x4, pregLoop);
                Add(x1, x1, x3, pregLoop);
                DataCopy(
                    ((__local_mem__ float*)tempUbAddr + (remainderLoopCount + i) * aLength + aLoopOffset), x1,
                    pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
            for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                curBinaryAddInnerLoop = curBinaryAddInnerLoop / 2;
                for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
                    DataCopy(x1, ((__local_mem__ float*)tempUbAddr) + j * aLength + aLoopOffset);
                    DataCopy(
                        x2, ((__local_mem__ float*)tempUbAddr) + (j + curBinaryAddInnerLoop) * aLength + aLoopOffset);
                    Add(x1, x1, x2, pregLoop);
                    DataCopy(((__local_mem__ float*)tempUbAddr + j * aLength + aLoopOffset), x1, pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            }
            DataCopy(x1, ((__local_mem__ float*)tempUbAddr) + aLoopOffset);
            DataCopy(((__local_mem__ float*)calUbAddr + aLoopOffset), x1, pregLoop);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::TwoRowAddWithTail(
    RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg, uint32_t offset1, uint32_t offset2,
    uint32_t offset3, uint32_t offset4, RegTensor<float>& rem, RegTensor<float>& nextRow, RegTensor<float>& remNextRow)
{
    DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
    DataCopy(rem, ((__local_mem__ float*)(input) + (offset2)));
    Add(dst, dst, rem, preg);
    DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset3)));
    DataCopy(remNextRow, ((__local_mem__ float*)(input) + (offset4)));
    Add(nextRow, nextRow, remNextRow, preg);
    Add(dst, dst, nextRow, preg);
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradBase<T, U>::TwoRowAdd(
    RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg, uint32_t offset1, uint32_t offset2,
    RegTensor<float>& nextRow)
{
    DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
    DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset2)));
    Add(dst, dst, nextRow, preg);
}
} // namespace GroupNormGrad

#endif