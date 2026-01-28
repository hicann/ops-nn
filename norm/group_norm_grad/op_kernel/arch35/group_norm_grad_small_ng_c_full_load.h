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
 * \file group_norm_grad_small_ng_c_full_load.h
 * \brief
 */
#ifndef GROUP_NORM_GRAD_SMALL_NG_C_FULL_LOAD_H
#define GROUP_NORM_GRAD_SMALL_NG_C_FULL_LOAD_H
#pragma once

#include "kernel_operator.h"
#include "group_norm_grad_base.h"
#include "group_norm_grad_common.h"

namespace GroupNormGrad {
template <typename T, typename U>
class GroupNormGradSmallNGCFullLoad : public GroupNormGradBase<T, U> {
public:
    __aicore__ inline GroupNormGradSmallNGCFullLoad() : GroupNormGradBase<T, U>(){};
    __aicore__ inline ~GroupNormGradSmallNGCFullLoad(){};
    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta,
        GM_ADDR workspace, const GroupNormGradRegBaseTilingData* __restrict tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitBuffer();
    __aicore__ inline void ComputeStage0();
    __aicore__ inline void ComputeStage1(int32_t taskIdx);
    __aicore__ inline void ComputeStage2();
    __aicore__ inline void VFMode0DbetaDs(
        const LocalTensor<T>& x, const LocalTensor<T>& dy, const LocalTensor<float>& dbeta,
        const LocalTensor<float>& dgamma, const uint32_t loopIdx, const uint32_t curCNum);
    __aicore__ inline void VFMode0DbetaDsBinaryFold(
        const LocalTensor<T>& x, const LocalTensor<T>& dy, const LocalTensor<float>& dbeta,
        const LocalTensor<float>& dgamma, const uint32_t loopIdx, const uint32_t curCNum);
    __aicore__ inline void VFComputeStage1Dx(
        const LocalTensor<T>& dstTensor, const LocalTensor<T>& xTensor, const LocalTensor<T>& dyTensor,
        const LocalTensor<float>& gammaTensor, const float C2, const float C3, const uint32_t loopIdx,
        const uint32_t curCNum);
    __aicore__ inline void VFComputeStage1Ds(
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& dsTensor, const LocalTensor<float>& dbetaTensor);

    uint32_t stage0TaskNumPerCore_;
    uint32_t stage0TaskNumPerTailCore_;
    uint32_t stage0TailCore_;
    uint32_t stage0CoreUsed_;
};

template <typename T, typename U>
__aicore__ inline void GroupNormGradSmallNGCFullLoad<T, U>::Init(
    GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta,
    GM_ADDR workspace, const GroupNormGradRegBaseTilingData* __restrict tilingData, TPipe* pipeIn)
{
    this->pipe = pipeIn;
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    this->curBlockIdx_ = GetBlockIdx();
    // get tiling data
    this->ParseTilingData(tilingData);
    this->stage0TaskNumPerCore_ = tilingData->gNGBaseParams.stage0TaskNumPerCore;
    this->stage0TaskNumPerTailCore_ = tilingData->gNGBaseParams.stage0TaskNumPerTailCore;
    this->stage0TailCore_ = tilingData->gNGBaseParams.stage0TailCore;
    this->stage0CoreUsed_ = tilingData->gNGBaseParams.stage0CoreUsed;
    // init gm
    this->elemTPerBlock_ = this->BLOCK_BYTES / sizeof(T);
    this->elemUPerBlock_ = this->BLOCK_BYTES / sizeof(U);
    this->dyGm_.SetGlobalBuffer((__gm__ T*)dy, this->allEleNum_);
    this->xGm_.SetGlobalBuffer((__gm__ T*)x, this->allEleNum_);
    this->dxGm_.SetGlobalBuffer((__gm__ T*)dx, this->allEleNum_);
    this->meanGm_.SetGlobalBuffer((__gm__ U*)mean, this->NXG_);
    this->rstdGm_.SetGlobalBuffer((__gm__ U*)rstd, this->NXG_);
    this->gammaGm_.SetGlobalBuffer((__gm__ U*)gamma, this->C_);
    this->dgammaGm_.SetGlobalBuffer((__gm__ U*)dgamma, this->C_);
    this->dbetaGm_.SetGlobalBuffer((__gm__ U*)dbeta, this->C_);
    this->dgammaWorkSpace_.SetGlobalBuffer((__gm__ float*)workspace, this->workSpaceSize_);
    this->dbetaWorkspace_.SetGlobalBuffer((__gm__ float*)workspace + this->workSpaceSize_, this->workSpaceSize_);
    // init ub
    InitBuffer();
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradSmallNGCFullLoad<T, U>::Process()
{
    // stage 0
    if (this->curBlockIdx_ < this->stage0CoreUsed_) {
        if (this->curBlockIdx_ >= this->stage0TailCore_) {
            this->curCoreTaskNum_ = this->stage0TaskNumPerTailCore_;
            this->startTaskId_ = this->stage0TaskNumPerTailCore_ * this->curBlockIdx_ + this->stage0TailCore_;
        } else {
            this->curCoreTaskNum_ = this->stage0TaskNumPerCore_;
            this->startTaskId_ = this->stage0TaskNumPerCore_ * this->curBlockIdx_;
        }
        ComputeStage0();
    }
    SyncAll();
    // stage1
    if (this->curBlockIdx_ < this->stage1CoreUsed_) {
        if (this->curBlockIdx_ >= this->tailCore_) {
            this->curCoreTaskNum_ = this->taskNumPerTailCore_;
            this->startTaskId_ = this->taskNumPerTailCore_ * this->curBlockIdx_ + this->tailCore_;
        } else {
            this->curCoreTaskNum_ = this->taskNumPerCore_;
            this->startTaskId_ = this->taskNumPerCore_ * this->curBlockIdx_;
        }
        // 每个核处理curCoreTaskNum_个G,
        for (int32_t taskIdx = 0; taskIdx < this->curCoreTaskNum_; taskIdx++) {
            this->LoadMeanRstd(taskIdx + this->startTaskId_);
            ComputeStage1(taskIdx + this->startTaskId_);
        }
    }
    // stage2
    if (this->dgammaIsRequire_ || this->dbetaIsRequire_) {
        this->ComputeStage2();
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradSmallNGCFullLoad<T, U>::InitBuffer()
{
    // for cast mean, rstd, for store sum1, sum2
    this->pipe->InitBuffer(this->tempMeanBuf_, this->BLOCK_BYTES);
    this->pipe->InitBuffer(this->tempRstdBuf_, this->BLOCK_BYTES);
    uint32_t capElemAllocSpace =
        CeilAlign(static_cast<uint32_t>(this->eleNumPerC_ * this->mode1UbCapCNum_), this->elemTPerBlock_) *
        sizeof(T);
    uint32_t cGAllocSpace = CeilAlign(this->C_G_, this->elemUPerBlock_) * sizeof(float);
    // VEC_IN
    this->pipe->InitBuffer(this->inQueDy_, DOUBLE_BUFFER, capElemAllocSpace);
    this->pipe->InitBuffer(this->inQueX_, DOUBLE_BUFFER, capElemAllocSpace);
    this->pipe->InitBuffer(this->inQueGamma_, 1, cGAllocSpace);
    this->pipe->InitBuffer(this->inQueDs_, 1, cGAllocSpace);
    this->pipe->InitBuffer(this->inQueDb_, 1, cGAllocSpace);
    // VEC_OUT
    this->pipe->InitBuffer(this->outQueDx_, DOUBLE_BUFFER, capElemAllocSpace);
    this->pipe->InitBuffer(this->outQueDs_, 1, cGAllocSpace);
    this->pipe->InitBuffer(this->outQueDgamma_, 1, cGAllocSpace);
    this->pipe->InitBuffer(this->outQueDbeta_, 1, cGAllocSpace);
    if constexpr (IsSameType<U, half>::value || IsSameType<U, bfloat16_t>::value) {
        uint32_t cGAllocSpaceNum = CeilAlign(this->C_G_, this->elemUPerBlock_);
        this->pipe->InitBuffer(this->tBufGamma_, cGAllocSpaceNum * sizeof(U));
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradSmallNGCFullLoad<T, U>::ComputeStage0()
{
    LocalTensor<float> dbetaTensor = this->outQueDbeta_.template AllocTensor<float>();
    LocalTensor<float> dsTensor = this->outQueDs_.template AllocTensor<float>();
    uint32_t baseOffset = this->startTaskId_ * this->eleNumPerC_;
    uint32_t ubLoopCnt = (this->curCoreTaskNum_ + this->mode1UbCapCNum_ - 1) / this->mode1UbCapCNum_;
    bool isLessEqualVL = this->eleNumPerC_ <= this->VecLen_;
    LocalTensor<T> dyTensor;
    LocalTensor<T> xTensor;
    for (uint32_t loopIdx = 0; loopIdx < ubLoopCnt; loopIdx++) {
        int64_t offset = baseOffset + loopIdx * this->eleNumPerC_ * this->mode1UbCapCNum_;
        auto curCNum = this->mode1UbCapCNum_;
        if ((loopIdx == ubLoopCnt - 1) && (this->curCoreTaskNum_ % this->mode1UbCapCNum_ != 0)) {
            curCNum = this->curCoreTaskNum_ % this->mode1UbCapCNum_;
        }

        xTensor = this->inQueX_.template AllocTensor<T>();
        dyTensor = this->inQueDy_.template AllocTensor<T>();
        this->CopyInDyAndX(dyTensor, xTensor, offset, this->eleNumPerC_ * curCNum);        
        this->inQueX_.EnQue(xTensor);
        this->inQueDy_.EnQue(dyTensor);

        dyTensor = this->inQueDy_.template DeQue<T>();
        xTensor = this->inQueX_.template DeQue<T>();
        if (isLessEqualVL) {
            VFMode0DbetaDs(xTensor, dyTensor, dbetaTensor, dsTensor, loopIdx, curCNum);
        } else {
            VFMode0DbetaDsBinaryFold(xTensor, dyTensor, dbetaTensor, dsTensor, loopIdx, curCNum);
        }
        this->inQueX_.FreeTensor(xTensor);
        this->inQueDy_.FreeTensor(dyTensor);
    }
    this->outQueDbeta_.EnQue(dbetaTensor);
    this->outQueDs_.EnQue(dsTensor);
    dsTensor = this->outQueDs_.template DeQue<float>();
    dbetaTensor = this->outQueDbeta_.template DeQue<float>();
    // st dbeta and ds
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->curCoreTaskNum_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(this->dbetaWorkspace_[this->startTaskId_], dbetaTensor, copyParams);
    DataCopyPad(this->dgammaWorkSpace_[this->startTaskId_], dsTensor, copyParams);

    this->outQueDbeta_.FreeTensor(dbetaTensor);
    this->outQueDs_.FreeTensor(dsTensor);
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradSmallNGCFullLoad<T, U>::ComputeStage1(int32_t taskIdx)
{
    int64_t dbDsOffset = taskIdx * this->C_G_;
    // load db and ds, 原始数据是 N*C  的，即N*G*D
    LocalTensor<float> dbetaTensor = this->inQueDb_.template AllocTensor<float>();
    LocalTensor<float> dsTensor = this->inQueDs_.template AllocTensor<float>();
    DataCopyPadExtParams<float> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = this->C_G_ * sizeof(float);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopyPad(dbetaTensor, this->dbetaWorkspace_[dbDsOffset], copyInParams, dataCopyPadExtParams);
    DataCopyPad(dsTensor, this->dgammaWorkSpace_[dbDsOffset], copyInParams, dataCopyPadExtParams);
    this->inQueDb_.EnQue(dbetaTensor);
    this->inQueDs_.EnQue(dsTensor);
    dbetaTensor = this->inQueDb_.template DeQue<float>();
    dsTensor = this->inQueDs_.template DeQue<float>();

    if (this->dxIsRequire_) {
        float sum1 = 0;
        float sum2 = 0;
        uint32_t channelIdx = (taskIdx % this->G_) * this->C_G_;
        uint32_t baseOffset = taskIdx * this->eleNumPerG_;
        uint32_t ubCapEleNum = CeilAlign(static_cast<uint32_t>(this->eleNumPerC_), this->elemTPerBlock_);
        this->LoadDataToUb(this->inQueGamma_, this->tBufGamma_, this->gammaGm_, channelIdx, this->C_G_);
        LocalTensor<float> gammaTensor = this->inQueGamma_.template DeQue<float>();
        this->ComputeSum1Sum2(dbetaTensor, dsTensor, gammaTensor, sum1, sum2);
        float s = 1.0f / this->eleNumPerG_;
        float C2 = (sum2 * this->meanScalar_ - sum1) * this->rstdScalar_ * this->rstdScalar_ * this->rstdScalar_ * s;
        float C3 = -C2 * this->meanScalar_ - sum2 * this->rstdScalar_ * s;

        if (this->dgammaIsRequire_ || this->dbetaIsRequire_) {
            // 需要计算(ds - mean*db)*rstd
            LocalTensor<float> dgammaTensor = this->outQueDgamma_.template AllocTensor<float>();
            VFComputeStage1Ds(dgammaTensor, dsTensor, dbetaTensor);
            this->outQueDgamma_.EnQue(dgammaTensor);
            dgammaTensor = this->outQueDgamma_.template DeQue<float>();
            DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(this->C_G_ * sizeof(float)), 0, 0, 0};
            DataCopyPad(this->dgammaWorkSpace_[dbDsOffset], dgammaTensor, outCopyParams);
            this->outQueDgamma_.FreeTensor(dgammaTensor);
        }

        this->inQueDb_.FreeTensor(dbetaTensor);
        this->inQueDs_.FreeTensor(dsTensor);

        LocalTensor<T> xTensor;
        LocalTensor<T> dyTensor;
        uint32_t ubLoopCnt = (this->C_G_ + this->mode1UbCapCNum_ - 1) / this->mode1UbCapCNum_;
        for (uint32_t loopIdx = 0; loopIdx < ubLoopCnt; loopIdx++) {
            int64_t offset = baseOffset + loopIdx * this->mode1UbCapCNum_ * this->eleNumPerC_;
            auto curCNum = this->mode1UbCapCNum_;
            if ((loopIdx == ubLoopCnt - 1) && (this->C_G_ % this->mode1UbCapCNum_ != 0)) {
                curCNum = this->C_G_ % this->mode1UbCapCNum_;
            }

            xTensor = this->inQueX_.template AllocTensor<T>();
            dyTensor = this->inQueDy_.template AllocTensor<T>();
            this->CopyInDyAndX(dyTensor, xTensor, offset, this->eleNumPerC_ * curCNum);
            this->inQueX_.EnQue(xTensor);
            this->inQueDy_.EnQue(dyTensor);

            xTensor = this->inQueX_.template DeQue<T>();
            dyTensor = this->inQueDy_.template DeQue<T>();
            LocalTensor<T> dxTensor = this->outQueDx_.template AllocTensor<T>();
            VFComputeStage1Dx(dxTensor, xTensor, dyTensor, gammaTensor, C2, C3, loopIdx, curCNum);
            this->inQueX_.FreeTensor(xTensor);
            this->inQueDy_.FreeTensor(dyTensor);
            this->outQueDx_.EnQue(dxTensor);
            this->StoreDxToGm(this->outQueDx_, offset, this->eleNumPerC_ * curCNum);
        }
        this->inQueGamma_.FreeTensor(gammaTensor);
    } else if (this->dgammaIsRequire_) {
        LocalTensor<float> dgammaTensor = this->outQueDgamma_.template AllocTensor<float>();
        VFComputeStage1Ds(dgammaTensor, dsTensor, dbetaTensor);
        this->outQueDgamma_.EnQue(dgammaTensor);
        dgammaTensor = this->outQueDgamma_.template DeQue<float>();
        DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(this->C_G_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(this->dgammaWorkSpace_[dbDsOffset], dgammaTensor, outCopyParams);

        this->outQueDgamma_.FreeTensor(dgammaTensor);
        this->inQueDb_.FreeTensor(dbetaTensor);
        this->inQueDs_.FreeTensor(dsTensor);
    }
}

/*
  dbeta = reduceSum(dy)
  ds = reduceSum(dy * x)
*/
template <typename T, typename U>
__aicore__ inline void GroupNormGradSmallNGCFullLoad<T, U>::VFMode0DbetaDs(
    const LocalTensor<T>& x, const LocalTensor<T>& dy, const LocalTensor<float>& dbeta, const LocalTensor<float>& ds,
    const uint32_t loopIdx, const uint32_t curCNum)
{
    __ubuf__ T* ubX = (__ubuf__ T*)x.GetPhyAddr();
    __ubuf__ T* ubDy = (__ubuf__ T*)dy.GetPhyAddr();
    __ubuf__ float* ubDbeta = (__ubuf__ float*)dbeta.GetPhyAddr();
    __ubuf__ float* ubDs = (__ubuf__ float*)ds.GetPhyAddr();
    uint32_t eleNumPerC = this->eleNumPerC_;
    uint32_t cIdxOffSet = loopIdx * this->mode1UbCapCNum_;
    __ubuf__ T* curUbX;
    __ubuf__ T* curUbDy;

    __VEC_SCOPE__
    {
        UnalignReg uSrcX;
        UnalignReg uSrcDy;
        RegTensor<float> vregDbeta;
        RegTensor<float> vregDs;
        RegTensor<float> tempDbeta;
        RegTensor<float> tempDs;
        RegTensor<float> vregX;
        RegTensor<float> vregDy;
        for (uint16_t idx = 0; idx < static_cast<uint16_t>(curCNum); idx++) {
            MaskReg preg;
            uint32_t sreg = (uint32_t)eleNumPerC;
            uint32_t sregvl = (uint32_t)this->VecLen_;
            uint32_t ubOffSet = idx * eleNumPerC;
            MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
            curUbX = ubX + ubOffSet;
            curUbDy = ubDy + ubOffSet;
            Duplicate(vregDbeta, 0, pregAll);
            Duplicate(vregDs, 0, pregAll);
            DataCopyUnAlignPre(uSrcX, curUbX);
            DataCopyUnAlignPre(uSrcDy, curUbDy);

            preg = UpdateMask<float>(sreg);
            LoadUnAlignOneTensor<T>(curUbX, vregX, uSrcX, preg, sregvl);
            LoadUnAlignOneTensor<T>(curUbDy, vregDy, uSrcDy, preg, sregvl);
            Mul(vregX, vregX, vregDy, preg);
            Add(tempDbeta, vregDbeta, vregDy, preg);
            Add(tempDs, vregDs, vregX, preg);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vregDbeta, tempDbeta, preg);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vregDs, tempDs, preg);

            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            ReduceSum(vregDbeta, vregDbeta, pregAll);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubDbeta + cIdxOffSet + idx, vregDbeta, pregMerge);
            ReduceSum(vregDs, vregDs, pregAll);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubDs + cIdxOffSet + idx, vregDs, pregMerge);
        }
    }
}

/*
  dbeta = reduceSum(dy)
  ds = reduceSum(dy * x)
*/
template <typename T, typename U>
__aicore__ inline void GroupNormGradSmallNGCFullLoad<T, U>::VFMode0DbetaDsBinaryFold(
    const LocalTensor<T>& x, const LocalTensor<T>& dy, const LocalTensor<float>& dbeta, const LocalTensor<float>& ds,
    const uint32_t loopIdx, const uint32_t curCNum)
{
    __ubuf__ T* ubX = (__ubuf__ T*)x.GetPhyAddr();
    __ubuf__ T* ubDy = (__ubuf__ T*)dy.GetPhyAddr();
    __ubuf__ float* ubDbeta = (__ubuf__ float*)dbeta.GetPhyAddr();
    __ubuf__ float* ubDs = (__ubuf__ float*)ds.GetPhyAddr();
    uint32_t binaryQuotientOffset = this->binaryAddQuotient_;
    uint32_t binaryAddRemainder = this->eleNumPerC_ - this->binaryAddQuotient_;
    uint16_t remainderLoop = CeilDiv(binaryAddRemainder, this->VecLen_);
    uint16_t remainderGeneral = remainderLoop == 0 ? 0 : remainderLoop - 1;
    // the binary fold, the head half o the 64-aligned
    uint16_t quotientLoop = CeilDiv(this->binaryAddQuotient_, this->VecLen_);
    uint16_t binaryAddKLoop = this->binaryAddK_;
    uint16_t binaryAddLoop = ((this->binaryAddQuotient_ / this->VecLen_) / this->VecLen_);
    uint32_t binaryAddLastNum = this->binaryAddLastNum_;
    LocalTensor<float> binaryDbetaTensor = this->outQueDx_.template AllocTensor<float>();
    // 复用 outQueDx_ 切分为前后2半用，float16时，大小是float32的一半，折半2次
    uint32_t HxWSpaceOffset = this->eleNumPerC_ / DOUBLE_BUFFER / DOUBLE_BUFFER;
    HxWSpaceOffset = CeilAlign(HxWSpaceOffset, this->PF32_PER_BLOCK);
    LocalTensor<float> binarydsTensor = binaryDbetaTensor[HxWSpaceOffset];
    __ubuf__ T* ubXR = (__ubuf__ T*)ubX + binaryQuotientOffset;
    __ubuf__ T* ubDyR = (__ubuf__ T*)ubDy + binaryQuotientOffset;
    __ubuf__ float* ubBinaryDbeta = (__ubuf__ float*)binaryDbetaTensor.GetPhyAddr();
    __ubuf__ float* ubBinaryDs = (__ubuf__ float*)binarydsTensor.GetPhyAddr();
    uint32_t cgIdxOffSet = loopIdx * this->mode1UbCapCNum_;
    uint32_t eleNumPerC = this->eleNumPerC_;
    uint32_t sregvl = this->VecLen_;
    __ubuf__ T* curUbX;
    __ubuf__ T* curUbXR;
    __ubuf__ T* curUbDy;
    __ubuf__ T* curUbDyR;

    __VEC_SCOPE__
    {
        UnalignReg uSrcX;
        UnalignReg uSrcXR;
        UnalignReg uSrcDy;
        UnalignReg uSrcDyR;
        RegTensor<float> vregDbeta;
        RegTensor<float> vregDs;
        RegTensor<float> vregX;
        RegTensor<float> vregXQ;
        RegTensor<float> vregXR;
        RegTensor<float> vregDy;
        RegTensor<float> vregDyQ;
        RegTensor<float> vregDyR;
        RegTensor<float> tempX;
        RegTensor<float> tempDy;
        for (uint16_t cgIdx = 0; cgIdx < static_cast<uint16_t>(curCNum); cgIdx++) {
            uint32_t sreg0 = binaryAddRemainder;
            uint32_t ubOffSet = cgIdx * eleNumPerC;
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            curUbX = ubX + ubOffSet;
            curUbXR = ubXR + ubOffSet;
            curUbDy = ubDy + ubOffSet;
            curUbDyR = ubDyR + ubOffSet;
            Duplicate(vregDbeta, 0, pregMain);
            Duplicate(vregDs, 0, pregMain);
            DataCopyUnAlignPre(uSrcX, curUbX);
            DataCopyUnAlignPre(uSrcXR, curUbXR);
            DataCopyUnAlignPre(uSrcDy, curUbDy);
            DataCopyUnAlignPre(uSrcDyR, curUbDyR);
            for (uint16_t i = 0; i < remainderGeneral; i++) {
                MaskReg pregLoop = UpdateMask<float>(sreg0);
                LoadUnAlignOneTensor<T>(curUbX, vregXQ, uSrcX, pregMain, sregvl);
                LoadUnAlignOneTensor<T>(curUbXR, vregXR, uSrcXR, pregLoop, sregvl);
                LoadUnAlignOneTensor<T>(curUbDy, vregDyQ, uSrcDy, pregMain, sregvl);
                LoadUnAlignOneTensor<T>(curUbDyR, vregDyR, uSrcDyR, pregLoop, sregvl);
                Mul(vregXQ, vregXQ, vregDyQ, pregMain);
                Mul(vregXR, vregXR, vregDyR, pregLoop);
                // add Quotient add remainder
                Add(vregXQ, vregXQ, vregXR, pregLoop);
                Add(vregDyQ, vregDyQ, vregDyR, pregLoop);
                ReduceSum(vregDs, vregXQ, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubBinaryDs + i, vregDs, pregMerge);
                ReduceSum(vregDbeta, vregDyQ, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubBinaryDbeta + i, vregDbeta, pregMerge);
            }
            // step2: the tail (last 64 or less than 64) blocks reduce to 1.
            {
                MaskReg pregLoop = UpdateMask<float>(sreg0);
                DataCopyUnAlignPre(uSrcX, curUbX);
                DataCopyUnAlignPre(uSrcXR, curUbXR);
                DataCopyUnAlignPre(uSrcDy, curUbDy);
                DataCopyUnAlignPre(uSrcDyR, curUbDyR);
                LoadUnAlignOneTensor<T>(curUbX, vregXQ, uSrcX, pregMain, sregvl);
                LoadUnAlignOneTensor<T>(curUbXR, vregXR, uSrcXR, pregLoop, sregvl);
                LoadUnAlignOneTensor<T>(curUbDy, vregDyQ, uSrcDy, pregMain, sregvl);
                LoadUnAlignOneTensor<T>(curUbDyR, vregDyR, uSrcDyR, pregLoop, sregvl);
                Mul(vregXQ, vregXQ, vregDyQ, pregMain);
                Mul(vregXR, vregXR, vregDyR, pregLoop);
                // add Quotient add remainder
                Add(tempX, vregXQ, vregXR, pregLoop);
                Add(tempDy, vregDyQ, vregDyR, pregLoop);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vregXQ, tempX, pregLoop);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vregDyQ, tempDy, pregLoop);
                ReduceSum(vregDs, vregXQ, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubBinaryDs + remainderGeneral, vregDs, pregMerge);
                ReduceSum(vregDbeta, vregDyQ, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ubBinaryDbeta + remainderGeneral, vregDbeta, pregMerge);
            }
            // step3: non-overlapping portions of the first half reduce by 64, this part always 64 align
            DataCopyUnAlignPre(uSrcX, curUbX);
            DataCopyUnAlignPre(uSrcDy, curUbDy);
            for (uint16_t i = 0; i < static_cast<uint16_t>(quotientLoop - remainderLoop); i++) {
                LoadUnAlignOneTensor<T>(curUbX, vregX, uSrcX, pregMain, sregvl);
                LoadUnAlignOneTensor<T>(curUbDy, vregDy, uSrcDy, pregMain, sregvl);
                Mul(vregX, vregX, vregDy, pregMain);
                ReduceSum(vregDs, vregX, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubBinaryDs + remainderLoop + i, vregDs, pregMerge);
                ReduceSum(vregDbeta, vregDy, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ubBinaryDbeta + remainderLoop + i, vregDbeta, pregMerge);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            // step4: binary folding reduce calculation
            pregMain = CreateMask<float, MaskPattern::ALL>();
            uint16_t curBinaryAddLoop = binaryAddLoop;
            for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                curBinaryAddLoop = curBinaryAddLoop / 2;
                for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
                    DataCopy(vregXQ, ((__ubuf__ float*)ubBinaryDs + j * sregvl));
                    DataCopy(vregXR, ((__ubuf__ float*)ubBinaryDs + (j + curBinaryAddLoop) * sregvl));
                    Add(vregXQ, vregXQ, vregXR, pregMain);
                    DataCopy(ubBinaryDs + j * sregvl, vregXQ, pregMain);
                    DataCopy(vregDyQ, ((__ubuf__ float*)ubBinaryDbeta + j * sregvl));
                    DataCopy(vregDyR, ((__ubuf__ float*)ubBinaryDbeta + (j + curBinaryAddLoop) * sregvl));
                    Add(vregDyQ, vregDyQ, vregDyR, pregMain);
                    DataCopy(ubBinaryDbeta + j * sregvl, vregDyQ, pregMain);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            }
            // step5: vcadd reduce to 1
            {
                uint32_t sreg2 = binaryAddLastNum;
                MaskReg pregLoop = UpdateMask<float>(sreg2);
                DataCopy(vregDs, ((__ubuf__ float*)ubBinaryDs));
                ReduceSum(vregDs, vregDs, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubDs + cgIdxOffSet + cgIdx, vregDs, pregMerge);
                DataCopy(vregDbeta, ((__ubuf__ float*)ubBinaryDbeta));
                ReduceSum(vregDbeta, vregDbeta, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubDbeta + cgIdxOffSet + cgIdx, vregDbeta, pregMerge);
            }
        }
    } // end VF
    this->outQueDx_.FreeTensor(binaryDbetaTensor);
}

/*
xTensor = xTensor * mulScalar
xTensor = xTensor + addScalar
dyTensor = dyTensor * rstd * gamma
dyTensor = dyTensor - xTensor
*/
template <typename T, typename U>
__aicore__ inline void GroupNormGradSmallNGCFullLoad<T, U>::VFComputeStage1Dx(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& xTensor, const LocalTensor<T>& dyTensor,
    const LocalTensor<float>& gammaTensor, const float C2, const float C3, const uint32_t loopIdx,
    const uint32_t curCNum)
{
    __ubuf__ T* ubX = (__ubuf__ T*)xTensor.GetPhyAddr();
    __ubuf__ T* ubDy = (__ubuf__ T*)dyTensor.GetPhyAddr();
    __ubuf__ T* ubDst = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float* ubGamma = (__ubuf__ float*)gammaTensor.GetPhyAddr();
    uint32_t eleNumPerC = this->eleNumPerC_;
    float rstdScalar = this->rstdScalar_;
    uint32_t sregvl = (uint32_t)this->VecLen_;
    uint16_t repeatTimes = CeilDiv(eleNumPerC, sregvl);
    uint16_t loopCnt = eleNumPerC / sregvl;
    uint32_t tailNum = eleNumPerC - (uint32_t)loopCnt * sregvl;
    __ubuf__ T* curUbDst;
    __ubuf__ T* curUbX;
    __ubuf__ T* curUbDy;

    __VEC_SCOPE__
    {
        UnalignReg uSrcX;
        UnalignReg uSrcDy;
        UnalignReg uValue;
        RegTensor<float> vregX;
        RegTensor<float> vregDy;
        RegTensor<float> vregGamma;
        for (uint16_t idx = 0; idx < static_cast<uint16_t>(curCNum); idx++) {
            MaskReg preg;
            uint32_t ubOffSet = idx * eleNumPerC;
            curUbX = ubX + ubOffSet;
            curUbDy = ubDy + ubOffSet;
            curUbDst = ubDst + ubOffSet;
            uint32_t dataLen = loopCnt * sregvl;
            DataCopy<float, LoadDist::DIST_BRC_B32>(vregGamma, ubGamma + loopIdx * this->mode1UbCapCNum_ + idx);
            DataCopyUnAlignPre(uSrcX, curUbX);
            DataCopyUnAlignPre(uSrcDy, curUbDy);
            for (uint16_t i = 0; i < (uint16_t)loopCnt; ++i) {
                preg = UpdateMask<float>(dataLen);
                LoadUnAlignOneTensor<T>(curUbX, vregX, uSrcX, preg, sregvl);
                LoadUnAlignOneTensor<T>(curUbDy, vregDy, uSrcDy, preg, sregvl);
                Muls(vregX, vregX, C2, preg);
                Mul(vregDy, vregDy, vregGamma, preg);
                Muls(vregDy, vregDy, rstdScalar, preg);
                Add(vregX, vregX, vregDy, preg);
                Adds(vregX, vregX, C3, preg);
                StoreUnAlignOneTensor<T>(curUbDst, vregX, uValue, preg, sregvl);
            }
            {
                uint32_t tail = tailNum;
                preg = UpdateMask<float>(tail);
                DataCopyUnAlignPre(uSrcX, curUbX);
                DataCopyUnAlignPre(uSrcDy, curUbDy);
                LoadUnAlignOneTensor<T>(curUbX, vregX, uSrcX, preg, tailNum);
                LoadUnAlignOneTensor<T>(curUbDy, vregDy, uSrcDy, preg, tailNum);
                Muls(vregX, vregX, C2, preg);
                Mul(vregDy, vregDy, vregGamma, preg);
                Muls(vregDy, vregDy, rstdScalar, preg);
                Add(vregX, vregX, vregDy, preg);
                Adds(vregX, vregX, C3, preg);
                StoreUnAlignOneTensor<T>(curUbDst, vregX, uValue, preg, tailNum);
            }
            DataCopyUnAlignPost(curUbDst, uValue, 0);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradSmallNGCFullLoad<T, U>::VFComputeStage1Ds(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& dsTensor, const LocalTensor<float>& dbetaTensor)
{
    __ubuf__ float* ubDb = (__ubuf__ float*)dbetaTensor.GetPhyAddr();
    __ubuf__ float* ubDs = (__ubuf__ float*)dsTensor.GetPhyAddr();
    __ubuf__ float* ubDst = (__ubuf__ float*)dstTensor.GetPhyAddr();
    float meanScalar = this->meanScalar_;
    float rstdScalar = this->rstdScalar_;

    uint32_t calCount = this->C_G_;
    uint16_t repeatTimes = CeilDiv(calCount, this->VecLen_);

    __VEC_SCOPE__
    {
        RegTensor<float> vregDb;
        RegTensor<float> vregDs;
        MaskReg preg;
        uint32_t sreg = (uint32_t)calCount;
        uint32_t sregvl = (uint32_t)this->VecLen_;
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = UpdateMask<float>(sreg);
            DataCopy(vregDs, ubDs + i * sregvl);
            DataCopy(vregDb, ubDb + i * sregvl);
            Muls(vregDb, vregDb, meanScalar, preg);
            Sub(vregDs, vregDs, vregDb, preg);
            Muls(vregDs, vregDs, rstdScalar, preg);
            DataCopy(ubDst + i * sregvl, vregDs, preg);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradSmallNGCFullLoad<T, U>::ComputeStage2()
{
    // wait core
    SyncAll();
    this->pipe->Reset();
    if (this->curBlockIdx_ < this->stage2CoreUsed_) {
        // stage2mode must be 1
        this->InitStage2Mode1Buffer();
        uint32_t stage2CurCoreCNum =
            (this->curBlockIdx_ == (this->stage2CoreUsed_ - 1)) ? this->cTailBlockFactor_ : this->cBlockFactor_;
        uint32_t quotient = (stage2CurCoreCNum + this->cFactor_ - 1) / this->cFactor_;
        for (uint32_t ubLoopIdx = 0; ubLoopIdx < quotient; ubLoopIdx++) {
            uint32_t cOffset = ubLoopIdx * this->cFactor_ + this->curBlockIdx_ * this->cBlockFactor_;
            uint32_t currentC =
                (ubLoopIdx == (quotient - 1)) ? (stage2CurCoreCNum - (quotient - 1) * this->cFactor_) : this->cFactor_;
            this->stage2Mode1Process(cOffset, currentC);
        }
    }
}
} // namespace GroupNormGrad
#endif