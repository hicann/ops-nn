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
 * \file group_norm_grad_c_full_load.h
 * \brief
 */
#ifndef GROUP_NORM_GRAD_C_FULL_LOAD_H
#define GROUP_NORM_GRAD_C_FULL_LOAD_H

#pragma once

#include "kernel_operator.h"
#include "group_norm_grad_base.h"
#include "group_norm_grad_common.h"

namespace GroupNormGrad {
template <typename T, typename U>
class GroupNormGradCFullLoad : public GroupNormGradBase<T, U> {
public:
    __aicore__ inline GroupNormGradCFullLoad() : GroupNormGradBase<T, U>(){};
    __aicore__ inline ~GroupNormGradCFullLoad(){};
    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta,
        GM_ADDR workspace, const GroupNormGradRegBaseTilingData* __restrict tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitBuffer(const GroupNormGradRegBaseTilingData* tilingData);
    __aicore__ inline void Compute(int32_t taskIdx);
    __aicore__ inline void VFMode1DbetaDs(
        const LocalTensor<T>& x, const LocalTensor<T>& dy, const LocalTensor<float>& dbeta,
        const LocalTensor<float>& dgamma, const uint32_t loopIdx, const uint32_t curCNum);
    __aicore__ inline void VFMode1DbetaDsBinaryFold(
        const LocalTensor<T>& x, const LocalTensor<T>& dy, const LocalTensor<float>& dbeta,
        const LocalTensor<float>& dgamma, const uint32_t loopIdx, const uint32_t curCNum);
    __aicore__ inline void ComputeMode1Dx(
        int32_t taskIdx, LocalTensor<float>& dbetaTensor, LocalTensor<float>& dsTensor);
    __aicore__ inline void VFComputeMode1Dx(
        const LocalTensor<T>& dstTensor, const LocalTensor<T>& xTensor, const LocalTensor<T>& dyTensor,
        const LocalTensor<float>& gammaTensor, const float C2, const float C3, const uint32_t loopIdx,
        const uint32_t curCNum);
};

template <typename T, typename U>
__aicore__ inline void GroupNormGradCFullLoad<T, U>::Init(
    GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta,
    GM_ADDR workspace, const GroupNormGradRegBaseTilingData* __restrict tilingData, TPipe* pipeIn)
{
    this->pipe = pipeIn;
    this->InitCommon(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, workspace, tilingData);
    this->InitOutPutOrWorkSpace();
    InitBuffer(tilingData);
    // wait core
    SyncAll();
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradCFullLoad<T, U>::Process()
{
    // tiling strategy, pipeline parallel
    if (this->curBlockIdx_ < this->stage1CoreUsed_) {
        for (int32_t taskIdx = 0; taskIdx < this->curCoreTaskNum_; taskIdx++) {
            this->LoadMeanRstd(taskIdx + this->startTaskId_);
            Compute(taskIdx + this->startTaskId_);
        }
    }
    this->Stage2Process();
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradCFullLoad<T, U>::InitBuffer(const GroupNormGradRegBaseTilingData* tilingData)
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
    // VEC_OUT
    this->pipe->InitBuffer(this->outQueDx_, DOUBLE_BUFFER, capElemAllocSpace);
    this->pipe->InitBuffer(this->outQueDs_, 1, cGAllocSpace);
    this->pipe->InitBuffer(this->outQueDgamma_, 1, cGAllocSpace);
    this->pipe->InitBuffer(this->outQueDbeta_, 1, cGAllocSpace);
    if constexpr (IsSameType<U, half>::value || IsSameType<U, bfloat16_t>::value) {
        uint32_t cGAllocSpaceNum = CeilAlign(this->C_G_, this->elemUPerBlock_);
        this->pipe->InitBuffer(this->tBufGamma_, cGAllocSpaceNum * sizeof(U));
        this->pipe->InitBuffer(this->tBufDbeta_, cGAllocSpaceNum * sizeof(U));
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradCFullLoad<T, U>::Compute(int32_t taskIdx)
{
    LocalTensor<float> dbetaTensor = this->outQueDbeta_.template AllocTensor<float>();
    LocalTensor<float> dsTensor = this->outQueDs_.template AllocTensor<float>();
    uint32_t baseOffset = taskIdx * this->eleNumPerG_;
    uint32_t ubLoopCnt = (this->C_G_ + this->mode1UbCapCNum_ - 1) / this->mode1UbCapCNum_;
    LocalTensor<T> dyTensor;
    LocalTensor<T> xTensor;
    for (uint32_t loopIdx = 0; loopIdx < ubLoopCnt; loopIdx++) {
        int64_t offset = baseOffset + loopIdx * this->eleNumPerC_ * this->mode1UbCapCNum_;
        auto curCNum = this->mode1UbCapCNum_;
        if ((loopIdx == ubLoopCnt - 1) && (this->C_G_ % this->mode1UbCapCNum_ != 0)) {
            curCNum = this->C_G_ % this->mode1UbCapCNum_;
        }
        uint32_t curEleNum = CeilAlign(static_cast<uint32_t>(this->eleNumPerC_ * curCNum), this->elemTPerBlock_);
        xTensor = this->inQueX_.template AllocTensor<T>();
        dyTensor = this->inQueDy_.template AllocTensor<T>();
        DataCopy(xTensor, this->xGm_[offset], curEleNum);
        DataCopy(dyTensor, this->dyGm_[offset], curEleNum);
        this->inQueX_.EnQue(xTensor);
        this->inQueDy_.EnQue(dyTensor);
        dyTensor = this->inQueDy_.template DeQue<T>();
        xTensor = this->inQueX_.template DeQue<T>();
        if (this->eleNumPerC_ <= this->VecLen_) {
            VFMode1DbetaDs(xTensor, dyTensor, dbetaTensor, dsTensor, loopIdx, curCNum);
        } else {
            VFMode1DbetaDsBinaryFold(xTensor, dyTensor, dbetaTensor, dsTensor, loopIdx, curCNum);
        }
        this->inQueX_.FreeTensor(xTensor);
        this->inQueDy_.FreeTensor(dyTensor);
    }
    this->outQueDbeta_.EnQue(dbetaTensor);
    this->outQueDs_.EnQue(dsTensor);
    dsTensor = this->outQueDs_.template DeQue<float>();
    dbetaTensor = this->outQueDbeta_.template DeQue<float>();
    this->StoreDgammaDbeta(taskIdx, dsTensor, dbetaTensor, this->meanScalar_, this->rstdScalar_);
    if (this->dxIsRequire_) {
        ComputeMode1Dx(taskIdx, dbetaTensor, dsTensor);
    } else {
        this->outQueDbeta_.FreeTensor(dbetaTensor);
        this->outQueDs_.FreeTensor(dsTensor);
    }
}

/*
  dbeta = reduceSum(dy)
  xHat = x * rstd - mean * rstd
  dgamma = reduceSum(dy * xHat)
*/
template <typename T, typename U>
__aicore__ inline void GroupNormGradCFullLoad<T, U>::VFMode1DbetaDs(
    const LocalTensor<T>& x, const LocalTensor<T>& dy, const LocalTensor<float>& dbeta,
    const LocalTensor<float>& dgamma, const uint32_t loopIdx, const uint32_t curCNum)
{
    __ubuf__ T* ubX = (__ubuf__ T*)x.GetPhyAddr();
    __ubuf__ T* ubDy = (__ubuf__ T*)dy.GetPhyAddr();
    __ubuf__ float* ubDbeta = (__ubuf__ float*)dbeta.GetPhyAddr();
    __ubuf__ float* ubDgamma = (__ubuf__ float*)dgamma.GetPhyAddr();
    uint32_t eleNumPerC = this->eleNumPerC_;
    uint16_t repeatTimes = CeilDiv(eleNumPerC, this->VecLen_);
    uint32_t cgIdxOffSet = loopIdx * this->mode1UbCapCNum_;
    __ubuf__ T* curUbX;
    __ubuf__ T* curUbDy;

    __VEC_SCOPE__
    {
        UnalignReg uSrcX;
        UnalignReg uSrcDy;
        RegTensor<float> vregDbeta;
        RegTensor<float> vregDgamma;
        RegTensor<float> tempDbeta;
        RegTensor<float> tempDgamma;
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
            Duplicate(vregDgamma, 0, pregAll);
            DataCopyUnAlignPre(uSrcX, curUbX);
            DataCopyUnAlignPre(uSrcDy, curUbDy);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                preg = UpdateMask<float>(sreg);
                LoadUnAlignOneTensor<T>(curUbX, vregX, uSrcX, preg, sregvl);
                LoadUnAlignOneTensor<T>(curUbDy, vregDy, uSrcDy, preg, sregvl);
                Mul(vregX, vregX, vregDy, preg);
                Add(tempDbeta, vregDbeta, vregDy, preg);
                Add(tempDgamma, vregDgamma, vregX, preg);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vregDbeta, tempDbeta, preg);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vregDgamma, tempDgamma, preg);
            }
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            ReduceSum(vregDbeta, vregDbeta, pregAll);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubDbeta + cgIdxOffSet + idx, vregDbeta, pregMerge);
            ReduceSum(vregDgamma, vregDgamma, pregAll);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubDgamma + cgIdxOffSet + idx, vregDgamma, pregMerge);
        }
    }
}

/*
  dbeta = reduceSum(dy)
  temp = xHat * rstd - mean * rstd
  dgamma = reduceSum(dy * temp)
*/
template <typename T, typename U>
__aicore__ inline void GroupNormGradCFullLoad<T, U>::VFMode1DbetaDsBinaryFold(
    const LocalTensor<T>& x, const LocalTensor<T>& dy, const LocalTensor<float>& dbeta,
    const LocalTensor<float>& dgamma, const uint32_t loopIdx, const uint32_t curCNum)
{
    __ubuf__ T* ubX = (__ubuf__ T*)x.GetPhyAddr();
    __ubuf__ T* ubDy = (__ubuf__ T*)dy.GetPhyAddr();
    __ubuf__ float* ubDbeta = (__ubuf__ float*)dbeta.GetPhyAddr();
    __ubuf__ float* ubDgamma = (__ubuf__ float*)dgamma.GetPhyAddr();
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
    __ubuf__ float* ubBinaryDgamma = (__ubuf__ float*)binarydsTensor.GetPhyAddr();
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
        RegTensor<float> vregDgamma;
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
            Duplicate(vregDgamma, 0, pregMain);
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
                ReduceSum(vregDgamma, vregXQ, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubBinaryDgamma + i, vregDgamma, pregMerge);
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
                ReduceSum(vregDgamma, vregXQ, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ubBinaryDgamma + remainderGeneral, vregDgamma, pregMerge);
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
                ReduceSum(vregDgamma, vregX, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ubBinaryDgamma + remainderLoop + i, vregDgamma, pregMerge);
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
                    DataCopy(vregXQ, ((__ubuf__ float*)ubBinaryDgamma + j * sregvl));
                    DataCopy(vregXR, ((__ubuf__ float*)ubBinaryDgamma + (j + curBinaryAddLoop) * sregvl));
                    Add(vregXQ, vregXQ, vregXR, pregMain);
                    DataCopy(ubBinaryDgamma + j * sregvl, vregXQ, pregMain);
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
                DataCopy(vregDgamma, ((__ubuf__ float*)ubBinaryDgamma));
                ReduceSum(vregDgamma, vregDgamma, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ubDgamma + cgIdxOffSet + cgIdx, vregDgamma, pregMerge);
                DataCopy(vregDbeta, ((__ubuf__ float*)ubBinaryDbeta));
                ReduceSum(vregDbeta, vregDbeta, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubDbeta + cgIdxOffSet + cgIdx, vregDbeta, pregMerge);
            }
        }
    } // end VF
    this->outQueDx_.FreeTensor(binaryDbetaTensor);
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradCFullLoad<T, U>::ComputeMode1Dx(
    int32_t taskIdx, LocalTensor<float>& dbetaTensor, LocalTensor<float>& dsTensor)
{
    float sum1 = 0;
    float sum2 = 0;
    uint32_t channelIdx = (taskIdx % this->G_) * this->C_G_;
    uint32_t baseOffset = taskIdx * this->eleNumPerG_;
    uint32_t ubCapEleNum = CeilAlign(static_cast<uint32_t>(this->eleNumPerC_), this->elemTPerBlock_);
    this->LoadDataToUb(
        this->inQueGamma_, this->tBufGamma_, this->gammaGm_, channelIdx,
        CeilAlign(this->C_G_, this->elemUPerBlock_));
    LocalTensor<float> gammaTensor = this->inQueGamma_.template DeQue<float>();
    this->ComputeSum1Sum2(dbetaTensor, dsTensor, gammaTensor, sum1, sum2);
    float s = 1.0f / this->eleNumPerG_;
    float C2 = (sum2 * this->meanScalar_ - sum1) * this->rstdScalar_ * this->rstdScalar_ * this->rstdScalar_ * s;
    float C3 = -C2 * this->meanScalar_ - sum2 * this->rstdScalar_ * s;
    this->outQueDbeta_.FreeTensor(dbetaTensor);
    this->outQueDs_.FreeTensor(dsTensor);
    LocalTensor<T> xTensor;
    LocalTensor<T> dyTensor;
    uint32_t ubLoopCnt = (this->C_G_ + this->mode1UbCapCNum_ - 1) / this->mode1UbCapCNum_;
    for (uint32_t loopIdx = 0; loopIdx < ubLoopCnt; loopIdx++) {
        uint32_t offset = baseOffset + loopIdx * this->mode1UbCapCNum_ * this->eleNumPerC_;
        auto curCNum = this->mode1UbCapCNum_;
        if ((loopIdx == ubLoopCnt - 1) && (this->C_G_ % this->mode1UbCapCNum_ != 0)) {
            curCNum = this->C_G_ % this->mode1UbCapCNum_;
        }
        xTensor = this->inQueX_.template AllocTensor<T>();
        dyTensor = this->inQueDy_.template AllocTensor<T>();
        DataCopy(
            xTensor, this->xGm_[offset],
            CeilAlign(static_cast<uint32_t>(this->eleNumPerC_ * curCNum), this->elemTPerBlock_));
        DataCopy(
            dyTensor, this->dyGm_[offset],
            CeilAlign(static_cast<uint32_t>(this->eleNumPerC_ * curCNum), this->elemTPerBlock_));
        this->inQueX_.EnQue(xTensor);
        this->inQueDy_.EnQue(dyTensor);
        xTensor = this->inQueX_.template DeQue<T>();
        dyTensor = this->inQueDy_.template DeQue<T>();
        LocalTensor<T> dxTensor = this->outQueDx_.template AllocTensor<T>();
        VFComputeMode1Dx(dxTensor, xTensor, dyTensor, gammaTensor, C2, C3, loopIdx, curCNum);
        this->inQueX_.FreeTensor(xTensor);
        this->inQueDy_.FreeTensor(dyTensor);
        this->outQueDx_.EnQue(dxTensor);
        this->StoreDxToGm(this->outQueDx_, offset, this->eleNumPerC_ * curCNum);
    }
    this->inQueGamma_.FreeTensor(gammaTensor);
}

/*
xTensor = xTensor * mulScalar
xTensor = xTensor + addScalar
dyTensor = dyTensor * rstd * gamma
dyTensor = dyTensor - xTensor
*/
template <typename T, typename U>
__aicore__ inline void GroupNormGradCFullLoad<T, U>::VFComputeMode1Dx(
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
} // namespace GroupNormGrad
#endif
