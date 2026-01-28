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
 * \file group_norm_grad_recompute.h
 * \brief
 */
#ifndef GROUP_NORM_GRAD_RECOMPUTE_H
#define GROUP_NORM_GRAD_RECOMPUTE_H
#pragma once

#include "kernel_operator.h"
#include "group_norm_grad_base.h"
#include "group_norm_grad_common.h"

namespace GroupNormGrad {
struct ComputeDxParams {
    float C2;
    float C3;
    float gamma;
    uint32_t processNum;
};

template <typename T, typename U>
class GroupNormGradReCompute : public GroupNormGradBase<T, U> {
public:
    __aicore__ inline GroupNormGradReCompute() : GroupNormGradBase<T, U>(){};
    __aicore__ inline ~GroupNormGradReCompute(){};
    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta,
        GM_ADDR workspace, const GroupNormGradRegBaseTilingData* __restrict tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitBuffer(const GroupNormGradRegBaseTilingData* tilingData);
    __aicore__ inline void Compute(int32_t taskIdx);
    __aicore__ inline void VFMode2DbetaDs(
        const LocalTensor<T>& xMain, const LocalTensor<T>& xFold, const LocalTensor<T>& dyMain,
        const LocalTensor<T>& dyFold, const uint32_t mainReduceNum, const uint32_t foldReduceNum,
        const uint32_t loopIdx, const LocalTensor<float>& cacheDbeta, const LocalTensor<float>& cacheDgama);
    __aicore__ inline void Mode2DbetaDs(
        const LocalTensor<float>& dbeta, const LocalTensor<float>& dgamma, const uint32_t C_G_idx,
        const int64_t offset);
    __aicore__ inline void ComputeMode2Dx(
        int32_t taskIdx, LocalTensor<float>& dbetaTensor, LocalTensor<float>& dsTensor);
    __aicore__ inline void VFComputeMode2Dx(
        const LocalTensor<T>& dstTensor, const LocalTensor<T>& xTensor, const LocalTensor<T>& dyTensor,
        const ComputeDxParams& dxParams);
};

template <typename T, typename U>
__aicore__ inline void GroupNormGradReCompute<T, U>::Init(
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
__aicore__ inline void GroupNormGradReCompute<T, U>::Process()
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
__aicore__ inline void GroupNormGradReCompute<T, U>::InitBuffer(const GroupNormGradRegBaseTilingData* tilingData)
{
    // for cast mean, rstd, for store sum1, sum2
    this->pipe->InitBuffer(this->tempMeanBuf_, this->BLOCK_BYTES);
    this->pipe->InitBuffer(this->tempRstdBuf_, this->BLOCK_BYTES);
    uint32_t capElemSpace = CeilAlign(this->mode2UbCapEle_, this->elemTPerBlock_) * sizeof(T);
    uint32_t cGAllocSpace = CeilAlign(this->C_G_, this->elemUPerBlock_) * sizeof(float);
    this->pipe->InitBuffer(this->cacheDbetaBuf_, this->MAX_CACHE_DEPTH * this->CACHE_BUFF_SIZE * sizeof(float));
    this->pipe->InitBuffer(this->cacheDgammaBuf_, this->MAX_CACHE_DEPTH * this->CACHE_BUFF_SIZE * sizeof(float));
    // VEC_IN
    this->pipe->InitBuffer(this->inQueDy_, DOUBLE_BUFFER, capElemSpace);
    this->pipe->InitBuffer(this->inQueX_, DOUBLE_BUFFER, capElemSpace);
    this->pipe->InitBuffer(this->inQueGamma_, 1, cGAllocSpace);
    // VEC_OUT
    this->pipe->InitBuffer(this->outQueDx_, DOUBLE_BUFFER, capElemSpace);
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
__aicore__ inline void GroupNormGradReCompute<T, U>::Compute(int32_t taskIdx)
{
    // deque input tensors from VECIN queue
    LocalTensor<float> dbetaTensor = this->outQueDbeta_.template AllocTensor<float>();
    LocalTensor<float> dsTensor = this->outQueDs_.template AllocTensor<float>();
    int64_t baseOffset = taskIdx * this->eleNumPerG_;
    for (uint32_t C_G_idx = 0; C_G_idx < this->C_G_; C_G_idx++) {
        int64_t offset = baseOffset + C_G_idx * this->eleNumPerC_;
        Mode2DbetaDs(dbetaTensor, dsTensor, C_G_idx, offset);
    }
    this->outQueDbeta_.EnQue(dbetaTensor);
    this->outQueDs_.EnQue(dsTensor);
    dsTensor = this->outQueDs_.template DeQue<float>();
    dbetaTensor = this->outQueDbeta_.template DeQue<float>();
    this->StoreDgammaDbeta(taskIdx, dsTensor, dbetaTensor, this->meanScalar_, this->rstdScalar_);
    if (this->dxIsRequire_) {
        ComputeMode2Dx(taskIdx, dbetaTensor, dsTensor);
    } else {
        this->outQueDbeta_.FreeTensor(dbetaTensor);
        this->outQueDs_.FreeTensor(dsTensor);
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradReCompute<T, U>::VFMode2DbetaDs(
    const LocalTensor<T>& xMain, const LocalTensor<T>& xFold, const LocalTensor<T>& dyMain,
    const LocalTensor<T>& dyFold, const uint32_t mainReduceNum, const uint32_t foldReduceNum, const uint32_t loopIdx,
    const LocalTensor<float>& cacheDbeta, const LocalTensor<float>& cacheDgama)
{
    __ubuf__ T* ubXMain = (__ubuf__ T*)xMain.GetPhyAddr();
    __ubuf__ T* ubXFold = (__ubuf__ T*)xFold.GetPhyAddr();
    __ubuf__ T* ubDyMain = (__ubuf__ T*)dyMain.GetPhyAddr();
    __ubuf__ T* ubDyFold = (__ubuf__ T*)dyFold.GetPhyAddr();
    __ubuf__ float* ubDbeta = (__ubuf__ float*)cacheDbeta.GetPhyAddr();
    __ubuf__ float* ubDgamma = (__ubuf__ float*)cacheDgama.GetPhyAddr();
    LocalTensor<float> binaryDbetaTensor = this->outQueDx_.template AllocTensor<float>();
    // 复用 outQueDx_ 切分为前后2半用，float16时，大小是float32的一半，折半2次
    uint32_t backHalfOffset = this->mode2OneLoopSize_ / DOUBLE_BUFFER / DOUBLE_BUFFER;
    backHalfOffset = CeilAlign(backHalfOffset, this->PF32_PER_BLOCK);
    LocalTensor<float> binarydsTensor = binaryDbetaTensor[backHalfOffset];
    __ubuf__ float* ubBinaryDbeta = (__ubuf__ float*)binaryDbetaTensor.GetPhyAddr();
    __ubuf__ float* ubBinaryDgamma = (__ubuf__ float*)binarydsTensor.GetPhyAddr();
    uint32_t overLapLoopTimes = 0;
    uint32_t remainerLoopTimes = 0;
    uint32_t unFoldLoopTimes = 0;
    uint16_t binaryAddKLoop = 0;
    uint16_t binaryAddLoop = 0;
    uint32_t binaryUbLastNum = 0;
    uint32_t foldAddNum = 0;
    uint32_t binaryAddRemainder = 0;
    uint32_t unFoldAddNum = 0;
    uint32_t sregvl = this->VecLen_;

    constexpr uint16_t ulongBitLenOfClzInstr = 64;

    if (loopIdx < this->mode2FoldLoopCnt_ - 1) {
        foldAddNum = mainReduceNum;
        overLapLoopTimes = mainReduceNum / this->VecLen_;
        binaryAddLoop = ((mainReduceNum / this->VecLen_) / this->VecLen_);
        binaryAddKLoop = binaryAddLoop == 0 ? 0 : ulongBitLenOfClzInstr - ScalarCountLeadingZero(binaryAddLoop - 1);
        binaryUbLastNum = overLapLoopTimes <= this->VecLen_ ? overLapLoopTimes : this->VecLen_;
    } else if (loopIdx == this->mode2FoldLoopCnt_ - 1) {
        foldAddNum = foldReduceNum;
        unFoldAddNum = (mainReduceNum - foldReduceNum) / this->VecLen_ * this->VecLen_;
        overLapLoopTimes = foldReduceNum / this->VecLen_;
        remainerLoopTimes = foldReduceNum % this->VecLen_ == 0 ? 0 : 1;
        unFoldLoopTimes = unFoldAddNum / this->VecLen_;
        uint32_t foldLoopTimes = overLapLoopTimes + remainerLoopTimes + unFoldLoopTimes;
        binaryAddLoop = ((mainReduceNum / this->VecLen_) / this->VecLen_);
        binaryAddKLoop = binaryAddLoop == 0 ? 0 : ulongBitLenOfClzInstr - ScalarCountLeadingZero(binaryAddLoop - 1);
        binaryUbLastNum = foldLoopTimes <= this->VecLen_ ? foldLoopTimes : this->VecLen_;
    } else {
        unFoldAddNum = Aligned(mainReduceNum, this->VecLen_);
        unFoldLoopTimes = mainReduceNum / this->VecLen_;
        binaryAddLoop = ((mainReduceNum / this->VecLen_) / this->VecLen_);
        binaryAddKLoop = binaryAddLoop == 0 ? 0 : ulongBitLenOfClzInstr - ScalarCountLeadingZero(binaryAddLoop - 1);
        binaryUbLastNum = unFoldLoopTimes <= this->VecLen_ ? unFoldLoopTimes : this->VecLen_;
    }

    __VEC_SCOPE__
    {
        RegTensor<float> vregDbeta;
        RegTensor<float> vregDgamma;
        RegTensor<float> vregX;
        RegTensor<float> vregXM;
        RegTensor<float> vregXF;
        RegTensor<float> vregDy;
        RegTensor<float> vregDyM;
        RegTensor<float> vregDyF;
        RegTensor<float> tempX;
        RegTensor<float> tempDy;
        uint32_t sreg0 = foldAddNum;
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        // step1: reduce main and fold overlapping block by 64
        for (uint16_t i = 0; i < static_cast<uint16_t>(overLapLoopTimes); i++) {
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            LoadTwoTensorForDtypeT<T>(ubXMain, ubXFold, vregXM, vregXF, pregMain, pregLoop, i * sregvl, i * sregvl);
            LoadTwoTensorForDtypeT<T>(ubDyMain, ubDyFold, vregDyM, vregDyF, pregMain, pregLoop, i * sregvl, i * sregvl);
            Mul(vregXM, vregXM, vregDyM, pregMain);
            Mul(vregXF, vregXF, vregDyF, pregLoop);
            // add Quotient add remainder
            Add(vregXM, vregXM, vregXF, pregMain);
            Add(vregDyM, vregDyM, vregDyF, pregMain);
            ReduceSum(vregDgamma, vregXM, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubBinaryDgamma + i, vregDgamma, pregMerge);
            ReduceSum(vregDbeta, vregDyM, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubBinaryDbeta + i, vregDbeta, pregMerge);
        }
        // step2:  reduce the fold  tail (last 64 or less than 64) blocks reduce to 1.
        for (uint16_t i = 0; i < static_cast<uint16_t>(remainerLoopTimes); i++) {
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            LoadTwoTensorForDtypeT<T>(
                ubXMain, ubXFold, vregXM, vregXF, pregMain, pregLoop, overLapLoopTimes * sregvl,
                overLapLoopTimes * sregvl);
            LoadTwoTensorForDtypeT<T>(
                ubDyMain, ubDyFold, vregDyM, vregDyF, pregMain, pregLoop, overLapLoopTimes * sregvl,
                overLapLoopTimes * sregvl);
            Mul(vregXM, vregXM, vregDyM, pregMain);
            Mul(vregXF, vregXF, vregDyF, pregLoop);
            // add Quotient add remainder
            Add(tempX, vregXM, vregXF, pregLoop);
            Add(tempDy, vregDyM, vregDyF, pregLoop);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vregXM, tempX, pregLoop);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vregDyM, tempDy, pregLoop);
            ReduceSum(vregDgamma, vregXM, pregMain);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                ubBinaryDgamma + overLapLoopTimes, vregDgamma, pregMerge);
            ReduceSum(vregDbeta, vregDyM, pregMain);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubBinaryDbeta + overLapLoopTimes, vregDbeta, pregMerge);
        }
        // step3: non-overlapping portions of the first half reduce by 64, this part always 64 align
        uint32_t sreg2 = unFoldAddNum;
        for (uint16_t i = 0; i < static_cast<uint16_t>(unFoldLoopTimes); i++) {
            MaskReg pregLoop = UpdateMask<float>(sreg2);
            LoadOneTensorForDtypeT<T>(ubXMain, vregXM, pregLoop, (i + overLapLoopTimes + remainerLoopTimes) * sregvl);
            LoadOneTensorForDtypeT<T>(ubDyMain, vregDyM, pregLoop, (i + overLapLoopTimes + remainerLoopTimes) * sregvl);
            Mul(vregXM, vregXM, vregDyM, pregLoop);
            ReduceSum(vregDgamma, vregXM, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                ubBinaryDgamma + i + overLapLoopTimes + remainerLoopTimes, vregDgamma, pregMerge);
            ReduceSum(vregDbeta, vregDyM, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                ubBinaryDbeta + i + overLapLoopTimes + remainerLoopTimes, vregDbeta, pregMerge);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        // step4: binary folding reduce calculation
        pregMain = CreateMask<float, MaskPattern::ALL>();
        uint16_t curBinaryAddLoop = binaryAddLoop;
        for (uint16_t i = 0; i < binaryAddKLoop; i++) {
            curBinaryAddLoop = curBinaryAddLoop / 2;
            for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
                DataCopy(vregXM, ((__ubuf__ float*)ubBinaryDgamma + j * sregvl));
                DataCopy(vregXF, ((__ubuf__ float*)ubBinaryDgamma + (j + curBinaryAddLoop) * sregvl));
                Add(vregXM, vregXM, vregXF, pregMain);
                DataCopy(((__ubuf__ float*)ubBinaryDgamma + j * sregvl), vregXM, pregMain);

                DataCopy(vregDyM, ((__ubuf__ float*)ubBinaryDbeta + j * sregvl));
                DataCopy(vregDyF, ((__ubuf__ float*)ubBinaryDbeta + (j + curBinaryAddLoop) * sregvl));
                Add(vregDyM, vregDyM, vregDyF, pregMain);
                DataCopy(((__ubuf__ float*)ubBinaryDbeta + j * sregvl), vregDyM, pregMain);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        // step5: vcadd reduce to 1
        {
            uint32_t sreg3 = binaryUbLastNum;
            uint32_t pos = loopIdx & 0xFF;
            MaskReg pregLoop = UpdateMask<float>(sreg3);
            DataCopy(vregDgamma, ((__ubuf__ float*)ubBinaryDgamma));
            ReduceSum(vregDgamma, vregDgamma, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubDgamma + pos, vregDgamma, pregMerge);
            DataCopy(vregDbeta, ((__ubuf__ float*)ubBinaryDbeta));
            ReduceSum(vregDbeta, vregDbeta, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubDbeta + pos, vregDbeta, pregMerge);
        }
    } // end VF
    this->outQueDx_.FreeTensor(binaryDbetaTensor);
}

/*
  dbeta = ReduceSum(dy)
  temp = xHat * rstd - mean * rstd
  dgamma = ReduceSum(dy * temp)
*/
template <typename T, typename U>
__aicore__ inline void GroupNormGradReCompute<T, U>::Mode2DbetaDs(
    const LocalTensor<float>& dbeta, const LocalTensor<float>& dgamma, const uint32_t C_G_idx, const int64_t offset)
{
    LocalTensor<T> xMainTensor, xFoldTensor;
    LocalTensor<T> dyMainTensor, dyFoldTensor;
    LocalTensor<float> cacheDbeta = this->cacheDbetaBuf_.template Get<float>();
    LocalTensor<float> cacheDgama = this->cacheDgammaBuf_.template Get<float>();
    int64_t loopIdx = 0;
    Duplicate<float>(cacheDbeta, 0.0f, this->CACHE_BUFF_SIZE * this->MAX_CACHE_DEPTH);
    Duplicate<float>(cacheDgama, 0.0f, this->CACHE_BUFF_SIZE * this->MAX_CACHE_DEPTH);
    int64_t baseFoldOffset = offset + this->mode2MainLoopCnt_ * this->mode2OneLoopSize_;
    for (loopIdx = 0; loopIdx < this->mode2MainLoopCnt_; ++loopIdx) {
        int64_t mainOffset = offset + loopIdx * this->mode2OneLoopSize_;
        int64_t foldOffset = baseFoldOffset + loopIdx * this->mode2OneLoopSize_;
        auto mainReduceNum = this->mode2OneLoopSize_;
        auto foldReduceNum = this->mode2OneLoopSize_;
        xMainTensor = this->inQueX_.template AllocTensor<T>();
        dyMainTensor = this->inQueDy_.template AllocTensor<T>();
        DataCopyPadExtParams<T> dataCopyPadExtParamsX;
        dataCopyPadExtParamsX.isPad = false;
        dataCopyPadExtParamsX.leftPadding = 0;
        dataCopyPadExtParamsX.rightPadding = 0;
        dataCopyPadExtParamsX.paddingValue = 0;
        DataCopyExtParams copyInParamsX;
        copyInParamsX.blockCount = 1;
        copyInParamsX.blockLen =  mainReduceNum * sizeof(T);
        copyInParamsX.srcStride = 0;
        copyInParamsX.dstStride = 0;
        DataCopyPad(xMainTensor, this->xGm_[mainOffset], copyInParamsX, dataCopyPadExtParamsX);
        DataCopyPad(dyMainTensor, this->dyGm_[mainOffset], copyInParamsX, dataCopyPadExtParamsX);
        xFoldTensor = xMainTensor[this->mode2OneLoopSize_];
        dyFoldTensor = dyMainTensor[this->mode2OneLoopSize_];
        if (loopIdx < this->mode2FoldLoopCnt_ - 1) {
            this->CopyInDyAndX(dyFoldTensor, xFoldTensor, foldOffset, this->mode2OneLoopSize_);
        } else if (loopIdx == this->mode2FoldLoopCnt_ - 1) {
            foldReduceNum = this->mode2FoldTail_;
            this->CopyInDyAndX(dyFoldTensor, xFoldTensor, foldOffset, this->mode2FoldTail_);
        } else {
            foldReduceNum = 0;
        }
        this->inQueX_.EnQue(xMainTensor);
        this->inQueDy_.EnQue(dyMainTensor);
        xMainTensor = this->inQueX_.template DeQue<T>();
        dyMainTensor = this->inQueDy_.template DeQue<T>();
        VFMode2DbetaDs(
            xMainTensor, xFoldTensor, dyMainTensor, dyFoldTensor, mainReduceNum, foldReduceNum, loopIdx, cacheDbeta,
            cacheDgama);
        this->inQueX_.FreeTensor(xMainTensor);
        this->inQueDy_.FreeTensor(dyMainTensor);
        this->UpdateCache(loopIdx, cacheDbeta, cacheDgama);
    }

    if (loopIdx < this->CACHE_BUFF_SIZE) {
        this->CustomReduceSum(dbeta, cacheDbeta[this->CACHE_LEVEL0_IDX], C_G_idx);
        this->CustomReduceSum(dgamma, cacheDgama[this->CACHE_LEVEL0_IDX], C_G_idx);
    } else if (loopIdx < this->CACHE_BUFF_SIZE * this->CACHE_BUFF_SIZE) {
        this->CustomReduceSum(dbeta, cacheDbeta[this->CACHE_LEVEL1_IDX], C_G_idx);
        this->CustomReduceSum(dgamma, cacheDgama[this->CACHE_LEVEL1_IDX], C_G_idx);
    } else {
        this->CustomReduceSum(dbeta, cacheDbeta[this->CACHE_LEVEL2_IDX], C_G_idx);
        this->CustomReduceSum(dgamma, cacheDgama[this->CACHE_LEVEL2_IDX], C_G_idx);
    }
}

template <typename T, typename U>
__aicore__ inline void GroupNormGradReCompute<T, U>::ComputeMode2Dx(
    int32_t taskIdx, LocalTensor<float>& dbetaTensor, LocalTensor<float>& dsTensor)
{
    float sum1 = 0;
    float sum2 = 0;
    uint32_t channelIdx = (taskIdx % this->G_) * this->C_G_;
    this->LoadDataToUb(this->inQueGamma_, this->tBufGamma_, this->gammaGm_, channelIdx, this->C_G_);
    LocalTensor<float> gammaTensor = this->inQueGamma_.template DeQue<float>();
    this->ComputeSum1Sum2(dbetaTensor, dsTensor, gammaTensor, sum1, sum2);
    float s = 1.0f / this->eleNumPerG_;
    float C2 = (sum2 * this->meanScalar_ - sum1) * this->rstdScalar_ * this->rstdScalar_ * this->rstdScalar_ * s;
    float C3 = -C2 * this->meanScalar_ - sum2 * this->rstdScalar_ * s;
    this->outQueDbeta_.FreeTensor(dbetaTensor);
    this->outQueDs_.FreeTensor(dsTensor);
    float gamma = 0;
    LocalTensor<T> xTensor;
    LocalTensor<T> dyTensor;
    for (int32_t iter_C_G_idx = 0; iter_C_G_idx < this->C_G_ * this->mode2UbIterNum_; iter_C_G_idx++) {
        int64_t offset = taskIdx * this->eleNumPerG_ + iter_C_G_idx / this->mode2UbIterNum_ * this->eleNumPerC_ +
                         iter_C_G_idx % this->mode2UbIterNum_ * this->mode2UbCapEle_;
        gamma = gammaTensor.GetValue(iter_C_G_idx / this->mode2UbIterNum_);
        TEventID eventIDSToV = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(eventIDSToV);
        if (iter_C_G_idx % this->mode2UbIterNum_ != this->mode2UbIterNum_ - 1) {
            xTensor = this->inQueX_.template AllocTensor<T>();
            dyTensor = this->inQueDy_.template AllocTensor<T>();
            this->CopyInDyAndX(dyTensor, xTensor, offset, this->mode2UbCapEle_);
            this->inQueX_.EnQue(xTensor);
            this->inQueDy_.EnQue(dyTensor);

            xTensor = this->inQueX_.template DeQue<T>();
            dyTensor = this->inQueDy_.template DeQue<T>();
            WaitFlag<HardEvent::S_V>(eventIDSToV);
            LocalTensor<T> dxTensor = this->outQueDx_.template AllocTensor<T>();
            ComputeDxParams dxParams = {C2, C3, gamma, this->mode2UbCapEle_};
            VFComputeMode2Dx(dxTensor, xTensor, dyTensor, dxParams);
            this->inQueX_.FreeTensor(xTensor);
            this->inQueDy_.FreeTensor(dyTensor);
            this->outQueDx_.EnQue(dxTensor);
            this->StoreDxToGm(this->outQueDx_, offset, this->mode2UbCapEle_);
        } else {
            xTensor = this->inQueX_.template AllocTensor<T>();
            dyTensor = this->inQueDy_.template AllocTensor<T>();
            DataCopyPadExtParams<T> dataCopyPadExtParamsX;
            dataCopyPadExtParamsX.isPad = false;
            dataCopyPadExtParamsX.leftPadding = 0;
            dataCopyPadExtParamsX.rightPadding = 0;
            dataCopyPadExtParamsX.paddingValue = 0;
            DataCopyExtParams copyInParamsX;
            copyInParamsX.blockCount = 1;
            copyInParamsX.blockLen =  this->mode2UbTailNum_ * sizeof(T);
            copyInParamsX.srcStride = 0;
            copyInParamsX.dstStride = 0;
            DataCopyPad(xTensor, this->xGm_[offset], copyInParamsX, dataCopyPadExtParamsX);
            DataCopyPad(dyTensor, this->dyGm_[offset], copyInParamsX, dataCopyPadExtParamsX);
            this->inQueX_.EnQue(xTensor);
            this->inQueDy_.EnQue(dyTensor);
            xTensor = this->inQueX_.template DeQue<T>();
            dyTensor = this->inQueDy_.template DeQue<T>();
            WaitFlag<HardEvent::S_V>(eventIDSToV);
            LocalTensor<T> dxTensor = this->outQueDx_.template AllocTensor<T>();
            ComputeDxParams dxParams = {C2, C3, gamma, this->mode2UbTailNum_};
            VFComputeMode2Dx(dxTensor, xTensor, dyTensor, dxParams);
            this->inQueX_.FreeTensor(xTensor);
            this->inQueDy_.FreeTensor(dyTensor);
            this->outQueDx_.EnQue(dxTensor);
            this->StoreDxToGm(this->outQueDx_, offset, this->mode2UbTailNum_);
        }
    }
    this->inQueGamma_.FreeTensor(gammaTensor);
}

/*
    sum1 = dbeta * gamma
    sum2 = ds * gamma
    C2 = (sum2 * mean - sum1) * rstd^3 / D * HxW
    C3 = -c2 * mean - sum2 * rstd / D * HxW
    C1 = rstd * gamma
    Dx = C1 * dy + C2 * X + C3
*/
template <typename T, typename U>
__aicore__ inline void GroupNormGradReCompute<T, U>::VFComputeMode2Dx(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& xTensor, const LocalTensor<T>& dyTensor,
    const ComputeDxParams& dxParams)
{
    __ubuf__ T* ubX = (__ubuf__ T*)xTensor.GetPhyAddr();
    __ubuf__ T* ubDy = (__ubuf__ T*)dyTensor.GetPhyAddr();
    __ubuf__ T* ubDst = (__ubuf__ T*)dstTensor.GetPhyAddr();
    float C2 = dxParams.C2;
    float C3 = dxParams.C3;
    auto gamma = dxParams.gamma;
    uint32_t calCount = dxParams.processNum;
    float dyMulScalar = this->rstdScalar_ * gamma;
    uint16_t repeatTimes = CeilDiv(calCount, this->VecLen_);

    __VEC_SCOPE__
    {
        RegTensor<float> vregX;
        RegTensor<float> vregDy;
        uint32_t sreg = (uint32_t)calCount;
        MaskReg preg;
        uint32_t sregvl = (uint32_t)this->VecLen_;
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = UpdateMask<float>(sreg);
            LoadOneTensorForDtypeT<T>(ubX, vregX, preg, i * sregvl);
            LoadOneTensorForDtypeT<T>(ubDy, vregDy, preg, i * sregvl);
            Muls(vregX, vregX, C2, preg);
            Muls(vregDy, vregDy, dyMulScalar, preg);
            Add(vregX, vregX, vregDy, preg);
            Adds(vregX, vregX, C3, preg);
            StoreOneTensorForDtypeT<T>(ubDst, vregX, preg, i * sregvl);
        }
    }
}
} // namespace GroupNormGrad
#endif