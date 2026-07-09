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
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR dx,
                                GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace,
                                const GroupNormGradRegBaseTilingData* __restrict tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitBuffer(const GroupNormGradRegBaseTilingData* tilingData);
    __aicore__ inline void Compute(int32_t taskIdx);
    __aicore__ inline void ComputeMode1Dx(
        int32_t taskIdx, LocalTensor<float>& dbetaTensor, LocalTensor<float>& dsTensor);
};

template <typename T, typename U>
__aicore__ inline void GroupNormGradCFullLoad<T, U>::Init(GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x,
                                                          GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta,
                                                          GM_ADDR workspace,
                                                          const GroupNormGradRegBaseTilingData* __restrict tilingData,
                                                          TPipe* pipeIn)
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
    uint32_t capElemAllocSpace = CeilAlign(static_cast<uint32_t>(this->eleNumPerC_ * this->mode1UbCapCNum_),
                                           this->elemTPerBlock_) *
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
    int64_t baseOffset = taskIdx * this->eleNumPerG_;
    uint32_t ubLoopCnt = (this->C_G_ + this->mode1UbCapCNum_ - 1) / this->mode1UbCapCNum_;
    LocalTensor<T> dyTensor;
    LocalTensor<T> xTensor;
    for (uint32_t loopIdx = 0; loopIdx < ubLoopCnt; loopIdx++) {
        int64_t offset = baseOffset + loopIdx * this->eleNumPerC_ * this->mode1UbCapCNum_;
        auto curCNum = this->mode1UbCapCNum_;
        if ((loopIdx == ubLoopCnt - 1) && (this->C_G_ % this->mode1UbCapCNum_ != 0)) {
            curCNum = this->C_G_ % this->mode1UbCapCNum_;
        }

        xTensor = this->inQueX_.template AllocTensor<T>();
        dyTensor = this->inQueDy_.template AllocTensor<T>();
        this->CopyInDyAndX(dyTensor, xTensor, offset, this->eleNumPerC_ * curCNum);
        this->inQueX_.EnQue(xTensor);
        this->inQueDy_.EnQue(dyTensor);

        dyTensor = this->inQueDy_.template DeQue<T>();
        xTensor = this->inQueX_.template DeQue<T>();
        if (this->eleNumPerC_ <= this->VecLen_) {
            VFComputeDbetaDs<T>(xTensor, dyTensor, dbetaTensor, dsTensor, this->eleNumPerC_, this->VecLen_,
                loopIdx * this->mode1UbCapCNum_, static_cast<uint16_t>(curCNum));
        } else {
            this->VFDbetaDgammaBinaryFoldCommon(xTensor, dyTensor, dbetaTensor, dsTensor,
                                                loopIdx * this->mode1UbCapCNum_, curCNum);
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
  temp = xHat * rstd - mean * rstd
  dgamma = reduceSum(dy * temp)
*/
template <typename T, typename U>
__aicore__ inline void GroupNormGradCFullLoad<T, U>::ComputeMode1Dx(int32_t taskIdx, LocalTensor<float>& dbetaTensor,
                                                                    LocalTensor<float>& dsTensor)
{
    float sum1 = 0;
    float sum2 = 0;
    int64_t channelIdx = (taskIdx % this->G_) * this->C_G_;
    int64_t baseOffset = taskIdx * this->eleNumPerG_;
    uint32_t ubCapEleNum = CeilAlign(static_cast<uint32_t>(this->eleNumPerC_), this->elemTPerBlock_);
    this->LoadDataToUb(this->inQueGamma_, this->tBufGamma_, this->gammaGm_, channelIdx, this->C_G_);
    LocalTensor<float> gammaTensor = this->inQueGamma_.template DeQue<float>();
    this->ComputeSum1Sum2(dbetaTensor, dsTensor, gammaTensor, sum1, sum2);
    float s = 1.0f / this->eleNumPerG_;
    float C2 = (sum2 * this->meanScalar_ + (0 - sum1)) * this->rstdScalar_ * this->rstdScalar_ * this->rstdScalar_ * s;
    float C3 = (0 - C2) * this->meanScalar_ + (0 - sum2 * this->rstdScalar_ * s);
    this->outQueDbeta_.FreeTensor(dbetaTensor);
    this->outQueDs_.FreeTensor(dsTensor);
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
        this->VFComputeMode1DxCommon(dxTensor, xTensor, dyTensor, gammaTensor, C2, C3, loopIdx * this->mode1UbCapCNum_,
                                     curCNum);
        this->inQueX_.FreeTensor(xTensor);
        this->inQueDy_.FreeTensor(dyTensor);
        this->outQueDx_.EnQue(dxTensor);
        this->StoreDxToGm(this->outQueDx_, offset, this->eleNumPerC_ * curCNum);
    }
    this->inQueGamma_.FreeTensor(gammaTensor);
}

} // namespace GroupNormGrad
#endif
