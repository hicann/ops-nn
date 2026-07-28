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
 * \file instance_norm_grad_recompute.h
 * \brief recompute kernel: [M, cTile] overflows UB; M is tiled, pass1 streams+accumulates,
 *        pass2 reloads x/dy to write pd_x. Column accumulators live in UB across M tiles.
 */
#ifndef INSTANCE_NORM_GRAD_RECOMPUTE_H
#define INSTANCE_NORM_GRAD_RECOMPUTE_H
#pragma once

#include "instance_norm_grad_base.h"

namespace InstanceNormGrad {
template <typename T>
class InstanceNormGradReCompute : public InstanceNormGradBase<T> {
public:
    __aicore__ inline InstanceNormGradReCompute() : InstanceNormGradBase<T>() {}

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR variance, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pd_x,
                                GM_ADDR pd_gamma, GM_ADDR pd_beta, GM_ADDR workspace,
                                const InstanceNormGradTilingData* __restrict tiling, TPipe* pipeIn)
    {
        this->InitCommon(dy, x, variance, mean, gamma, pd_x, pd_gamma, pd_beta, workspace, tiling, pipeIn);
        this->InitStage1Buffers();
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx_ < this->stage1CoreUsed_) {
            for (uint32_t t = 0; t < this->curCoreTaskNum_; ++t) {
                ProcessTask(this->startTask_ + t);
            }
        }
        this->Stage2Process();
    }

private:
    __aicore__ inline uint32_t RowsOfIter(uint32_t iter) const
    {
        return (iter == this->mUbIterNum_ - 1) ? this->mUbTailNum_ : this->mUbTile_;
    }

    __aicore__ inline void ProcessTask(int64_t taskId)
    {
        int64_t n = 0;
        int64_t cStart = 0;
        uint32_t cLen = 0;
        this->GetTaskCoords(taskId, n, cStart, cLen);
        uint32_t rowStride = this->RowStrideT(cLen);

        this->LoadTaskParams(n, cStart, cLen);

        __local_mem__ float* meanUb = (__local_mem__ float*)this->meanBuf_.template Get<float>().GetPhyAddr();
        __local_mem__ float* rstdUb = (__local_mem__ float*)this->rstdBuf_.template Get<float>().GetPhyAddr();
        __local_mem__ float* gammaUb = (__local_mem__ float*)this->gammaBuf_.template Get<float>().GetPhyAddr();
        __local_mem__ float* pdVarUb = (__local_mem__ float*)this->pdVarBuf_.template Get<float>().GetPhyAddr();
        __local_mem__ float* pdMeanUb = (__local_mem__ float*)this->pdMeanBuf_.template Get<float>().GetPhyAddr();
        __local_mem__ float* accDgUb = (__local_mem__ float*)this->accDgammaBuf_.template Get<float>().GetPhyAddr();
        __local_mem__ float* accDbUb = (__local_mem__ float*)this->accDbetaBuf_.template Get<float>().GetPhyAddr();
        __local_mem__ float* cDgUb = (__local_mem__ float*)this->cDgammaBuf_.template Get<float>().GetPhyAddr();
        __local_mem__ float* cDbUb = (__local_mem__ float*)this->cDbetaBuf_.template Get<float>().GetPhyAddr();

        // ---- pass1: stream M tiles, fold into UB column accumulators ----
        for (uint32_t it = 0; it < this->mUbIterNum_; ++it) {
            uint32_t rows = RowsOfIter(it);
            uint32_t mStart = it * this->mUbTile_;
            LocalTensor<T> xLocal = this->inQueX_.template AllocTensor<T>();
            LocalTensor<T> dyLocal = this->inQueDy_.template AllocTensor<T>();
            this->CopyInTile(this->xGm_, xLocal, n, mStart, rows, cStart, cLen);
            this->CopyInTile(this->dyGm_, dyLocal, n, mStart, rows, cStart, cLen);
            this->inQueX_.EnQue(xLocal);
            this->inQueDy_.EnQue(dyLocal);
            xLocal = this->inQueX_.template DeQue<T>();
            dyLocal = this->inQueDy_.template DeQue<T>();
            Pass1Accumulate<T>((__local_mem__ T*)xLocal.GetPhyAddr(), (__local_mem__ T*)dyLocal.GetPhyAddr(), meanUb,
                               rstdUb, gammaUb, pdVarUb, pdMeanUb, accDgUb, accDbUb, cDgUb, cDbUb, rows, cLen,
                               rowStride);
            this->inQueX_.FreeTensor(xLocal);
            this->inQueDy_.FreeTensor(dyLocal);
        }
        this->WritePartialOrOutput(n, cStart, cLen);

        // ---- pass2: reload M tiles, write pd_x ----
        float oneOverM = 1.0f / static_cast<float>(this->M_);
        float twoOverM = 2.0f * oneOverM;
        for (uint32_t it = 0; it < this->mUbIterNum_; ++it) {
            uint32_t rows = RowsOfIter(it);
            uint32_t mStart = it * this->mUbTile_;
            LocalTensor<T> xLocal = this->inQueX_.template AllocTensor<T>();
            LocalTensor<T> dyLocal = this->inQueDy_.template AllocTensor<T>();
            this->CopyInTile(this->xGm_, xLocal, n, mStart, rows, cStart, cLen);
            this->CopyInTile(this->dyGm_, dyLocal, n, mStart, rows, cStart, cLen);
            this->inQueX_.EnQue(xLocal);
            this->inQueDy_.EnQue(dyLocal);
            xLocal = this->inQueX_.template DeQue<T>();
            dyLocal = this->inQueDy_.template DeQue<T>();
            LocalTensor<T> pdxLocal = this->outQuePdx_.template AllocTensor<T>();
            ComputePdx<T>((__local_mem__ T*)xLocal.GetPhyAddr(), (__local_mem__ T*)dyLocal.GetPhyAddr(),
                          (__local_mem__ T*)pdxLocal.GetPhyAddr(), meanUb, rstdUb, gammaUb, pdVarUb, pdMeanUb, rows,
                          cLen, rowStride, twoOverM, oneOverM);
            this->outQuePdx_.EnQue(pdxLocal);
            pdxLocal = this->outQuePdx_.template DeQue<T>();
            this->CopyOutPdx(pdxLocal, n, mStart, rows, cStart, cLen);
            this->outQuePdx_.FreeTensor(pdxLocal);
            this->inQueX_.FreeTensor(xLocal);
            this->inQueDy_.FreeTensor(dyLocal);
        }
    }
};
} // namespace InstanceNormGrad
#endif // INSTANCE_NORM_GRAD_RECOMPUTE_H
