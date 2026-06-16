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
 * \file rms_norm_dynamic_quant_base.h
 * \brief
 */

#ifndef RMS_NORM_DYNAMIC_QUANT_BASE_CLASS_H_
#define RMS_NORM_DYNAMIC_QUANT_BASE_CLASS_H_

#include "rms_norm_dynamic_quant_helper.h"

template <typename T, typename T_Y, int TILING_KEY, int BUFFER_NUM = 1>
class KernelRmsNormDynamicQuantBase {
public:
    __aicore__ inline KernelRmsNormDynamicQuantBase()
    {}

    __aicore__ inline void InitBaseParams(const RmsNormDynamicQuantTilingData* tiling)
    {
        this->numCore = tiling->useCore;
        this->numFirstDim = tiling->numFirstDim;
        this->numLastDim = tiling->numLastDim;
        this->numLastDimAligned = tiling->numLastDimAligned;

        this->firstDimPerCore = tiling->firstDimPerCore;
        this->firstDimPerCoreTail = tiling->firstDimPerCoreTail;
        this->firstDimPerLoop = tiling->firstDimPerLoop;

        this->lastDimSliceLen = tiling->lastDimSliceLen;
        this->lastDimLoopNum = tiling->lastDimLoopNum;
        this->lastDimSliceLenTail = tiling->lastDimSliceLenTail;
        this->betaFlag = tiling->betaFlag;
        this->eps = tiling->epsilon;
        this->aveNum = tiling->avgFactor;

        blockIdx_ = GetBlockIdx();
        if (blockIdx_ != this->numCore - 1) {
            this->rowWork = this->firstDimPerCore;
            this->rowStep = this->firstDimPerLoop;
        } else {
            this->rowWork = this->firstDimPerCoreTail;
            this->rowStep = TWO_NUMS_MIN(this->firstDimPerLoop, this->rowWork);
        }
        this->rowTail_ = (this->rowWork % this->rowStep == 0) ? this->rowStep : (this->rowWork % this->rowStep);
        this->gmOffset_ = this->firstDimPerCore * this->numLastDim;

        this->smoothExist = tiling->smoothNum1;

        // dynamic quant max value
        if constexpr (IsSameType<T_Y, int8_t>::value) {
            this->quantMaxVal = DYNAMIC_QUANT_DIVIDEND;
        } else {
            this->quantMaxVal = DYNAMIC_QUANT_DIVIDEND_INT4;
        }
        this->outQuantFlag = tiling->outQuant1Flag;
    }

    __aicore__ inline void InitInGlobalTensors(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR smooth, GM_ADDR beta)
    {
        xGm.SetGlobalBuffer((__gm__ T*)(x) + blockIdx_ * this->gmOffset_);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma);
        smoothGm.SetGlobalBuffer((__gm__ T*)smooth);
        if (this->betaFlag == 1) {
            betaGm.SetGlobalBuffer((__gm__ T*)beta);
        }
    }

    __aicore__ inline void InitOutGlobalTensors(GM_ADDR y, GM_ADDR outScale)
    {
        int64_t yBufferSize = blockIdx_ * this->gmOffset_;
        if constexpr (IsSameType<T_Y, int4b_t>::value) {
            yBufferSize = yBufferSize / 2;
        }
        yGm.SetGlobalBuffer((__gm__ T_Y*)(y) + yBufferSize);
        outScaleGm.SetGlobalBuffer((__gm__ float*)outScale + blockIdx_ * this->firstDimPerCore);
    }

    __aicore__ inline void InitWorkSpaceGlobalTensors(GM_ADDR workspace)
    {}

protected:
    GlobalTensor<T> xGm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> smoothGm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T_Y> yGm;
    GlobalTensor<float> outScaleGm;

    uint32_t betaFlag;
    uint64_t numCore;
    uint64_t numFirstDim;
    uint64_t numLastDim;
    uint64_t numLastDimAligned;
    uint64_t firstDimPerCore;
    uint64_t firstDimPerCoreTail;
    uint64_t firstDimPerLoop;
    uint64_t lastDimSliceLen;
    uint64_t lastDimLoopNum;
    uint64_t lastDimSliceLenTail;

    float eps;
    float aveNum;

    uint64_t blockIdx_;
    uint64_t gmOffset_;
    uint64_t rowTail_;
    uint64_t rowStep;
    uint64_t rowWork;

    bool smoothExist;
    int32_t outQuantFlag;
    float quantMaxVal;
};

#endif // RMS_NORM_DYNAMIC_QUANT_BASE_CLASS_H_
