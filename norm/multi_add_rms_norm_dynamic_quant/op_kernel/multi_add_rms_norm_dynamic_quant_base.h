/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file multi_add_rms_norm_dynamic_quant_base.h
 * \brief
 */

#ifndef __multi_add_rms_norm_dynamic_quant_BASE_CLASS_H_
#define __multi_add_rms_norm_dynamic_quant_BASE_CLASS_H_

#include "multi_add_rms_norm_dynamic_quant_helper.h"

namespace MultiAddRmsNormDynQnt {
constexpr uint8_t UNIT_64_BYTE_POW = 3;
constexpr uint8_t X1_LIST_MAX_SIZE = 5;

template <typename T>
__aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> UNIT_64_BYTE_POW);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelMultiAddRmsNormDynamicQuantBase
{
public:
    using xSrcGMList = GlobalTensor<T>[X1_LIST_MAX_SIZE];
    __aicore__ inline KernelMultiAddRmsNormDynamicQuantBase()
    {}

    __aicore__ inline void InitBaseParams(const MultiAddRmsNormDynamicQuantTilingData* tiling)
    {
        this->numCore = tiling->useCore;
        this->numFirstDim = tiling->numFirstDim;
        this->numLastDim = tiling->numLastDim;
        this->numLastDimAligned = tiling->numLastDimAligned; // Quantize better be aligned to 32 elements

        this->firstDimPerCore = tiling->firstDimPerCore;
        this->firstDimPerCoreTail = tiling->firstDimPerCoreTail;
        this->firstDimPerLoop = tiling->firstDimPerLoop;

        this->lastDimSliceLen = tiling->lastDimSliceLen;
        this->lastDimLoopNum = tiling->lastDimLoopNum;
        this->lastDimSliceLenTail = tiling->lastDimSliceLenTail;

        this->eps = tiling->epsilon;
        this->aveNum = tiling->avgFactor;
        this->x1Num = tiling->x1Num; // demands tiling data upgrade

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

        this->smooth1Exist = tiling->smoothNum >= 1;
        // 2 dynamic quant operator required 2 scale buffer.
        this->smooth2Exist = tiling->smoothNum == 2;
    }

    __aicore__ inline void InitInGlobalTensors(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooth1, GM_ADDR smooth2)
    {
        uint64_t block_offset = blockIdx_ * this->gmOffset_;
        // Init x1List Tensors
        for (int32_t i = 0; i < this->x1Num; ++i) {
            x1GmList[i].SetGlobalBuffer(GetTensorAddr<T>(i, x1) + block_offset);
        }
        // Init x2List Tensors
        for (int32_t i = 0; i < this->x2Num; ++i) {
            x2GmList[i].SetGlobalBuffer((__gm__ T*)x2 + block_offset);
        }

        x1Gm = x1GmList[0]; // x1Gm.SetGlobalBuffer((__gm__ T*)(x1) + block_offset);
        x2Gm = x2GmList[0]; // x2Gm.SetGlobalBuffer((__gm__ T*)(x2) + block_offset);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma);
        smooth1Gm.SetGlobalBuffer((__gm__ T*)smooth1);
        smooth2Gm.SetGlobalBuffer((__gm__ T*)smooth2);
    }

    __aicore__ inline void InitOutGlobalTensors(
        GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR y, GM_ADDR outScale1, GM_ADDR outScale2)
    {
        y1Gm.SetGlobalBuffer((__gm__ int8_t*)(y1) + blockIdx_ * this->gmOffset_);
        y2Gm.SetGlobalBuffer((__gm__ int8_t*)(y2) + blockIdx_ * this->gmOffset_);
        xGm.SetGlobalBuffer((__gm__ T*)(x) + blockIdx_ * this->gmOffset_);
        yGm.SetGlobalBuffer((__gm__ T*)(y) + blockIdx_ * this->gmOffset_);
        outScale1Gm.SetGlobalBuffer((__gm__ float*)outScale1 + blockIdx_ * this->firstDimPerCore);
        outScale2Gm.SetGlobalBuffer((__gm__ float*)outScale2 + blockIdx_ * this->firstDimPerCore);
    }

    __aicore__ inline void InitWorkSpaceGlobalTensors(GM_ADDR workspace)
    {}

protected:
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> smooth1Gm;
    GlobalTensor<T> smooth2Gm;

    GlobalTensor<int8_t> y1Gm;
    GlobalTensor<int8_t> y2Gm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> outScale1Gm;
    GlobalTensor<float> outScale2Gm;

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
    // xlist
    uint32_t x1Num;
    uint32_t x2Num{1};

    xSrcGMList x1GmList{}; // Redirectable GM for xList copy-in
    xSrcGMList x2GmList{}; // Redirectable GM for xList copy-in

    float eps;
    float aveNum;

    uint64_t blockIdx_;
    uint64_t gmOffset_;
    uint64_t rowTail_;
    uint64_t rowStep;
    uint64_t rowWork;

    bool smooth1Exist;
    bool smooth2Exist;
};
} // namespace MultiAddRmsNormDynQnt
#endif // __multi_add_rms_norm_dynamic_quant_BASE_CLASS_H_
