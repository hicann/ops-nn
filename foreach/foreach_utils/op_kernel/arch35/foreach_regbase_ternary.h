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
 * \file foreach_regbase_ternary.h
 * \brief
 */
#ifndef FOREACH_REGBASE_TERNARY_H
#define FOREACH_REGBASE_TERNARY_H

#include "foreach_regbase_common.h"

using namespace AscendC;

template <typename T, typename Tiling, typename Predicate>
class ForeachRegbaseTernary
{
public:
    __aicore__ inline ForeachRegbaseTernary(Predicate& p) : pred_(p){};
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR outputs, GM_ADDR workspace, const Tiling* tilingData,
        TPipe* tPipe)
    {
        blockIdx_ = GetBlockIdx();
        inDesc_ = ListTensorDesc((__gm__ void*)inputs);
        tensorOneDesc_ = ListTensorDesc((__gm__ void*)tensor1);
        tensorTwoDesc_ = ListTensorDesc((__gm__ void*)tensor2);
        outDesc_ = ListTensorDesc((__gm__ void*)outputs);
        ParseTilingData(tilingData);
        tPipe->InitBuffer(inQueue_, BUFFER_NUM, inputsTensorUbSize_ * sizeof(T));
        tPipe->InitBuffer(tensorOneQueue_, BUFFER_NUM, inputsTensorUbSize_ * sizeof(T));
        tPipe->InitBuffer(tensorTwoQueue_, BUFFER_NUM, inputsTensorUbSize_ * sizeof(T));
        tPipe->InitBuffer(outQueue_, BUFFER_NUM, inputsTensorUbSize_ * sizeof(T));
        maxDataCount_ = inputsTensorUbSize_;
    }

    __aicore__ inline void Process()
    {
        for (uint16_t i = tensorStart_; i <= tensorEnd_; i++) {
            int64_t cursorStart = 0;
            int64_t cursorEnd = tensorDataCountList_[i] - 1;
            int64_t dataCount = 0;
            if (i == tensorStart_) {
                cursorStart = tensorStartOffset_;
            }
            if (i == tensorEnd_) {
                cursorEnd = tensorEndOffset_;
            }
            dataCount = cursorEnd - cursorStart + 1;
            inTensorGM_.SetGlobalBuffer(GetTensorAddr(i, inDesc_) + cursorStart);
            tensorOneGM_.SetGlobalBuffer(GetTensorAddr(i, tensorOneDesc_) + cursorStart);
            tensorTwoGM_.SetGlobalBuffer(GetTensorAddr(i, tensorTwoDesc_) + cursorStart);
            outTensorGM_.SetGlobalBuffer(GetTensorAddr(i, outDesc_) + cursorStart);
            SingleTensorProcess(dataCount);
        }
    }

    __aicore__ inline void ParseTilingData(const Tiling* tilingData)
    {
        inputsTensorUbSize_ = tilingData->inputsTensorUbSize;
        tensorDataCountList_ = (int64_t*)tilingData->tensorDataCountList;
        tensorStart_ = tilingData->tensorStartList[blockIdx_];
        tensorEnd_ = tilingData->tensorEndList[blockIdx_];
        tensorStartOffset_ = tilingData->tensorStartOffsetList[blockIdx_];
        tensorEndOffset_ = tilingData->tensorEndOffsetList[blockIdx_];
    }
    __aicore__ inline void SingleTensorProcess(int64_t dataCount)
    {
        // Batch handling and calculation.
        int64_t quotient = CeilDivision(dataCount, maxDataCount_);
        for (int64_t i = 0; i < quotient; i++) {
            int64_t currentDataCount =
                (i == (quotient - 1)) ? (dataCount - (quotient - 1) * maxDataCount_) : maxDataCount_;
            CopyIn(i, currentDataCount);
            Compute(currentDataCount);
            CopyOut(i, currentDataCount);
        }
    }

    __aicore__ inline void CopyIn(int64_t index, int64_t dataCount)
    {
        LocalTensor<T> dataLocal = inQueue_.AllocTensor<T>();
        LocalTensor<T> dataLocalTensorOne = tensorOneQueue_.AllocTensor<T>();
        LocalTensor<T> dataLocalTensorTwo = tensorTwoQueue_.AllocTensor<T>();
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = dataCount * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(dataLocal, inTensorGM_[1ULL * index * maxDataCount_], copyInParams, dataCopyPadExtParams);
        DataCopyPad(dataLocalTensorOne, tensorOneGM_[1ULL * index * maxDataCount_], copyInParams, dataCopyPadExtParams);
        DataCopyPad(dataLocalTensorTwo, tensorTwoGM_[1ULL * index * maxDataCount_], copyInParams, dataCopyPadExtParams);
        inQueue_.EnQue(dataLocal);
        tensorOneQueue_.EnQue(dataLocalTensorOne);
        tensorTwoQueue_.EnQue(dataLocalTensorTwo);
    }

    __aicore__ inline void Compute(int64_t dataCount)
    {
        LocalTensor<T> inLocal = inQueue_.template DeQue<T>();
        LocalTensor<T> inLocalTensorOne = tensorOneQueue_.template DeQue<T>();
        LocalTensor<T> inLocalTensorTwo = tensorTwoQueue_.template DeQue<T>();
        LocalTensor<T> outLocal = outQueue_.template AllocTensor<T>();
        pred_.Compute(inLocal, inLocalTensorOne, inLocalTensorTwo, outLocal, dataCount);
        inQueue_.FreeTensor(inLocal);
        tensorOneQueue_.FreeTensor(inLocalTensorOne);
        tensorTwoQueue_.FreeTensor(inLocalTensorTwo);
        outQueue_.template EnQue<T>(outLocal);
    }

    __aicore__ inline void CopyOut(int64_t index, int64_t dataCount)
    {
        LocalTensor<T> retLocal = outQueue_.DeQue<T>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = dataCount * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(outTensorGM_[1ULL * index * maxDataCount_], retLocal, copyInParams);
        outQueue_.FreeTensor(retLocal);
    }

    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, ListTensorDesc desc)
    {
        return (__gm__ T*)desc.GetDataPtr<__gm__ T>(index);
    }

protected:
    static constexpr int32_t BUFFER_NUM = 2;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> tensorOneQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> tensorTwoQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;

    GlobalTensor<T> inTensorGM_;
    GlobalTensor<T> tensorOneGM_;
    GlobalTensor<T> tensorTwoGM_;
    GlobalTensor<T> outTensorGM_;

    int64_t blockIdx_ = 0;

    uint32_t maxDataCount_ = {0};
    // tiling params
    uint64_t inputsTensorUbSize_ = 0;
    int64_t* tensorDataCountList_ = nullptr;
    uint16_t tensorStart_ = {0};
    uint16_t tensorEnd_ = {0};
    int64_t tensorStartOffset_ = {0};
    int64_t tensorEndOffset_ = {0};

    ListTensorDesc inDesc_;
    ListTensorDesc tensorOneDesc_;
    ListTensorDesc tensorTwoDesc_;
    ListTensorDesc outDesc_;

private:
    Predicate& pred_;
};

#endif // FOREACH_REGBASE_TERNARY_H