/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_AXIS_ONE_COPY_H
#define KTH_VALUE_AXIS_ONE_COPY_H

#include "kernel_operator.h"
#include "kth_value_tiling_data.h"

namespace KthValue {
using namespace AscendC;

template <typename T>
class KthValueAxisOneCopy {
public:
    __aicore__ inline KthValueAxisOneCopy() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace,
                                const KthValueTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyValue(int64_t offset, uint32_t validElems);
    __aicore__ inline void CopyIndex(int64_t offset, uint32_t validElems);

    static constexpr uint32_t BUFFER_NUM = 2;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> valueQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> idxQueue_;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<T> outValueGm_;
    GlobalTensor<int64_t> outIdxGm_;

    uint32_t blockIdx_ = 0;
    uint32_t blockDim_ = 0;
    uint32_t copyElemsPerLoop_ = 0;
    uint32_t loopTimes_ = 0;
    int64_t totalElems_ = 0;
};

template <typename T>
__aicore__ inline void KthValueAxisOneCopy<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace,
                                                    const KthValueTilingData* tilingData, TPipe* pipe)
{
    (void)workspace;
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }

    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    copyElemsPerLoop_ = tilingData->keyParams0;
    loopTimes_ = tilingData->keyParams1;
    totalElems_ = tilingData->unsortedDimNum * tilingData->lastAxisNum;

    inputXGm_.SetGlobalBuffer((__gm__ T*)x);
    outValueGm_.SetGlobalBuffer((__gm__ T*)y);
    outIdxGm_.SetGlobalBuffer((__gm__ int64_t*)idx);

    if (copyElemsPerLoop_ == 0) {
        return;
    }

    pipe->InitBuffer(valueQueue_, BUFFER_NUM, copyElemsPerLoop_ * sizeof(T));
    pipe->InitBuffer(idxQueue_, BUFFER_NUM, copyElemsPerLoop_ * sizeof(int64_t));
}

template <typename T>
__aicore__ inline void KthValueAxisOneCopy<T>::CopyValue(int64_t offset, uint32_t validElems)
{
    LocalTensor<T> valueLocal = valueQueue_.AllocTensor<T>();
    DataCopyExtParams copyParam{1, static_cast<uint32_t>(validElems * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(valueLocal, inputXGm_[offset], copyParam, padParams);
    valueQueue_.EnQue(valueLocal);

    valueLocal = valueQueue_.DeQue<T>();
    DataCopyPad(outValueGm_[offset], valueLocal, copyParam);
    valueQueue_.FreeTensor(valueLocal);
}

template <typename T>
__aicore__ inline void KthValueAxisOneCopy<T>::CopyIndex(int64_t offset, uint32_t validElems)
{
    LocalTensor<int64_t> idxLocal = idxQueue_.AllocTensor<int64_t>();
    Duplicate(idxLocal, static_cast<int64_t>(0), validElems);
    idxQueue_.EnQue(idxLocal);

    idxLocal = idxQueue_.DeQue<int64_t>();
    DataCopyExtParams copyParam{1, static_cast<uint32_t>(validElems * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(outIdxGm_[offset], idxLocal, copyParam);
    idxQueue_.FreeTensor(idxLocal);
}

template <typename T>
__aicore__ inline void KthValueAxisOneCopy<T>::Process()
{
    if (blockIdx_ >= blockDim_ || blockDim_ == 0 || copyElemsPerLoop_ == 0 || totalElems_ <= 0) {
        return;
    }

    uint32_t loopsPerCore = (loopTimes_ + blockDim_ - 1U) / blockDim_;
    uint32_t startLoop = blockIdx_ * loopsPerCore;
    uint32_t endLoop = (startLoop + loopsPerCore < loopTimes_) ? (startLoop + loopsPerCore) : loopTimes_;
    for (uint32_t loopId = startLoop; loopId < endLoop; ++loopId) {
        int64_t offset = static_cast<int64_t>(loopId) * static_cast<int64_t>(copyElemsPerLoop_);
        int64_t remain = totalElems_ - offset;
        if (remain <= 0) {
            continue;
        }
        uint32_t validElems = remain >= static_cast<int64_t>(copyElemsPerLoop_) ? copyElemsPerLoop_ :
                                                                                  static_cast<uint32_t>(remain);
        CopyValue(offset, validElems);
        CopyIndex(offset, validElems);
    }
}

} // namespace KthValue

#endif
