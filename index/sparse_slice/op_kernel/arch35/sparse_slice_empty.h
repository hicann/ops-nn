/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sparse_slice_empty.h
 * \brief
 */

#ifndef SPARSE_SLICE_EMPTY_H
#define SPARSE_SLICE_EMPTY_H

#include "sparse_slice_common.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"

namespace SparseSlice {
using namespace AscendC;

template <typename T>
class SparseSliceEmpty : public SparseSliceBase
{
public:
    __aicore__ inline SparseSliceEmpty() {};
    __aicore__ inline void Init(GM_ADDR indices, GM_ADDR values,
                               GM_ADDR shape, GM_ADDR start, GM_ADDR size,
                               GM_ADDR yIndices, GM_ADDR yValues, GM_ADDR yShape,
                               GM_ADDR outputShape1,
                               GM_ADDR workspace, const SparseSliceTilingData& tilingData, TPipe *inPipe);
    __aicore__ inline void Process();

private:
    TPipe *pipe;
    TBuf<QuePosition::VECCALC> yIndicesTmpUb_;
    TBuf<QuePosition::VECCALC> yValuesTmpUb_;
    TBuf<QuePosition::VECCALC> yShapeTmpUb_;
    TBuf<QuePosition::VECCALC> yShapeShapeTmpUb_;
    GlobalTensor<int64_t> yIndicesGm_;
    GlobalTensor<T> yValuesGm_;
    GlobalTensor<int64_t> yShapeGm_;
    GlobalTensor<int64_t> outputShape1Gm_;
};

template <typename T>
__aicore__ inline void SparseSliceEmpty<T>::Init(
    GM_ADDR indices, GM_ADDR values, GM_ADDR shape, GM_ADDR start, GM_ADDR size,
    GM_ADDR yIndices, GM_ADDR yValues, GM_ADDR yShape,
    GM_ADDR outputShape1,
    GM_ADDR workspace, const SparseSliceTilingData& tilingData, TPipe *inPipe)
{
    pipe = inPipe;
    blockIdx_ = GetBlockIdx();
    SparseSliceBase::ParseTilingData(tilingData);
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    
    yIndicesGm_.SetGlobalBuffer((__gm__ int64_t*)(yIndices), 0);
    yValuesGm_.SetGlobalBuffer((__gm__ T*)(yValues), 0);
    yShapeGm_.SetGlobalBuffer((__gm__ int64_t*)(yShape), rankNumbers_);
    outputShape1Gm_.SetGlobalBuffer((__gm__ int64_t*)(outputShape1), DIGIT_NINE * DIGIT_THREE);
    pipe->InitBuffer(yShapeTmpUb_, rankNumbers_ * sizeof(int64_t));
    pipe->InitBuffer(yIndicesTmpUb_, EMPTY_RESERVED * sizeof(int64_t));
    pipe->InitBuffer(yValuesTmpUb_, EMPTY_RESERVED * sizeof(int64_t));
    pipe->InitBuffer(yShapeShapeTmpUb_, EMPTY_RESERVED * sizeof(int64_t));
}

template <typename T>
__aicore__ inline void SparseSliceEmpty<T>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    LocalTensor<int64_t> yShapeTensor = yShapeTmpUb_.Get<int64_t>();
    for (int32_t i = 0; i < rankNumbers_; i++) {
        yShapeTensor.SetValue(i, yShape_[i]);
    }

    event_t eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventId1);
    WaitFlag<HardEvent::S_MTE3>(eventId1);
    DataCopyPad(yShapeGm_[0], yShapeTensor[0], {1, static_cast<uint16_t>(rankNumbers_ * sizeof(int64_t)), 0, 0});

    LocalTensor<int64_t> outputIndicesShapeTensor = yIndicesTmpUb_.Get<int64_t>();
    // 此处输出y_indices的shape为[0, rankNumbers_]
    outputIndicesShapeTensor.SetValue(DIGIT_ZERO, Y_INDICES_SHAPE_DIM_BASE); // 输出shape的维度
    outputIndicesShapeTensor.SetValue(DIGIT_ONE, DIGIT_ZERO); // 输出的第一个维度
    outputIndicesShapeTensor.SetValue(DIGIT_TWO, rankNumbers_); // 输出的第二个维度
    event_t eventId2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventId2);
    WaitFlag<HardEvent::S_MTE3>(eventId2);
    DataCopyPad(outputShape1Gm_[0], outputIndicesShapeTensor[0], {1, static_cast<uint16_t>(DIGIT_THREE * sizeof(int64_t)), 0, 0});

    LocalTensor<int64_t> outputValuesShapeTensor = yValuesTmpUb_.Get<int64_t>();
    // 此处输出y_values的shape为[0]
    outputValuesShapeTensor.SetValue(DIGIT_ZERO, DIGIT_ONE);
    outputValuesShapeTensor.SetValue(DIGIT_ONE, DIGIT_ZERO);
    event_t eventId3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventId3);
    WaitFlag<HardEvent::S_MTE3>(eventId3);
    DataCopyPad(outputShape1Gm_[DIGIT_NINE], outputValuesShapeTensor[0], {1, static_cast<uint16_t>(DIGIT_TWO * sizeof(int64_t)), 0, 0});

    LocalTensor<int64_t> outputShapeShapeTensor = yShapeShapeTmpUb_.Get<int64_t>();
    // 此处输出y_shape的shape为[rankNumbers_]
    outputShapeShapeTensor.SetValue(DIGIT_ZERO, DIGIT_ONE);
    outputShapeShapeTensor.SetValue(DIGIT_ONE, rankNumbers_);
    event_t eventId4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventId4);
    WaitFlag<HardEvent::S_MTE3>(eventId4);
    DataCopyPad(outputShape1Gm_[DIGIT_EIGHTEEN], outputShapeShapeTensor[0], {1, static_cast<uint16_t>(DIGIT_TWO * sizeof(int64_t)), 0, 0});
}

} //  namespace SparseSlice
#endif // SPARSE_SLICE_EMPTY_H