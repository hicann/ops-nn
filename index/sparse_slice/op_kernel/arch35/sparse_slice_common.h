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
 * \file sparse_slice_common.h
 * \brief
 */

#ifndef SPARSE_SLICE_COMMON_H
#define SPARSE_SLICE_COMMON_H

#include "kernel_operator.h"
namespace SparseSlice {
using namespace AscendC;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t DIGIT_ZERO = 0;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_THREE = 3;
constexpr int64_t DIGIT_NINE = 9;
constexpr int64_t DIGIT_EIGHTEEN = 18;
constexpr int64_t DIGIT_TWENTY_FOUR = 24;
constexpr int64_t EMPTY_RESERVED = 256;
constexpr int64_t Y_INDICES_SHAPE_DIM_BASE = 0x80000002;

class SparseSliceBase {
public:
    __aicore__ inline SparseSliceBase(){};
    __aicore__ inline void ParseTilingData(const SparseSliceTilingData& tilingData);

    uint32_t blockIdx_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t valueNumbers_ = 0;
    int64_t rankNumbers_ = 0;
    int64_t valuePerUb_ = 0;
    int64_t valuePerCore_ = 0;
    int64_t valuePerTail_ = 0;
    int64_t yShape_[DIGIT_TWENTY_FOUR];
    int64_t sliceStart_[DIGIT_TWENTY_FOUR];
    int64_t sliceEnd_[DIGIT_TWENTY_FOUR];
};

__aicore__ inline void SparseSliceBase::ParseTilingData(const SparseSliceTilingData& tilingData)
{
    usedCoreNum_ = tilingData.usedCoreNum;
    valueNumbers_ = tilingData.valueNumbers;
    rankNumbers_ = tilingData.rankNumbers;
    valuePerUb_ = tilingData.valuePerUb;
    valuePerCore_ = tilingData.valuePerCore;
    valuePerTail_ = tilingData.valuePerTail;
    for (int32_t i = 0; i < rankNumbers_; i++) {
        yShape_[i] = tilingData.yShape[i];
        sliceStart_[i] = tilingData.sliceStart[i];
        sliceEnd_[i] = tilingData.sliceEnd[i];
    }
}

} // namespace SparseSlice
#endif // SPARSE_SLICE_COMMON_H