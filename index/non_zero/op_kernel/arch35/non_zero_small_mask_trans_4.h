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
 * \file non_zero_small_mask_trans_4.h
 * \brief
 */

#ifndef CANN_NON_ZERO_SMALL_MASK_TRANS_4_H
#define CANN_NON_ZERO_SMALL_MASK_TRANS_4_H

#include "non_zero_base.h"

namespace NonZero {
using namespace AscendC;

template <typename T1, typename T2, int TILING_KEY>
class NonZeroSmallMask4 : public NonZeroBase<T1, T2, TILING_KEY> {
public:
    __aicore__ inline NonZeroSmallMask4(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR outShape, GM_ADDR workspace, const NonZeroTilingData* tilingData);
    __aicore__ inline void Process();
};

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroSmallMask4<T1, T2, TILING_KEY>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR outShape, GM_ADDR workspace, const NonZeroTilingData* tilingData)
{
    this->InitBase(x, y, outShape, workspace, tilingData);
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroSmallMask4<T1, T2, TILING_KEY>::Process()
{
    this->BaseProcess();
} // namespace NonZero
} // namespace NonZero
#endif // CANN_RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_ND_SMALL_C_H
