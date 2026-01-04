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
 * \file non_zero_null.h
 * \brief
 */
#ifndef NON_ZERO_NULL_H
#define NON_ZERO_NULL_H

#include "non_zero_base.h"

namespace NonZero {
using namespace AscendC;
template <typename T1, typename T2>
class NonZeroNullInputTensor {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR outShape, GM_ADDR workspace, const NonZeroTilingData* tilingData);
    __aicore__ inline void Process();

private:
    const NonZeroTilingData* tilingData_;
    TPipe pipe_;
    uint32_t blockIdx_ = 0;
    GlobalTensor<uint64_t> shapeGm_;
    TBuf<QuePosition::VECCALC> addUb_;
};

template <typename T1, typename T2>
__aicore__ inline void NonZeroNullInputTensor<T1, T2>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR outShape, GM_ADDR workspace, const NonZeroTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    shapeGm_.SetGlobalBuffer((__gm__ uint64_t*)outShape);
    tilingData_ = tilingData;
    pipe_.InitBuffer(addUb_, ADD_UB_SIZE);
}
template <typename T1, typename T2>
__aicore__ inline void NonZeroNullInputTensor<T1, T2>::Process()
{
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }
    LocalTensor<uint64_t> tensorNew = addUb_.Get<uint64_t>();
    if (tilingData_->needTranspose) {
        tensorNew.SetValue(DIM16, NON_ZERO_OUTSHAPE_DIM_BASE);
        tensorNew.SetValue(DIM17, 0);
        tensorNew.SetValue(DIM18, tilingData_->inputDims);
    } else {
        tensorNew.SetValue(DIM16, NON_ZERO_OUTSHAPE_DIM_BASE);
        tensorNew.SetValue(DIM17, tilingData_->inputDims);
        tensorNew.SetValue(DIM18, 0);
    }
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    DataCopyPad(shapeGm_, tensorNew[DIM16], {1, 24, 0, 0});

    return;
}
} // namespace NonZero
#endif