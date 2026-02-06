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
 * \file gather_v2_empty.h
 * \brief
 */
#ifndef GATHER_V2_EMPTY
#define GATHER_V2_EMPTY

#include "basic_api/kernel_vec_intf.h"
#include "op_kernel/platform_util.h"

namespace gatherv2 {
using namespace AscendC;

template <typename T>
class Gatherv2Empty
{
public:
    __aicore__ inline Gatherv2Empty(){};
    __aicore__ inline void Init(GM_ADDR y, const GatherV2TilingDataEmptyInput* tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> yGm_;

    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t lastCoreElements_;

    int64_t extentElements_;
    uint32_t blockIdx_;
    const GatherV2TilingDataEmptyInput* tilingData_;
};

template <typename T>
__aicore__ inline void Gatherv2Empty<T>::Init(GM_ADDR y, const GatherV2TilingDataEmptyInput* tilingData)
{
    tilingData_ = tilingData;
    yGm_.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T>
__aicore__ inline void Gatherv2Empty<T>::Process()
{
    needCoreNum_ = tilingData_->needCoreNum;
    perCoreElements_ = tilingData_->perCoreElements;
    lastCoreElements_ = tilingData_->lastCoreElements;
    extentElements_ = perCoreElements_;

    blockIdx_ = GetBlockIdx();
    if (blockIdx_ + 1 == (uint32_t)needCoreNum_) {
        extentElements_ = lastCoreElements_;
    }

    InitOutput<T>(yGm_, extentElements_, 0);
}
} // namespace gatherv2
#endif // GATHER_V2_EMPTY
