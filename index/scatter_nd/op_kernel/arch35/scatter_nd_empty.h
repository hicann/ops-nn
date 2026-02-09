/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_empty.h
 * \brief scatterNd kernel for empty tensor
 */

#ifndef SCATTER_ND_EMPTY_H
#define SCATTER_ND_EMPTY_H

#include "kernel_operator.h"

namespace ScatterNdEmpty {
using namespace AscendC;

template<typename U>
class ScatterNdEmptyImpl {
public:
    __aicore__ inline ScatterNdEmptyImpl(const ScatterNdTilingData& tilingData) :
        tilingData_(tilingData) {};
    __aicore__ inline void Init(GM_ADDR y);

private:
    const ScatterNdTilingData& tilingData_;
    AscendC::GlobalTensor<U> yGmInit_;
};

template<typename U>
__aicore__ inline void ScatterNdEmptyImpl<U>::Init(GM_ADDR y)
{
    uint32_t coreNum = GetBlockNum();
    uint64_t initPerCore = tilingData_.outputSize / coreNum;
    uint64_t initTailCore = tilingData_.outputSize - (coreNum - 1) * initPerCore;
    uint64_t initCoreReal = GetBlockIdx() == (coreNum - 1) ? initTailCore : initPerCore;
    uint64_t yGmOffset = GetBlockIdx() * initPerCore;

    yGmInit_.SetGlobalBuffer((__gm__ U *)(y) + yGmOffset);
    InitGlobalMemory(yGmInit_, initCoreReal, static_cast<U>(0));
}

} //namespace ScatterNdEmpty

#endif //SCATTER_ND_EMPTY_H