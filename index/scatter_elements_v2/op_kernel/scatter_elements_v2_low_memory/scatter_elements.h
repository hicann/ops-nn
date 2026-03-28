/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements.h
 * \brief 完成二维，且在尾轴上的scatter_elements操作
 */

#ifndef SCATTER_ELEMENTS_H
#define SCATTER_ELEMENTS_H
#include "scatter_elements_cache.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
using namespace std;

template <typename T, typename U, const uint32_t MODE, const bool IsScalar>
class ScatterElements {
public:
    __aicore__ inline ScatterElements() {}

    __aicore__ inline void Init(GlobalTensor<T>& x, GlobalTensor<U>& indices,
                                GlobalTensor<T>& updates, LocalTensor<uint8_t>& allUbLocal, GM_ADDR workspace) {
        this->xGm = x;
        this->indicesGm = indices;
        this->updatesGm = updates;
        this->allUbLocal = allUbLocal;
        this->workspace = workspace;
    }

    __aicore__ inline void SetXInfo(uint64_t xDim0, uint64_t xDim1) {
        this->xDim0 = xDim0;
        this->xDim1 = xDim1;
    }

    __aicore__ inline void SetCoreNums(int32_t coreNums) {
        this->coreNums = coreNums;
    }

    __aicore__ inline void SetIndicesInfo(uint64_t indicesDim0, uint64_t indicesDim1) {
        this->indicesDim0 = indicesDim0;
        this->indicesDim1 = indicesDim1;
    }

    __aicore__ inline void SetUpdatesInfo(uint64_t updatesDim0, uint64_t updatesDim1) {
        this->updatesDim0 = updatesDim0;
        this->updatesDim1 = updatesDim1;
    }

    __aicore__ inline void Process() {
        if (this->xDim1 < X_LOCAL_LENGTH) {
            if constexpr ((std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) && MODE == 1) {
                ScatterElementsCacheOp<T, U, float, MODE, IsScalar> op;
                op.Init(this->xGm, this->indicesGm, this->updatesGm, this->allUbLocal, this->workspace);
                op.SetXInfo(this->xDim0, this->xDim1);
                op.SetIndicesInfo(this->indicesDim0, this->indicesDim1);
                op.SetUpdatesInfo(this->updatesDim0, this->updatesDim1);
                op.SetCoreNums(this->coreNums);
                op.Process();
            } else {
                ScatterElementsCacheOp<T, U, T, MODE, IsScalar> op;
                op.Init(this->xGm, this->indicesGm, this->updatesGm, this->allUbLocal, this->workspace);
                op.SetXInfo(this->xDim0, this->xDim1);
                op.SetIndicesInfo(this->indicesDim0, this->indicesDim1);
                op.SetUpdatesInfo(this->updatesDim0, this->updatesDim1);
                op.SetCoreNums(this->coreNums);
                op.Process();
            }
        }
    }

private:
    LocalTensor<uint8_t> allUbLocal;
    GlobalTensor<U> indicesGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> updatesGm;
    GM_ADDR workspace;

    int32_t coreNums = 0;
    uint64_t xDim0 = 0;
    uint64_t xDim1 = 0;
    uint64_t updatesDim0 = 0;
    uint64_t updatesDim1 = 0;
    uint64_t indicesDim0 = 0;
    uint64_t indicesDim1 = 0;
};
}
#endif