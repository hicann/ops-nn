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
 * \file scatter_elements_two_dims.h
 * \brief 完成二维，且在尾轴上的scatter_elements操作。T: float half, int32_t，不需要转置的场景只有bool能进来
 */

#ifndef SCATTER_ELEMENTS_TWO_DIMS_H
#define SCATTER_ELEMENTS_TWO_DIMS_H
#include "scatter_elements_v2_cache_base.h"
#include "scatter_elements_v2_cache_scatter_add.h"
#include "scatter_elements_v2_cache_scatter.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
using namespace std;

template <typename T, typename U>
class ScatterElementsTwoDims {
public:
    __aicore__ inline ScatterElementsTwoDims() {}

    __aicore__ inline void Init(GlobalTensor<T>& x, GlobalTensor<U>& indices,
                                GlobalTensor<T>& updates, LocalTensor<uint8_t>& allUbLocal) {
        this->xGm = x;
        this->indicesGm = indices;
        this->updatesGm = updates;
        this->allUbLocal = allUbLocal;
    }

    __aicore__ inline void SetMode(int32_t mode) {
        this->mode = mode;
    }

    __aicore__ inline void SetUpdatesIsScalar(int32_t updatesIsScalar) {
        this->updatesIsScalar = updatesIsScalar;
    }

    SCATTER_ELEMENTS_SET_INFO_METHODS()

    __aicore__ inline void Process() {
        if (this->xDim1 < X_LOCAL_LENGTH) {
            this->DoCacheScatterElements();
        }
    }

#define EXECUTE_SCATTER_OP(OpClass) \
    do { \
        OpClass<T, U> op; \
        op.Init(this->xGm, this->indicesGm, this->updatesGm, this->allUbLocal); \
        op.SetXInfo(this->xDim0, this->xDim1); \
        op.SetIndicesInfo(this->indicesDim0, this->indicesDim1); \
        op.SetUpdatesInfo(this->updatesDim0, this->updatesDim1); \
        op.SetCoreNums(this->coreNums); \
        op.Process(); \
    } while(0)

private:
    __aicore__ inline void DoCacheScatterElements() {
        if (this->mode == 0 && this->updatesIsScalar) {
            EXECUTE_SCATTER_OP(CacheXScatterValue);
        }
        if (this->mode == 1 && this->updatesIsScalar) {
            EXECUTE_SCATTER_OP(CacheXScatterAddValue);
        }
        if (this->mode == 0 && !this->updatesIsScalar) {
            EXECUTE_SCATTER_OP(CacheXScatter);
        }
        if (this->mode == 1 && !this->updatesIsScalar) {
            EXECUTE_SCATTER_OP(CacheXScatterAdd);
        }
    }

private:
    LocalTensor<uint8_t> allUbLocal;
    GlobalTensor<U> indicesGm;
    GlobalTensor<T> updatesGm;
    GlobalTensor<T> xGm;
    int32_t mode = 0;
    int32_t coreNums = 0;
    bool updatesIsScalar = false;
    uint64_t xDim0 = 0;
    uint64_t xDim1 = 0;
    uint64_t indicesDim0 = 0;
    uint64_t indicesDim1 = 0;
    uint64_t updatesDim0 = 0;
    uint64_t updatesDim1 = 0;
};
}
#endif