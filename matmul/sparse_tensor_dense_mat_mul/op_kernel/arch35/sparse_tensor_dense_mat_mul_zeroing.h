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
 * \file sparse_tensor_dense_mat_mul_zeroing.h
 * \brief
 */
#ifndef SPARSE_TENSOR_DENSE_MAT_MUL_ZEROING_H
#define SPARSE_TENSOR_DENSE_MAT_MUL_ZEROING_H

#include "sparse_tensor_dense_mat_mul_base.h"

namespace SparseTensorDenseMatMul {
using namespace AscendC;

template <typename T_VAL>
class SparseTensorDenseMatMulZeroing {
public:
    __aicore__ inline SparseTensorDenseMatMulZeroing(TPipe* pipe, const SparseTensorDenseMatMulTilingData* tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void InitAndProcessZeroing(GM_ADDR y);

private:
    TPipe* pipe_{nullptr};
    const SparseTensorDenseMatMulTilingData* tilingData_;
};

template <typename T_VAL>
__aicore__ inline void SparseTensorDenseMatMulZeroing<T_VAL>::InitAndProcessZeroing(GM_ADDR y)
{
    uint32_t currCoreIdx = static_cast<uint32_t>(GetBlockIdx());
    GlobalTensor<T_VAL> yOutputGm;
    yOutputGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ T_VAL*>(y) + currCoreIdx * tilingData_->initAndOutFormerCoreElemNum);

    if (currCoreIdx < tilingData_->initAndOutUsedCoreNum) {
        uint64_t initEleNumPerCore = (currCoreIdx == tilingData_->initAndOutUsedCoreNum - 1) ?
                                         tilingData_->initAndOutTailCoreElemNum :
                                         tilingData_->initAndOutFormerCoreElemNum;
        InitGlobalMemory(yOutputGm, initEleNumPerCore, static_cast<T_VAL>(0));
    }
}

} // namespace SparseTensorDenseMatMul

#endif // SPARSE_TENSOR_DENSE_MAT_MUL_ZEROING_H
