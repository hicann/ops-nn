/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantized_batch_norm_base.h
 * \brief
 */

#ifndef QUANTIZED_BATCH_NORM_BASE_H
#define QUANTIZED_BATCH_NORM_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace QuantizedBatchNormOps {
using namespace AscendC;

template <typename T1, typename T2>
class QuantizedBatchNormBase {
public:
    __aicore__ inline QuantizedBatchNormBase()
    {}

protected:
    /* global memory address */
    GlobalTensor<T1> xGm;
    GlobalTensor<T2> weightGm;
    GlobalTensor<T2> biasGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<float> varGm;
    GlobalTensor<float> inScaleGm;
    GlobalTensor<int32_t> inZeroPointGm;
    GlobalTensor<float> outScaleGm;
    GlobalTensor<int32_t> outZeroPointGm;

    GlobalTensor<T1> yGm;

    /* variable */
    float epsilon = 1e-5;

    /* ascendc variable */
    TPipe* pipe_ = nullptr;
    uint32_t blockIdx = GetBlockIdx();
    uint32_t useCoreNum = GetBlockNum();
    // 公共函数声明
};
// 公共函数实现

} // namespace QuantizedBatchNormOps
#endif // QUANTIZED_BATCH_NORM_BASE_H