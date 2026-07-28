/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been technically reviewed for functional accuracy.
 */

/*!
 * \file l2_normalize_grad_empty.h
 * \brief L2NormalizeGrad empty-tensor kernel (TilingKey 8000).
 *
 * When x.numel() == 0 the output dx is also empty; there is nothing to compute or write.
 * This is a no-op kernel so the empty case does not fall through the reduction tiling.
 */
#ifndef L2_NORMALIZE_GRAD_EMPTY_H
#define L2_NORMALIZE_GRAD_EMPTY_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "l2_normalize_grad_tiling_data.h"

namespace L2NormalizeGrad {
using namespace AscendC;

class L2NormalizeGradEmpty {
public:
    __aicore__ inline L2NormalizeGradEmpty(TPipe* pipe, const L2NormalizeGradTilingData* tilingData) {}
    __aicore__ inline void Init(__gm__ uint8_t* dx) {}
    __aicore__ inline void Process() {}
};
} // namespace L2NormalizeGrad
#endif // L2_NORMALIZE_GRAD_EMPTY_H
