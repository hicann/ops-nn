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
 * \file quantize.cpp
 * \brief Quantize kernel entry, ascend910b (DAV_2201) standard model.
 *
 * Per-binary dtypes come from compile-time macros DTYPE_X / DTYPE_SCALES / DTYPE_ZERO_POINTS / DTYPE_Y.
 * The schedule is selected at COMPILE time by the tiling-key template argument `perMode` (declared in
 * quantize_tiling_key.h): one binary is instantiated per perMode value, and the host selects it by composing
 * the matching key with GET_TPL_TILING_KEY. No runtime tiling-key branching.
 */

#include "quantize.h"
#include "quantize_tiling_key.h"

using namespace AscendC;

template <uint32_t perMode>
__global__ __aicore__ void quantize(GM_ADDR x, GM_ADDR scales, GM_ADDR zero_points, GM_ADDR y, GM_ADDR workspace,
                                    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if constexpr (perMode == QUANTIZE_PER_TENSOR) {
        QuantizeOp::QuantizePerTensor<DTYPE_X, DTYPE_SCALES, DTYPE_ZERO_POINTS, DTYPE_Y> op;
        op.Init(x, scales, zero_points, y, &tilingData);
        op.Process();
    } else {
        QuantizeOp::QuantizePerChannel<DTYPE_X, DTYPE_SCALES, DTYPE_ZERO_POINTS, DTYPE_Y> op;
        op.Init(x, scales, zero_points, y, &tilingData);
        op.Process();
    }
}
