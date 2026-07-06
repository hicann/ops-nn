/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file average_pooling_grad.cpp
 * \brief AveragePoolingGrad kernel entry.
 */

#include "average_pooling_grad.h"
#if defined(ORIG_DTYPE_GRAD_OUTPUT) && ORIG_DTYPE_GRAD_OUTPUT == DT_FLOAT && !defined(__CCE_KT_TEST__)
#include "average_pooling_grad_float_tiled.h"
#include "average_pooling_grad_float_no_overlap.h"
#endif

__global__ __aicore__ void average_pooling_grad(GM_ADDR orig_input_shape, GM_ADDR grad_output, GM_ADDR grad_input,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    (void)orig_input_shape;
    REGISTER_TILING_DEFAULT(AveragePoolingGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(AveragePoolingGradTilingData, tilingData, tiling);

#if defined(ORIG_DTYPE_GRAD_OUTPUT) && ORIG_DTYPE_GRAD_OUTPUT == DT_FLOAT && !defined(__CCE_KT_TEST__)
    if (TILING_KEY_IS(1)) {
        NsAveragePoolingGrad::AveragePoolingGradFloatTiled op;
        op.Init(orig_input_shape, grad_output, grad_input, tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(2)) {
        NsAveragePoolingGrad::AveragePoolingGradFloatNoOverlap op;
        op.Init(orig_input_shape, grad_output, grad_input, tilingData);
        op.Process();
        return;
    }
#endif

    NsAveragePoolingGrad::AveragePoolingGrad<DTYPE_GRAD_OUTPUT> op;
    op.Init(grad_output, grad_input, &tilingData);
    op.Process();
}
