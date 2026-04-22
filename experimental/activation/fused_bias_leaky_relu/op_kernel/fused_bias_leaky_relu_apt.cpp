/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_bias_leaky_relu_apt.cpp
 * \brief FusedBiasLeakyRelu kernel entry (arch35)
 *
 * Template parameters (matching ASCENDC_TPL_ARGS_DECL in tiling_key.h):
 *   - D_T_X: Data type, from ASCENDC_TPL_DATATYPE_DECL
 *   - BUFFER_MODE: Buffer mode (0=single, 1=double), from ASCENDC_TPL_UINT_DECL
 */

#include "arch35/fused_bias_leaky_relu.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void fused_bias_leaky_relu(
    GM_ADDR x,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FusedBiasLeakyReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedBiasLeakyReluTilingData, tilingData, tiling);
    NsFusedBiasLeakyRelu::FusedBiasLeakyRelu<D_T_X, BUFFER_MODE> op;
    op.Init(x, bias, y, &tilingData);
    op.Process();
}
