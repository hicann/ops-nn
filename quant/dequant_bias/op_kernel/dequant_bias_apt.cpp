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
 * \file dequant_bias_apt.cpp
 * \brief DequantBias 算子 kernel 入口 (arch35 架构, apt 选编)
 *
 * 模板参数:
 *   HAS_BIAS:     0=无bias, 1=有bias
 *   HAS_ACTIVATE: 0=无activate_scale, 1=有activate_scale
 *   DTYPE_BIAS_ID: bias 类型ID (0=fp32, 1=fp16, 2=bf16, 3=int32)
 */

#include "arch35/dequant_bias_tiling_key_arch35.h"
#include "arch35/dequant_bias_kernel.h"

/* Bias type selector from tiling key parameter */
template <uint64_t ID>
struct BiasTypeSelector {};
template <> struct BiasTypeSelector<TPL_BIAS_FLOAT> { using type = float; };
template <> struct BiasTypeSelector<TPL_BIAS_HALF> { using type = half; };
template <> struct BiasTypeSelector<TPL_BIAS_BFLOAT> { using type = bfloat16_t; };
template <> struct BiasTypeSelector<TPL_BIAS_INT> { using type = int32_t; };

template <uint64_t HAS_BIAS, uint64_t HAS_ACTIVATE, uint64_t DTYPE_BIAS_ID>
__global__ __aicore__ void dequant_bias(
    GM_ADDR x, GM_ADDR weight_scale, GM_ADDR activate_scale,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DequantBiasArch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(DequantBiasArch35TilingData, tilingData, tiling);

    using DTYPE_B = typename BiasTypeSelector<DTYPE_BIAS_ID>::type;
    NsDequantBias::DequantBiasKernel<int32_t, DTYPE_WEIGHT_SCALE, DTYPE_B, DTYPE_Y,
                                     HAS_BIAS, HAS_ACTIVATE> op;
    op.Init(x, weight_scale, activate_scale, bias, y, tiling);
    op.Process();
}
