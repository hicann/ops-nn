/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file quant_matmul_activation_quant.cpp
 * \brief
 */
#pragma once
#include "quant_matmul_activation_quant_tiling_data.h"
#include "quant_matmul_activation_quant_tiling_key.h"
#include "quant_matmul_activation_quant.h"
#include "tensor_api/tensor.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
using namespace AscendC;

template <int8_t A_TRANS, int8_t B_TRANS, uint64_t FULL_LOAD_MODE = 0, int8_t B_FORMAT = 1>
__global__ __aicore__ void quant_matmul_activation_quant(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale,
                                                         GM_ADDR x2_scale, GM_ADDR y, GM_ADDR y_scale,
                                                         GM_ADDR workspace, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(QMMAQ::QuantMatmulActivationQuantTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    constexpr bool aTran = (A_TRANS == 1);
    constexpr bool bTran = (B_TRANS == 1);
    constexpr bool bIsNz = (B_FORMAT == 1);
    using layoutA = AscendC::Std::conditional_t<aTran, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using layoutB_NZ = AscendC::Std::conditional_t<bTran, AscendC::Te::ZNLayoutPtn, AscendC::Te::NZLayoutPtn>;
    using layoutB_ND = AscendC::Std::conditional_t<bTran, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using layoutB = AscendC::Std::conditional_t<bIsNz, layoutB_NZ, layoutB_ND>;

    QuantMatmulActivationQuantKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, layoutA, layoutB, AscendC::Te::NDExtLayoutPtn,
                                     FULL_LOAD_MODE>(x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace,
                                                     &tilingData);
}
