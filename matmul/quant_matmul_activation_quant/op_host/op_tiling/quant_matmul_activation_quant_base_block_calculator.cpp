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
 * \file quant_matmul_activation_quant_base_block_calculator.cpp
 * \brief Base-block alignment for GELU and paired-column SwiGLU MX quantization.
 */
#include "quant_matmul_activation_quant_base_block_calculator.h"

#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/quant_batch_matmul_v3_tiling_util.h"
#include "../quant_matmul_activation_quant_host_utils.h"

namespace optiling {

using QuantMatmulActivationQuantTilingConstant::GELU_BASEN_ALIGN;
using QuantMatmulActivationQuantTilingConstant::SWIGLU_BASEN_ALIGN;

QuantBaseBlockCalculator::QuantBaseBlockCalculator(const QuantBatchMatmulInfo& inputParams,
                                                   const QuantBatchMatmulV3CompileInfo& compileInfo,
                                                   uint64_t batchCoreCnt, bool isSwiglu)
    : BaseBlockCalculator(inputParams, compileInfo, batchCoreCnt), isSwiglu_(isSwiglu)
{}

uint64_t QuantBaseBlockCalculator::GetBaseNAlignSize(uint64_t innerAlignSize) const
{
    if (isSwiglu_) {
        return SWIGLU_BASEN_ALIGN;
    }
    return this->inputParams_.transB ? GELU_BASEN_ALIGN :
                                       GetShapeWithDataType(innerAlignSize, this->inputParams_.bDtype);
}

} // namespace optiling
