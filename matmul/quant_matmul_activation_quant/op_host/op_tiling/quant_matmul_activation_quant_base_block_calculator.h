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
 * \file quant_matmul_activation_quant_base_block_calculator.h
 * \brief Base-block alignment for GELU and paired-column SwiGLU MX quantization.
 */
#pragma once

#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/base_block_calculator.h"

namespace optiling {

class QuantBaseBlockCalculator : public BaseBlockCalculator {
public:
    QuantBaseBlockCalculator(const QuantBatchMatmulInfo& inputParams, const QuantBatchMatmulV3CompileInfo& compileInfo,
                             uint64_t batchCoreCnt = 1UL, bool isSwiglu = false);
    ~QuantBaseBlockCalculator() override = default;

protected:
    uint64_t GetBaseNAlignSize(uint64_t innerAlignSize) const override;

private:
    bool isSwiglu_;
};

} // namespace optiling
