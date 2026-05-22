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
 * \file l1_tiling_data_calculator.h
 * \brief
 */
#pragma once

#include <cstdint>

#include "../quant_batch_matmul_v3_tiling_base.h"

namespace optiling {

struct L1TilingData {
    uint64_t depthKa_ = 0UL;
    uint64_t depthKb_ = 0UL;
    uint64_t stepKa_ = 0UL;
    uint64_t stepKb_ = 0UL;
    uint64_t scaleFactorA_ = 1UL;
    uint64_t scaleFactorB_ = 1UL;
};

enum class L1TilingMode
{
    DEFAULT = 0,
    A_L1_FULL_LOAD
};

class L1TilingDataCalculator {
public:
    L1TilingDataCalculator(
        const QuantBatchMatmulInfo& inputParams, const QuantBatchMatmulV3CompileInfo& compileInfo, uint64_t baseM,
        uint64_t baseN, uint64_t baseK);
    virtual ~L1TilingDataCalculator() = default;

    bool Compute(L1TilingMode mode);
    const L1TilingData& GetOutput() const;

private:
    bool ValidateInput() const;
    bool InitLeftL1Size();
    bool ComputeL1TilingDefault();
    bool ComputeL1TilingAL1FullLoad();
    bool CalStepKs();
    bool CalScaleFactors(uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleASize, uint64_t baseScaleBSize);
    uint64_t GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit) const;
    uint64_t GetDepthB1AfullLoad(uint64_t leftSize) const;
    uint64_t GetScaleFactorBAfullLoad(uint64_t leftSize) const;

    const QuantBatchMatmulInfo& inputParams_;
    const QuantBatchMatmulV3CompileInfo& compileInfo_;
    uint64_t baseM_ = 0UL;
    uint64_t baseN_ = 0UL;
    uint64_t baseK_ = 0UL;
    uint64_t leftL1Size_ = 0UL;
    L1TilingData l1TilingData_;
};

} // namespace optiling
