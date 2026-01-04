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
 * \file quant_batch_matmul_v3_checker_base.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_V3_CHECKER_BASE_H
#define QUANT_BATCH_MATMUL_V3_CHECKER_BASE_H
#include "../quant_batch_matmul_v3_tiling_base.h"

namespace optiling {

class QuantBatchMatmulV3CheckerBase {
public:
    explicit QuantBatchMatmulV3CheckerBase(gert::TilingContext *context, const QuantBatchMatmulInfo &inputParams)
        : context_(context), inputParams_(inputParams)
    {
    }

    virtual ~QuantBatchMatmulV3CheckerBase() = default;
    virtual bool CheckDtype() const
    {
        return false;
    }
    virtual bool CheckShape(
        const std::vector<gert::Shape*>& /* mandtoryShape */, const gert::StorageShape* /* biasShape */,
        const gert::StorageShape* /* pertokenShape */, const std::vector<int64_t>& /* dimValueOfMKN */) const
    {
        return false;
    }

protected:
    gert::TilingContext *context_ = nullptr;
    QuantBatchMatmulInfo inputParams_;

    // input index
    constexpr static uint32_t X1_INDEX = 0;
    constexpr static uint32_t X2_INDEX = 1;
    constexpr static uint32_t SCALE_INDEX = 2;
    constexpr static uint32_t OFFSET_INDEX = 3;
    constexpr static uint32_t BIAS_INDEX = 4;
    constexpr static uint32_t PERTOKEN_SCALE_INDEX = 5;

    // 根据V3/V4原型获取输入index
    virtual uint32_t GetX1Idx() const
    {
        return X1_INDEX;
    }
    virtual uint32_t GetX2Idx() const
    {
        return X2_INDEX;
    }
    virtual uint32_t GetScaleIdx() const
    {
        return SCALE_INDEX;
    }
    virtual uint32_t GetOffsetIdx() const
    {
        return OFFSET_INDEX;
    }
    virtual uint32_t GetBiasIdx() const
    {
        return BIAS_INDEX;
    }
    virtual uint32_t GetPertokenIdx() const
    {
        return PERTOKEN_SCALE_INDEX;
    }
};
}  // namespace optiling
#endif  // QUANT_BATCH_MATMUL_V3_CHECKER_BASE_H