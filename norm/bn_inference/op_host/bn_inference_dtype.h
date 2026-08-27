/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN_INFERENCE_DTYPE_H
#define BN_INFERENCE_DTYPE_H

#include <array>
#include "graph/types.h"

namespace BNInferenceSupport {
struct DtypeCombination {
    ge::DataType x = ge::DT_UNDEFINED;
    ge::DataType statistics = ge::DT_UNDEFINED;
    ge::DataType momentum = ge::DT_UNDEFINED;
    ge::DataType affine = ge::DT_UNDEFINED;

    constexpr DtypeCombination(ge::DataType xValue = ge::DT_UNDEFINED, ge::DataType statisticsValue = ge::DT_UNDEFINED,
                               ge::DataType momentumValue = ge::DT_UNDEFINED,
                               ge::DataType affineValue = ge::DT_UNDEFINED)
        : x(xValue), statistics(statisticsValue), momentum(momentumValue), affine(affineValue)
    {}
};

// Ascend 950 verified joint dtype rows, including the public same-dtype baseline. This is not a Cartesian product.
static const std::array<DtypeCombination, 11> DTYPE_COMBINATIONS = {{
    DtypeCombination(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT),
    DtypeCombination(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16),
    DtypeCombination(ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16),
    DtypeCombination(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16),
    DtypeCombination(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16),
    DtypeCombination(ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16),
    DtypeCombination(ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16),
    DtypeCombination(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT),
    DtypeCombination(ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT),
    DtypeCombination(ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16),
    DtypeCombination(ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16),
}};

// Map canndev's private layout families to public layouts on Ascend 950 and preserve public ND.
// Private NC1HWC0/NDC1HWC0 formats are intentionally not exposed by this implementation.
static const std::array<ge::Format, 5> FEATURE_FORMATS = {
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_ND,
};
} // namespace BNInferenceSupport

#endif // BN_INFERENCE_DTYPE_H
