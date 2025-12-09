/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "layer_norm_beta_gamma_backprop_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LayerNormBetaGammaBackpropV2);

constexpr size_t BETA_GAMMAX_NUM = 2;

const std::array<aclTensor*, BETA_GAMMAX_NUM> LayerNormBetaGammaBackpropV2(
    const aclTensor* gradOut, const aclTensor* gammaRes, const aclIntArray* gammaShape, aclOpExecutor* executor)
{
    L0_DFX(LayerNormBetaGammaBackpropV2, gradOut, gammaRes, gammaShape);
    op::Shape gradShape = gradOut->GetViewShape();
    for (size_t index = 0; index < gradShape.GetDimNum() - gammaShape->Size(); index++) {
        gradShape[index] = 1;
    }
    auto gradWeightOut = executor->AllocTensor(gradShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto gradBiasOut = executor->AllocTensor(gradShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);

    // 当前算子的二进制匹配逻辑需要在l0op中处理
    constexpr int64_t MATCH_SHAPE = -2;
    FVector<int64_t> matchShape{MATCH_SHAPE};
    if (gradShape.GetDimNum() == gammaShape->Size()) {
        matchShape.emplace_back(MATCH_SHAPE);
    }
    aclIntArray* matchArray = executor->AllocIntArray(matchShape.data(), matchShape.size());
    ADD_TO_LAUNCHER_LIST_AICORE(
        LayerNormBetaGammaBackpropV2, OP_INPUT(gradOut, gammaRes), OP_OUTPUT(gradWeightOut, gradBiasOut),
        OP_ATTR(gammaShape, matchArray));

    return {gradWeightOut, gradBiasOut};
}

} // namespace l0op
