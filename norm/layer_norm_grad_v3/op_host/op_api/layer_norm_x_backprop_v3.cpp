/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "layer_norm_x_backprop_v3.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LayerNormXBackpropV3);

const std::array<aclTensor*, X_V3_OUT_NUM> LayerNormXBackpropV3(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* rstd, const aclTensor* mean,
    const aclTensor* weight, aclOpExecutor* executor)
{
    L0_DFX(LayerNormXBackpropV3, gradOut, input, rstd, mean, weight);
    auto gradInputOut = executor->AllocTensor(input->GetViewShape(), input->GetDataType(), Format::FORMAT_ND);
    // kernel only support second output dtype float32
    auto gammaTempOut = executor->AllocTensor(input->GetViewShape(), DataType::DT_FLOAT, Format::FORMAT_ND);

    ADD_TO_LAUNCHER_LIST_AICORE(
        LayerNormXBackpropV3, OP_INPUT(gradOut, input, rstd, mean, weight), OP_OUTPUT(gradInputOut, gammaTempOut));

    return {gradInputOut, gammaTempOut};
}

} // namespace l0op
