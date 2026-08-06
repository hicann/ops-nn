/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_emu_split_weight.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(MatmulEmuSplitWeight);

const aclTensor* MatmulEmuSplitWeight(const aclTensor* x, const aclTensor* wHigh, const aclTensor* wLow,
                                      op::DataType outDtype, float wLowScale, int32_t yDtype, bool transposeX,
                                      bool transposeW, aclOpExecutor* executor)
{
    L0_DFX(MatmulEmuSplitWeight, x, wHigh, wLow, outDtype, wLowScale, yDtype, transposeX, transposeW);

    auto out = executor->AllocTensor(outDtype, Format::FORMAT_ND, Format::FORMAT_ND);
    OP_CHECK_NULL(out, return nullptr);

    auto ret = INFER_SHAPE(MatmulEmuSplitWeight, OP_INPUT(x, wHigh, wLow), OP_OUTPUT(out),
                           OP_ATTR(wLowScale, transposeX, transposeW, yDtype));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
        return nullptr;
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(MatmulEmuSplitWeight, OP_INPUT(x, wHigh, wLow), OP_OUTPUT(out),
                                      OP_ATTR(wLowScale, transposeX, transposeW, yDtype));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "MatmulEmuSplitWeight ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}

} // namespace l0op
