/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "situ_glu_grad.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SituGluGrad);

const aclTensor* SituGluGrad(const aclTensor* gradY, const aclTensor* x, int64_t dim, float beta, float linearBeta,
                             bool activateLeft, aclOpExecutor* executor)
{
    L0_DFX(SituGluGrad, gradY, x, dim, beta, linearBeta, activateLeft);

    // 计算结果写入连续临时tensor，shape与x一致，由ViewCopy按out的视图信息回写
    auto gradX = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), Format::FORMAT_ND);
    if (gradX == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc gradX tensor failed.");
        return nullptr;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SituGluGrad, OP_INPUT(gradY, x), OP_OUTPUT(gradX),
                                           OP_ATTR(dim, beta, linearBeta, activateLeft));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "SituGluGrad add to aicore launch list failed.");

    return gradX;
}

} // namespace l0op
