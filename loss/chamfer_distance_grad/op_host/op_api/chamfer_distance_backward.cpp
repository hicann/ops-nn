/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "chamfer_distance_backward.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChamferDistanceGrad);

const std::tuple<aclTensor*, aclTensor*> ChamferDistanceGrad(
    const aclTensor* xyz1, const aclTensor* xyz2, const aclTensor* idx1, const aclTensor* idx2,
    const aclTensor* gradDist1, const aclTensor* gradDist2, aclOpExecutor* executor)
{
    L0_DFX(ChamferDistanceGrad, xyz1, xyz2, idx1, idx2, gradDist1, gradDist2);
    auto gradXyz1 = executor->AllocTensor(xyz1->GetViewShape(), xyz1->GetDataType(), op::Format::FORMAT_ND);
    auto gradXyz2 = executor->AllocTensor(xyz2->GetViewShape(), xyz2->GetDataType(), op::Format::FORMAT_ND);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChamferDistanceGrad, OP_INPUT(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2), OP_OUTPUT(gradXyz1, gradXyz2));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ChamferDistanceGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tie(gradXyz1, gradXyz2);
}
} // namespace l0op