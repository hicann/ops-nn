/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_foreach_div_scalar.h"
#include "op_api_ut_common/tensor_desc.h"

// scalar 元素个数不为 1：前置校验返回 161002（原经 V2 桥接误报 561103
TEST(ForeachDivScalar, ascend910B2_scalar_elements_not_one)
{
    auto x = TensorListDesc({TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND)}).ToAclType();
    auto scalar = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    auto out = TensorListDesc({TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND)}).ToAclType();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(aclnnForeachDivScalarGetWorkspaceSize(x.get(), scalar.get(), out.get(), &workspaceSize, &executor),
              ACLNN_ERR_PARAM_INVALID);
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
}
