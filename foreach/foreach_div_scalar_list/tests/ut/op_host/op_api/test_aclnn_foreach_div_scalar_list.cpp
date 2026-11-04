/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_foreach_div_scalar_list.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"

TEST(ForeachDivScalarList, ascend910B2_non_contiguous_input_output)
{
    auto x = TensorListDesc({TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 1, {9})}).ToAclType();
    auto out = TensorListDesc({TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 1, {9})}).ToAclType();
    auto scalar = ScalarDesc(2.0f).ToAclType();
    const aclScalar* values[] = {scalar.release()};
    auto scalars = aclCreateScalarList(values, 1);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(aclnnForeachDivScalarListGetWorkspaceSize(x.get(), scalars, out.get(), &workspaceSize, &executor),
              ACLNN_SUCCESS);
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
    aclDestroyScalarList(scalars);
}
