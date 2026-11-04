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
#include "../../../../op_host/op_api/aclnn_foreach_mul_scalar_list.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"

namespace {
void ExpectScalarDtypeStatus(aclDataType tensorType, const ScalarDesc& scalarDesc, aclnnStatus expected)
{
    const auto tensor = TensorDesc({2, 4}, tensorType, ACL_FORMAT_ND);
    auto x = TensorListDesc({tensor, tensor}).ToAclType();
    auto out = TensorListDesc({tensor, tensor}).ToAclType();
    auto validScalar = tensorType == ACL_INT32 ? ScalarDesc(int64_t{2}).ToAclType() : ScalarDesc(2.0f).ToAclType();
    auto scalar = scalarDesc.ToAclType();
    ASSERT_NE(validScalar.get(), nullptr);
    ASSERT_NE(scalar.get(), nullptr);
    // Keep the first scalar valid so mismatches in later list entries are also checked.
    const aclScalar* values[] = {validScalar.release(), scalar.release()};
    auto scalars = aclCreateScalarList(values, 2);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(aclnnForeachMulScalarListGetWorkspaceSize(x.get(), scalars, out.get(), &workspaceSize, &executor),
              expected);
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
    aclDestroyScalarList(scalars);
}
} // namespace

TEST(ForeachMulScalarList, ascend910B2_non_contiguous_input_output)
{
    auto x = TensorListDesc({TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 1, {9})}).ToAclType();
    auto out = TensorListDesc({TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 1, {9})}).ToAclType();
    auto scalar = ScalarDesc(2.0f).ToAclType();
    const aclScalar* values[] = {scalar.release()};
    auto scalars = aclCreateScalarList(values, 1);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(aclnnForeachMulScalarListGetWorkspaceSize(x.get(), scalars, out.get(), &workspaceSize, &executor),
              ACLNN_SUCCESS);
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
    aclDestroyScalarList(scalars);
}

TEST(ForeachMulScalarList, ascend910B2_supported_scalar_dtypes)
{
    for (auto tensorType : {ACL_FLOAT16, ACL_FLOAT, ACL_BF16}) {
        SCOPED_TRACE(static_cast<int>(tensorType));
        ExpectScalarDtypeStatus(tensorType, ScalarDesc(0.5f), ACLNN_SUCCESS);
    }
    ExpectScalarDtypeStatus(ACL_INT32, ScalarDesc(int64_t{2}), ACLNN_SUCCESS);
}

TEST(ForeachMulScalarList, ascend910B2_int32_rejects_floating_scalars)
{
    for (const auto& scalar : {ScalarDesc(0.5f), ScalarDesc(2.9f), ScalarDesc(0.5), ScalarDesc(2.9), ScalarDesc(1e100),
                               ScalarDesc(2.0f, ACL_FLOAT16)}) {
        ExpectScalarDtypeStatus(ACL_INT32, scalar, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST(ForeachMulScalarList, ascend910B2_float_tensors_reject_non_float_scalars)
{
    for (auto tensorType : {ACL_FLOAT16, ACL_FLOAT, ACL_BF16}) {
        SCOPED_TRACE(static_cast<int>(tensorType));
        for (const auto& scalar : {ScalarDesc(int64_t{2}), ScalarDesc(int32_t{2}), ScalarDesc(2.0f, ACL_FLOAT16)}) {
            ExpectScalarDtypeStatus(tensorType, scalar, ACLNN_ERR_PARAM_INVALID);
        }
    }
}

TEST(ForeachMulScalarList, ascend910B2_int32_rejects_non_int64_integral_scalars)
{
    for (const auto& scalar : {ScalarDesc(int32_t{2}), ScalarDesc(uint64_t{2}), ScalarDesc(true)}) {
        ExpectScalarDtypeStatus(ACL_INT32, scalar, ACLNN_ERR_PARAM_INVALID);
    }
}
