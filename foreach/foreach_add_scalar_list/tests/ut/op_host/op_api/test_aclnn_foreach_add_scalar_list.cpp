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
#include "../../../../op_host/op_api/aclnn_foreach_add_scalar_list.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"

namespace {
void RunGetWorkspaceSize(const TensorListDesc& xDesc, const std::vector<ScalarDesc>& scalarDescs,
                         const TensorListDesc& outDesc, aclnnStatus expected)
{
    auto x = xDesc.ToAclType();
    auto out = outDesc.ToAclType();
    std::vector<const aclScalar*> values;
    for (auto& s : scalarDescs) {
        auto v = s.ToAclType();
        ASSERT_NE(v.get(), nullptr);
        values.push_back(v.release());
    }
    auto scalars = aclCreateScalarList(values.data(), values.size());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(aclnnForeachAddScalarListGetWorkspaceSize(x.get(), scalars, out.get(), &workspaceSize, &executor),
              expected);
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
    aclDestroyScalarList(scalars);
}
} // namespace

// 非连续输入+非连续输出：应正常通过（修复目标场景）
TEST(ForeachAddScalarList, ascend910B2_non_contiguous_input_output)
{
    auto x = TensorListDesc({TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 1, {9})}).ToAclType();
    auto out = TensorListDesc({TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 1, {9})}).ToAclType();
    auto scalar = ScalarDesc(2.0f).ToAclType();
    const aclScalar* values[] = {scalar.release()};
    auto scalars = aclCreateScalarList(values, 1);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(aclnnForeachAddScalarListGetWorkspaceSize(x.get(), scalars, out.get(), &workspaceSize, &executor),
              ACLNN_SUCCESS);
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
    aclDestroyScalarList(scalars);
}

// 仅输出非连续：应正常通过
TEST(ForeachAddScalarList, ascend910B2_non_contiguous_output_only)
{
    RunGetWorkspaceSize(TensorListDesc({TensorDesc({4, 6})}), {ScalarDesc(2.0f)},
                        TensorListDesc({TensorDesc({4, 6}, ACL_FLOAT, ACL_FORMAT_ND, {12, 2}, 0, {4, 12})}),
                        ACLNN_SUCCESS);
}

// 数据类型支持：FLOAT/FLOAT16/BFLOAT16 + INT32（配INT64标量）
TEST(ForeachAddScalarList, ascend910B2_supported_dtypes)
{
    for (auto tensorType : {ACL_FLOAT16, ACL_FLOAT, ACL_BF16}) {
        SCOPED_TRACE(static_cast<int>(tensorType));
        RunGetWorkspaceSize(TensorListDesc({TensorDesc({2, 4}, tensorType)}), {ScalarDesc(2.0f)},
                            TensorListDesc({TensorDesc({2, 4}, tensorType)}), ACLNN_SUCCESS);
    }
    RunGetWorkspaceSize(TensorListDesc({TensorDesc({2, 4}, ACL_INT32)}), {ScalarDesc(int64_t{2})},
                        TensorListDesc({TensorDesc({2, 4}, ACL_INT32)}), ACLNN_SUCCESS);
}

// INT32张量只接受INT64标量
TEST(ForeachAddScalarList, ascend910B2_int32_rejects_float_scalar)
{
    RunGetWorkspaceSize(TensorListDesc({TensorDesc({2, 4}, ACL_INT32)}), {ScalarDesc(2.0f)},
                        TensorListDesc({TensorDesc({2, 4}, ACL_INT32)}), ACLNN_ERR_PARAM_INVALID);
}

// 浮点张量不接受INT64标量
TEST(ForeachAddScalarList, ascend910B2_float_rejects_int64_scalar)
{
    RunGetWorkspaceSize(TensorListDesc({TensorDesc({2, 4}, ACL_FLOAT)}), {ScalarDesc(int64_t{2})},
                        TensorListDesc({TensorDesc({2, 4}, ACL_FLOAT)}), ACLNN_ERR_PARAM_INVALID);
}

// 浮点张量接受DOUBLE标量（内部转FLOAT32参与计算）
TEST(ForeachAddScalarList, ascend910B2_float_accepts_double_scalar)
{
    RunGetWorkspaceSize(TensorListDesc({TensorDesc({2, 4}, ACL_FLOAT)}), {ScalarDesc(2.0)},
                        TensorListDesc({TensorDesc({2, 4}, ACL_FLOAT)}), ACLNN_SUCCESS);
}

// scalars元素个数必须与x中Tensor个数相等
TEST(ForeachAddScalarList, ascend910B2_scalar_count_mismatch)
{
    RunGetWorkspaceSize(TensorListDesc({TensorDesc({2, 4}), TensorDesc({2, 4})}), {ScalarDesc(2.0f)},
                        TensorListDesc({TensorDesc({2, 4}), TensorDesc({2, 4})}), ACLNN_ERR_PARAM_INVALID);
}

// 空TensorList不合法
TEST(ForeachAddScalarList, ascend910B2_empty_list_rejected)
{
    auto* emptyX = aclCreateTensorList(nullptr, 0);
    auto* emptyOut = aclCreateTensorList(nullptr, 0);
    auto scalar = ScalarDesc(2.0f).ToAclType();
    const aclScalar* values[] = {scalar.release()};
    auto scalars = aclCreateScalarList(values, 1);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(aclnnForeachAddScalarListGetWorkspaceSize(emptyX, scalars, emptyOut, &workspaceSize, &executor),
              ACLNN_ERR_PARAM_INVALID);
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
    aclDestroyScalarList(scalars);
    aclDestroyTensorList(emptyX);
    aclDestroyTensorList(emptyOut);
}

// out的Tensor个数允许大于等于x
TEST(ForeachAddScalarList, ascend910B2_out_longer_than_x)
{
    RunGetWorkspaceSize(TensorListDesc({TensorDesc({2, 4})}), {ScalarDesc(2.0f)},
                        TensorListDesc({TensorDesc({2, 4}), TensorDesc({2, 4})}), ACLNN_SUCCESS);
}

// out 的 Tensor 个数小于 x 时应拒绝（161002）
TEST(ForeachAddScalarList, ascend910B2_out_shorter_than_x_rejected)
{
    RunGetWorkspaceSize(TensorListDesc({TensorDesc({2, 4}), TensorDesc({2, 4})}), {ScalarDesc(2.0f), ScalarDesc(2.0f)},
                        TensorListDesc({TensorDesc({2, 4})}), ACLNN_ERR_PARAM_INVALID);
}

// out[i] 元素数小于 x[i] 时应拒绝（资料口径：shape size 大于等于 x）
TEST(ForeachAddScalarList, ascend910B2_out_smaller_numel_rejected)
{
    RunGetWorkspaceSize(TensorListDesc({TensorDesc({2, 4})}), {ScalarDesc(2.0f)}, TensorListDesc({TensorDesc({4})}),
                        ACLNN_ERR_PARAM_INVALID);
}

// out[i] 元素数不小于 x[i]（同元素数异 shape）应放行
TEST(ForeachAddScalarList, ascend910B2_out_equal_numel_different_shape_allowed)
{
    RunGetWorkspaceSize(TensorListDesc({TensorDesc({2, 4})}), {ScalarDesc(2.0f)}, TensorListDesc({TensorDesc({8})}),
                        ACLNN_SUCCESS);
}
