/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_foreach_zero_inplace.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {

// Phase-one tests do not execute kernels; inplace data checks require ST.
class L2ForeachZeroInplaceTest : public testing::TestWithParam<op::SocVersion> {
protected:
    void SetUp() override { versionManager_ = std::make_unique<op::SocVersionManager>(GetParam()); }

    void ExpectWorkspaceStatus(const TensorListDesc& x, aclnnStatus expected, bool empty = false)
    {
        auto ut = OP_API_UT(aclnnForeachZeroInplace, INPUT(x), OUTPUT());
        uint64_t workspaceSize = 1;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), expected);
        if (empty) {
            EXPECT_EQ(workspaceSize, 0U);
        }
    }

    static aclnnStatus GetWorkspaceStatus(const aclTensorList* x, uint64_t* workspaceSize)
    {
        aclOpExecutor* executor = nullptr;
        auto status = aclnnForeachZeroInplaceGetWorkspaceSize(x, workspaceSize, &executor);
        std::unique_ptr<aclOpExecutor> executorGuard(executor);
        return status;
    }

private:
    std::unique_ptr<op::SocVersionManager> versionManager_;
};

TEST_P(L2ForeachZeroInplaceTest, supported_contiguous_dtypes)
{
    for (auto dtype : {ACL_FLOAT, ACL_FLOAT16, ACL_BF16, ACL_INT16, ACL_INT32}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        auto x = TensorDesc({2, 3}, dtype, ACL_FORMAT_ND);
        ExpectWorkspaceStatus(TensorListDesc({x}), ACLNN_SUCCESS);
    }
}

TEST_P(L2ForeachZeroInplaceTest, transposed_view)
{
    auto x = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {3, 2});
    ExpectWorkspaceStatus(TensorListDesc({x}), ACLNN_SUCCESS);
}

TEST_P(L2ForeachZeroInplaceTest, strided_view_with_offset)
{
    for (auto dtype : {ACL_FLOAT, ACL_FLOAT16, ACL_BF16, ACL_INT16, ACL_INT32}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        auto x = TensorDesc({2, 3}, dtype, ACL_FORMAT_ND, {6, 2}, 1, {12});
        ExpectWorkspaceStatus(TensorListDesc({x}), ACLNN_SUCCESS);
    }
}

TEST_P(L2ForeachZeroInplaceTest, contiguous_view_in_larger_storage)
{
    // IsContiguous alone cannot identify a view backed by a fused parameter buffer.
    auto x = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {3, 1}, 5, {20});
    ExpectWorkspaceStatus(TensorListDesc({x}), ACLNN_SUCCESS);
}

TEST_P(L2ForeachZeroInplaceTest, mixed_layouts_and_empty_view)
{
    auto contiguous = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto transposed = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {3, 2});
    auto strided = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {6, 2}, 1, {12});
    auto fused = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {3, 1}, 5, {20});
    auto empty = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND, {3, 1}, 2, {16});
    ExpectWorkspaceStatus(TensorListDesc({contiguous, transposed, strided, fused, empty}), ACLNN_SUCCESS);
}

TEST_P(L2ForeachZeroInplaceTest, scalar_view)
{
    auto x = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND, {}, 3, {8});
    ExpectWorkspaceStatus(TensorListDesc({x}), ACLNN_SUCCESS);
}

TEST_P(L2ForeachZeroInplaceTest, maximum_rank)
{
    auto x = TensorDesc(std::vector<int64_t>(8, 1), ACL_FLOAT, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({x}), ACLNN_SUCCESS);
}

TEST_P(L2ForeachZeroInplaceTest, empty_list_is_noop)
{
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> x(aclCreateTensorList(nullptr, 0),
                                                                      &aclDestroyTensorList);
    ASSERT_NE(x.get(), nullptr);
    uint64_t workspaceSize = 1;
    EXPECT_EQ(GetWorkspaceStatus(x.get(), &workspaceSize), ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
}

TEST_P(L2ForeachZeroInplaceTest, all_empty_views_are_noop)
{
    auto empty = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto emptySubview = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND, {3, 1}, 2, {16});
    ExpectWorkspaceStatus(TensorListDesc({empty, emptySubview}), ACLNN_SUCCESS, true);
}

TEST_P(L2ForeachZeroInplaceTest, maximum_list_length)
{
    auto x = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc(256, x), ACLNN_SUCCESS);
}

TEST_P(L2ForeachZeroInplaceTest, excessive_list_length)
{
    auto x = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc(257, x), ACLNN_ERR_INNER_TILING_ERROR);
}

TEST_P(L2ForeachZeroInplaceTest, null_list)
{
    uint64_t workspaceSize = 0;
    EXPECT_EQ(GetWorkspaceStatus(nullptr, &workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_P(L2ForeachZeroInplaceTest, null_element_after_valid_tensor)
{
    const aclTensor* items[] = {TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), nullptr};
    // aclDestroyTensorList handles null entries and owns the remaining tensor.
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> x(aclCreateTensorList(items, 2),
                                                                      &aclDestroyTensorList);
    ASSERT_NE(x.get(), nullptr);
    uint64_t workspaceSize = 0;
    EXPECT_EQ(GetWorkspaceStatus(x.get(), &workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_P(L2ForeachZeroInplaceTest, null_workspace_size)
{
    auto x = TensorListDesc({TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)}).ToAclType();
    EXPECT_EQ(GetWorkspaceStatus(x.get(), nullptr), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_P(L2ForeachZeroInplaceTest, null_executor_output)
{
    auto x = TensorListDesc({TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)}).ToAclType();
    uint64_t workspaceSize = 0;
    EXPECT_EQ(aclnnForeachZeroInplaceGetWorkspaceSize(x.get(), &workspaceSize, nullptr), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_P(L2ForeachZeroInplaceTest, unsupported_dtype)
{
    auto x = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({x}), ACLNN_ERR_PARAM_INVALID);
}

TEST_P(L2ForeachZeroInplaceTest, mixed_dtypes)
{
    auto x0 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x1 = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({x0, x1}), ACLNN_ERR_INNER_TILING_ERROR);
}

TEST_P(L2ForeachZeroInplaceTest, private_format)
{
    auto x = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    ExpectWorkspaceStatus(TensorListDesc({x}), ACLNN_ERR_PARAM_INVALID);
}

TEST_P(L2ForeachZeroInplaceTest, excessive_rank)
{
    auto x = TensorDesc(std::vector<int64_t>(9, 1), ACL_FLOAT, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({x}), ACLNN_ERR_INNER_TILING_ERROR);
}

INSTANTIATE_TEST_SUITE_P(A2AndA3, L2ForeachZeroInplaceTest,
                         testing::Values(op::SocVersion::ASCEND910B, op::SocVersion::ASCEND910_93));

} // namespace
