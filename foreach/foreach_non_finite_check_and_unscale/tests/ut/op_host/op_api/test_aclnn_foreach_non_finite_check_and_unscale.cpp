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
#include "../../../../op_host/op_api/aclnn_foreach_non_finite_check_and_unscale.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {

// Phase-one tests do not execute kernels; foundInf accumulation and inplace data checks require ST.
class L2ForeachNonFiniteCheckAndUnscaleTest : public testing::TestWithParam<op::SocVersion> {
protected:
    void SetUp() override { versionManager_ = std::make_unique<op::SocVersionManager>(GetParam()); }

    void ExpectWorkspaceStatus(const TensorListDesc& scaledGrads, const TensorDesc& foundInf,
                               const TensorDesc& invScale, aclnnStatus expected)
    {
        auto ut = OP_API_UT(aclnnForeachNonFiniteCheckAndUnscale, INPUT(scaledGrads, foundInf, invScale), OUTPUT());
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), expected);
    }

    void ExpectWorkspaceStatus(const TensorListDesc& scaledGrads, aclnnStatus expected)
    {
        auto foundInf = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
        auto invScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
        ExpectWorkspaceStatus(scaledGrads, foundInf, invScale, expected);
    }

    static aclnnStatus GetWorkspaceStatus(const aclTensorList* scaledGrads, const aclTensor* foundInf,
                                          const aclTensor* invScale, uint64_t* workspaceSize)
    {
        aclOpExecutor* executor = nullptr;
        auto status = aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(scaledGrads, foundInf, invScale,
                                                                           workspaceSize, &executor);
        std::unique_ptr<aclOpExecutor> executorGuard(executor);
        return status;
    }

private:
    std::unique_ptr<op::SocVersionManager> versionManager_;
};

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, supported_contiguous_dtypes)
{
    for (auto dtype : {ACL_FLOAT, ACL_FLOAT16, ACL_BF16}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        auto grads = TensorDesc({2, 3}, dtype, ACL_FORMAT_ND);
        ExpectWorkspaceStatus(TensorListDesc({grads}), ACLNN_SUCCESS);
    }
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, transposed_gradients)
{
    auto grads = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {3, 2});
    ExpectWorkspaceStatus(TensorListDesc({grads}), ACLNN_SUCCESS);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, strided_gradients_with_offset)
{
    for (auto dtype : {ACL_FLOAT, ACL_FLOAT16, ACL_BF16}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        auto grads = TensorDesc({2, 3}, dtype, ACL_FORMAT_ND, {6, 2}, 1, {12});
        ExpectWorkspaceStatus(TensorListDesc({grads}), ACLNN_SUCCESS);
    }
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, contiguous_gradients_in_larger_storage)
{
    // Storage rank differs from view rank even though the view strides are contiguous.
    auto grads = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {3, 1}, 5, {20});
    ExpectWorkspaceStatus(TensorListDesc({grads}), ACLNN_SUCCESS);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, mixed_gradient_layouts_and_scalar_subviews)
{
    auto contiguous = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto transposed = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {3, 2});
    auto strided = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {6, 2}, 1, {12});
    auto fused = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {3, 1}, 5, {20});
    auto foundInf = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND, {2}, 7, {16});
    auto invScale = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND, {4, 2}, 3, {8});
    ExpectWorkspaceStatus(TensorListDesc({contiguous, transposed, strided, fused}), foundInf, invScale, ACLNN_SUCCESS);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, scalar_subviews_with_contiguous_gradients)
{
    auto grads = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto foundInf = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND, {2}, 7, {16});
    auto invScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND, {3}, 3, {8});
    ExpectWorkspaceStatus(TensorListDesc({grads}), foundInf, invScale, ACLNN_SUCCESS);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, zero_dimensional_tensors)
{
    auto grads = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto foundInf = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invScale = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({grads}), foundInf, invScale, ACLNN_SUCCESS);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, maximum_rank)
{
    auto grads = TensorDesc(std::vector<int64_t>(8, 1), ACL_FLOAT, ACL_FORMAT_ND);
    auto foundInf = TensorDesc(std::vector<int64_t>(8, 1), ACL_FLOAT, ACL_FORMAT_ND);
    auto invScale = TensorDesc(std::vector<int64_t>(8, 1), ACL_FLOAT, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({grads}), foundInf, invScale, ACLNN_SUCCESS);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, maximum_list_length)
{
    auto grads = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc(256, grads), ACLNN_SUCCESS);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, empty_list)
{
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> grads(aclCreateTensorList(nullptr, 0),
                                                                          &aclDestroyTensorList);
    ASSERT_NE(grads.get(), nullptr);
    auto foundInf = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    auto invScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    uint64_t workspaceSize = 0;
    EXPECT_EQ(GetWorkspaceStatus(grads.get(), foundInf.get(), invScale.get(), &workspaceSize),
              ACLNN_ERR_INNER_TILING_ERROR);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, empty_gradient)
{
    auto grads = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({grads}), ACLNN_ERR_INNER_TILING_ERROR);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, empty_subview_after_valid_gradient)
{
    auto grads = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto empty = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND, {3, 1}, 2, {16});
    ExpectWorkspaceStatus(TensorListDesc({grads, empty}), ACLNN_ERR_INNER_TILING_ERROR);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, excessive_list_length)
{
    auto grads = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc(257, grads), ACLNN_ERR_INNER_TILING_ERROR);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, null_inputs)
{
    auto grads = TensorListDesc({TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)}).ToAclType();
    auto foundInf = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    auto invScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    uint64_t workspaceSize = 0;
    EXPECT_EQ(GetWorkspaceStatus(nullptr, foundInf.get(), invScale.get(), &workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(GetWorkspaceStatus(grads.get(), nullptr, invScale.get(), &workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(GetWorkspaceStatus(grads.get(), foundInf.get(), nullptr, &workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, null_element_after_valid_gradient)
{
    const aclTensor* items[] = {TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), nullptr};
    // Use the ACL deleter directly because the UT descriptor deleter dereferences each entry.
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> grads(aclCreateTensorList(items, 2),
                                                                          &aclDestroyTensorList);
    ASSERT_NE(grads.get(), nullptr);
    auto foundInf = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    auto invScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    uint64_t workspaceSize = 0;
    EXPECT_EQ(GetWorkspaceStatus(grads.get(), foundInf.get(), invScale.get(), &workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, unsupported_gradient_dtype)
{
    auto grads = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({grads}), ACLNN_ERR_PARAM_INVALID);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, mixed_gradient_dtypes)
{
    auto grads0 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grads1 = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({grads0, grads1}), ACLNN_ERR_INNER_TILING_ERROR);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, invalid_found_inf_dtype)
{
    auto grads = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto foundInf = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto invScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({grads}), foundInf, invScale, ACLNN_ERR_PARAM_INVALID);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, invalid_inv_scale_dtype)
{
    auto grads = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto foundInf = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invScale = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({grads}), foundInf, invScale, ACLNN_ERR_PARAM_INVALID);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, invalid_found_inf_element_count)
{
    auto grads = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    for (int64_t size : {0, 2}) {
        SCOPED_TRACE(size);
        auto foundInf = TensorDesc({size}, ACL_FLOAT, ACL_FORMAT_ND);
        ExpectWorkspaceStatus(TensorListDesc({grads}), foundInf, invScale, ACLNN_ERR_INNER_TILING_ERROR);
    }
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, invalid_inv_scale_element_count)
{
    auto grads = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto foundInf = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    for (int64_t size : {0, 2}) {
        SCOPED_TRACE(size);
        auto invScale = TensorDesc({size}, ACL_FLOAT, ACL_FORMAT_ND);
        ExpectWorkspaceStatus(TensorListDesc({grads}), foundInf, invScale, ACLNN_ERR_INNER_TILING_ERROR);
    }
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, excessive_gradient_rank)
{
    auto grads = TensorDesc(std::vector<int64_t>(9, 1), ACL_FLOAT, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({grads}), ACLNN_ERR_INNER_TILING_ERROR);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, excessive_found_inf_rank)
{
    auto grads = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto foundInf = TensorDesc(std::vector<int64_t>(9, 1), ACL_FLOAT, ACL_FORMAT_ND);
    auto invScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({grads}), foundInf, invScale, ACLNN_ERR_INNER_TILING_ERROR);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, excessive_inv_scale_rank)
{
    auto grads = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto foundInf = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invScale = TensorDesc(std::vector<int64_t>(9, 1), ACL_FLOAT, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({grads}), foundInf, invScale, ACLNN_ERR_INNER_TILING_ERROR);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, private_gradient_format)
{
    auto grads = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    ExpectWorkspaceStatus(TensorListDesc({grads}), ACLNN_ERR_PARAM_INVALID);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, private_found_inf_format)
{
    auto grads = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto foundInf = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    auto invScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    ExpectWorkspaceStatus(TensorListDesc({grads}), foundInf, invScale, ACLNN_ERR_PARAM_INVALID);
}

TEST_P(L2ForeachNonFiniteCheckAndUnscaleTest, private_inv_scale_format)
{
    auto grads = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto foundInf = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invScale = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    ExpectWorkspaceStatus(TensorListDesc({grads}), foundInf, invScale, ACLNN_ERR_PARAM_INVALID);
}

INSTANTIATE_TEST_SUITE_P(A2AndA3, L2ForeachNonFiniteCheckAndUnscaleTest,
                         testing::Values(op::SocVersion::ASCEND910B, op::SocVersion::ASCEND910_93));

} // namespace
