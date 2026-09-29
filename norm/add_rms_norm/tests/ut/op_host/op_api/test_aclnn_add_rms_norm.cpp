/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include <limits>
#include <memory>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_add_rms_norm.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

class l2_add_rms_norm_test : public testing::Test {
protected:
    std::unique_ptr<op::SocVersionManager> versionManager_;
    void SetUp() override { versionManager_ = std::make_unique<op::SocVersionManager>(op::SocVersion::ASCEND950); }
    void TearDown() override { versionManager_.reset(); }
    static void SetUpTestCase() { cout << "l2_add_rms_norm_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_add_rms_norm_test TearDown" << endl; }

public:
    void CommonTest(const vector<int64_t>& xShape, const vector<int64_t>& weightShape, const vector<int64_t>& rstdShape,
                    aclDataType dtype, aclnnStatus expectRet, double epsilon = 0.00001)
    {
        auto x1 = TensorDesc(xShape, dtype, ACL_FORMAT_ND);
        auto x2 = TensorDesc(xShape, dtype, ACL_FORMAT_ND);
        auto weight = TensorDesc(weightShape, dtype, ACL_FORMAT_ND);
        auto yOut = TensorDesc(xShape, dtype, ACL_FORMAT_ND);
        auto rstdOut = TensorDesc(rstdShape, ACL_FLOAT, ACL_FORMAT_ND);
        auto xOut = TensorDesc(xShape, dtype, ACL_FORMAT_ND);
        uint64_t workspace_size = 0;
        auto ut = OP_API_UT(aclnnAddRmsNorm, INPUT(x1, x2, weight, epsilon), OUTPUT(yOut, rstdOut, xOut));
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, expectRet);
    }

    void ShapeTest(const vector<int64_t>& x1Shape, const vector<int64_t>& x2Shape, const vector<int64_t>& gammaShape,
                   const vector<int64_t>& yShape, const vector<int64_t>& rstdShape, const vector<int64_t>& xOutShape,
                   aclnnStatus expectRet)
    {
        auto x1 = TensorDesc(x1Shape, ACL_FLOAT16, ACL_FORMAT_ND);
        auto x2 = TensorDesc(x2Shape, ACL_FLOAT16, ACL_FORMAT_ND);
        auto gamma = TensorDesc(gammaShape, ACL_FLOAT16, ACL_FORMAT_ND);
        auto yOut = TensorDesc(yShape, ACL_FLOAT16, ACL_FORMAT_ND);
        auto rstdOut = TensorDesc(rstdShape, ACL_FLOAT, ACL_FORMAT_ND);
        auto xOut = TensorDesc(xOutShape, ACL_FLOAT16, ACL_FORMAT_ND);
        uint64_t workspaceSize = 0;
        auto ut = OP_API_UT(aclnnAddRmsNorm, INPUT(x1, x2, gamma, 0.00001), OUTPUT(yOut, rstdOut, xOut));
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), expectRet);
    }

    void DtypeTest(aclDataType x1Dtype, aclDataType x2Dtype, aclDataType gammaDtype, aclDataType yDtype,
                   aclDataType rstdDtype, aclDataType xOutDtype, aclnnStatus expectRet)
    {
        auto x1 = TensorDesc({2, 4, 8}, x1Dtype, ACL_FORMAT_ND);
        auto x2 = TensorDesc({2, 4, 8}, x2Dtype, ACL_FORMAT_ND);
        auto gamma = TensorDesc({8}, gammaDtype, ACL_FORMAT_ND);
        auto yOut = TensorDesc({2, 4, 8}, yDtype, ACL_FORMAT_ND);
        auto rstdOut = TensorDesc({2, 4, 1}, rstdDtype, ACL_FORMAT_ND);
        auto xOut = TensorDesc({2, 4, 8}, xOutDtype, ACL_FORMAT_ND);
        uint64_t workspaceSize = 0;
        auto ut = OP_API_UT(aclnnAddRmsNorm, INPUT(x1, x2, gamma, 0.00001), OUTPUT(yOut, rstdOut, xOut));
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), expectRet);
    }

    void FormatTest(aclFormat x1Format, aclFormat x2Format, aclFormat gammaFormat, aclFormat yFormat,
                    aclFormat rstdFormat, aclFormat xOutFormat, aclnnStatus expectRet)
    {
        auto x1 = TensorDesc({2, 4, 8}, ACL_FLOAT16, x1Format);
        auto x2 = TensorDesc({2, 4, 8}, ACL_FLOAT16, x2Format);
        auto gamma = TensorDesc({8}, ACL_FLOAT16, gammaFormat);
        auto yOut = TensorDesc({2, 4, 8}, ACL_FLOAT16, yFormat);
        auto rstdOut = TensorDesc({2, 4, 1}, ACL_FLOAT, rstdFormat);
        auto xOut = TensorDesc({2, 4, 8}, ACL_FLOAT16, xOutFormat);
        uint64_t workspaceSize = 0;
        auto ut = OP_API_UT(aclnnAddRmsNorm, INPUT(x1, x2, gamma, 0.00001), OUTPUT(yOut, rstdOut, xOut));
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), expectRet);
    }
};

TEST_F(l2_add_rms_norm_test, ascend910B2_success)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    // data type cases
    CommonTest({4, 16, 128}, {128}, {4, 16, 1}, ACL_FLOAT, ACLNN_SUCCESS);
    CommonTest({4, 4, 128}, {128}, {4, 4, 1}, ACL_FLOAT16, ACLNN_SUCCESS);
    CommonTest({4, 16, 12288}, {12288}, {4, 16, 1}, ACL_FLOAT16, ACLNN_SUCCESS);
    CommonTest({4, 0, 128}, {128}, {4, 0, 1}, ACL_FLOAT, ACLNN_SUCCESS);
}

TEST_F(l2_add_rms_norm_test, timer_s1_bf16_success)
{
    CommonTest({360, 1024}, {1024}, {360, 1}, ACL_BF16, ACLNN_SUCCESS);
}

TEST_F(l2_add_rms_norm_test, ascend950_reduce_empty_success)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    // A>0、R=0 时 ACLNN 必须继续构造 AddRmsNorm 节点，由 A5 的 key=5000 写出非空 rstd。
    CommonTest({16, 0}, {0}, {16, 1}, ACL_FLOAT16, ACLNN_SUCCESS);
}

TEST_F(l2_add_rms_norm_test, timer_s1_dtype_invalid)
{
    CommonTest({360, 1024}, {1024}, {360, 1}, ACL_DOUBLE, ACLNN_ERR_PARAM_INVALID);
}
TEST_F(l2_add_rms_norm_test, ascend910B2_param_invalid)
{
    // invalid dtype
    CommonTest({4, 4, 128}, {128}, {4, 4, 1}, ACL_DOUBLE, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_rms_norm_test, gamma_shape_invalid)
{
    // gamma must exactly match the trailing dimensions of x1.
    ShapeTest({2, 4, 8}, {2, 4, 8}, {4, 7}, {2, 4, 8}, {2, 1, 1}, {2, 4, 8}, ACLNN_ERR_PARAM_INVALID);
    ShapeTest({2, 4, 8}, {2, 4, 8}, {1, 8}, {2, 4, 8}, {2, 1, 1}, {2, 4, 8}, ACLNN_ERR_PARAM_INVALID);
    // gamma rank must not be greater than x1 rank. This used to underflow size_t and hang.
    ShapeTest({8}, {8}, {1, 8}, {8}, {1}, {8}, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_rms_norm_test, rstd_shape_invalid)
{
    // rstd must preserve every non-normalized prefix dimension and replace every normalized dimension with 1.
    ShapeTest({2, 4, 8}, {2, 4, 8}, {8}, {2, 4, 8}, {2, 1, 1}, {2, 4, 8}, ACLNN_ERR_PARAM_INVALID);
    ShapeTest({2, 4, 8}, {2, 4, 8}, {8}, {2, 4, 8}, {3, 4, 1}, {2, 4, 8}, ACLNN_ERR_PARAM_INVALID);
    ShapeTest({2, 4, 8}, {2, 4, 8}, {4, 8}, {2, 4, 8}, {2, 4, 1}, {2, 4, 8}, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_rms_norm_test, tensor_rank_and_shape_invalid)
{
    ShapeTest({}, {}, {}, {}, {}, {}, ACLNN_ERR_PARAM_INVALID);
    ShapeTest({1, 1, 1, 1, 1, 1, 1, 1, 8}, {1, 1, 1, 1, 1, 1, 1, 1, 8}, {8}, {1, 1, 1, 1, 1, 1, 1, 1, 8},
              {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 8}, ACLNN_ERR_PARAM_INVALID);
    ShapeTest({2, 4, 8}, {2, 5, 8}, {8}, {2, 4, 8}, {2, 4, 1}, {2, 4, 8}, ACLNN_ERR_PARAM_INVALID);
    ShapeTest({2, 4, 8}, {2, 4, 8}, {8}, {2, 5, 8}, {2, 4, 1}, {2, 4, 8}, ACLNN_ERR_PARAM_INVALID);
    ShapeTest({2, 4, 8}, {2, 4, 8}, {8}, {2, 4, 8}, {2, 4, 1}, {2, 5, 8}, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_rms_norm_test, epsilon_invalid_and_boundary)
{
    CommonTest({2, 4, 8}, {8}, {2, 4, 1}, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID, -0.00001);
    CommonTest({2, 4, 8}, {8}, {2, 4, 1}, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID,
               std::numeric_limits<double>::quiet_NaN());
    CommonTest({2, 4, 8}, {8}, {2, 4, 1}, ACL_FLOAT16, ACLNN_SUCCESS, 0.0);
}

TEST_F(l2_add_rms_norm_test, dtype_relation_invalid)
{
    DtypeTest(ACL_FLOAT16, ACL_BF16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID);
    DtypeTest(ACL_FLOAT16, ACL_FLOAT16, ACL_BF16, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID);
    DtypeTest(ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_BF16, ACL_FLOAT, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID);
    DtypeTest(ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID);
    DtypeTest(ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT, ACL_BF16, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_rms_norm_test, base_formats_supported)
{
    FormatTest(ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACLNN_SUCCESS);
    // Keep matching input/output formats, as supplied by framework callers.
    FormatTest(ACL_FORMAT_NCL, ACL_FORMAT_NCL, ACL_FORMAT_ND, ACL_FORMAT_NCL, ACL_FORMAT_NCL, ACL_FORMAT_NCL,
               ACLNN_SUCCESS);
}

TEST_F(l2_add_rms_norm_test, legacy_soc_ncl_and_epsilon_compatibility)
{
    for (auto soc : {op::SocVersion::ASCEND310P, op::SocVersion::ASCEND910B, op::SocVersion::ASCEND910_93}) {
        op::SocVersionManager versionManager(soc);
        SCOPED_TRACE(static_cast<int>(soc));
        FormatTest(ACL_FORMAT_NCL, ACL_FORMAT_NCL, ACL_FORMAT_ND, ACL_FORMAT_NCL, ACL_FORMAT_NCL, ACL_FORMAT_NCL,
                   ACLNN_SUCCESS);
        CommonTest({2, 4, 8}, {8}, {2, 4, 1}, ACL_FLOAT16, ACLNN_SUCCESS, -0.00001);
    }
}

TEST_F(l2_add_rms_norm_test, ascend950_optional_output_modes_rejected)
{
    auto x1 = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2 = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto validGamma = TensorDesc({1, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto yOut = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rstdOut = TensorDesc({2, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xOut = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;

    auto preMode = OP_API_UT(aclnnAddRmsNorm, INPUT(x1, x2, validGamma, 0.00001),
                             OUTPUT(yOut, (aclTensor*)nullptr, xOut));
    EXPECT_EQ(preMode.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
    auto postMode = OP_API_UT(aclnnAddRmsNorm, INPUT(x1, x2, validGamma, 0.00001),
                              OUTPUT(yOut, (aclTensor*)nullptr, (aclTensor*)nullptr));
    EXPECT_EQ(postMode.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
    auto missingXOut = OP_API_UT(aclnnAddRmsNorm, INPUT(x1, x2, validGamma, 0.00001),
                                 OUTPUT(yOut, rstdOut, (aclTensor*)nullptr));
    EXPECT_EQ(missingXOut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_add_rms_norm_test, tensor_nullptr_invalid)
{
    auto x = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gamma = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto yOut = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rstdOut = TensorDesc({2, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xOut = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;

    auto nullX1 = OP_API_UT(aclnnAddRmsNorm, INPUT((aclTensor*)nullptr, x, gamma, 0.00001),
                            OUTPUT(yOut, rstdOut, xOut));
    EXPECT_EQ(nullX1.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
    auto nullX2 = OP_API_UT(aclnnAddRmsNorm, INPUT(x, (aclTensor*)nullptr, gamma, 0.00001),
                            OUTPUT(yOut, rstdOut, xOut));
    EXPECT_EQ(nullX2.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
    auto nullGamma = OP_API_UT(aclnnAddRmsNorm, INPUT(x, x, (aclTensor*)nullptr, 0.00001), OUTPUT(yOut, rstdOut, xOut));
    EXPECT_EQ(nullGamma.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
    auto nullY = OP_API_UT(aclnnAddRmsNorm, INPUT(x, x, gamma, 0.00001), OUTPUT((aclTensor*)nullptr, rstdOut, xOut));
    EXPECT_EQ(nullY.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_add_rms_norm_test, common_output_nullptr_invalid)
{
    auto x1 = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto x2 = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto gamma = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto yOut = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto rstdOut = TensorDesc({2, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    auto xOut = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    EXPECT_EQ(aclnnAddRmsNormGetWorkspaceSize(x1.get(), x2.get(), gamma.get(), 0.00001, yOut.get(), rstdOut.get(),
                                              xOut.get(), nullptr, &executor),
              ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(aclnnAddRmsNormGetWorkspaceSize(x1.get(), x2.get(), gamma.get(), 0.00001, yOut.get(), rstdOut.get(),
                                              xOut.get(), &workspaceSize, nullptr),
              ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_add_rms_norm_test, ascend950_exact_tail_and_empty_outputs_success)
{
    // A dimension of 1 is valid when gamma exactly matches the input tail.
    CommonTest({2, 1, 8}, {1, 8}, {2, 1, 1}, ACL_FLOAT16, ACLNN_SUCCESS);
    // Empty tensors are valid outputs, unlike null output pointers.
    CommonTest({0, 8}, {8}, {0, 1}, ACL_FLOAT16, ACLNN_SUCCESS);
}

TEST_F(l2_add_rms_norm_test, ascend950_required_outputs_with_valid_gamma)
{
    for (const auto& shape : {vector<int64_t>{2, 8}, vector<int64_t>{0, 8}}) {
        SCOPED_TRACE(shape[0]);
        auto x = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND);
        auto gamma = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto y = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND);
        auto rstd = TensorDesc({shape[0], 1}, ACL_FLOAT, ACL_FORMAT_ND);
        auto xOut = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND);
        uint64_t workspaceSize = 0;
        auto missingRstd = OP_API_UT(aclnnAddRmsNorm, INPUT(x, x, gamma, 0.00001),
                                     OUTPUT(y, (aclTensor*)nullptr, xOut));
        EXPECT_EQ(missingRstd.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
        auto missingX = OP_API_UT(aclnnAddRmsNorm, INPUT(x, x, gamma, 0.00001), OUTPUT(y, rstd, (aclTensor*)nullptr));
        EXPECT_EQ(missingX.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
        auto missingBoth = OP_API_UT(aclnnAddRmsNorm, INPUT(x, x, gamma, 0.00001),
                                     OUTPUT(y, (aclTensor*)nullptr, (aclTensor*)nullptr));
        EXPECT_EQ(missingBoth.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
    }
}

TEST_F(l2_add_rms_norm_test, legacy_soc_optional_output_modes_success)
{
    for (auto soc : {op::SocVersion::ASCEND310P, op::SocVersion::ASCEND910B, op::SocVersion::ASCEND910_93}) {
        op::SocVersionManager versionManager(soc);
        SCOPED_TRACE(static_cast<int>(soc));
        auto x = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto gamma = TensorDesc({1, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto y = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto xOut = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
        uint64_t workspaceSize = 0;
        auto preMode = OP_API_UT(aclnnAddRmsNorm, INPUT(x, x, gamma, 0.00001), OUTPUT(y, (aclTensor*)nullptr, xOut));
        EXPECT_EQ(preMode.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
        auto postMode = OP_API_UT(aclnnAddRmsNorm, INPUT(x, x, gamma, 0.00001),
                                  OUTPUT(y, (aclTensor*)nullptr, (aclTensor*)nullptr));
        EXPECT_EQ(postMode.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    }
}
