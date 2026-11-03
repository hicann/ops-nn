/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cfloat>

#include <array>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_rotate_quant.h"
#include "../../../op_api/aclnn_rotate_quant_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_rotate_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_rotate_quant_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_rotate_quant_test TearDown" << endl; }
};

// x 为 nullptr
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_x_nullptr_fail)
{
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(
        aclnnRotateQuant,
        INPUT((aclTensor*)nullptr, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// rot 为 nullptr
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_rot_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(
        aclnnRotateQuant,
        INPUT(x_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// y 为 nullptr
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_y_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
                        OUTPUT((aclTensor*)nullptr, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// scale 为 nullptr
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_scale_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
                        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// x 和 rot 数据类型不一致
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_dtype_mismatch_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// x 数据类型不支持 (FP32)
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_x_dtype_unsupported_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// y 数据类型不支持 (BF16)
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_y_dtype_unsupported_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// scale 数据类型不支持 (BF16 instead of FP32)
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_scale_dtype_unsupported_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// rot 不是方阵
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_rot_not_square_fail)
{
    TensorDesc x_desc = TensorDesc({4, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// x 维度不是2D
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_x_dim_fail)
{
    TensorDesc x_desc = TensorDesc({2, 4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({2, 4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// rot 维度不是2D
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_rot_dim_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({1, 64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// y shape 与 x shape 不一致
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_y_shape_fail)
{
    TensorDesc x_desc = TensorDesc({4, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 128}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// scale shape 与 M 不一致
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_scale_shape_fail)
{
    TensorDesc x_desc = TensorDesc({4, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// x N 不能被 rot K 整除
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_n_not_divisible_fail)
{
    TensorDesc x_desc = TensorDesc({4, 240}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({32, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 240}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// empty tensor
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_empty_tensor)
{
    TensorDesc x_desc = TensorDesc({0, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({0, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// SoC 版本错误
TEST_F(l2_rotate_quant_test, ascend910B2_rotate_quant_bf16_int8_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t axis = -1;
    char* roundMode = nullptr;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;

    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_rotate_quant_test, ascend910B_rotate_quant_success_int8)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({4, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_rotate_quant_test, ascend910B_rotate_quant_invalid_axis)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, 0, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rotate_quant_test, ascend910B_rotate_quant_invalid_round_mode)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({4, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("trunc");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rotate_quant_test, ascend910B_rotate_quant_invalid_scale_alg)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({4, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 1, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rotate_quant_test, ascend910B_rotate_quant_invalid_dst_type_max)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({4, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 1.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rotate_quant_test, ascend910B_rotate_quant_invalid_trans_true)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({4, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, true),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rotate_quant_test, ascend910B_rotate_quant_nz_format_invalid)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({4, 256}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {16, 1, 16, 16});
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rotate_quant_test, ascend910B_rotate_quant_alpha_not_null_int_mode)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({4, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc alpha_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha_desc, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rotate_quant_test, ascend910B_rotate_quant_int32_out_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({4, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 32}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("floor");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_rotate_quant_test, ascend910B_rotate_quant_int32_out_shape_invalid)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({4, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rotate_quant_test, ascend950_rotate_quant_mx_fp8_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// MX路径空Tensor（M=0）: 维度非负即可，走IsEmpty早退返回成功，不应被dim>0校验拦截
TEST_F(l2_rotate_quant_test, ascend950_rotate_quant_mx_fp8_empty_tensor_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    TensorDesc x_desc = TensorDesc({0, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({0, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({0, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_rotate_quant_test, ascend950_rotate_quant_mx_invalid_rot_k)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rotate_quant_test, ascend950_rotate_quant_k256_all_mx_dtypes)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    for (auto inType : {ACL_FLOAT16, ACL_BF16}) {
        for (auto outType : {ACL_FLOAT4_E2M1, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2}) {
            for (bool multiRot : {false, true}) {
                TensorDesc x = TensorDesc({65, 768}, inType, ACL_FORMAT_ND);
                TensorDesc rot = multiRot ? TensorDesc({3, 256, 256}, inType, ACL_FORMAT_ND) :
                                            TensorDesc({256, 256}, inType, ACL_FORMAT_ND);
                TensorDesc y = TensorDesc({65, 768}, outType, ACL_FORMAT_ND);
                TensorDesc scale = TensorDesc({65, 12, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
                auto ut = OP_API_UT(aclnnRotateQuant,
                                    INPUT(x, rot, (aclTensor*)nullptr, -1, (char*)nullptr, 0, 0.0, false),
                                    OUTPUT(y, scale));
                uint64_t size = 0;
                EXPECT_EQ(ut.TestGetWorkspaceSize(&size), ACLNN_SUCCESS);
                auto v2 = OP_API_UT(
                    aclnnRotateQuantV2,
                    INPUT(x, rot, (aclTensor*)nullptr, (aclTensor*)nullptr, 0, -1, (char*)nullptr, 0, 0.0, false),
                    OUTPUT(y, scale));
                EXPECT_EQ(v2.TestGetWorkspaceSize(&size), ACLNN_SUCCESS);
            }
        }
    }
}

TEST_F(l2_rotate_quant_test, ascend950_grouped_rotate_quant_k256_all_mx_dtypes)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    for (auto inType : {ACL_FLOAT16, ACL_BF16}) {
        for (auto outType : {ACL_FLOAT4_E2M1, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2}) {
            for (int64_t type : {0, 1}) {
                TensorDesc x = TensorDesc({65, 768}, inType, ACL_FORMAT_ND);
                TensorDesc rot = TensorDesc({7, 256, 256}, inType, ACL_FORMAT_ND);
                TensorDesc groups = TensorDesc({7}, ACL_INT64, ACL_FORMAT_ND);
                TensorDesc y = TensorDesc({65, 768}, outType, ACL_FORMAT_ND);
                TensorDesc scale = TensorDesc({65, 12, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
                auto ut = OP_API_UT(aclnnRotateQuantV2,
                                    INPUT(x, rot, (aclTensor*)nullptr, groups, type, -1, (char*)nullptr, 0, 0.0, false),
                                    OUTPUT(y, scale));
                uint64_t size = 0;
                EXPECT_EQ(ut.TestGetWorkspaceSize(&size), ACLNN_SUCCESS);
            }
        }
    }
}

TEST_F(l2_rotate_quant_test, ascend950_rotate_quant_mx_invalid_round_mode)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("floor");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rotate_quant_test, ascend950_rotate_quant_mx_fp4_invalid_dst_type_max)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 2, 3.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rotate_quant_test, ascend950_grouped_rotate_quant_v2_metadata)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    for (int64_t type : {0, 1}) {
        TensorDesc x = TensorDesc({29, 96}, ACL_BF16, ACL_FORMAT_ND);
        TensorDesc rot = TensorDesc({7, 32, 32}, ACL_BF16, ACL_FORMAT_ND);
        TensorDesc groups = TensorDesc({7}, ACL_INT64, ACL_FORMAT_ND);
        TensorDesc y = TensorDesc({29, 96}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
        TensorDesc scale = TensorDesc({29, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnRotateQuantV2,
                            INPUT(x, rot, (aclTensor*)nullptr, groups, type, -1, (char*)nullptr, 0, 0.0, false),
                            OUTPUT(y, scale));
        uint64_t size = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&size), ACLNN_SUCCESS);
    }
}

TEST_F(l2_rotate_quant_test, ascend950_grouped_rotate_quant_v2_per_n_rotation)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    for (int64_t k : {32, 256}) {
        for (auto inType : {ACL_FLOAT16, ACL_BF16}) {
            for (auto outType : {ACL_FLOAT4_E2M1, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2}) {
                for (int64_t type : {0, 1}) {
                    for (int64_t q : {1, 3}) {
                        TensorDesc x = TensorDesc({65, q * k}, inType, ACL_FORMAT_ND);
                        TensorDesc rot = TensorDesc({7, q, k, k}, inType, ACL_FORMAT_ND);
                        TensorDesc groups = TensorDesc({7}, ACL_INT64, ACL_FORMAT_ND);
                        TensorDesc y = TensorDesc({65, q * k}, outType, ACL_FORMAT_ND);
                        TensorDesc scale = TensorDesc({65, (q * k + 63) / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
                        auto ut = OP_API_UT(
                            aclnnRotateQuantV2,
                            INPUT(x, rot, (aclTensor*)nullptr, groups, type, -1, (char*)nullptr, 0, 0.0, false),
                            OUTPUT(y, scale));
                        uint64_t size = 0;
                        EXPECT_EQ(ut.TestGetWorkspaceSize(&size), ACLNN_SUCCESS);
                    }
                }
            }
        }
    }
}

TEST_F(l2_rotate_quant_test, ascend950_grouped_rotate_quant_v2_rejects_invalid_4d_rotation)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    TensorDesc x = TensorDesc({7, 96}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc groups = TensorDesc({7}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({7, 96}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc scale = TensorDesc({7, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    for (int64_t q : {0, 1, 2, 4}) {
        TensorDesc rot = TensorDesc({7, q, 32, 32}, ACL_BF16, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnRotateQuantV2,
                            INPUT(x, rot, (aclTensor*)nullptr, groups, 1, -1, (char*)nullptr, 0, 0.0, false),
                            OUTPUT(y, scale));
        uint64_t size = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&size), ACLNN_ERR_PARAM_INVALID);
    }
    TensorDesc rot = TensorDesc({7, 3, 32, 32}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuantV2,
                        INPUT(x, rot, (aclTensor*)nullptr, (aclTensor*)nullptr, 0, -1, (char*)nullptr, 0, 0.0, false),
                        OUTPUT(y, scale));
    uint64_t size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rotate_quant_test, ascend950_grouped_rotate_quant_v2_invalid_metadata)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    // Vary E, rotation E, group-list dtype, and representation independently.
    struct Case {
        int64_t e, rotE, type;
        aclDataType dtype;
    };
    for (const auto& c : std::vector<Case>{{0, 0, 0, ACL_INT64},
                                           {1025, 1025, 0, ACL_INT64},
                                           {7, 3, 0, ACL_INT64},
                                           {7, 7, 2, ACL_INT64},
                                           {7, 7, 0, ACL_INT32}}) {
        TensorDesc x = TensorDesc({29, 96}, ACL_BF16, ACL_FORMAT_ND);
        TensorDesc rot = TensorDesc({c.rotE, 32, 32}, ACL_BF16, ACL_FORMAT_ND);
        TensorDesc groups = TensorDesc({c.e}, c.dtype, ACL_FORMAT_ND);
        TensorDesc y = TensorDesc({29, 96}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
        TensorDesc scale = TensorDesc({29, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnRotateQuantV2,
                            INPUT(x, rot, (aclTensor*)nullptr, groups, c.type, -1, (char*)nullptr, 0, 0.0, false),
                            OUTPUT(y, scale));
        uint64_t size = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&size), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_rotate_quant_test, ascend950_rotate_quant_v2_null_groups_keep_n_axis_rotation)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    TensorDesc x = TensorDesc({7, 96}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot = TensorDesc({3, 32, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({7, 96}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc scale = TensorDesc({7, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuantV2,
                        INPUT(x, rot, (aclTensor*)nullptr, (aclTensor*)nullptr, 0, -1, (char*)nullptr, 0, 0.0, false),
                        OUTPUT(y, scale));
    uint64_t size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&size), ACLNN_SUCCESS);
}

TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_v2_rejects_grouped_mode)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x = TensorDesc({7, 96}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot = TensorDesc({3, 32, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc groups = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({7, 96}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuantV2,
                        INPUT(x, rot, (aclTensor*)nullptr, groups, 0, -1, (char*)nullptr, 0, 0.0, false),
                        OUTPUT(y, scale));
    uint64_t size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rotate_quant_test, ascend350_rotate_quant_v2_rejects_groups_on_same_arch)
{
    // Ascend350 shares DAV_3510; checking only the NPU architecture is insufficient.
    NpuArchManager archManager(NpuArch::DAV_3510);
    SocVersionManager versionManager(SocVersion::ASCEND350);
    TensorDesc x = TensorDesc({7, 96}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot = TensorDesc({3, 32, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc groups = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({7, 96}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc scale = TensorDesc({7, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuantV2,
                        INPUT(x, rot, (aclTensor*)nullptr, groups, 1, -1, (char*)nullptr, 0, 0.0, false),
                        OUTPUT(y, scale));
    uint64_t size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rotate_quant_test, ascend350_rotate_quant_v2_null_groups_keep_legacy_mx)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    SocVersionManager versionManager(SocVersion::ASCEND350);
    TensorDesc x = TensorDesc({7, 96}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot = TensorDesc({3, 32, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({7, 96}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc scale = TensorDesc({7, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuantV2,
                        INPUT(x, rot, (aclTensor*)nullptr, (aclTensor*)nullptr, 0, -1, (char*)nullptr, 0, 0.0, false),
                        OUTPUT(y, scale));
    uint64_t size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&size), ACLNN_SUCCESS);
}
