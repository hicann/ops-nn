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
#include "op_common/op_host/util/op_const_def.h"

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

TEST_F(l2_rotate_quant_test, dav3510_rotate_quant_mx_fp8_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
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
TEST_F(l2_rotate_quant_test, dav3510_rotate_quant_mx_fp8_empty_tensor_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
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

TEST_F(l2_rotate_quant_test, dav3510_rotate_quant_mx_invalid_rot_k)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
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

TEST_F(l2_rotate_quant_test, dav3510_rotate_quant_k256_all_mx_dtypes)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
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

TEST_F(l2_rotate_quant_test, dav3510_grouped_rotate_quant_k256_all_mx_dtypes)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
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

TEST_F(l2_rotate_quant_test, dav3510_rotate_quant_mx_invalid_round_mode)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
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

TEST_F(l2_rotate_quant_test, dav3510_rotate_quant_mx_fp4_invalid_dst_type_max)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
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

TEST_F(l2_rotate_quant_test, dav3510_grouped_rotate_quant_v2_metadata)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
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

TEST_F(l2_rotate_quant_test, dav3510_grouped_rotate_quant_v2_per_n_rotation)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
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

TEST_F(l2_rotate_quant_test, dav3510_grouped_rotate_quant_v2_rejects_invalid_4d_rotation)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
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

TEST_F(l2_rotate_quant_test, dav3510_grouped_rotate_quant_v2_invalid_metadata)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
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

TEST_F(l2_rotate_quant_test, dav3510_rotate_quant_v2_null_groups_keep_n_axis_rotation)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
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

// ==================== ascend960dt (v121) hif4 success cases ====================

// hif4: bf16 x, K=64, scale shape (M, CeilDiv(N,64), 1)
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_bf16_k64_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// hif4: fp16 x, K=128, scale shape (M, CeilDiv(N,64), 1) = (4, 2, 1)
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_fp16_k128_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({128, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 128}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 2, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// hif4 success: V2接口(groupList为空)与V1同走hif4校验路径
TEST_F(l2_rotate_quant_test, ascend960dt_rotate_quant_v2_hif4_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnRotateQuantV2,
        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, 0, -1, (char*)nullptr, 0, 0.0, false),
        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// hif4 success: 3D x, yOut与x逐维同shape, scale为(*, CeilDiv(N,64), 1)
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_3d_x_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({2, 4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({2, 4, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({2, 4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, (char*)nullptr, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// hif4 success: 空tensor(含0维)走快速返回
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_empty_tensor_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({0, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({0, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({0, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// hif4 fail: 3D rot not allowed (hif4 only supports 2D rotation, per-axis=-1 quant)
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_bf16_k64_3d_rot_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({2, 64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 128}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 2, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// ==================== ascend960dt (v121) mxfp4 success cases ====================

// mxfp4: bf16 x, K=64, scaleAlg=0, scale shape (M, CeilDiv(N,64), 2)
TEST_F(l2_rotate_quant_test, ascend960dt_mxfp4_bf16_k64_alg0_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// mxfp4: fp16 x, K=128, scaleAlg=2, dstTypeMax=8.0, scale shape (4, 2, 2)
TEST_F(l2_rotate_quant_test, ascend960dt_mxfp4_fp16_k128_alg2_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({128, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 2, 8.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// mxfp4: bf16 x, K=32, scaleAlg=0, scale shape (8, 2, 2) (N=96, CeilDiv(96,64)=2)
TEST_F(l2_rotate_quant_test, ascend960dt_mxfp4_bf16_k32_n96_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({8, 96}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({32, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({8, 96}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({8, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// ==================== ascend960dt (v121) mxfp8 success cases ====================

// mxfp8: fp8_e4m3 output, scaleAlg=0
TEST_F(l2_rotate_quant_test, ascend960dt_mxfp8_e4m3_bf16_k64_alg0_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
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

// mxfp8: fp8_e5m2 output, scaleAlg=1
TEST_F(l2_rotate_quant_test, ascend960dt_mxfp8_e5m2_fp16_k128_alg1_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({128, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 1, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// ==================== ascend960dt (v121) hif4 fail cases ====================

// hif4 fail: K=32 not allowed (must be 64 or 128)
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_k32_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({32, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 32}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// hif4 fail: scaleAlg=2 not allowed (must be 0)
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_scaleAlg2_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 2, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// hif4 fail: roundMode="floor" not allowed (must be "rint")
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_roundMode_floor_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("floor");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// hif4 fail: scale dtype mismatch (FLOAT8_E8M0 instead of HIFLOAT4_SCALE)
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_scale_dtype_e8m0_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// hif4 fail: dstTypeMax != 0.0
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_dstTypeMax_nonzero_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 8.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// hif4 fail: scale shape mismatch (expected (4,1,1), given (4,1))
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_scale_shape_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// hif4 fail: hif4场景(scale非E8M0)alpha必须为nullptr
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_alpha_not_null_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc alpha_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha_desc, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// hif4 fail: HIFLOAT4输出仅支持DAV_9201/DAV_9202, 950平台拒绝
TEST_F(l2_rotate_quant_test, ascend950_hif4_output_rejected_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// hif4 fail: rotation为2D但非方阵(dim0 != K)
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_rot_not_square_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({128, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// ==================== ascend960dt (v121) hif4 yOut shape cases ====================

// hif4输出: yOut的shape与x相同(仅dtype位宽不同), 正常路径成功
TEST_F(l2_rotate_quant_test, ascend960dt_rotate_quant_hif4_y_shape_equal_x_success)
{
    for (auto soc : {SocVersion::ASCEND960DT, SocVersion::ASCEND960PR}) {
        SocVersionManager versionManager(soc);
        TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
        TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
        TensorDesc y_desc = TensorDesc({4, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
        TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnRotateQuant,
                            INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, (char*)nullptr, 0, 0.0, false),
                            OUTPUT(y_desc, scale_desc));
        uint64_t workspace_size = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
    }
}

// hif4输出: yOut的shape与x不一致(偏大)时必须报参数错
TEST_F(l2_rotate_quant_test, ascend960dt_rotate_quant_hif4_y_shape_larger_than_x_fail)
{
    for (auto soc : {SocVersion::ASCEND960DT, SocVersion::ASCEND960PR}) {
        SocVersionManager versionManager(soc);
        TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
        TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
        TensorDesc y_desc = TensorDesc({4, 128}, ACL_HIFLOAT4, ACL_FORMAT_ND);
        TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnRotateQuant,
                            INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, (char*)nullptr, 0, 0.0, false),
                            OUTPUT(y_desc, scale_desc));
        uint64_t workspace_size = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
    }
}

// hif4输出: yOut的shape与x不一致(偏小)时必须报参数错
TEST_F(l2_rotate_quant_test, ascend960dt_rotate_quant_hif4_y_shape_smaller_than_x_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 32}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, (char*)nullptr, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// hif4输出(V2接口): yOut的shape与x不一致时同样必须报参数错
TEST_F(l2_rotate_quant_test, ascend960dt_rotate_quant_v2_hif4_y_shape_not_equal_x_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 128}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnRotateQuantV2,
        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, 0, -1, (char*)nullptr, 0, 0.0, false),
        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// hif4输出: 3D x时yOut需与x逐维同shape, 中间维不一致必须报参数错
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_3d_x_y_shape_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({2, 4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({2, 8, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({2, 4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, (char*)nullptr, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// ==================== ascend960dt (v121) mxfp4/mxfp8 fail cases ====================

// mxfp4 fail: K=48 not in {32, 64, 128}
TEST_F(l2_rotate_quant_test, ascend960dt_mxfp4_k48_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 48}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({48, 48}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 48}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// mxfp4 fail: scaleAlg=2 but dstTypeMax=3.0 out of [6.0, 12.0]
TEST_F(l2_rotate_quant_test, ascend960dt_mxfp4_alg2_dstTypeMax_3_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 2, 3.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// mxfp4 fail: scale dtype mismatch (HIFLOAT4_SCALE instead of FLOAT8_E8M0)
TEST_F(l2_rotate_quant_test, ascend960dt_mxfp4_scale_hif4scale_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 2}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// mxfp8 fail: roundMode="round" not allowed (must be "rint")
TEST_F(l2_rotate_quant_test, ascend960dt_mxfp8_roundMode_round_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("round");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// mxfp8 fail: scaleAlg=2 not allowed for fp8 (must be 0 or 1)
TEST_F(l2_rotate_quant_test, ascend960dt_mxfp8_scaleAlg2_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 2, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// ==================== ascend960dt (v121) common fail cases ====================

// v121 fail: INT8 output not supported on v121 (MX arch)
TEST_F(l2_rotate_quant_test, ascend960dt_int8_output_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, nullptr, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// v121 fail: trans=true not supported
TEST_F(l2_rotate_quant_test, ascend960dt_trans_true_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 1, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, true),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// v121 fail: N not divisible by K
TEST_F(l2_rotate_quant_test, ascend960dt_hif4_n_not_divisible_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({4, 100}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 100}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4, 2, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    char* roundMode = const_cast<char*>("rint");
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, (aclTensor*)nullptr, -1, roundMode, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

// v121 fail: groupList仅支持Ascend950, 960平台hif4下V2接口传入groupList必须拒绝
TEST_F(l2_rotate_quant_test, ascend960dt_rotate_quant_v2_hif4_reject_groups_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND960DT);
    TensorDesc x_desc = TensorDesc({7, 96}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({3, 32, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc groups_desc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({7, 96}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({7, 2, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuantV2,
                        INPUT(x_desc, rot_desc, (aclTensor*)nullptr, groups_desc, 0, -1, (char*)nullptr, 0, 0.0, false),
                        OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}
