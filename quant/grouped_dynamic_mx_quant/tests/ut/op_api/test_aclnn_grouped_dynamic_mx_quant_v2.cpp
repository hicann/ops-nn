/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_grouped_dynamic_mx_quant_v2.cpp
 * \brief
 */
#include <float.h>

#include <array>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_grouped_dynamic_mx_quant_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_grouped_dynamic_mx_quant_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_grouped_dynamic_mx_quant_v2_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_grouped_dynamic_mx_quant_v2_test TearDown" << endl; }
};

TEST_F(l2_grouped_dynamic_mx_quant_v2_test, ascend950_grouped_dynamic_mx_quant_v2_bf16_E4M3)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E4M3FN);
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    const char* roundMode = "rint";

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnGroupedDynamicMxQuantV2,
                        INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize, scaleAlg, dstTypeMax),
                        OUTPUT(y_desc, mxscale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_v2_test, ascend950_grouped_dynamic_mx_quant_v2_bf16_E5M2)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    const char* roundMode = "rint";

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnGroupedDynamicMxQuantV2,
                        INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize, scaleAlg, dstTypeMax),
                        OUTPUT(y_desc, mxscale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_v2_test, ascend950_grouped_dynamic_mx_quant_v2_fp16_E4M3)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E4M3FN);
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    const char* roundMode = "rint";

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnGroupedDynamicMxQuantV2,
                        INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize, scaleAlg, dstTypeMax),
                        OUTPUT(y_desc, mxscale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_v2_test, ascend950_grouped_dynamic_mx_quant_v2_fp16_E5M2)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    const char* roundMode = "rint";

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnGroupedDynamicMxQuantV2,
                        INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize, scaleAlg, dstTypeMax),
                        OUTPUT(y_desc, mxscale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_v2_test, ascend950_grouped_dynamic_mx_quant_v2_fp16_E4M3FN_odd_tail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E4M3FN);
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    const char* roundMode = "rint";

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnGroupedDynamicMxQuantV2,
                        INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize, scaleAlg, dstTypeMax),
                        OUTPUT(y_desc, mxscale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_v2_test, ascend950_grouped_dynamic_mx_quant_v2_fp16_E5M2_odd_tail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    const char* roundMode = "rint";

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnGroupedDynamicMxQuantV2,
                        INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize, scaleAlg, dstTypeMax),
                        OUTPUT(y_desc, mxscale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_v2_test, ascend950_grouped_dynamic_mx_quant_v2_fp16_E4M3FN_dynamic_tail)
{
    TensorDesc x_desc = TensorDesc({64, -1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, -1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({3, -1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E4M3FN);
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    const char* roundMode = "rint";

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnGroupedDynamicMxQuantV2,
                        INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize, scaleAlg, dstTypeMax),
                        OUTPUT(y_desc, mxscale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_v2_test, ascend950_grouped_dynamic_mx_quant_v2_fp16_E5M2_dynamic_tail)
{
    TensorDesc x_desc = TensorDesc({64, -1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, -1}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({3, -1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    const char* roundMode = "rint";

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnGroupedDynamicMxQuantV2,
                        INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize, scaleAlg, dstTypeMax),
                        OUTPUT(y_desc, mxscale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_v2_test, ascend950_grouped_dynamic_mx_quant_v2_fp16_E2M1_even_tail)
{
    TensorDesc x_desc = TensorDesc({64, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 4}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({3, 4, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT4_E2M1);
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    const char* roundMode = "rint";

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnGroupedDynamicMxQuantV2,
                        INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize, scaleAlg, dstTypeMax),
                        OUTPUT(y_desc, mxscale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_v2_test, ascend950_grouped_dynamic_mx_quant_v2_fp16_E1M2_even_tail)
{
    TensorDesc x_desc = TensorDesc({64, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 4}, ACL_FLOAT4_E1M2, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({3, 4, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT4_E1M2);
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    const char* roundMode = "rint";

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnGroupedDynamicMxQuantV2,
                        INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize, scaleAlg, dstTypeMax),
                        OUTPUT(y_desc, mxscale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_v2_test, ascend950_grouped_dynamic_mx_quant_v2_fp16_E2M1_odd_tail_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT4_E2M1);
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    const char* roundMode = "rint";

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnGroupedDynamicMxQuantV2,
                        INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize, scaleAlg, dstTypeMax),
                        OUTPUT(y_desc, mxscale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_v2_test, ascend950_grouped_dynamic_mx_quant_v2_fp16_E1M2_odd_tail_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT4_E1M2, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT4_E1M2);
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    const char* roundMode = "rint";

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnGroupedDynamicMxQuantV2,
                        INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize, scaleAlg, dstTypeMax),
                        OUTPUT(y_desc, mxscale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_v2_test, ascend950_grouped_dynamic_mx_quant_v2_fp16_E2M1_dynamic_tail)
{
    TensorDesc x_desc = TensorDesc({64, -1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, -1}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({3, -1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT4_E2M1);
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    const char* roundMode = "rint";

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnGroupedDynamicMxQuantV2,
                        INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize, scaleAlg, dstTypeMax),
                        OUTPUT(y_desc, mxscale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_v2_test, ascend950_grouped_dynamic_mx_quant_v2_fp16_E1M2_dynamic_tail)
{
    TensorDesc x_desc = TensorDesc({64, -1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, -1}, ACL_FLOAT4_E1M2, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({3, -1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT4_E1M2);
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    const char* roundMode = "rint";

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnGroupedDynamicMxQuantV2,
                        INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize, scaleAlg, dstTypeMax),
                        OUTPUT(y_desc, mxscale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
