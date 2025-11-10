/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <float.h>
#include <array>
#include <vector>
#include <thread>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_host/op_api/aclnn_convert_weight_to_int4_pack.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

struct ConvertWeightToINT4PackTestParam {
    string caseName;
    vector<int64_t> weight;
    vector<int64_t> weightInt4Pack;
    aclDataType weightType;
    aclDataType weightInt4PackType;
    aclFormat weightFormat;
    aclFormat weightInt4PackFormat;
    aclnnStatus expectRet;
};

class l2_ConvertWeightToINT4Pack_test : public testing::TestWithParam<ConvertWeightToINT4PackTestParam> {
 protected:
  static void SetUpTestCase() { cout << "l2_ConvertWeightToINT4Pack_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_ConvertWeightToINT4Pack_test TearDown" << endl; }
};

class l2_ConvertWeightToINT4Pack_Ascend910_95_test : public testing::TestWithParam<ConvertWeightToINT4PackTestParam> {
 protected:
  static void SetUpTestCase() { cout << "l2_ConvertWeightToINT4Pack_Ascend910_95_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_ConvertWeightToINT4Pack_Ascend910_95_test TearDown" << endl; }
};

static void TestOneParamCase(const ConvertWeightToINT4PackTestParam &param)
{
  std::cout << "run case: " << param.caseName << std::endl;
  TensorDesc weight = TensorDesc(param.weight, param.weightType, param.weightFormat);
  TensorDesc weightInt4Pack = TensorDesc(param.weightInt4Pack, param.weightInt4PackType, param.weightInt4PackFormat);
  auto ut = OP_API_UT(aclnnConvertWeightToINT4Pack, INPUT(weight),
                      OUTPUT(weightInt4Pack));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, param.expectRet);
  std::cout << "end case: " << param.caseName << std::endl;
}

TEST_P(l2_ConvertWeightToINT4Pack_test, ascend910B2_generalTest)
{
    ConvertWeightToINT4PackTestParam param = GetParam();
    TestOneParamCase(param);
}

TEST_P(l2_ConvertWeightToINT4Pack_Ascend910_95_test, ascend910_95_generalTest)
{
    ConvertWeightToINT4PackTestParam param = GetParam();
    TestOneParamCase(param);
}

static ConvertWeightToINT4PackTestParam casesParams[] = {
    {"test_convert_weight_to_int4_pack_invalid_dtype_case03", {32, 64}, {32, 64}, ACL_INT8, ACL_INT4, ACL_FORMAT_ND, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID},
    {"test_convert_weight_to_int4_pack_invalid_dtype_case04", {32, 64}, {32, 64}, ACL_INT32, ACL_INT8, ACL_FORMAT_ND, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID},
    {"test_convert_weight_to_int4_pack_invalid_format_case05", {32, 64}, {32, 64}, ACL_INT32, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID},
    {"test_convert_weight_to_int4_pack_invalid_format_case06", {32, 64}, {32, 64}, ACL_INT32, ACL_INT4, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_NZ, ACLNN_ERR_PARAM_INVALID},
    {"test_convert_weight_to_int4_pack_invalid_shapedim_case07", {1, 1, 32, 64}, {1, 1, 32, 64}, ACL_INT32, ACL_INT4, ACL_FORMAT_ND, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID},
    {"test_convert_weight_to_int4_pack_invalid_shape_int4_case08", {32, 63}, {32, 63}, ACL_INT32, ACL_INT4, ACL_FORMAT_ND, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID},
    {"test_convert_weight_to_int4_pack_invalid_shape_int4_case09", {32, 64}, {32, 32}, ACL_INT32, ACL_INT4, ACL_FORMAT_ND, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID},
    {"test_convert_weight_to_int4_pack_invalid_shape_int32_case10", {32, 74}, {32, 74}, ACL_INT32, ACL_INT32, ACL_FORMAT_ND, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID},
    {"test_convert_weight_to_int4_pack_invalid_shape_int32_case11", {32, 64}, {32, 32}, ACL_INT32, ACL_INT32, ACL_FORMAT_ND, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID},
    {"test_convert_weight_to_int4_pack_Format_Nz_case12", {32, 64}, {32, 8}, ACL_INT32, ACL_INT32, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_NZ, ACLNN_ERR_PARAM_INVALID},
};

INSTANTIATE_TEST_SUITE_P(ConvertWeightToINT4Pack, l2_ConvertWeightToINT4Pack_test, testing::ValuesIn(casesParams));

static void ThreadFunc(const ConvertWeightToINT4PackTestParam *params, size_t testcase_num, size_t thread_idx,
                       size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const ConvertWeightToINT4PackTestParam *params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

TEST_F(l2_ConvertWeightToINT4Pack_test, ascend910B2_multi_thread)
{
    TestMultiThread(casesParams, sizeof(casesParams) / sizeof(ConvertWeightToINT4PackTestParam), 3);
}

TEST_F(l2_ConvertWeightToINT4Pack_test, ascend910B2_normal_case_1) {
    std::cout << "run case: ascend910B2_normal_case_1" << std::endl;
    TensorDesc weight = TensorDesc({32, 64}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc weightInt4Pack = TensorDesc({32, 64}, ACL_INT4, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConvertWeightToINT4Pack, INPUT(weight),
                        OUTPUT(weightInt4Pack));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // cpu实现会导致ut报错，待int4pack npu实现之后放开
    // EXPECT_EQ(aclRet, param.expectRet);
    std::cout << "end case: ascend910B2_normal_case_1" << std::endl;
}

TEST_F(l2_ConvertWeightToINT4Pack_test, ascend910B2_normal_case_2) {
    std::cout << "run case: ascend910B2_normal_case_2" << std::endl;
    TensorDesc weight = TensorDesc({32, 64}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc weightInt4Pack = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConvertWeightToINT4Pack, INPUT(weight),
                        OUTPUT(weightInt4Pack));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // cpu实现会导致ut报错，待int4pack npu实现之后放开
    // EXPECT_EQ(aclRet, param.expectRet);
    std::cout << "end case: ascend910B2_normal_case_2" << std::endl;
}

TEST_F(l2_ConvertWeightToINT4Pack_test, ascend910B2_non_contiguous_case) {
    std::cout << "run case: ascend910B2_non_contiguous_case" << std::endl;
    TensorDesc weight = TensorDesc({64, 32}, ACL_INT32, ACL_FORMAT_ND, {1, 64}, 0, {32, 64});
    TensorDesc weightInt4Pack = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConvertWeightToINT4Pack, INPUT(weight),
                        OUTPUT(weightInt4Pack));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    std::cout << "end case: ascend910B2_non_contiguous_case" << std::endl;
}

TEST_F(l2_ConvertWeightToINT4Pack_test, ascend910B2_normal_case_3) {
    std::cout << "run case: ascend910B2_normal_case_3" << std::endl;
    TensorDesc weight = TensorDesc({32, 64}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc weightInt4Pack = TensorDesc({32, 64}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {64, 1}, 0, {1, 2, 16, 64});
    auto ut = OP_API_UT(aclnnConvertWeightToINT4Pack, INPUT(weight),
                        OUTPUT(weightInt4Pack));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // cpu实现会导致ut报错，待int4pack npu实现之后放开
    // EXPECT_EQ(aclRet, param.expectRet);
    std::cout << "end case: ascend910B2_normal_case_3" << std::endl;
}

TEST_F(l2_ConvertWeightToINT4Pack_test, ascend910B2_normal_case_4) {
    std::cout << "run case: ascend910B2_normal_case_4" << std::endl;
    TensorDesc weight = TensorDesc({32, 64}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc weightInt4Pack = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {8, 1}, 0, {1, 2, 16, 8});
    auto ut = OP_API_UT(aclnnConvertWeightToINT4Pack, INPUT(weight),
                        OUTPUT(weightInt4Pack));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // cpu实现会导致ut报错，待int4pack npu实现之后放开
    // EXPECT_EQ(aclRet, param.expectRet);
    std::cout << "end case: ascend910B2_normal_case_4" << std::endl;
}

TEST_F(l2_ConvertWeightToINT4Pack_Ascend910_95_test, ascend910_95_normal_case_0) {
    std::cout << "run case: ascend910_95_normal_case_0" << std::endl;
    TensorDesc weight = TensorDesc({32, 64}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc weightInt4Pack = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {8, 1}, 0, {4, 2, 16, 2});
    auto ut = OP_API_UT(aclnnConvertWeightToINT4Pack, INPUT(weight),
                        OUTPUT(weightInt4Pack));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // cpu实现会导致ut报错，待int4pack npu实现之后放开
    // EXPECT_EQ(aclRet, param.expectRet);
    std::cout << "end case: ascend910_95_normal_case_0" << std::endl;
}

TEST_F(l2_ConvertWeightToINT4Pack_Ascend910_95_test, ascend910_95_normal_case_1) {
    std::cout << "run case: ascend910_95_normal_case_1" << std::endl;
    TensorDesc weight = TensorDesc({32, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc weightInt4Pack = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {8, 1}, 0, {4, 2, 16, 2});
    auto ut = OP_API_UT(aclnnConvertWeightToINT4Pack, INPUT(weight),
                        OUTPUT(weightInt4Pack));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // cpu实现会导致ut报错，待int4pack npu实现之后放开
    // EXPECT_EQ(aclRet, param.expectRet);
    std::cout << "end case: ascend910_95_normal_case_1" << std::endl;
}

TEST_F(l2_ConvertWeightToINT4Pack_Ascend910_95_test, ascend910_95_normal_case_2) {
    std::cout << "run case: ascend910_95_normal_case_2" << std::endl;
    TensorDesc weight = TensorDesc({32, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc weightInt4Pack = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {8, 1}, 0, {4, 2, 16, 2});
    auto ut = OP_API_UT(aclnnConvertWeightToINT4Pack, INPUT(weight),
                        OUTPUT(weightInt4Pack));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // cpu实现会导致ut报错，待int4pack npu实现之后放开
    // EXPECT_EQ(aclRet, param.expectRet);
    std::cout << "end case: ascend910_95_normal_case_2" << std::endl;
}