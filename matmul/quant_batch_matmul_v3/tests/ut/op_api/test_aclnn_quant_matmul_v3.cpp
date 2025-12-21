/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <vector>
#include <thread>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_quant_matmul_v3.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

struct QuantBatchMatmulV3TestParam {
    string caseName;
    vector<int64_t> x1;
    vector<int64_t> x2;
    vector<int64_t> scale;
    vector<int64_t> offset;
    vector<int64_t> bias;
    vector<int64_t> out;
    vector<int64_t> x1_stride;
    vector<int64_t> x2_stride;
    aclDataType scaleType;
    aclDataType outType;
    // out
    aclnnStatus expect_ret;
};

class l2_QuantBatchMatmulV3_test : public testing::TestWithParam<QuantBatchMatmulV3TestParam> {
 protected:
  static void SetUpTestCase() { cout << "l2_QuantBatchMatmulV3_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_QuantBatchMatmulV3_test TearDown" << endl; }
};
static void TestOneParamCase(const QuantBatchMatmulV3TestParam &param)
{
  std::cout << "run case " << param.caseName << std::endl;
  TensorDesc x1_desc = TensorDesc(param.x1, ACL_INT8, ACL_FORMAT_ND, param.x1_stride);
  TensorDesc x2_desc = TensorDesc(param.x2, ACL_INT8, ACL_FORMAT_ND, param.x2_stride);
  TensorDesc scale_desc = TensorDesc(param.scale, param.scaleType, ACL_FORMAT_ND);
  TensorDesc offset_desc = TensorDesc(param.offset, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc(param.bias, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc(param.out, param.outType, ACL_FORMAT_ND);
  bool hasOffset = param.offset.size() == 0 ? false : true;
  bool hasBias = param.bias.size() == 0 ? false : true;
  auto mandtoryInput = std::make_tuple(x1_desc, x2_desc, scale_desc);
  aclnnStatus aclRet = ACLNN_ERR_PARAM_INVALID;
  uint64_t workspace_size = 0;
  if (hasOffset && hasBias) {
    auto ut = OP_API_UT(aclnnQuantMatmulV3,
                        std::tuple_cat(mandtoryInput, std::make_tuple(offset_desc, bias_desc, false, false)),
                        OUTPUT(out_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  } else if (hasOffset) {
    auto ut = OP_API_UT(aclnnQuantMatmulV3,
                        std::tuple_cat(mandtoryInput, std::make_tuple(offset_desc, nullptr, false, false)),
                        OUTPUT(out_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  } else if (hasBias) {
    auto ut = OP_API_UT(aclnnQuantMatmulV3,
                        std::tuple_cat(mandtoryInput, std::make_tuple(nullptr, bias_desc, false, false)),
                        OUTPUT(out_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  } else {
    auto ut = OP_API_UT(aclnnQuantMatmulV3,
                        std::tuple_cat(mandtoryInput, std::make_tuple(nullptr, nullptr, false, false)),
                        OUTPUT(out_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  }
//   EXPECT_EQ(aclRet, param.expect_ret);
  std::cout << "end case " << param.caseName << std::endl;
}

TEST_P(l2_QuantBatchMatmulV3_test, ascend910B2_generalTest)
{
    QuantBatchMatmulV3TestParam param = GetParam();
    TestOneParamCase(param);
}
static QuantBatchMatmulV3TestParam casesParams[] = {
    {"ascend910B2_test_quant_bmm_v3_A8W8FP16_false_false", {16, 32}, {32, 16}, {16}, {}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_FLOAT16, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v3_A8W8FP16_false_true", {16, 32}, {32, 16}, {16}, {}, {16}, {16, 16}, {32, 1}, {1, 16}, ACL_UINT64, ACL_FLOAT16, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v3_A8W8FP16_false_true_max_dim_error", {2, 2, 2, 2, 2, 16, 32}, {2, 2, 2, 2, 2, 32, 16}, {16}, {}, {16}, {2, 2, 2, 2, 2, 16, 16}, {2, 2, 2, 2, 2, 32, 1}, {2, 2, 2, 2, 2, 1, 16}, ACL_UINT64, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8_x1_speical_out_case", {16, 32}, {32, 16}, {16}, {}, {16}, {16, 1, 16}, {32, 1}, {1, 16}, ACL_UINT64, ACL_FLOAT16, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v3_A8W8FP16_hasoffset_false_true", {16, 32}, {32, 16}, {16}, {}, {16}, {16, 16}, {32, 1}, {1, 16}, ACL_UINT64, ACL_FLOAT16, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v3_A8W8FP16_true_false", {16, 32}, {32, 16}, {16}, {}, {16}, {16, 16}, {1, 16}, {16, 1}, ACL_UINT64, ACL_FLOAT16, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v3_A8W8C8_false_false", {16, 32}, {32, 16}, {16}, {16}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_INT8, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v3_A8W8C8_scale_fp32_false_false", {16, 32}, {32, 16}, {16}, {16}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_FLOAT, ACL_INT8, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8C8_bias_three_dim_false_false", {1, 16, 32}, {1, 32, 16}, {16}, {16}, {1, 1, 16}, {1, 16, 16}, {1, 32, 1}, {1, 16, 1}, ACL_UINT64, ACL_INT8, ACLNN_SUCCESS},
    // {"ascend910B2_test_quant_bmm_v3_A8W8FP16_x1_dim_error", {32}, {32, 16}, {16}, {}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_FLOAT16, ACLNN_ERR_INNER_NULLPTR},
    // {"ascend910B2_test_quant_bmm_v3_A8W8FP16_x2_dim_error", {16, 32}, {16}, {16}, {}, {16}, {16, 16}gen, {32, 1}, {16, 1}, ACL_UINT64, ACL_FLOAT16, ACLNN_ERR_INNER_NULLPTR},
    {"ascend910B2_test_quant_bmm_v3_A8W8FP16_k_dim_error", {16, 2}, {32, 16}, {16}, {}, {16}, {16, 16}, {2, 1}, {16, 1}, ACL_UINT64, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8FP16_scale_dim_error", {16, 32}, {32, 16}, {2}, {}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8C8_bias_three_dim_false_false_error", {16, 32}, {32, 16}, {16}, {16}, {1, 1, 16}, {16, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_INT8, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8C8_bias_batch_dim_infered_error", {1, 16, 32}, {2, 32, 16}, {16}, {16}, {3, 1, 16}, {2, 16, 16}, {1, 32, 1}, {1, 16, 1}, ACL_UINT64, ACL_INT8, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8C8_bias_second_dim_error", {1, 16, 32}, {2, 32, 16}, {16}, {16}, {2, 2, 16}, {2, 16, 16}, {1, 32, 1}, {1, 16, 1}, ACL_UINT64, ACL_INT8, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8C8_input_batch_dim_error", {3, 16, 32}, {2, 32, 16}, {16}, {16}, {16}, {2, 16, 16}, {1, 32, 1}, {1, 16, 1}, ACL_UINT64, ACL_INT8, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8C8_bias_third_dim_error", {1, 16, 32}, {2, 32, 16}, {16}, {16}, {2, 1, 10}, {2, 16, 16}, {1, 32, 1}, {1, 16, 1}, ACL_UINT64, ACL_INT8, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8C8_output_shape_m_dim_error", {16, 32}, {32, 16}, {16}, {16}, {16}, {10, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_INT8, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8C8_output_shape_n_dim_error", {16, 32}, {32, 16}, {16}, {16}, {16}, {16, 10}, {32, 1}, {16, 1}, ACL_UINT64, ACL_INT8, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8C8_output_batch_dim_error", {1, 16, 32}, {2, 32, 16}, {16}, {16}, {16}, {3, 16, 16}, {1, 32, 1}, {1, 16, 1}, ACL_UINT64, ACL_INT8, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8FP16_outpu_dim_error", {16, 32}, {32, 16}, {16}, {}, {16}, {16, 16, 1}, {32, 1}, {16, 1}, ACL_UINT64, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8BF16_wrong_scale_dtype_false_false_error", {16, 32}, {32, 16}, {16}, {}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_BF16, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8C8_wrong_scale_dtype_false_false_error", {16, 32}, {32, 16}, {16}, {16}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_BF16, ACL_INT8, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8FP16_wrong_x1_dimtrue_false_error", {1, 1, 1, 1, 1, 1, 1, 1, 16, 32}, {32, 16}, {16}, {}, {16}, {16, 16}, {1, 1, 1, 1, 1, 1, 1, 1, 32, 1}, {16, 1}, ACL_UINT64, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8FP16_wrong_k_dim_false_false_error", {16, 32}, {33, 16}, {16}, {}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8FP16_wrong_scale_dim_false_false_error", {16, 32}, {32, 16}, {15}, {}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8C8_wrong_offset_dim_false_false_error", {16, 32}, {32, 16}, {16}, {15}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_INT8, ACLNN_ERR_PARAM_INVALID},
    {"ascend910B2_test_quant_bmm_v3_A8W8C8_wrong_inner_axis_false_false_error", {16, 70000}, {70000, 16}, {16}, {16}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_INT8, ACLNN_ERR_PARAM_INVALID},
};

INSTANTIATE_TEST_SUITE_P(QuantBatchMatmulV3, l2_QuantBatchMatmulV3_test, testing::ValuesIn(casesParams));

static void ThreadFunc(const QuantBatchMatmulV3TestParam *params, size_t testcase_num, size_t thread_idx,
                       size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const QuantBatchMatmulV3TestParam *params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

// TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_multi_thread)
// {
//     TestMultiThread(casesParams, sizeof(casesParams) / sizeof(QuantBatchMatmulV3TestParam), 3);
// }

TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_test_QuantBatchMatmulV3_special_case_01)
{
    TensorDesc x1_desc = TensorDesc({1, 1}, ACL_INT8, ACL_FORMAT_ND, {1, 1});
    TensorDesc x2_desc = TensorDesc({1, 1}, ACL_INT8, ACL_FORMAT_ND, {1, 1});
    TensorDesc scale_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV3, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_test_quant_bmm_v3_bias_bf16_error_case_02)
{
    TensorDesc x1_desc = TensorDesc({1, 16, 32}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({2, 32, 16}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc bias_desc = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({2, 16, 16}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnQuantMatmulV3, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                        bias_desc, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_test_QuantBatchMatmulV3_wrong_bias_dtype)
{
    TensorDesc x1_desc = TensorDesc({1, 1}, ACL_INT8, ACL_FORMAT_ND, {1, 1});
    TensorDesc x2_desc = TensorDesc({1, 1}, ACL_INT8, ACL_FORMAT_ND, {1, 1});
    TensorDesc scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV3, INPUT(x1_desc, x2_desc, scale_desc, nullptr, bias_desc, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_test_QuantBatchMatmulV3_check_wrong_nz_shape)
{
    TensorDesc x1_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {1, 1}, 0, {10, 10, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnQuantMatmulV3, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                        nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_test_QuantBatchMatmulV3_check_nullptr)
{
    TensorDesc x1_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnQuantMatmulV3, INPUT(x1_desc, x2_desc, nullptr, offset_desc,
                        nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_test_case_a4w4_false_false_case)
{
    TensorDesc x1_desc = TensorDesc({64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 280}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV3, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_test_QuantBatchMatmulV3_check_transposeX1)
{
    // A4W4场景不支持transposeX1=true
    TensorDesc x1_desc = TensorDesc({8, 64}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 280}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV3, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, true, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_test_QuantBatchMatmulV3_check_k1_k2_match)
{
    // A4W4场景scale_dim需要和展开成int4后的n_dim匹配
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 35}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({35}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV3, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_test_QuantBatchMatmulV3_check_output_dtype)
{
    // A4W4场景暂时只支持FLOAT16的输出
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 35}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV3, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_test_QuantBatchMatmulV3_check_x2_dim)
{
    // A4W4场景暂时只支持2维的x2输入
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({2, 8, 35}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV3, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_test_QuantBatchMatmulV3_check_x2_format)
{
    // A4W4场景暂时只支持ND的x2输入
    TensorDesc x1_desc = TensorDesc({16, 32}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({32, 16}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {1, 1}, 0, {10, 10, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV3, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_test_QuantBatchMatmulV3_check_x1_x2_dtype_same)
{
    // x1和x2的数据类型必须一致
    TensorDesc x1_desc = TensorDesc({64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 280}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV3, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_test_QuantBatchMatmulV3_check_x1_shape)
{
    // x1 int4输入时内轴需要为偶数
    TensorDesc x1_desc = TensorDesc({64, 7}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({7, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({8}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV3, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV3_test, ascend910B4_x1_nz_transdata)
{
    // x1 nz transdata
    TensorDesc x1_desc = TensorDesc({320, 3696}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {320, 3696}).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({3696, 8192}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {1, 1}, 0, {256, 231, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({320, 8192}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV3, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
