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
#include <thread>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_quant_matmul_v4.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

struct QuantBatchMatmulV4TestParam {
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

class l2_QuantBatchMatmulV4_test : public testing::TestWithParam<QuantBatchMatmulV4TestParam> {
 protected:
  static void SetUpTestCase() { cout << "l2_QuantBatchMatmulV4_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_QuantBatchMatmulV4_test TearDown" << endl; }
};

class l2_QuantBatchMatmulV4_test_310P : public testing::TestWithParam<QuantBatchMatmulV4TestParam> {
 protected:
  static void SetUpTestCase() { cout << "l2_QuantBatchMatmulV4_test_310P SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_QuantBatchMatmulV4_test_310P TearDown" << endl; }
};

class l2_QuantBatchMatmulV4_test_910_95 : public testing::TestWithParam<QuantBatchMatmulV4TestParam> {
  protected:
      static void SetUpTestCase() { cout << "l2_QuantBatchMatmulV4_test_910_95 SetUp" << endl; }

      static void TearDownTestCase() { cout << "l2_QuantBatchMatmulV4_test_910_95 TearDown" << endl; }
  };

static void TestOneParamCase(const QuantBatchMatmulV4TestParam &param)
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
  bool hasPertoken = param.bias.size() == 0 ? false : true;
  auto mandtoryInput = std::make_tuple(x1_desc, x2_desc, scale_desc);
  aclnnStatus aclRet = ACLNN_ERR_PARAM_INVALID;
  uint64_t workspace_size = 0;
  if (hasOffset && hasBias) {
    auto ut = OP_API_UT(aclnnQuantMatmulV4,
                        std::tuple_cat(mandtoryInput, std::make_tuple(offset_desc, nullptr, bias_desc, false, false)),
                        OUTPUT(out_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  } else if (hasOffset) {
    auto ut = OP_API_UT(aclnnQuantMatmulV4,
                        std::tuple_cat(mandtoryInput, std::make_tuple(offset_desc, nullptr, nullptr, false, false)),
                        OUTPUT(out_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  } else if (hasBias) {
    auto ut = OP_API_UT(aclnnQuantMatmulV4,
                        std::tuple_cat(mandtoryInput, std::make_tuple(nullptr, nullptr, bias_desc, false, false)),
                        OUTPUT(out_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  } else {
    auto ut = OP_API_UT(aclnnQuantMatmulV4,
                        std::tuple_cat(mandtoryInput, std::make_tuple(nullptr, nullptr, nullptr, false, false)),
                        OUTPUT(out_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  }
  // EXPECT_EQ(aclRet, param.expect_ret);
  std::cout << "end case " << param.caseName << std::endl;
}

TEST_P(l2_QuantBatchMatmulV4_test, ascend910B2_generalTest)
{
    QuantBatchMatmulV4TestParam param = GetParam();
    TestOneParamCase(param);
}
TEST_P(l2_QuantBatchMatmulV4_test_310P, ascend310P_generalTest)
{
    QuantBatchMatmulV4TestParam param = GetParam();
    TestOneParamCase(param);
}
TEST_P(l2_QuantBatchMatmulV4_test_910_95, ascend910_95_generalTest)
{
    QuantBatchMatmulV4TestParam param = GetParam();
    TestOneParamCase(param);
}

static QuantBatchMatmulV4TestParam casesParams[] = {
     //caseName, x1, x2, scale, offset, bias, out, x1_stride, x2_stride, scaleType, outType, expect_ret
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
    // {"ascend910B2_test_quant_bmm_v3_A8W8FP16_x2_dim_error", {16, 32}, {16}, {16}, {}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_FLOAT16, ACLNN_ERR_INNER_NULLPTR},
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
    {"ascend910B2_test_quant_bmm_v3_A8W8INT32_false_false_scale_bf16", {16, 32}, {32, 16}, {16}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_BF16, ACL_INT32, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v3_A8W8INT32_false_false_scale_bf16", {16, 32}, {32, 16}, {16}, {}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_BF16, ACL_INT32, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v3_A8W8INT32_false_false_bias_int32", {16, 32}, {32, 16}, {16}, {}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_FLOAT, ACL_INT32, ACLNN_SUCCESS},
    //out int32 wrong scale type
    {"ascend910B2_test_quant_bmm_v3_A8W8INT32_false_false_wrong_scale", {16, 32}, {32, 16}, {16}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_INT32, ACLNN_ERR_PARAM_INVALID},
};

static QuantBatchMatmulV4TestParam casesParamsAscend310P[] = {
     //caseName, x1, x2, scale, offset, bias, out, x1_stride, x2_stride, scaleType, outType, expect_ret
    {"ascend310P_test_quant_bmm_v3_A8W8FP16_false_false", {16, 32}, {32, 16}, {16}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_FLOAT16, ACLNN_SUCCESS},
    {"ascend310P_test_quant_bmm_v3_A8W8FP16_false_false_scale_type_wrong", {16, 32}, {32, 16}, {16}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_BF16, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"ascend310P_test_quant_bmm_v3_A8W8FP16_false_false_out_type_wrong", {16, 32}, {32, 16}, {16}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_UINT64, ACL_BF16, ACLNN_ERR_PARAM_INVALID},
};

static QuantBatchMatmulV4TestParam casesParams910_95[] = {
    //caseName, x1, x2, scale, offset, bias, out, x1_stride, x2_stride, scaleType, outType, expect_ret
    {"ascend910_95_test_quant_bmm_v5_A8W8CB16_pertoken_scale_bfloat16_bias_none", {16, 32}, {32, 16}, {1}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_BF16, ACL_BF16, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v3_A8W8CB16_pertoken_scale_bfloat16_bias_int32", {16, 32}, {32, 16}, {16}, {}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_BF16, ACL_BF16, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v3_A8W8CB16_pertoken_scale_bfloat16_bias_float", {16, 32}, {32, 16}, {1}, {}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_BF16, ACL_BF16, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v3_A8W8CB16_pertoken_scale_bfloat16_bias_bfloat16", {16, 32}, {32, 16}, {16}, {}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_BF16, ACL_BF16, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v3_A8W8CB16_pertoken_scale_float_bias_float", {16, 32}, {32, 16}, {1}, {}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_FLOAT,  ACL_BF16, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v3_A8W8CB16_pertoken_scale_float_bias_bfloat16", {16, 32}, {32, 16}, {16}, {}, {16}, {16, 16}, {32, 1}, {16, 1}, ACL_FLOAT, ACL_BF16, ACLNN_SUCCESS},
};


INSTANTIATE_TEST_SUITE_P(QuantBatchMatmulV4, l2_QuantBatchMatmulV4_test, testing::ValuesIn(casesParams));
INSTANTIATE_TEST_SUITE_P(Ascend310P_QuantBatchMatmulV4, l2_QuantBatchMatmulV4_test_310P, testing::ValuesIn(casesParamsAscend310P));
INSTANTIATE_TEST_SUITE_P(Ascend910_95_QuantBatchMatmulV4, l2_QuantBatchMatmulV4_test_910_95, testing::ValuesIn(casesParams910_95));

static void ThreadFunc(const QuantBatchMatmulV4TestParam *params, size_t testcase_num, size_t thread_idx,
                       size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const QuantBatchMatmulV4TestParam *params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_multi_thread)
// {
//     TestMultiThread(casesParams, sizeof(casesParams) / sizeof(QuantBatchMatmulV4TestParam), 3);
// }

// TEST_F(l2_QuantBatchMatmulV4_test_310P, ascend310P_multi_thread)
// {
//     TestMultiThread(casesParamsAscend310P, sizeof(casesParamsAscend310P) / sizeof(QuantBatchMatmulV4TestParam), 3);
// }

TEST_F(l2_QuantBatchMatmulV4_test_910_95, ascend910_95_multi_thread)
{
    TestMultiThread(casesParams910_95, sizeof(casesParams910_95) / sizeof(QuantBatchMatmulV4TestParam), 3);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_special_case_01)
{
    TensorDesc x1_desc = TensorDesc({1, 1}, ACL_INT8, ACL_FORMAT_ND, {1, 1});
    TensorDesc x2_desc = TensorDesc({1, 1}, ACL_INT8, ACL_FORMAT_ND, {1, 1});
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_0) {
  // 含pertoken的调用
  std::cout << "run case 0" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  std::cout << "end case 0" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_1) {
  //覆盖pertoken的类型异常检测
  std::cout << "run case 1" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 1" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_2) {
  //覆盖pertoken的形状异常检测
  std::cout << "run case 2" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 2" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_3) {
  //覆盖格式转换
  std::cout << "run case ascend910B2_test_case_3" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case ascend910B2_test_case_3" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_4) {
  // 覆盖：当offset存在时，out只能是int8
  std::cout << "run case 4" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);;
  auto pertoken_desc = nullptr;
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 4" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_5) {
  //覆盖：out是bf16，scale可以bf16或fp32
  std::cout << "run case 5" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  std::cout << "end case 5" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_6) {
  //覆盖：scale是bf16，out不是bf16
  std::cout << "run case 6" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 6" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_7) {
  //覆盖：pertoken存在时，out只能是int8
  std::cout << "run case 7" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 7" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_8) {
  // 覆盖：pertoken存在时，out是float16，scale必须是float
  std::cout << "run case 8" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 8" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_9) {
  // scale dim(0) != 1 && != n
  std::cout << "run case 9" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 9" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_10) {
  // 覆盖：bias dim数量不是3也不是1
  std::cout << "run case 10" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({1, 1, 1, 1, 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 10" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_11) {
  // 覆盖：bias是一维，但是值不等于x2ndim
  std::cout << "run case 11" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({256}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 11" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_12) {
  // 覆盖：output batch need to be only 1 dim when bias dim is 3
  std::cout << "run case 12" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({1, 1 , 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({2, 1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 12" << std::endl;
} //bias还没写完

//checkout out shape
TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_13) {
  // inferedOutDimNum != outDimNum
  std::cout << "run case 13" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 1, 1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 13" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_14) {
  // outMDim != x1MDim
  std::cout << "run case 14" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({32, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 14" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_15) {
  // outNDim != x2NDim
  std::cout << "run case 15" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 32}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 15" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_16) {
  // k != k
  std::cout << "run case 16" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({513, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 16" << std::endl;
}

//offset
TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_17) {
  // offset dim(0) != x2NDim
  std::cout << "run case 17" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto pertoken_desc = nullptr;
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 17" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_18) {
  // 多维 令bias不对
  std::cout << "run case 18" << std::endl;
  TensorDesc x1_desc = TensorDesc({2, 1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({2, 512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({64}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 18" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_19) {
  // check dim range false
  std::cout << "run case 19" << std::endl;
  TensorDesc x1_desc = TensorDesc({2, 1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({2, 512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 19" << std::endl;
}

// bias搞一下
TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_20) {
  // biasSecondDim != 1
  std::cout << "run case 20" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({1, 1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128, 1, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 20" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_21) {
  // biasSecondDim != 1
  std::cout << "run case 21" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({1, 1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto offset_desc = nullptr;
  TensorDesc pertoken_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({1, 128, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({1, 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 21" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_22) {
  std::cout << "run case 22" << std::endl;
  TensorDesc x1_desc = TensorDesc({1, 16, 32}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({2, 32, 16}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto pertoken_desc = nullptr;
  TensorDesc bias_desc = TensorDesc({2, 1, 10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc out_desc = TensorDesc({2, 16, 16}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  std::cout << "end case 22" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_23)
{
    std::cout << "run case 23" << std::endl;
    TensorDesc x1_desc = TensorDesc({1, 70000}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({70000, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({1, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    std::cout << "run case 23" << std::endl;
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_bias_bf16_error_case_24)
{
    std::cout << "run case 24" << std::endl;
    TensorDesc x1_desc = TensorDesc({1, 16, 32}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({2, 32, 16}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto pertoken_desc = nullptr;
    TensorDesc bias_desc = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({2, 16, 16}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, offset_desc,
                        pertoken_desc, bias_desc, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    std::cout << "end case 24" << std::endl;
}


TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_wrong_bias_dtype_case_25)
{
    TensorDesc x1_desc = TensorDesc({1, 1}, ACL_INT8, ACL_FORMAT_ND, {1, 1});
    TensorDesc x2_desc = TensorDesc({1, 1}, ACL_INT8, ACL_FORMAT_ND, {1, 1});
    TensorDesc scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, bias_desc, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_wrong_nz_shape_case_26)
{
    TensorDesc x1_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {1, 1}, 0, {10, 10, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_27)
{
    TensorDesc x1_desc = TensorDesc({64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 280}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_28)
{
    TensorDesc x1_desc = TensorDesc({64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({280, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, true),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_29)
{
    // A4W4场景无pertoken_scale时不支持transposeX1=true
    TensorDesc x1_desc = TensorDesc({8, 64}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 280}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, true, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_30)
{
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 280}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_31)
{
    // A4W4场景scale_dim需要和展开成int4后的n_dim匹配
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 35}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({35}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_32)
{
    // A4W4场景不支持INT8输出
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 35}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_33)
{
    // A4W4场景无pertoken_scale时暂时只支持2维的x2输入
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({2, 8, 35}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_34)
{
    // A4W4场景无pertoken_scale时暂时只支持ND的x2输入
    TensorDesc x1_desc = TensorDesc({16, 32}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({32, 16}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {1, 1}, 0, {10, 10, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_35)
{
    TensorDesc x1_desc = TensorDesc({64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({32}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_36)
{
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({64}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_37)
{
    // x1和x2的数据类型必须一致
    TensorDesc x1_desc = TensorDesc({64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 280}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_38)
{
    // x1 int4输入时内轴需要为偶数
    TensorDesc x1_desc = TensorDesc({64, 7}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({7, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({8}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_39)
{
    // x2 int4输入时内轴需要为偶数
    TensorDesc x1_desc = TensorDesc({64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 13}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({13}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 13}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_40)
{
    // x1 int4输入无pertoken_scale时bias只支持1维
    TensorDesc x1_desc = TensorDesc({2, 64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 14}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({14}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({2, 1, 14}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 64, 14}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, bias_desc, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_41)
{
    // 新增支持组合：A4W4_pertoken_scale_out_fp16
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                        pertoken_desc, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_42)
{
  // 白盒用例：bias不支持int8数据类型
  TensorDesc x1_desc = TensorDesc({16, 681}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {16 * 681}).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({681, 2298}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {681 * 2298}).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({2298}, ACL_BF16, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({2298}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc out_desc = TensorDesc({2, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                      nullptr, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_43)
{
  // 白盒用例：非pertoken场景下bias不支持fp16
  TensorDesc x1_desc = TensorDesc({9984, 3072}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {9984 * 3072}).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({3072, 4864}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {3072 * 4864}).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({4864}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc out_desc = TensorDesc({2, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                      nullptr, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_44)
{
  // 白盒用例：非pertoken场景下当bias为fp32时，out一定为bf16
  TensorDesc x1_desc = TensorDesc({2, 512}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {2 * 512}).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {128 * 512}).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc out_desc = TensorDesc({2, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                      nullptr, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_45)
{
  // 白盒用例：bias为bf16时，out不能为fp16
  TensorDesc x1_desc = TensorDesc({2, 512}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {2 * 512}).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {128 * 512}).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc pertoken_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc out_desc = TensorDesc({2, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_46)
{
  // 白盒用例：bias为fp16时，out不能为bf16
  TensorDesc x1_desc = TensorDesc({2, 512}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {2 * 512}).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {128 * 512}).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc pertoken_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc out_desc = TensorDesc({2, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_47)
{
  // 白盒用例：bias为fp32时，out不能为int8
  TensorDesc x1_desc = TensorDesc({2, 512}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {2 * 512}).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {128 * 512}).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc out_desc = TensorDesc({2, 128}, ACL_INT8, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                      nullptr, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_48)
{
  // 白盒用例：scale为bf16时，out不能为fp16
  TensorDesc x1_desc = TensorDesc({2, 512}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {2 * 512}).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {128 * 512}).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
  TensorDesc pertoken_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc out_desc = TensorDesc({2, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_49)
// {
//   // 新增支持数据类型组合：scale_fp32_bias_fp32_out_fp16
//   TensorDesc x1_desc = TensorDesc({2, 512}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {2 * 512}).ValueRange(-1, 1);
//   TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {128 * 512}).ValueRange(-1, 1);
//   TensorDesc scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
//   TensorDesc pertoken_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
//   TensorDesc bias_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
//   TensorDesc out_desc = TensorDesc({2, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                       pertoken_desc, bias_desc, false, false),
//                       OUTPUT(out_desc));
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_50)
// {
//   // 新增支持数据类型组合：scale_fp32_bias_fp16_out_fp16
//   TensorDesc x1_desc = TensorDesc({2, 512}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {2 * 512}).ValueRange(-1, 1);
//   TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {128 * 512}).ValueRange(-1, 1);
//   TensorDesc scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
//   TensorDesc pertoken_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
//   TensorDesc bias_desc = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-5, 5);
//   TensorDesc out_desc = TensorDesc({2, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                       pertoken_desc, bias_desc, false, false),
//                       OUTPUT(out_desc));
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_51)
// {
//   // 新增支持数据类型组合：scale_fp32_bias_fp32_out_bf16
//   TensorDesc x1_desc = TensorDesc({2, 512}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {2 * 512}).ValueRange(-1, 1);
//   TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {128 * 512}).ValueRange(-1, 1);
//   TensorDesc scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
//   TensorDesc pertoken_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
//   TensorDesc bias_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
//   TensorDesc out_desc = TensorDesc({2, 128}, ACL_BF16, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                       pertoken_desc, bias_desc, false, false),
//                       OUTPUT(out_desc));
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_52)
// {
//   // 新增支持数据类型组合：scale_bf16_bias_fp32_out_bf16
//   TensorDesc x1_desc = TensorDesc({2, 512}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {2 * 512}).ValueRange(-1, 1);
//   TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {128 * 512}).ValueRange(-1, 1);
//   TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
//   TensorDesc pertoken_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
//   TensorDesc bias_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
//   TensorDesc out_desc = TensorDesc({2, 128}, ACL_BF16, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                       pertoken_desc, bias_desc, false, false),
//                       OUTPUT(out_desc));
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_torch_api_case_01)
// {
//     TensorDesc x1_desc = TensorDesc({2, 512}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {2 * 512}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {128 * 512}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc pertoken_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
//     TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
//     TensorDesc out_desc = TensorDesc({2, 128}, ACL_BF16, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                         pertoken_desc, bias_desc, false, false),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_53)
{
  // out int32 bias非int32异常用例
  TensorDesc x1_desc = TensorDesc({2, 512}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {2 * 512}).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {128 * 512}).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
  TensorDesc pertoken_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc out_desc = TensorDesc({2, 128}, ACL_INT32, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                      nullptr, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_54)
{
  // out int32 有pertoken,异常用例
  TensorDesc x1_desc = TensorDesc({2, 512}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {2 * 512}).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {128 * 512}).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
  TensorDesc pertoken_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
  TensorDesc out_desc = TensorDesc({2, 128}, ACL_INT32, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                      pertoken_desc, bias_desc, false, false),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_55)
// {
//     //out int32 tilingkey 0
//     TensorDesc x1_desc = TensorDesc({24, 8192}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {24 * 8192}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({8192, 11264}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {11264 * 8192}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc bias_desc = TensorDesc({11264}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
//     TensorDesc out_desc = TensorDesc({24, 11264}, ACL_INT32, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                         nullptr, bias_desc, false, false),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_56)
// {
//     //out int32 tilingkey 1000
//     TensorDesc x1_desc = TensorDesc({928, 2752}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {928 * 2752}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({2752, 8192}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {8192 * 2752}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc bias_desc = TensorDesc({8192}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
//     TensorDesc out_desc = TensorDesc({928, 8192}, ACL_INT32, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                         nullptr, bias_desc, false, false),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_57)
// {
//     //out int32 tilingkey 1100
//     TensorDesc x1_desc = TensorDesc({32768, 2560}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {32768 * 2560}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({2560, 5120}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {5120 * 2560}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc bias_desc = TensorDesc({5120}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
//     TensorDesc out_desc = TensorDesc({32768, 5120}, ACL_INT32, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                         nullptr, bias_desc, false, false),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_58)
// {
//     //out int32 tilingkey 1
//     TensorDesc x1_desc = TensorDesc({24, 8192}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {24 * 8192}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({11264, 8192}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {11264 * 8192}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc bias_desc = TensorDesc({11264}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
//     TensorDesc out_desc = TensorDesc({24, 11264}, ACL_INT32, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                         nullptr, bias_desc, false, true),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_59)
// {
//     //out int32 tilingkey 1001
//     TensorDesc x1_desc = TensorDesc({1024, 1408}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {1024 * 1408}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({11264, 1408}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {11264 * 1408}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc bias_desc = TensorDesc({11264}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
//     TensorDesc out_desc = TensorDesc({1024, 11264}, ACL_INT32, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                         nullptr, bias_desc, false, true),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_60)
// {
//     //out int32 tilingkey 1101
//     TensorDesc x1_desc = TensorDesc({32768, 2560}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {32768 * 2560}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({5120, 2560}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {5120 * 2560}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc bias_desc = TensorDesc({5120}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
//     TensorDesc out_desc = TensorDesc({32768, 5120}, ACL_INT32, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                         nullptr, bias_desc, false, true),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_61)
// {
//     //out int32 tilingkey 10
//     TensorDesc x1_desc = TensorDesc({1164, 16}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {16 * 1164}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({1164, 6656}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {1164, 6656}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc bias_desc = TensorDesc({6656}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
//     TensorDesc out_desc = TensorDesc({16, 6656}, ACL_INT32, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                         nullptr, bias_desc, true, false),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_62)
// {
//     //out int32 tilingkey 1010
//     TensorDesc x1_desc = TensorDesc({1408, 1024}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {1024 * 1408}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({1408, 11264}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {1408 * 11264}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc bias_desc = TensorDesc({11264}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
//     TensorDesc out_desc = TensorDesc({1024, 11264}, ACL_INT32, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                         nullptr, bias_desc, true, false),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_63)
// {
//     //out int32 tilingkey 1110
//     TensorDesc x1_desc = TensorDesc({2560, 32768}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {32768 * 2560}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({2560, 5120}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {2560 * 5120}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc bias_desc = TensorDesc({5120}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
//     TensorDesc out_desc = TensorDesc({32768, 5120}, ACL_INT32, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                         nullptr, bias_desc, true, false),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_64)
// {
//     //out int32 tilingkey 11
//     TensorDesc x1_desc = TensorDesc({6656, 16}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {16 * 6656}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({4480, 6656}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {4480 * 6656}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc bias_desc = TensorDesc({4480}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
//     TensorDesc out_desc = TensorDesc({16, 4480}, ACL_INT32, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
//                         nullptr, bias_desc, true, true),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_65)
{
    //out int32 tilingkey 1011
    TensorDesc x1_desc = TensorDesc({1408, 1024}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {1024 * 1408}).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({11264, 1408}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {1, 1}, 0, {44, 352, 32, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({11264}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({1024, 11264}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                        nullptr, bias_desc, true, true),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_66)
{
    //out int32 tilingkey 1111
    TensorDesc x1_desc = TensorDesc({2560, 32768}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {32768 * 2560}).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({5120, 2560}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {1, 1}, 0, {80, 160, 32, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({5120}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({32768, 5120}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                        nullptr, bias_desc, true, true),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_67)
{
    // 新增支持组合：A4W4_pertoken_scale_out_bf16
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                        pertoken_desc, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_68)
{
    //A4W4有pertoken_scale不支持transposex1=True
    TensorDesc x1_desc = TensorDesc({8, 64}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 280}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                        pertoken_desc, nullptr, true, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_69)
{
    // A4W4 pertoken_scale 不支持3维x2输入
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({2, 8, 35}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({2, 64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                        pertoken_desc, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_70)
{
    // A4W4 pertoken_scale 不支持NZ x2输入
    TensorDesc x1_desc = TensorDesc({128, 128}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({128, 128}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {128, 1}, 0, {2, 8, 16, 64}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({128, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                        pertoken_desc, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_71)
{
    // A4W4 pertoken_scale 不支持3维bias
    TensorDesc x1_desc = TensorDesc({2, 64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 14}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc bias_desc = TensorDesc({2, 1, 14}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 64, 14}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                        pertoken_desc, bias_desc, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B4_test_case_72)
{
    // x1 nz transdata
    TensorDesc x1_desc = TensorDesc({320, 3696}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {320, 3696}).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({3696, 8192}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {1, 1}, 0, {256, 231, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({320, 8192}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                        nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B4_test_case_73)
{
    // x1 nz transdata, m out of range
    TensorDesc x1_desc = TensorDesc({92, 3696}, ACL_INT8, ACL_FORMAT_ND, {1, 1}, 0, {92, 3696}).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({3696, 8192}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {1, 1}, 0, {256, 231, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({92, 8192}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                        nullptr, nullptr, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_74)
{
    // A4W4 pertoken_scale bf16 bias不支持fp16输出
    TensorDesc x1_desc = TensorDesc({2, 64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 14}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc bias_desc = TensorDesc({2, 1, 14}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 64, 14}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                        pertoken_desc, bias_desc, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV4_test, ascend910B2_test_case_75)
{
    // A4W4 pertoken_scale fp16 bias不支持bf16输出
    TensorDesc x1_desc = TensorDesc({2, 64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 14}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc bias_desc = TensorDesc({2, 1, 14}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 64, 14}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV4, INPUT(x1_desc, x2_desc, scale_desc, nullptr,
                        pertoken_desc, bias_desc, false, false),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
