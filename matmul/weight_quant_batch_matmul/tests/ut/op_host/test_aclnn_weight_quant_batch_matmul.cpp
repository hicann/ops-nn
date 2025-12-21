/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"


#include "../../../op_host/op_api/aclnn_weight_quant_batch_matmul.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"


using namespace std;

class l2_weight_quant_batch_matmul_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_weight_quant_batch_matmul_test SetUp" << endl; }
  static void TearDownTestCase() { cout << "l2_weight_quant_batch_matmul_test TearDown" << endl; }

  static void MatMulCommonTest(TensorDesc x1_desc, TensorDesc x2_desc, TensorDesc bias_desc,
                               TensorDesc addOffset_desc, TensorDesc mulScale_desc,
                               TensorDesc diagonalMatrix_desc,TensorDesc deqOffset_desc,TensorDesc deqScale_desc,
                               bool transpose_a, bool transpose_b,
                               TensorDesc out_desc, aclnnStatus expect_status) {
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmul,
                        INPUT(x1_desc, x2_desc, diagonalMatrix_desc,
                              deqOffset_desc, deqScale_desc, addOffset_desc, mulScale_desc,
                              bias_desc, transpose_a, transpose_b, 1.0, 1.0), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, expect_status);
    // SAMPLE: precision simulates
    if (expect_status == ACL_SUCCESS) {
      ut.TestPrecision();  // soc version  2. 二段接口
    }
  }
};


TEST_F(l2_weight_quant_batch_matmul_test, ascend910B2_test_mm_case_0) {
  // 使用**Desc描述host api输入输出
  TensorDesc x1_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc addOffset_desc = TensorDesc({1, }, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mulScale_desc = TensorDesc({1, }, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc diagonalMatrix_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc deqOffset_desc = TensorDesc({16, }, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc deqScale_desc = TensorDesc({16, }, ACL_UINT64, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  MatMulCommonTest(x1_desc, x2_desc, bias_desc, addOffset_desc, mulScale_desc,
                   diagonalMatrix_desc, deqOffset_desc, deqScale_desc, false, false,
                   out_desc,  ACL_SUCCESS);
}

TEST_F(l2_weight_quant_batch_matmul_test, ascend910B2_test_mm_case_1) {
  // 使用**Desc描述host api输入输出
  TensorDesc x1_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc addOffset_desc = TensorDesc({1, }, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mulScale_desc = TensorDesc({1, }, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc diagonalMatrix_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc deqOffset_desc = TensorDesc({16, }, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc deqScale_desc = TensorDesc({16, }, ACL_UINT64, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  MatMulCommonTest(x1_desc, x2_desc, bias_desc, addOffset_desc, mulScale_desc,
                   diagonalMatrix_desc, deqOffset_desc, deqScale_desc, false, false,
                   out_desc,  ACLNN_ERR_PARAM_INVALID);
}

// mm 空tensor
TEST_F(l2_weight_quant_batch_matmul_test, ascend910B2_test_mm_emptyinput) {
  // 使用**Desc描述host api输入输出
  TensorDesc x1_desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc addOffset_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mulScale_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc diagonalMatrix_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc deqOffset_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc deqScale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  MatMulCommonTest(x1_desc, x2_desc, bias_desc, addOffset_desc, mulScale_desc,
                   diagonalMatrix_desc, deqOffset_desc, deqScale_desc, false, false,
                   out_desc, ACLNN_SUCCESS);
}

// mm 空tensor1
TEST_F(l2_weight_quant_batch_matmul_test, ascend910B2_test_mm_emptyinput1) {
  // 使用**Desc描述host api输入输出
  TensorDesc x1_desc = TensorDesc({16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({0, 16}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc addOffset_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mulScale_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc diagonalMatrix_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc deqOffset_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc deqScale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  MatMulCommonTest(x1_desc, x2_desc, bias_desc, addOffset_desc, mulScale_desc,
                   diagonalMatrix_desc, deqOffset_desc, deqScale_desc, false, false,
                   out_desc, ACLNN_SUCCESS);
}

// fixpipe scale strange
TEST_F(l2_weight_quant_batch_matmul_test, ascend910B2_test_mm_wrongscale) {
  // 使用**Desc描述host api输入输出
  TensorDesc x1_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc addOffset_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mulScale_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc diagonalMatrix_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc deqOffset_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc deqScale_desc = TensorDesc({16, 2}, ACL_UINT64, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  MatMulCommonTest(x1_desc, x2_desc, bias_desc, addOffset_desc, mulScale_desc,
                   diagonalMatrix_desc, deqOffset_desc, deqScale_desc, false, false,
                   out_desc,  ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_test, ascend910B2_test_mm_wrongoutputshape0) {
  // 使用**Desc描述host api输入输出
  TensorDesc x1_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc addOffset_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mulScale_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc diagonalMatrix_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc deqOffset_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc deqScale_desc = TensorDesc({16, 1}, ACL_UINT64, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  MatMulCommonTest(x1_desc, x2_desc, bias_desc, addOffset_desc, mulScale_desc,
                   diagonalMatrix_desc, deqOffset_desc, deqScale_desc, false, false,
                   out_desc,  ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_test, ascend910B2_test_mm_wrongoutputshape1) {
  // 使用**Desc描述host api输入输出
  TensorDesc x1_desc = TensorDesc({2, 4, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({3, 16, 16}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc addOffset_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mulScale_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc diagonalMatrix_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc deqOffset_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc deqScale_desc = TensorDesc({16, 1}, ACL_UINT64, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({3, 4 ,16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  MatMulCommonTest(x1_desc, x2_desc, bias_desc, addOffset_desc, mulScale_desc,
                   diagonalMatrix_desc, deqOffset_desc, deqScale_desc, false, false,
                   out_desc,  ACLNN_ERR_PARAM_INVALID);
}

// single op scale N
TEST_F(l2_weight_quant_batch_matmul_test, ascend910B2_test_mm_scaleN_SingleOp) {
  // 使用**Desc描述host api输入输出
  TensorDesc x1_desc = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc addOffset_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mulScale_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc diagonalMatrix_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc deqOffset_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc deqScale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  MatMulCommonTest(x1_desc, x2_desc, bias_desc, addOffset_desc, mulScale_desc,
                   diagonalMatrix_desc, deqOffset_desc, deqScale_desc, false, false,
                   out_desc, ACLNN_SUCCESS);
}

//single op scale 1
TEST_F(l2_weight_quant_batch_matmul_test, ascend910B2_test_mm_scale1_SingleOp) {
  // 使用**Desc描述host api输入输出
  TensorDesc x1_desc = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc addOffset_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mulScale_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc diagonalMatrix_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc deqOffset_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc deqScale_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  MatMulCommonTest(x1_desc, x2_desc, bias_desc, addOffset_desc, mulScale_desc,
                   diagonalMatrix_desc, deqOffset_desc, deqScale_desc, false, false,
                   out_desc, ACLNN_SUCCESS);
}

// single op scale N bmm
TEST_F(l2_weight_quant_batch_matmul_test, ascend910B2_test_bmm_scaleN_SingleOp) {
  // 使用**Desc描述host api输入输出
  TensorDesc x1_desc = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc addOffset_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mulScale_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc diagonalMatrix_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc deqOffset_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc deqScale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  MatMulCommonTest(x1_desc, x2_desc, bias_desc, addOffset_desc, mulScale_desc,
                   diagonalMatrix_desc, deqOffset_desc, deqScale_desc, false, false,
                   out_desc, ACLNN_SUCCESS);
}

// single op scale N offset N bmm transposex1
TEST_F(l2_weight_quant_batch_matmul_test, ascend910B2_test_bmm_scaleN_offsetN_SingleOp_tranx1) {
  // 使用**Desc描述host api输入输出
  TensorDesc x1_desc = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc addOffset_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mulScale_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc diagonalMatrix_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc deqOffset_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc deqScale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  MatMulCommonTest(x1_desc, x2_desc, bias_desc, addOffset_desc, mulScale_desc,
                   diagonalMatrix_desc, deqOffset_desc, deqScale_desc, true, false,
                   out_desc, ACLNN_SUCCESS);
}

// single op scale N offset N bmm transposex2
TEST_F(l2_weight_quant_batch_matmul_test, ascend910B2_test_bmm_scaleN_SingleOp_tranx2) {
  // 使用**Desc描述host api输入输出
  TensorDesc x1_desc = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({8, 16}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({8,}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc addOffset_desc = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mulScale_desc = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc diagonalMatrix_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc deqOffset_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc deqScale_desc = TensorDesc({8, 1}, ACL_UINT64, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({128, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
  MatMulCommonTest(x1_desc, x2_desc, bias_desc, addOffset_desc, mulScale_desc,
                   diagonalMatrix_desc, deqOffset_desc, deqScale_desc, false, true,
                   out_desc, ACLNN_SUCCESS);
}

// single op scale N offset N mm transposex1
TEST_F(l2_weight_quant_batch_matmul_test, ascend910B2_test_mm_scaleN_offsetN_SingleOp_tranx1) {
  // 使用**Desc描述host api输入输出
  TensorDesc x1_desc = TensorDesc({ 16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({ 16, 16}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc addOffset_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mulScale_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc diagonalMatrix_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc deqOffset_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc deqScale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  MatMulCommonTest(x1_desc, x2_desc, bias_desc, addOffset_desc, mulScale_desc,
                   diagonalMatrix_desc, deqOffset_desc, deqScale_desc, true, false,
                   out_desc, ACLNN_SUCCESS);
}

// single op scale N offset N mm transposex2
TEST_F(l2_weight_quant_batch_matmul_test, ascend910B2_test_mm_scaleN_SingleOp_tranx2) {
  // 使用**Desc描述host api输入输出
  TensorDesc x1_desc = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({8, 16}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc bias_desc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc addOffset_desc = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mulScale_desc = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc diagonalMatrix_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
  TensorDesc deqOffset_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc deqScale_desc = TensorDesc({8, 1}, ACL_UINT64, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({128, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
  MatMulCommonTest(x1_desc, x2_desc, bias_desc, addOffset_desc, mulScale_desc,
                   diagonalMatrix_desc, deqOffset_desc, deqScale_desc, false, true,
                   out_desc, ACLNN_SUCCESS);
}