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

#include "../../../op_host/op_api/aclnn_sparse4to2quant_matmul_weight_nz.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_Sparse4to2QuantMatmul_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_Sparse4to2QuantMatmul_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_Sparse4to2QuantMatmul_test TearDown" << endl; }
};


TEST_F(l2_Sparse4to2QuantMatmul_test, ascend910B2_test_normal_case_01)
{
    TensorDesc x_desc = TensorDesc({64, 512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc sparse_weight_desc = TensorDesc({256, 256}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {8, 16, 16, 32});
    TensorDesc index_desc = TensorDesc({256, 32}, ACL_UINT8, ACL_FORMAT_ND);
    TensorDesc x_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc sparse_weight_scale_desc = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSparse4to2QuantMatmulWeightNz,
                        INPUT(x_desc, sparse_weight_desc, index_desc, x_scale_desc,
                              sparse_weight_scale_desc, bias_desc),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}


