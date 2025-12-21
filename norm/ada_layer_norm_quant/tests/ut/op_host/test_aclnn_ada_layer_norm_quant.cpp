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
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_ada_layer_norm_quant.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_ada_layer_norm_quant_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_ada_layer_norm_quant_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_ada_layer_norm_quant_test TearDown" << endl;
    }

public:
    void CommonTest(
        const vector<int64_t>& xShape, const vector<int64_t>& scaleShape, const vector<int64_t>& weightShape,
        const vector<int64_t>& outShape, const vector<int64_t>& quantScaleShape, aclDataType dtype,
        aclnnStatus expectRet)
    {
        auto x = TensorDesc(xShape, dtype, ACL_FORMAT_NCL);
        auto scale = TensorDesc(scaleShape, dtype, ACL_FORMAT_ND);
        auto shift = TensorDesc(scaleShape, dtype, ACL_FORMAT_ND);
        auto weight = TensorDesc(weightShape, dtype, ACL_FORMAT_ND);
        auto bias = TensorDesc(weightShape, dtype, ACL_FORMAT_ND);
        auto smoothScales = TensorDesc(weightShape, dtype, ACL_FORMAT_ND);
        auto out = TensorDesc(outShape, ACL_INT8, ACL_FORMAT_NCL);
        auto quantScale = TensorDesc(quantScaleShape, ACL_FLOAT, ACL_FORMAT_ND);
        const char* quantMode = "dynamic";
        uint64_t workspace_size = 0;
        auto ut = OP_API_UT(
            aclnnAdaLayerNormQuant, INPUT(x, scale, shift, weight, bias, smoothScales, 0.00001, quantMode),
            OUTPUT(out, quantScale, (aclTensor*)nullptr));
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

        EXPECT_EQ(aclRet, expectRet);
    }
};

TEST_F(l2_ada_layer_norm_quant_test, ascend910B2_success)
{
    // data type cases
    CommonTest({4, 16, 128}, {4, 128}, {128}, {4, 16, 128}, {4, 16}, ACL_FLOAT16, ACLNN_SUCCESS);
    CommonTest({4, 16, 128}, {4, 128}, {128}, {4, 16, 128}, {4, 16}, ACL_BF16, ACLNN_SUCCESS);
}

TEST_F(l2_ada_layer_norm_quant_test, ascend910B2_param_invalid)
{
    // H = 0
    CommonTest({4, 16, 0}, {4, 128}, {128}, {4, 16, 128}, {4, 16}, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID);
    // invalid dtype
    CommonTest({4, 16, 128}, {4, 128}, {128}, {4, 16, 128}, {4, 16}, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID);
    // invalid shape
    CommonTest({4, 16, 128}, {16, 128}, {128}, {4, 16, 128}, {4, 16}, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID);
    CommonTest({4, 16, 128}, {4, 128}, {512}, {4, 16, 128}, {4, 16}, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID);
    CommonTest({4, 16, 128}, {4, 128}, {128}, {4, 4, 128}, {4, 16}, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID);
    CommonTest({4, 16, 128}, {4, 128}, {128}, {4, 16, 128}, {4, 4}, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID);
}