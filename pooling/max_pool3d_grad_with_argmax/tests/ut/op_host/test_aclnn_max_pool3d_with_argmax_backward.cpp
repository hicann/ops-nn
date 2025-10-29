/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "opdev/op_log.h"
#include "../../../op_host/op_api/aclnn_max_pool3d_with_argmax_backward.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_max_pool3d_with_argmax_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_max_pool3d_with_argmax_backward_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_max_pool3d_with_argmax_backward_test TearDown" << std::endl;
    }
};

// 正常场景：数据类型为float
TEST_F(l2_max_pool3d_with_argmax_backward_test, ascend910B2_normal_float)
{
    vector<int64_t> self_dims = {2, 3, 2, 2, 2};
    vector<int64_t> kernel_dims = {2};
    vector<int64_t> stride_dims = {2};
    vector<int64_t> padding_dims = {0};
    vector<int64_t> dilation_dims = {1};
    vector<int64_t> out_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 4, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = true;

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmaxBackward,
        INPUT(
            tensor_gradOutput, tensor_self, indices_tensor_desc, kernel_desc, stride_desc, padding_desc, dilation_desc,
            ceilMode),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：数据类型是float16
TEST_F(l2_max_pool3d_with_argmax_backward_test, ascend910B2_normal_float16)
{
    vector<int64_t> self_dims = {2, 3, 2, 2, 2};
    vector<int64_t> kernel_dims = {2};
    vector<int64_t> stride_dims = {1};
    vector<int64_t> padding_dims = {0};
    vector<int64_t> dilation_dims = {1};
    vector<int64_t> out_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 1, 1, 1};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = true;

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmaxBackward,
        INPUT(
            tensor_gradOutput, tensor_self, indices_tensor_desc, kernel_desc, stride_desc, padding_desc, dilation_desc,
            ceilMode),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：ceilMode是false
TEST_F(l2_max_pool3d_with_argmax_backward_test, ascend910B2_normal_4d)
{
    vector<int64_t> self_dims = {2, 2, 2, 2};
    vector<int64_t> kernel_dims = {2};
    vector<int64_t> stride_dims = {1};
    vector<int64_t> padding_dims = {0};
    vector<int64_t> dilation_dims = {1};
    vector<int64_t> out_dims = {2, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1, 1};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmaxBackward,
        INPUT(
            tensor_gradOutput, tensor_self, indices_tensor_desc, kernel_desc, stride_desc, padding_desc, dilation_desc,
            ceilMode),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

// 异常场景：kernel_size为0
TEST_F(l2_max_pool3d_with_argmax_backward_test, ascend910B2_exception_kernel_size_0)
{
    vector<int64_t> self_dims = {2, 3, 2, 2, 2};
    vector<int64_t> kernel_dims = {0};
    vector<int64_t> stride_dims = {2};
    vector<int64_t> padding_dims = {1};
    vector<int64_t> dilation_dims = {1};
    vector<int64_t> out_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 1, 1, 1};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = true;

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmaxBackward,
        INPUT(
            tensor_gradOutput, tensor_self, indices_tensor_desc, kernel_desc, stride_desc, padding_desc, dilation_desc,
            ceilMode),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：padding_size为负数
TEST_F(l2_max_pool3d_with_argmax_backward_test, ascend910B2_exception_padding_size_negative)
{
    vector<int64_t> self_dims = {2, 3, 2, 2, 2};
    vector<int64_t> kernel_dims = {2};
    vector<int64_t> stride_dims = {2};
    vector<int64_t> padding_dims = {-1};
    vector<int64_t> dilation_dims = {1};
    vector<int64_t> out_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 1, 1, 1};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = true;

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmaxBackward,
        INPUT(
            tensor_gradOutput, tensor_self, indices_tensor_desc, kernel_desc, stride_desc, padding_desc, dilation_desc,
            ceilMode),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：dilation_size为0
TEST_F(l2_max_pool3d_with_argmax_backward_test, ascend910B2_exception_dilation_size_0)
{
    vector<int64_t> self_dims = {2, 3, 2, 2, 2};
    vector<int64_t> kernel_dims = {2};
    vector<int64_t> stride_dims = {2};
    vector<int64_t> padding_dims = {1};
    vector<int64_t> dilation_dims = {0};
    vector<int64_t> out_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 1, 1, 1};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = true;

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmaxBackward,
        INPUT(
            tensor_gradOutput, tensor_self, indices_tensor_desc, kernel_desc, stride_desc, padding_desc, dilation_desc,
            ceilMode),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}