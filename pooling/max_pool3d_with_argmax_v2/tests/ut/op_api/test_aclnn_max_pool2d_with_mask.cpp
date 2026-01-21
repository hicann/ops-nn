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

#include "opdev/op_log.h"
#include "../../../op_api/aclnn_max_pool2d_with_indices.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_max_pool2d_with_mask_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_max_pool2d_with_mask_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_max_pool2d_with_mask_test TearDown" << std::endl;
    }
};

// 正常场景：数据类型为float
TEST_F(l2_max_pool2d_with_mask_test, normal_float)
{
    vector<int64_t> self_dims = {2, 3, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 3, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = true;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：数据类型为float16
TEST_F(l2_max_pool2d_with_mask_test, normal_float16)
{
    vector<int64_t> self_dims = {2, 3, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 3, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    const bool ceilMode = true;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

// 正常场景：数据类型为bfloat16
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_normal_bfloat16)
{
    vector<int64_t> self_dims = {2, 3, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 3, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = true;

    auto self_desc = TensorDesc(self_dims, ACL_BF16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_BF16, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

// 正常场景：数据类型是float16，stride是空，其它参数是size是1，ceilMode是true
TEST_F(l2_max_pool2d_with_mask_test, normal_float16_stride_null)
{
    vector<int64_t> self_dims = {2, 3, 2, 2};
    vector<int64_t> kernel_dims = {2};
    vector<int64_t> stride_dims = {};
    vector<int64_t> padding_dims = {0};
    vector<int64_t> dilation_dims = {1};
    vector<int64_t> out_dims = {2, 3, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = true;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

// 该ut场景涉及l0op::UnsqueezeNd等打桩场景接口，无法正常运行，待后续工程解决后开放
// // 正常场景：format是CHW
// TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_normal_float16_CHW)
// {
//     vector<int64_t> self_dims = {3, 2, 2};
//     vector<int64_t> kernel_dims = {2};
//     vector<int64_t> stride_dims = {1};
//     vector<int64_t> padding_dims = {0};
//     vector<int64_t> dilation_dims = {1};
//     vector<int64_t> out_dims = {3, 1, 1};
//     vector<int64_t> indices_dims = {3, 4, 64};

//     auto kernel_desc = IntArrayDesc(kernel_dims);
//     auto stride_desc = IntArrayDesc(stride_dims);
//     auto padding_desc = IntArrayDesc(padding_dims);
//     auto dilation_desc = IntArrayDesc(dilation_dims);
//     bool ceilMode = true;

//     auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCL);
//     auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCL);
//     auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCL);

//     auto ut = OP_API_UT(
//         aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
//         OUTPUT(out_tensor_desc, indices_tensor_desc));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
//         EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
//     } else {
//         EXPECT_EQ(aclRet, ACL_SUCCESS);
//         // ut.TestPrecision();
//     }
// }


// 正常场景：kernel的size是2
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_normal_kernel_size2)
{
    vector<int64_t> self_dims = {2, 3, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 3, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = true;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // ut.TestPrecision();
    }
}

// 正常场景：ceilMode是true
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_normal_ceilMode_true)
{
    vector<int64_t> self_dims = {2, 3, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 3, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    const bool ceilMode = true;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // ut.TestPrecision();
    }
}

// 正常场景：ceilMode是false
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_normal_ceilMode_false)
{
    vector<int64_t> self_dims = {2, 3, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 3, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // ut.TestPrecision();
    }
}

// 空tensor，n可以是0，chw不可以是0
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_normal_nbatch0)
{
    vector<int64_t> self_dims = {0, 3, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {0, 3, 1, 1};
    vector<int64_t> indices_dims = {0, 3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // ut.TestPrecision();
    }
}

// 异常场景：空tensor，n可以是0，chw不可以是0
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_nInputPlane0)
{
    vector<int64_t> self_dims = {2, 0, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 0, 1, 1};
    vector<int64_t> indices_dims = {2, 0, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：数据类型是int
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_int)
{
    vector<int64_t> self_dims = {2, 3, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 3, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_INT32, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：self是nullptr
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_normal_self_nullptr)
{
    vector<int64_t> self_dims = {2, 3, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 3, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = nullptr;
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：kernelSize是nullptr
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_kernel_nullptr)
{
    vector<int64_t> self_dims = {2, 3, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 3, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 4, 64};

    auto kernel_desc = nullptr;
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：indices数据类型是float
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_indices_float)
{
    vector<int64_t> self_dims = {2, 3, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 3, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：out数据类型是int
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_out_int)
{
    vector<int64_t> self_dims = {2, 3, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 3, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：self的数据类型与out数据类型不一致
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_self_out_diff)
{
    vector<int64_t> self_dims = {2, 3, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 3, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：self是2维
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_self_2d)
{
    vector<int64_t> self_dims = {2, 3};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 3};
    vector<int64_t> indices_dims = {2, 3};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：self是5维
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_self_5d)
{
    vector<int64_t> self_dims = {2, 3, 2, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {2, 3, 1, 1, 2};
    vector<int64_t> indices_dims = {2, 3, 1, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：self是私有格式
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_self_privateFormat)
{
    vector<int64_t> self_dims = {3, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {3, 1, 1};
    vector<int64_t> indices_dims = {3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;
    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FRACTAL_Z_3D);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FRACTAL_Z_3D);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);
    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：kernelSize的元素个数是0
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_kenelSize_0)
{
    vector<int64_t> self_dims = {3, 2, 2, 2};
    vector<int64_t> kernel_dims = {};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {3, 1, 1, 2};
    vector<int64_t> indices_dims = {3, 1, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：kernel的值是0
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_kernelValue_0)
{
    vector<int64_t> self_dims = {3, 2, 2, 2};
    vector<int64_t> kernel_dims = {0, 1};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {3, 1, 1, 2};
    vector<int64_t> indices_dims = {3, 1, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：stride的值是0
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_strideValue_0)
{
    vector<int64_t> self_dims = {3, 2, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {0, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {3, 1, 1, 2};
    vector<int64_t> indices_dims = {2, 3, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：pad大于kernelSize/2
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_padValue)
{
    vector<int64_t> self_dims = {3, 2, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {2, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {3, 1, 1, 2};
    vector<int64_t> indices_dims = {3, 1, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：dilation是2
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_dilation)
{
    vector<int64_t> self_dims = {3, 2, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {2, 1};
    vector<int64_t> out_dims = {3, 1, 1, 2};
    vector<int64_t> indices_dims = {3, 1, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：padding的元素个数是0
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_paddingSize)
{
    vector<int64_t> self_dims = {3, 2, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {3, 1, 1, 2};
    vector<int64_t> indices_dims = {3, 1, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：dilation的元素个数是0
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_dilationSize)
{
    vector<int64_t> self_dims = {3, 2, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {};
    vector<int64_t> out_dims = {3, 1, 1, 2};
    vector<int64_t> indices_dims = {3, 1, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：indices的shape不正确
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_out_indices_diff)
{
    vector<int64_t> self_dims = {3, 2, 2, 2};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {3, 1, 1, 2};
    vector<int64_t> indices_dims = {2, 1, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：out的shape异常，输入shape是（2,3,2), kernelSize是3, stride是2，则out的shape是(2, 1, 0)
TEST_F(l2_max_pool2d_with_mask_test, ascend910B2_abnormal_out_shape)
{
    vector<int64_t> self_dims = {3, 2, 2, 2};
    vector<int64_t> kernel_dims = {3, 3};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> out_dims = {3, 1, 1, 2};
    vector<int64_t> indices_dims = {3, 1, 4, 64};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnMaxPool2dWithMask, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
