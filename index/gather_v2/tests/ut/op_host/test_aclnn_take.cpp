/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_take.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_take_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "Take Test Setup" << endl; }
    static void TearDownTestCase() { cout << "Take Test TearDown" << endl; }
};

// dtype_1_float32
TEST_F(l2_take_test, dtype_1_float32)
{
    auto selfDesc = TensorDesc({1, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{0, 10.123, -200, 8388608.1, 44.12345, 501000, 6501000, 7650999});
    auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 4, 2, 3});
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    
}

// dtype_2_float16
TEST_F(l2_take_test, dtype_2_float16)
{
    auto selfDesc = TensorDesc({1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND)
        .Value(vector<float>{0, 10, -200.12345, 30608.1, 432.2345, -501000, 65519, 7650999});
    auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 1, 5, 3});
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    
}

// dtype_2_bfloat16
TEST_F(l2_take_test, dtype_2_bfloat16)
{
    auto selfDesc = TensorDesc({1, 2, 2, 2}, ACL_BF16, ACL_FORMAT_ND)
        .Value(vector<float>{0, 10, -200.12345, 30608.1, 432.2345, -501000, 65519, 7650999});
    auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 1, 5, 3});
    auto outDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
      EXPECT_EQ(aclRet, ACL_SUCCESS);
      
    } else {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// dtype_3_uint64
TEST_F(l2_take_test, dtype_3_uint64)
{
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_UINT64, ACL_FORMAT_ND)
        .Value(vector<uint64_t>{0, 10, 200, 3000, 40000, 501000, 6501000, 7650999, 87650999, 987650999, 1088881111,
               11088481111, 120188489958, 1301288489958, 14012388489958, 150123488489958});
    auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{4, 1, 5, 3});
    auto outDesc = TensorDesc({2, 2}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    
}

// dtype_4_int64
TEST_F(l2_take_test, dtype_4_int64)
{
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_INT64, ACL_FORMAT_ND)
        .Value(vector<int64_t>{0, 10, 200, 3000, -40000, 501000, 6501000, -7650999, -87650999, -987650999, -1088881111,
               -11088481111, 120188489958, 1301288489958, 14012388489958, -150123488489958});
    auto indexDesc = TensorDesc(vector<int64_t>{2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{4, 5});
    auto outDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    
}

// dtype_5_uint32
TEST_F(l2_take_test, dtype_5_uint32)
{
    auto selfDesc = TensorDesc({1, 2, 4, 2}, ACL_UINT32, ACL_FORMAT_ND)
        .Value(vector<uint32_t>{0, 10, 200, 3000, 40000, 501000, 6501000, 7650999, 87650999, 987650999, 1088881111,
               4188489958, 1201, 130121, 140123, 1501234});
    auto indexDesc = TensorDesc(vector<int64_t>{2, 2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{1, 4, 8, 5});
    auto outDesc = TensorDesc({2, 2}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    
}

// dtype_6_int32
TEST_F(l2_take_test, dtype_6_int32)
{
    auto selfDesc = TensorDesc({1, 4, 4, 1}, ACL_INT32, ACL_FORMAT_ND)
        .Value(vector<int32_t>{0, 10, 200, 3000, 40000, 501000, -6501000, -7650999, 87650999, 987650999, -1088881111,
               2088489958, -2088489958, -130121, 140123, -1501234});
    auto indexDesc = TensorDesc(vector<int64_t>{2, 2, 2}, ACL_INT64, ACL_FORMAT_ND)
        .Value(vector<int64_t>{1, 4, 8, 5, 0, 1, 0, 7});
    auto outDesc = TensorDesc({2, 2, 2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    
}

// dtype_7_uint16
TEST_F(l2_take_test, dtype_7_uint16)
{
    auto selfDesc = TensorDesc({1, 4, 4, 1}, ACL_UINT16, ACL_FORMAT_NCHW)
        .Value(vector<uint16_t>{0, 10, 200, 3000, 40000, 50100, 65010, 7650, 8765, 9876, 10888,
               11888, 12884, 13012, 14012, 65535});
    auto indexDesc = TensorDesc(vector<int64_t>{2, 2, 2, 1}, ACL_INT64, ACL_FORMAT_NHWC)
        .Value(vector<int64_t>{1, 14, 8, 15, 0, 1, 10, 7});
    auto outDesc = TensorDesc({2, 2, 2, 1}, ACL_UINT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    
}

// dtype_8_int16
TEST_F(l2_take_test, dtype_8_int16)
{
    auto selfDesc = TensorDesc({1, 4, 4, 1}, ACL_INT16, ACL_FORMAT_NHWC)
        .Value(vector<int16_t>{0, 10, 200, 3000, -4000, -5100, 6501, 7650, 8765, 9876, 10888,
               -11888, 12884, 13012, -32768, 32767});
    auto indexDesc = TensorDesc(vector<int64_t>{2, 2, 2, 1}, ACL_INT64, ACL_FORMAT_NHWC)
        .Value(vector<int64_t>{1, 14, 8, 15, 0, 1, 10, 7});
    auto outDesc = TensorDesc({2, 2, 2, 1}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    
}

// dtype_9_uint8
TEST_F(l2_take_test, dtype_9_uint8)
{
    auto selfDesc = TensorDesc({1, 4, 4, 1}, ACL_UINT8, ACL_FORMAT_HWCN)
        .Value(vector<uint8_t>{0, 10, 200, 30, 40, 50, 60, 70, 85, 98, 108, 118, 128, 130, 149, 255});
    auto indexDesc = TensorDesc(vector<int64_t>{2, 2, 2, 1}, ACL_INT64, ACL_FORMAT_NHWC)
        .Value(vector<int64_t>{1, 14, 8, 15, 0, 1, 10, 7});
    auto outDesc = TensorDesc({2, 2, 2, 1}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    
}

// dtype_10_int8
TEST_F(l2_take_test, dtype_10_int8)
{
    auto selfDesc = TensorDesc({1, 4, 4, 1}, ACL_INT8, ACL_FORMAT_NDHWC)
        .Value(vector<int8_t>{0, -10, 20, -30, -40, -50, 60, 70, 85, 98, 108, 118, -128, 13, 14, 127});
    auto indexDesc = TensorDesc(vector<int64_t>{2, 2, 2, 1}, ACL_INT64, ACL_FORMAT_NHWC)
        .Value(vector<int64_t>{1, 14, 8, 15, 0, 1, 10, 7});
    auto outDesc = TensorDesc({2, 2, 2, 1}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    
}
// dtype_14_BOOL
TEST_F(l2_take_test, dtype_14_dtype_bool)
{
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_BOOL, ACL_FORMAT_ND);
    auto indexDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 2});
    auto outDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    
}

// 空tensor
TEST_F(l2_take_test, only_self_empty_tensor)
{
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indexDesc = TensorDesc(vector<int64_t>{2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 2});
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor
TEST_F(l2_take_test, only_index_empty_tensor)
{
    auto selfDesc = TensorDesc({2, 3, 8, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indexDesc = TensorDesc({0, 2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 2});
    auto outDesc = TensorDesc({0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 空tensor
TEST_F(l2_take_test, self_index_both_empty)
{
    auto selfDesc = TensorDesc({0, 3, 8, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indexDesc = TensorDesc({0, 2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 2});
    auto outDesc = TensorDesc({0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// self out 数据类型不一致
TEST_F(l2_take_test, abnormal_self_out_diff_dtype)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indexDesc = TensorDesc(vector<int64_t>{2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 2});
    auto outDesc = TensorDesc(indexDesc);
    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self nullptr
TEST_F(l2_take_test, abnormal_self_null_ptr)
{
    auto indexDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int>{0, 2});
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTake, INPUT(nullptr, indexDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// index nullptr
TEST_F(l2_take_test, abnormal_index_null_ptr)
{
    auto selfDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int>{0, 2});
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, nullptr), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2_take_test, abnormal_out_null_ptr)
{
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indexDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int>{0, 2});
    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// check shape self.dim() > 8
TEST_F(l2_take_test, normal_self_dim_gt_8)
{
    auto selfDesc = TensorDesc({1, 2, 3, 1, 2, 3, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indexDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 2});
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    
}

// check shape index.dim() > 8
TEST_F(l2_take_test, abnormal_index_dim_gt_8)
{
    auto selfDesc = TensorDesc({1, 16, 1, 1, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indexDesc = TensorDesc({1, 2, 3, 1, 2, 3, 1, 2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 2, 3, 1, 2, 3, 1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Check shape: out shape different from index shape
TEST_F(l2_take_test, abnormal_out_shape_diff_from_index)
{
    auto selfDesc = TensorDesc({1, 16, 1, 1, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indexDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Check private format
TEST_F(l2_take_test, abnormal_private_format)
{
    auto selfDesc = TensorDesc({1, 16, 1, 1, 16}, ACL_FLOAT16, ACL_FORMAT_NC1HWC0);
    auto indexDesc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 2, 1, 3, 5, 5, 7, 7});
    auto outDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 数据范围[-1, 1]
TEST_F(l2_take_test, data_value_between_negative_1_and_positive_1)
{
    auto selfDesc = TensorDesc({2, 4, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto indexDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{3, 5});
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    
}

// 非连续
TEST_F(l2_take_test, self_not_contiguous)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2})
                        .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto indexDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{1, 0, 2});
    auto outDesc = TensorDesc({3},  ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    
}

// Check private format
TEST_F(l2_take_test, abnormal_private_format_index)
{
    auto selfDesc = TensorDesc({1, 16, 1, 1, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indexDesc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_NC1HWC0).Value(vector<int>{0, 2, 1, 3, 5, 5, 7, 7});
    auto outDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Check private format
TEST_F(l2_take_test, abnormal_private_format_out)
{
    auto selfDesc = TensorDesc({1, 16, 1, 1, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indexDesc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 2, 1, 3, 5, 5, 7, 7});
    auto outDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_NC1HWC0);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Check private format
TEST_F(l2_take_test, abnormal_index_dtype)
{
    auto selfDesc = TensorDesc({1, 16, 1, 1, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indexDesc = TensorDesc({8}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int>{0, 2, 1, 3, 5, 5, 7, 7});
    auto outDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTake, INPUT(selfDesc, indexDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}