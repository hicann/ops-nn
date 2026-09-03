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

#include "../../../op_api/aclnn_fused_adam.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_fused_adam_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_fused_adam_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_fused_adam_test TearDown" << std::endl; }
};

TEST_F(l2_fused_adam_test, fused_adam_test_nullptr)
{
    vector<vector<int64_t>> selfDims = {{2, 2}, {1}};
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto params = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grads = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto maxExpAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto step = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 100.0);
    auto gradScaleOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto foundInfOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 0.0);
    auto lr = 0.01f;
    auto beta1 = 0.9f;
    auto beta2 = 0.999f;
    auto weightDecay = 0.01f;
    auto eps = 1e-8f;
    bool amsgrad = false;
    bool maximize = false;
    // 前6个参数是 aclTensorList*，需要用 TensorListDesc 构造
    auto paramsList = TensorListDesc({params});
    auto gradsList = TensorListDesc({grads});
    auto expAvgsList = TensorListDesc({expAvgs});
    auto expAvgSqsList = TensorListDesc({expAvgSqs});
    auto maxExpAvgSqsList = TensorListDesc({maxExpAvgSqs});
    auto stateStepsList = TensorListDesc({step});
    auto ut = OP_API_UT(
        aclnnFusedAdam,
        INPUT(nullptr, gradsList, expAvgsList, expAvgSqsList, maxExpAvgSqsList, stateStepsList, gradScaleOptional,
              foundInfOptional, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fused_adam_test, fused_adam_test_grad_nullptr)
{
    vector<vector<int64_t>> selfDims = {{2, 2}, {1}};
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto params = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grads = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto maxExpAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto step = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 100.0);
    auto gradScaleOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto foundInfOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 0.0);
    auto lr = 0.01f;
    auto beta1 = 0.9f;
    auto beta2 = 0.999f;
    auto weightDecay = 0.01f;
    auto eps = 1e-8f;
    bool amsgrad = false;
    bool maximize = false;
    // 前6个参数是 aclTensorList*，需要用 TensorListDesc 构造
    auto paramsList = TensorListDesc({params});
    auto gradsList = TensorListDesc({grads});
    auto expAvgsList = TensorListDesc({expAvgs});
    auto expAvgSqsList = TensorListDesc({expAvgSqs});
    auto maxExpAvgSqsList = TensorListDesc({maxExpAvgSqs});
    auto stateStepsList = TensorListDesc({step});
    auto ut = OP_API_UT(
        aclnnFusedAdam,
        INPUT(paramsList, nullptr, expAvgsList, expAvgSqsList, maxExpAvgSqsList, stateStepsList, gradScaleOptional,
              foundInfOptional, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fused_adam_test, fused_adam_test_attr_error)
{
    vector<vector<int64_t>> selfDims = {{2, 2}, {1}};
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto params = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grads = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto maxExpAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto step = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 100.0);
    auto gradScaleOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto foundInfOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 0.0);
    auto lr = -0.01f;
    auto beta1 = 0.9f;
    auto beta2 = 0.999f;
    auto weightDecay = 0.01f;
    auto eps = 1e-8f;
    bool amsgrad = false;
    bool maximize = false;
    // 前6个参数是 aclTensorList*，需要用 TensorListDesc 构造
    auto paramsList = TensorListDesc({params});
    auto gradsList = TensorListDesc({grads});
    auto expAvgsList = TensorListDesc({expAvgs});
    auto expAvgSqsList = TensorListDesc({expAvgSqs});
    auto maxExpAvgSqsList = TensorListDesc({maxExpAvgSqs});
    auto stateStepsList = TensorListDesc({step});
    auto ut = OP_API_UT(
        aclnnFusedAdam,
        INPUT(paramsList, gradsList, expAvgsList, expAvgSqsList, maxExpAvgSqsList, stateStepsList, gradScaleOptional,
              foundInfOptional, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_adam_test, fused_adam_test_diff_list_num)
{
    vector<vector<int64_t>> selfDims = {{2, 2}, {1}, {8, 8}};
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto params = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto params2 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grads = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto maxExpAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto step = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 100.0);
    auto gradScaleOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto foundInfOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 0.0);
    auto lr = 0.01f;
    auto beta1 = 0.9f;
    auto beta2 = 0.999f;
    auto weightDecay = 0.01f;
    auto eps = 1e-8f;
    bool amsgrad = false;
    bool maximize = false;
    // 前6个参数是 aclTensorList*，需要用 TensorListDesc 构造
    auto paramsList = TensorListDesc({params, params2});
    auto gradsList = TensorListDesc({grads});
    auto expAvgsList = TensorListDesc({expAvgs});
    auto expAvgSqsList = TensorListDesc({expAvgSqs});
    auto maxExpAvgSqsList = TensorListDesc({maxExpAvgSqs});
    auto stateStepsList = TensorListDesc({step});
    auto ut = OP_API_UT(
        aclnnFusedAdam,
        INPUT(paramsList, gradsList, expAvgsList, expAvgSqsList, maxExpAvgSqsList, stateStepsList, gradScaleOptional,
              foundInfOptional, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_adam_test, fused_adam_test_error_dtype)
{
    vector<vector<int64_t>> selfDims = {{2, 2}, {1}, {8, 8}};

    vector<vector<int64_t>> outDims = {{2, 2}};
    auto params = TensorDesc(selfDims[0], ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grads = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto maxExpAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto step = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 100.0);
    auto gradScaleOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto foundInfOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 0.0);
    auto lr = 0.01f;
    auto beta1 = 0.9f;
    auto beta2 = 0.999f;
    auto weightDecay = 0.01f;
    auto eps = 1e-8f;
    bool amsgrad = false;
    bool maximize = false;
    // 前6个参数是 aclTensorList*，需要用 TensorListDesc 构造
    auto paramsList = TensorListDesc({params});
    auto gradsList = TensorListDesc({grads});
    auto expAvgsList = TensorListDesc({expAvgs});
    auto expAvgSqsList = TensorListDesc({expAvgSqs});
    auto maxExpAvgSqsList = TensorListDesc({maxExpAvgSqs});
    auto stateStepsList = TensorListDesc({step});
    auto ut = OP_API_UT(
        aclnnFusedAdam,
        INPUT(paramsList, gradsList, expAvgsList, expAvgSqsList, maxExpAvgSqsList, stateStepsList, gradScaleOptional,
              foundInfOptional, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_adam_test, fused_adam_test_step_error_dtype)
{
    vector<vector<int64_t>> selfDims = {{2, 2}, {1}, {8, 8}};

    vector<vector<int64_t>> outDims = {{2, 2}};
    auto params = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grads = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto maxExpAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto step = TensorDesc(selfDims[1], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0.0, 100.0);
    auto gradScaleOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto foundInfOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 0.0);
    auto lr = 0.01f;
    auto beta1 = 0.9f;
    auto beta2 = 0.999f;
    auto weightDecay = 0.01f;
    auto eps = 1e-8f;
    bool amsgrad = false;
    bool maximize = false;
    // 前6个参数是 aclTensorList*，需要用 TensorListDesc 构造
    auto paramsList = TensorListDesc({params});
    auto gradsList = TensorListDesc({grads});
    auto expAvgsList = TensorListDesc({expAvgs});
    auto expAvgSqsList = TensorListDesc({expAvgSqs});
    auto maxExpAvgSqsList = TensorListDesc({maxExpAvgSqs});
    auto stateStepsList = TensorListDesc({step});
    auto ut = OP_API_UT(
        aclnnFusedAdam,
        INPUT(paramsList, gradsList, expAvgsList, expAvgSqsList, maxExpAvgSqsList, stateStepsList, gradScaleOptional,
              foundInfOptional, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_adam_test, fused_adam_test_diff_params_grad_dtype)
{
    vector<vector<int64_t>> selfDims = {{2, 2}, {1}, {8, 8}};
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto params = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grads = TensorDesc(selfDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto maxExpAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto step = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 100.0);
    auto gradScaleOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto foundInfOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 0.0);
    auto lr = 0.01f;
    auto beta1 = 0.9f;
    auto beta2 = 0.999f;
    auto weightDecay = 0.01f;
    auto eps = 1e-8f;
    bool amsgrad = false;
    bool maximize = false;
    // 前6个参数是 aclTensorList*，需要用 TensorListDesc 构造
    auto paramsList = TensorListDesc({params});
    auto gradsList = TensorListDesc({grads});
    auto expAvgsList = TensorListDesc({expAvgs});
    auto expAvgSqsList = TensorListDesc({expAvgSqs});
    auto maxExpAvgSqsList = TensorListDesc({maxExpAvgSqs});
    auto stateStepsList = TensorListDesc({step});
    auto ut = OP_API_UT(
        aclnnFusedAdam,
        INPUT(paramsList, gradsList, expAvgsList, expAvgSqsList, maxExpAvgSqsList, stateStepsList, gradScaleOptional,
              foundInfOptional, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_adam_test, fused_adam_test_diff_params_grad_shape)
{
    vector<vector<int64_t>> selfDims = {{2, 2}, {1}, {8, 8}};
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto params = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grads = TensorDesc(selfDims[2], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto maxExpAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto step = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 100.0);
    auto gradScaleOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto foundInfOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 0.0);
    auto lr = 0.01f;
    auto beta1 = 0.9f;
    auto beta2 = 0.999f;
    auto weightDecay = 0.01f;
    auto eps = 1e-8f;
    bool amsgrad = false;
    bool maximize = false;
    // 前6个参数是 aclTensorList*，需要用 TensorListDesc 构造
    auto paramsList = TensorListDesc({params});
    auto gradsList = TensorListDesc({grads});
    auto expAvgsList = TensorListDesc({expAvgs});
    auto expAvgSqsList = TensorListDesc({expAvgSqs});
    auto maxExpAvgSqsList = TensorListDesc({maxExpAvgSqs});
    auto stateStepsList = TensorListDesc({step});
    auto ut = OP_API_UT(
        aclnnFusedAdam,
        INPUT(paramsList, gradsList, expAvgsList, expAvgSqsList, maxExpAvgSqsList, stateStepsList, gradScaleOptional,
              foundInfOptional, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_adam_test, fused_adam_test_success)
{
    vector<vector<int64_t>> selfDims = {{2, 2}, {1}, {8, 8}};
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto params = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grads = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto expAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto maxExpAvgSqs = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto step = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 100.0);
    auto gradScaleOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto foundInfOptional = TensorDesc(selfDims[1], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 0.0);
    auto lr = 0.01f;
    auto beta1 = 0.9f;
    auto beta2 = 0.999f;
    auto weightDecay = 0.01f;
    auto eps = 1e-8f;
    bool amsgrad = false;
    bool maximize = false;
    // 前6个参数是 aclTensorList*，需要用 TensorListDesc 构造
    auto paramsList = TensorListDesc({params});
    auto gradsList = TensorListDesc({grads});
    auto expAvgsList = TensorListDesc({expAvgs});
    auto expAvgSqsList = TensorListDesc({expAvgSqs});
    auto maxExpAvgSqsList = TensorListDesc({maxExpAvgSqs});
    auto stateStepsList = TensorListDesc({step});
    auto ut = OP_API_UT(aclnnFusedAdam,
                        INPUT(paramsList, gradsList, expAvgsList, expAvgSqsList, maxExpAvgSqsList, stateStepsList,
                              nullptr, foundInfOptional, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}
