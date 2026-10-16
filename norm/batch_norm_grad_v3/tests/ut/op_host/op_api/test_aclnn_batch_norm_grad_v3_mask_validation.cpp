/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "opdev/platform.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "norm/batch_norm_grad_v3/op_host/op_api/aclnn_batch_norm_backward.h"

class BatchNormGradV3MaskValidationUt : public testing::Test {
protected:
    op::SocVersionManager platform_{op::SocVersion::ASCEND950};
    TensorDesc dy_{{2, 16, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCHW};
    TensorDesc x_{{2, 16, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCHW};
    TensorDesc param_{{16}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc dx_{{2, 16, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCHW};
};

TEST_F(BatchNormGradV3MaskValidationUt, InvalidLengthReturnsParamInvalid)
{
    for (const auto length : {0, 1, 2, 4}) {
        for (const bool training : {false, true}) {
            SCOPED_TRACE(testing::Message() << "length=" << length << ", training=" << training);
            auto mask = BoolArrayDesc(vector<bool>(length, true));
            auto ut = OP_API_UT(aclnnBatchNormBackward,
                                INPUT(dy_, x_, param_, param_, param_, param_, param_, training, 1e-5, mask),
                                OUTPUT(dx_, param_, param_));
            uint64_t workspace = 0;
            EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_INVALID);
        }
    }
}

TEST_F(BatchNormGradV3MaskValidationUt, NullPointersReturnParamNullptr)
{
    uint64_t workspace = 0;
    auto mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto nullMask = OP_API_UT(aclnnBatchNormBackward,
                              INPUT(dy_, x_, param_, param_, param_, param_, param_, true, 1e-5, nullptr),
                              OUTPUT(dx_, param_, param_));
    EXPECT_EQ(nullMask.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_NULLPTR);

    auto nullDx = OP_API_UT(aclnnBatchNormBackward,
                            INPUT(dy_, x_, param_, param_, param_, param_, param_, true, 1e-5, mask),
                            OUTPUT(nullptr, param_, param_));
    EXPECT_EQ(nullDx.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_NULLPTR);

    auto nullDweight = OP_API_UT(aclnnBatchNormBackward,
                                 INPUT(dy_, x_, param_, param_, param_, param_, param_, true, 1e-5, mask),
                                 OUTPUT(dx_, nullptr, param_));
    EXPECT_EQ(nullDweight.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_NULLPTR);

    auto nullDbias = OP_API_UT(aclnnBatchNormBackward,
                               INPUT(dy_, x_, param_, param_, param_, param_, param_, true, 1e-5, mask),
                               OUTPUT(dx_, param_, nullptr));
    EXPECT_EQ(nullDbias.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(BatchNormGradV3MaskValidationUt, DisabledOutputsMayBeNull)
{
    auto mask = BoolArrayDesc(vector<bool>{false, false, false});
    auto ut = OP_API_UT(aclnnBatchNormBackward,
                        INPUT(dy_, x_, param_, param_, param_, param_, param_, true, 1e-5, mask),
                        OUTPUT(nullptr, nullptr, nullptr));
    uint64_t workspace = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace), ACLNN_SUCCESS);
}
