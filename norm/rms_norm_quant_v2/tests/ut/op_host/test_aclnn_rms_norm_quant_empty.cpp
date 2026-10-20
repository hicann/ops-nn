/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "../../../../rms_norm_quant/op_host/op_api/aclnn_rms_norm_quant.h"
#include "opdev/platform.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

TEST(RmsNormQuantEmptyApi, ascend950_valid_empty_shapes_return_zero_workspace)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    for (const auto& shape : {std::vector<int64_t>{0, 64}, {3, 0}, {0}, {0, 0}, {2, 0, 64}, {2, 3, 0}}) {
        for (auto xType : {ACL_FLOAT16, ACL_BF16, ACL_FLOAT}) {
            for (auto yType : {ACL_INT8, ACL_INT4, ACL_INT32, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_HIFLOAT8}) {
                auto yShape = shape;
                if (yType == ACL_INT32) {
                    yShape.back() /= 8;
                }
                const auto h = shape.back();
                auto x = TensorDesc(shape, xType, ACL_FORMAT_ND);
                auto gamma = TensorDesc({1, h}, xType, ACL_FORMAT_ND);
                auto beta = TensorDesc({h}, xType, ACL_FORMAT_ND);
                auto scale = TensorDesc({h}, ACL_FLOAT, ACL_FORMAT_ND);
                auto offset = TensorDesc({h}, ACL_FLOAT, ACL_FORMAT_ND);
                auto y = TensorDesc(yShape, yType, ACL_FORMAT_ND);
                auto ut = OP_API_UT(aclnnRmsNormQuant, INPUT(x, gamma, beta, scale, offset, 1e-6), OUTPUT(y));
                uint64_t workspace = 123;
                ASSERT_EQ(ut.TestGetWorkspaceSize(&workspace), ACLNN_SUCCESS);
                EXPECT_EQ(workspace, 0U);
            }
        }
    }
}

TEST(RmsNormQuantEmptyApi, ascend950_empty_input_still_requires_non_null_tensors)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x = TensorDesc({0, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y = TensorDesc({0, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRmsNormQuant, INPUT(x, gamma, nullptr, scale, offset, 1e-6), OUTPUT(y));
    uint64_t workspace = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_NULLPTR);
}
