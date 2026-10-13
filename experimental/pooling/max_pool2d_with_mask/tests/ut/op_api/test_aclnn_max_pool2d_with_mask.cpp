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

#include "../../../op_api/aclnn_max_pool2d_with_mask.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace op {
extern thread_local SocVersion g_socVersion;
}

namespace {
class MaxPool2dWithMaskApi : public testing::Test {
protected:
    void SetUp() override { savedSoc_ = op::g_socVersion; }
    void TearDown() override { op::g_socVersion = savedSoc_; }

    aclnnStatus Workspace(aclDataType dtype, const std::vector<int64_t>& indicesShape = {1, 1, 1, 128},
                          int64_t kernelSize = 1)
    {
        auto self = TensorDesc({1, 1, 5, 5}, dtype, ACL_FORMAT_ND);
        auto out = TensorDesc({1, 1, 6 - kernelSize, 6 - kernelSize}, dtype, ACL_FORMAT_ND);
        auto indices = TensorDesc(indicesShape, ACL_INT8, ACL_FORMAT_ND);
        auto kernel = IntArrayDesc(std::vector<int64_t>{kernelSize});
        auto stride = IntArrayDesc(std::vector<int64_t>{1});
        auto padding = IntArrayDesc(std::vector<int64_t>{0});
        auto dilation = IntArrayDesc(std::vector<int64_t>{1});
        auto ut = OP_API_UT(aclnnMaxPool2dWithMask, INPUT(self, kernel, stride, padding, dilation, false),
                            OUTPUT(out, indices));
        uint64_t workspaceSize = 0;
        return ut.TestGetWorkspaceSize(&workspaceSize);
    }

private:
    op::SocVersion savedSoc_ = op::SocVersion::ASCEND910B;
};

TEST_F(MaxPool2dWithMaskApi, A2AcceptsBfloat16)
{
    op::g_socVersion = op::SocVersion::ASCEND910B;
    EXPECT_EQ(Workspace(ACL_BF16, {1, 1, 9, 64}, 3), ACLNN_SUCCESS);
}

TEST_F(MaxPool2dWithMaskApi, Ascend310PRejectsBfloat16DuringParameterCheck)
{
    op::g_socVersion = op::SocVersion::ASCEND310P;
    EXPECT_EQ(Workspace(ACL_BF16, {1, 1, 9, 64}, 3), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MaxPool2dWithMaskApi, BothPlatformsAcceptFloatAndHalf)
{
    for (const auto soc : {op::SocVersion::ASCEND910B, op::SocVersion::ASCEND310P}) {
        op::g_socVersion = soc;
        EXPECT_EQ(Workspace(ACL_FLOAT), ACLNN_SUCCESS);
        EXPECT_EQ(Workspace(ACL_FLOAT16), ACLNN_SUCCESS);
    }
}

TEST_F(MaxPool2dWithMaskApi, RejectsUndersizedOneByOneContainer)
{
    op::g_socVersion = op::SocVersion::ASCEND910B;
    EXPECT_EQ(Workspace(ACL_FLOAT, {1, 1, 1, 96}), ACLNN_ERR_PARAM_INVALID);
}
} // namespace
