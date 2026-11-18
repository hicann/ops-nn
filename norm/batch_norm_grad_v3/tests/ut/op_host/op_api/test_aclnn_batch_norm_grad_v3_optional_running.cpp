/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <tuple>
#include "gtest/gtest.h"
#include "opdev/platform.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "norm/batch_norm_grad_v3/op_host/op_api/aclnn_batch_norm_backward.h"

using RunningOptionalParam = std::tuple<op::SocVersion, aclDataType, int, int>;

class BatchNormGradV3OptionalRunning : public testing::TestWithParam<RunningOptionalParam> {};

TEST_P(BatchNormGradV3OptionalRunning, WorkspaceWithOptionalStatistics)
{
    const auto soc = std::get<0>(GetParam());
    const auto dtype = std::get<1>(GetParam());
    const int missing = std::get<2>(GetParam());
    const int rank = std::get<3>(GetParam());
    op::SocVersionManager platform(soc);
    const std::vector<int64_t> shape = rank == 7 ? std::vector<int64_t>{2, 8, 2, 1, 2, 1, 2} :
                                                   std::vector<int64_t>{2, 8, 2, 4};
    const auto format = rank == 7 ? ACL_FORMAT_ND : ACL_FORMAT_NCHW;
    auto dy = TensorDesc(shape, dtype, format);
    auto x = TensorDesc(shape, dtype, format);
    auto weight = TensorDesc({8}, dtype, ACL_FORMAT_ND);
    auto mean = TensorDesc({8}, dtype, ACL_FORMAT_ND);
    auto variance = TensorDesc({8}, dtype, ACL_FORMAT_ND);
    auto saved = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dx = TensorDesc(shape, dtype, format);
    auto dw = TensorDesc({8}, dtype, ACL_FORMAT_ND);
    auto db = TensorDesc({8}, dtype, ACL_FORMAT_ND);
    auto mask = BoolArrayDesc(std::vector<bool>{true, true, true});
    auto check = [&](auto runningMean, auto runningVar) {
        auto ut = OP_API_UT(aclnnBatchNormBackward,
                            INPUT(dy, x, weight, runningMean, runningVar, saved, saved, true, 1e-5, mask),
                            OUTPUT(dx, dw, db));
        uint64_t workspace = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace), ACLNN_SUCCESS);
    };
    if (missing == 0)
        check(mean, variance);
    if (missing == 1)
        check(nullptr, variance);
    if (missing == 2)
        check(mean, nullptr);
    if (missing == 3)
        check(nullptr, nullptr);
}

INSTANTIATE_TEST_SUITE_P(TargetPlatformsAndDtypes, BatchNormGradV3OptionalRunning,
                         testing::Combine(testing::Values(op::SocVersion::ASCEND950, op::SocVersion::ASCEND960DT,
                                                          op::SocVersion::ASCEND960PR),
                                          testing::Values(ACL_FLOAT16, ACL_BF16), testing::Values(0, 1, 2, 3),
                                          testing::Values(4, 7)));

TEST(BatchNormGradV3RunningValidation, ExplicitMismatchedStatisticsRemainInvalid)
{
    for (auto soc : {op::SocVersion::ASCEND950, op::SocVersion::ASCEND960DT, op::SocVersion::ASCEND960PR}) {
        op::SocVersionManager platform(soc);
        for (auto dtype : {ACL_FLOAT16, ACL_BF16}) {
            auto dy = TensorDesc({2, 8, 2, 4}, dtype, ACL_FORMAT_NCHW);
            auto weight = TensorDesc({8}, dtype, ACL_FORMAT_ND);
            auto mean = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
            auto variance = TensorDesc({8}, dtype, ACL_FORMAT_ND);
            auto saved = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
            auto dx = TensorDesc({2, 8, 2, 4}, dtype, ACL_FORMAT_NCHW);
            auto dw = TensorDesc({8}, dtype, ACL_FORMAT_ND);
            auto db = TensorDesc({8}, dtype, ACL_FORMAT_ND);
            auto mask = BoolArrayDesc(std::vector<bool>{true, true, true});
            auto ut = OP_API_UT(aclnnBatchNormBackward,
                                INPUT(dy, dy, weight, mean, variance, saved, saved, true, 1e-5, mask),
                                OUTPUT(dx, dw, db));
            uint64_t workspace = 0;
            EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_INVALID);
        }
    }
}
