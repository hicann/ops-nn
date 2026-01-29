/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/relu_grad_v2_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class ReluGradV2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReluGradV2 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ReluGradV2 TearDown" << std::endl;
    }
};

TEST_F(ReluGradV2, ReluGradV2_infershape_95_test)
{
    // set version
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    ge::op::ReluGradV2 op;

    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 16}, {1, 16}};

    auto tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {16, 16}, ge::FORMAT_ND, shape_range);

    auto tensor_desc2 =
        create_desc_shape_range({-1, -1}, ge::DT_UINT1, ge::FORMAT_ND, {16, 16}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("gradients", tensor_desc);
    op.UpdateInputDesc("mask", tensor_desc2);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("backprops");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {-1, -1};
    EXPECT_EQ(output_desc.GetShape().GetDimNum(), 2);

    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);

    auto output_desc1 = op.GetOutputDesc(0);
    EXPECT_EQ(output_desc1.GetShape().GetDimNum(), 2);
}
