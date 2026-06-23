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
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/group_norm_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class TestGroupNormInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 20;
        platformInfo.str_info.short_soc_version = "Ascend910B";
        optiCompilationInfo.soc_version = "Ascend910B";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};

TEST_F(TestGroupNormInfershape, group_norm_infershape_fp16_4d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNorm")->infer_shape;

    gert::Shape xShape = {2, 8, 4, 4};
    gert::Shape yShape = {};
    gert::Shape meanShape = {};
    gert::Shape rstdShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, nullptr, nullptr})
                      .OutputShapes({&yShape, &meanShape, &rstdShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(4)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto outputYShape = context->GetOutputShape(0);
    auto outputMeanShape = context->GetOutputShape(1);
    auto outputRstdShape = context->GetOutputShape(2);
    ASSERT_NE(outputYShape, nullptr);
    ASSERT_NE(outputMeanShape, nullptr);
    ASSERT_NE(outputRstdShape, nullptr);
    EXPECT_EQ(*outputYShape, xShape);
    EXPECT_EQ(outputMeanShape->GetDimNum(), 2);
    EXPECT_EQ(outputMeanShape->GetDim(0), 2);
    EXPECT_EQ(outputMeanShape->GetDim(1), 4);
    EXPECT_EQ(*outputRstdShape, *outputMeanShape);
}

TEST_F(TestGroupNormInfershape, group_norm_infershape_invalid_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNorm")->infer_shape;

    gert::Shape xShape = {8};
    gert::Shape yShape = {};
    gert::Shape meanShape = {};
    gert::Shape rstdShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, nullptr, nullptr})
                      .OutputShapes({&yShape, &meanShape, &rstdShape})
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(4)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TestGroupNormInfershape, group_norm_infershape_invalid_group)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNorm")->infer_shape;

    gert::Shape xShape = {2, 8, 4, 4};
    gert::Shape yShape = {};
    gert::Shape meanShape = {};
    gert::Shape rstdShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, nullptr, nullptr})
                      .OutputShapes({&yShape, &meanShape, &rstdShape})
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
