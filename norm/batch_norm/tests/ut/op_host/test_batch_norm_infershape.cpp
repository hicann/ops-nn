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
#include <iostream>
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "array_ops.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_info.h"
#include "../../../op_graph/batch_norm_proto.h"

class BatchNormInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BatchNormTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BatchNormTest TearDown" << std::endl;
    }
};

TEST_F(BatchNormInfershapeTest, batch_norm_infershape_test_0)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNorm"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNorm")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape input_x_shape = {2, 3, 4, 5};
    gert::Shape input_scale_shape = {5};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1})
                      .InputShapes({&input_x_shape, &input_scale_shape, &input_scale_shape,
                                    &input_scale_shape, &input_scale_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    auto output5 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(5);
    ASSERT_EQ(Ops::Base::ToString(*output0), "[2, 3, 4, 5]");
    ASSERT_EQ(Ops::Base::ToString(*output1), "[5]");
    ASSERT_EQ(Ops::Base::ToString(*output5), "[1]");
}

TEST_F(BatchNormInfershapeTest, batch_norm_inferdtype_test_0)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNorm"), nullptr);
    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNorm")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    ge::DataType dfp16 = ge::DT_FLOAT16;
    ge::DataType dfp32 = ge::DT_FLOAT;

    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(5)
                              .NodeIoNum(5, 6)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                              .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                              .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs(
                                  {{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                              .InputDataTypes({&dfp16, &dfp32, &dfp32, &dfp32, &dfp32})
                              .OutputDataTypes({&dfp16, &dfp32, &dfp32, &dfp32, &dfp32, &dfp32})
                              .Build();

    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), dfp16);
    EXPECT_EQ(context->GetInputDataType(1), dfp32);
}
