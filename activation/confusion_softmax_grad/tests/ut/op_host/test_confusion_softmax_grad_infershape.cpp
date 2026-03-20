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
#include "../../../op_graph/confusion_softmax_grad_proto.h"

class ConfusionSoftmaxGradInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ConfusionSoftmaxGradTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ConfusionSoftmaxGradTest TearDown" << std::endl;
    }
};

TEST_F(ConfusionSoftmaxGradInfershapeTest, confusion_softmax_grad_infershape_test_0)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ConfusionSoftmaxGrad"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ConfusionSoftmaxGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape grad_shape = {2, 3, 4, 5};
    gert::Shape x_shape = {2, 3, 4, 5};
    gert::Shape y_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1}, {1})
                      .InputShapes({&grad_shape, &x_shape})
                      .OutputShapes({&y_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output0), "[2, 3, 4, 5]");
}