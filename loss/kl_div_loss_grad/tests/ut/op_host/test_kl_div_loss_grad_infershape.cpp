/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "log/log.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "platform/platform_info.h"

// ----------------KlDivLossGrad--------------
class KlDivLossGradInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KlDivLossGradInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KlDivLossGradInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(KlDivLossGradInfershapeTest, kl_div_loss_grad_test1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_95";
    optiCompilationInfo.soc_version = "Ascend910_95";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_95"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("KlDivLossGrad")->infer_shape;
    gert::Shape gradShape = {3, 4, 5};
    gert::Shape inputShape = {3, 4, 5};
    gert::Shape targetShape = {3, 4, 5};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&gradShape, &inputShape, &targetShape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs(
                          {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                           {"ignore_index", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}