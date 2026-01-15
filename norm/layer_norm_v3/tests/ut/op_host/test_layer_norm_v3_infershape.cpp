/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "array_ops.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../op_graph/layer_norm_v3_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_info.h"

class LayerNormV3InfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LayerNormV3Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LayerNormV3Test TearDown" << std::endl;
    }
};

TEST_F(LayerNormV3InfershapeTest, layer_norm_v3_infershape_test_1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_95";
    optiCompilationInfo.soc_version = "Ascend910_95";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_95"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormV3")->infer_shape;

    gert::Shape input_x_shape = {30, 256, 512};
    gert::Shape input_gamma_shape = {512};
    gert::Shape input_beta_shape = {512};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_x_shape, &input_gamma_shape, &input_beta_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"begin_norm_dims", Ops::NN::AnyValue::CreateFrom<int64_t>(-4)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
