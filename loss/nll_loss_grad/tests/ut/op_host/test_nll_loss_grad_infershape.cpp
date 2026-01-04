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

// ----------------NLLLossGrad--------------
class nll_loss_grad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "nll_loss_grad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "nll_loss_grad TearDown" << std::endl;
  }
};

TEST_F(nll_loss_grad, nll_loss_grad_infershape_test1) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_95";
    optiCompilationInfo.soc_version = "Ascend910_95";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_95"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLossGrad")->infer_shape;

    gert::Shape xShape = {-1};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                        .NodeIoNum(5, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1})
                        .InputShapes({&xShape, &xShape, &xShape, &xShape, &xShape})
                        .OutputShapes({&output_shape})
                        .NodeAttrs({
                            {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                            {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}
                        })
                        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {-1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(nll_loss_grad, nll_loss_grad_infershape_test2) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_95";
    optiCompilationInfo.soc_version = "Ascend910_95";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_95"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLossGrad")->infer_shape;

    gert::Shape xShape = {-1, -1};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                        .NodeIoNum(5, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1})
                        .InputShapes({&xShape, &xShape, &xShape, &xShape, &xShape})
                        .OutputShapes({&output_shape})
                        .NodeAttrs({
                            {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                            {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}
                        })
                        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {-1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(nll_loss_grad, nll_loss_grad_infershape_test3) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_95";
    optiCompilationInfo.soc_version = "Ascend910_95";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_95"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLossGrad")->infer_shape;

    gert::Shape xShape = {-1, -1, -1, -1};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                        .NodeIoNum(5, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1})
                        .InputShapes({&xShape, &xShape, &xShape, &xShape, &xShape})
                        .OutputShapes({&output_shape})
                        .NodeAttrs({
                            {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                            {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}
                        })
                        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {-1, -1, -1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(nll_loss_grad, nll_loss_grad_infershape_test4) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_95";
    optiCompilationInfo.soc_version = "Ascend910_95";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_95"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLossGrad")->infer_shape;

    gert::Shape xShape = {-1, -1, -1};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                        .NodeIoNum(5, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1})
                        .InputShapes({&xShape, &xShape, &xShape, &xShape, &xShape})
                        .OutputShapes({&output_shape})
                        .NodeAttrs({
                            {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                            {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}
                        })
                        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(nll_loss_grad, success_dtype_infer_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLossGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLossGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref1 = ge::DT_FLOAT;
        ge::DataType input_ref2 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto holder = gert::InferDataTypeContextFaker()
            .NodeIoNum(5, 1)
            .IrInstanceNum({1,1,1,1,1})
            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({
                        {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                        {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}
                        })
            .InputDataTypes({ &input_ref1, &input_ref1, &input_ref2, &input_ref1, &input_ref1})
            .OutputDataTypes({ &output_ref })
            .Build();

        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(nll_loss_grad, fail_dtype_infer_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLossGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("NLLLossGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType input_ref2 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_INT32;
        auto holder = gert::InferDataTypeContextFaker()
            .NodeIoNum(5, 1)
            .IrInstanceNum({1,1,1,1,1})
            .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({
                        {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                        {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}
                        })
            .InputDataTypes({ &input_ref1, &input_ref1, &input_ref1, &input_ref1, &input_ref1})
            .OutputDataTypes({ &output_ref })
            .Build();

        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}