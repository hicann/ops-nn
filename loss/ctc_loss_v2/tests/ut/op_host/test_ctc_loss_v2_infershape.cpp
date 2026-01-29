/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_ctc_loss_v2_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/ctc_loss_v2_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

namespace {
// ----------------CTCLossV2-------------------
class CTCLossV2ProtoTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CTCLossV2 Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CTCLossV2 Proto Test TearDown" << std::endl;
    }
};

//   pass cases
TEST_F(CTCLossV2ProtoTest, ctc_loss_v2_infer_shape_test1_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("CTCLossV2")->infer_shape;

    int N = 8;
    int T = 16;
    int C = 24;
    int S = 8;

    gert::Shape logProbsShape = {T, N, C};
    gert::Shape targetsShape = {N, S};
    gert::Shape lengthsShape = {N};

    gert::Shape output_shape_0 = {};
    gert::Shape output_shape_1 = {};

    std::vector<int32_t> input_lengths_value = {2, 3, 7, 2, 4, 5, 6, 1};
    std::vector<int32_t> target_lengths_value = {6, 3, 1, 2, 4, 5, 1, 4};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&logProbsShape, &targetsShape, &lengthsShape, &lengthsShape})
                      .OutputShapes({&output_shape_0, &output_shape_1})
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc_0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape_0 = {N};
    // ASSERT_EQ(Ops::Base::ToString(*output_desc_0), Ops::Base::ToString(expected_output_shape_0));

    auto output_desc_1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_output_shape_1 = {N, T, 24};
    // ASSERT_EQ(Ops::Base::ToString(*output_desc_1), Ops::Base::ToString(expected_output_shape_1));
}

TEST_F(CTCLossV2ProtoTest, ctc_loss_v2_infer_shape_test2_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("CTCLossV2")->infer_shape;

    int N = -1;
    int T = -1;
    int C = -1;
    int S = -1;

    gert::Shape logProbsShape = {T, N, C};
    gert::Shape targetsShape = {N, S};
    gert::Shape lengthsShape = {3};

    gert::Shape output_shape_0 = {};
    gert::Shape output_shape_1 = {};

    std::vector<int32_t> input_lengths_value = {2, 3, 7};
    std::vector<int32_t> target_lengths_value = {7, 3, 1};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&logProbsShape, &targetsShape, &lengthsShape, &lengthsShape})
                      .OutputShapes({&output_shape_0, &output_shape_1})
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc_0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape_0 = {N};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_0), Ops::Base::ToString(expected_output_shape_0));

    auto output_desc_1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_output_shape_1 = {N, T, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_1), Ops::Base::ToString(expected_output_shape_1));
}

TEST_F(CTCLossV2ProtoTest, ctc_loss_v2_infer_shape_test3_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("CTCLossV2")->infer_shape;

    gert::Shape logProbsShape = {-2};
    gert::Shape targetsShape = {-2};
    gert::Shape lengthsShape = {3};

    gert::Shape output_shape_0 = {};
    gert::Shape output_shape_1 = {};

    std::vector<int32_t> input_lengths_value = {2, 3, 7};
    std::vector<int32_t> target_lengths_value = {7, 3, 1};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&logProbsShape, &targetsShape, &lengthsShape, &lengthsShape})
                      .OutputShapes({&output_shape_0, &output_shape_1})
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc_0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape_0 = {-2};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_0), Ops::Base::ToString(expected_output_shape_0));

    auto output_desc_1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_output_shape_1 = {-2};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_1), Ops::Base::ToString(expected_output_shape_1));
}

TEST_F(CTCLossV2ProtoTest, ctc_loss_v2_infer_shape_test4_success)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("CTCLossV2")->infer_shape;

    int N = -1;
    int T = -1;
    int C = -1;
    int S = -1;

    gert::Shape logProbsShape = {T, N, C};
    gert::Shape targetsShape = {N, S};
    gert::Shape lengthsShape = {0};

    gert::Shape output_shape_0 = {};
    gert::Shape output_shape_1 = {};

    std::vector<int32_t> input_lengths_value = {2, 3, 7};
    std::vector<int32_t> target_lengths_value = {7, 3, 1};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&logProbsShape, &targetsShape, &lengthsShape, &lengthsShape})
                      .OutputShapes({&output_shape_0, &output_shape_1})
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto output_desc_0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape_0 = {N};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_0), Ops::Base::ToString(expected_output_shape_0));

    auto output_desc_1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_output_shape_1 = {N, T, -1};
    ASSERT_EQ(Ops::Base::ToString(*output_desc_1), Ops::Base::ToString(expected_output_shape_1));
}

TEST_F(CTCLossV2ProtoTest, ctc_loss_v2_infer_dtype_test_success)
{
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("CTCLossV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref1 = ge::DT_FLOAT;
        ge::DataType input_ref2 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
            {"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}};
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .NodeIoNum(4, 2)
                                  .IrInstanceNum({1, 1, 1, 1})
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(attrs)
                                  .InputDataTypes({&input_ref1, &input_ref1, &input_ref2, &input_ref2})
                                  .OutputDataTypes({&output_ref, &output_ref})
                                  .Build();
        auto context = context_holder.template GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
    }
}

// failed
TEST_F(CTCLossV2ProtoTest, ctc_loss_v2_infer_shape_test1_failed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("CTCLossV2")->infer_shape;

    gert::Tensor* targetLengthsTensor = nullptr;
    int N = 8;
    int T = 16;
    int C = 24;
    int S = 8;

    gert::Shape logProbsShape = {T, N, C};
    gert::Shape targetsShape = {N, S};
    gert::Shape lengthsShape = {N};

    gert::Shape output_shape_0 = {};
    gert::Shape output_shape_1 = {};

    std::vector<int32_t> input_lengths_value = {2, 3, 7, 2, 4, 5, 6, 1};
    std::vector<int32_t> target_lengths_value = {6, 3, 1, 2, 4, 5, 1, 4};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&logProbsShape, &targetsShape, &lengthsShape, &lengthsShape})
                      .OutputShapes({&output_shape_0, &output_shape_1})
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(-2)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(CTCLossV2ProtoTest, ctc_loss_v2_infer_shape_test2_failed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("CTCLossV2")->infer_shape;

    int64_t targetLengthsSum = -1;
    int N = 8;
    int T = 16;
    int C = 24;
    int S = 8;

    gert::Shape logProbsShape = {T, N, C};
    gert::Shape targetsShape = {N, S};
    gert::Shape lengthsShape = {N};

    gert::Shape output_shape_0 = {};
    gert::Shape output_shape_1 = {};

    std::vector<int32_t> input_lengths_value = {2, 3, 7, 2, 4, 5, 6, 1};
    std::vector<int32_t> target_lengths_value = {6, 3, 1, 2, 4, 5, 1, 4};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&logProbsShape, &targetsShape, &lengthsShape, &lengthsShape})
                      .OutputShapes({&output_shape_0, &output_shape_1})
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(-2)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(CTCLossV2ProtoTest, ctc_loss_v2_infer_shape_test3_failed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("CTCLossV2")->infer_shape;

    int64_t targetLengthsSum = -1;
    int N = 8;
    int T = 16;
    int C = 24;
    int S = 8;

    gert::Shape logProbsShape = {T, N};
    gert::Shape targetsShape = {N, S};
    gert::Shape lengthsShape = {N};

    gert::Shape output_shape_0 = {};
    gert::Shape output_shape_1 = {};

    std::vector<int32_t> input_lengths_value = {2, 3, 7, 2, 4, 5, 6, 1};
    std::vector<int32_t> target_lengths_value = {6, 3, 1, 2, 4, 5, 1, 4};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&logProbsShape, &targetsShape, &lengthsShape, &lengthsShape})
                      .OutputShapes({&output_shape_0, &output_shape_1})
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(-2)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(CTCLossV2ProtoTest, ctc_loss_v2_infer_shape_test4_failed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("CTCLossV2")->infer_shape;

    int64_t targetLengthsSum = -1;
    int N = 8;
    int T = 16;
    int C = 24;
    int S = 8;

    gert::Shape logProbsShape = {T, N, C};
    gert::Shape targetsShape = {N, N, N};
    gert::Shape lengthsShape = {N};

    gert::Shape output_shape_0 = {};
    gert::Shape output_shape_1 = {};

    std::vector<int32_t> input_lengths_value = {2, 3, 7, 2, 4, 5, 6, 1};
    std::vector<int32_t> target_lengths_value = {6, 3, 1, 2, 4, 5, 1, 4};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&logProbsShape, &targetsShape, &lengthsShape, &lengthsShape})
                      .OutputShapes({&output_shape_0, &output_shape_1})
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(-2)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
} // namespace