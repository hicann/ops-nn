/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "../../../op_graph/grouped_dynamic_block_quant_proto.h"

using namespace ge;
using namespace op;

class GroupedDynamicBlockQuantProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedDynamicBlockQuantProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedDynamicBlockQuantProto TearDown" << std::endl;
    }
};

TEST_F(GroupedDynamicBlockQuantProto, inferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_group_list_ref = ge::DT_INT32;
        ge::DataType output_y_ref = ge::DT_FLOAT8_E4M3FN;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({
                                      {"min_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(0.0)},
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(36)},
                                      {"row_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                                      {"col_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                                      {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  })
                                  .InputDataTypes({&input_x_ref, &input_group_list_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
    }
}

TEST_F(GroupedDynamicBlockQuantProto, inferDtype_case_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_group_list_ref = ge::DT_INT32;
        ge::DataType output_y_ref = ge::DT_FLOAT8_E5M2;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({
                                      {"min_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(0.0)},
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)},
                                      {"row_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                                      {"col_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                                      {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  })
                                  .InputDataTypes({&input_x_ref, &input_group_list_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
    }
}

TEST_F(GroupedDynamicBlockQuantProto, inferDtype_case_3)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_group_list_ref = ge::DT_INT32;
        ge::DataType output_y_ref = ge::DT_HIFLOAT8;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({
                                      {"min_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(0.0)},
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("round")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)},
                                      {"row_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                                      {"col_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                                      {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  })
                                  .InputDataTypes({&input_x_ref, &input_group_list_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
    }
}

TEST_F(GroupedDynamicBlockQuantProto, infershape_case_1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant")->infer_shape;

    int64_t B = 6;
    int64_t H = 2000;
    int64_t W = 2000;
    int64_t groupNum = 10;
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    gert::StorageShape xShape = {{B, H, W}, {B, H, W}};
    gert::StorageShape groupListShape = {{groupNum}, {groupNum}};
    gert::StorageShape yShape = {{B, H, W}, {B, H, W}};
    gert::StorageShape scaleShape = {
        {B, H / rowBlockSize + groupNum, (W + colBlockSize - 1) / colBlockSize},
        {B, H / rowBlockSize + groupNum, (W + colBlockSize - 1) / colBlockSize}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({
                          {"min_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(0.0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                          {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)},
                          {"row_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(rowBlockSize)},
                          {"col_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(colBlockSize)},
                          {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &groupListShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* YShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* ScaleShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*YShape), "[6, 2000, 2000]");
    ASSERT_EQ(Ops::Base::ToString(*ScaleShape), "[6, 25, 16]");
}

TEST_F(GroupedDynamicBlockQuantProto, infershape_case_2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant")->infer_shape;

    int64_t H = 2000;
    int64_t W = 2000;
    int64_t groupNum = 10;
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    gert::StorageShape xShape = {{H, W}, {H, W}};
    gert::StorageShape groupListShape = {{groupNum}, {groupNum}};
    gert::StorageShape yShape = {{H, W}, {H, W}};
    gert::StorageShape scaleShape = {
        {H / rowBlockSize + groupNum, (W + colBlockSize - 1) / colBlockSize},
        {H / rowBlockSize + groupNum, (W + colBlockSize - 1) / colBlockSize}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({
                          {"min_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(0.0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                          {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)},
                          {"row_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(rowBlockSize)},
                          {"col_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(colBlockSize)},
                          {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &groupListShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* YShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* ScaleShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*YShape), "[2000, 2000]");
    ASSERT_EQ(Ops::Base::ToString(*ScaleShape), "[25, 16]");
}