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
 * \file test_unique_consecutive_proto.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "../../../op_graph/unique_consecutive_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

class UniqueConsecutiveTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "UniqueConsecutive Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "UniqueConsecutive Proto Test TearDown" << std::endl;
    }
};

TEST_F(UniqueConsecutiveTest, unique_consecutive_infer_shape_test1) {
  ge::op::UniqueConsecutive op;

  ge::DataType dtype = ge::DT_INT64;
  ge::Format format = ge::FORMAT_ND;
  
  auto input_tensor = create_desc_with_ori({3,4}, dtype, format, {3,4}, format);
  
  op.UpdateInputDesc("x", input_tensor);

  op.SetAttr("return_idx", false);
  op.SetAttr("return_counts", false);
  op.SetAttr("axis", 1000);
  Runtime2TestParam param{{"return_idx", "return_counts", "axis"}};
  EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  auto idx_desc = op.GetOutputDescByName("idx");
  auto count_desc = op.GetOutputDescByName("count");

  EXPECT_EQ(output_desc.GetShape().GetDimNum(), 1);
  EXPECT_EQ(idx_desc.GetShape().GetDimNum(), 2);
  EXPECT_EQ(count_desc.GetShape().GetDimNum(), 1);
}

TEST_F(UniqueConsecutiveTest, unique_consecutive_infer_shape_test2) {
  ge::op::UniqueConsecutive op;

  ge::DataType dtype = ge::DT_INT64;
  ge::Format format = ge::FORMAT_ND;
  
  auto input_tensor = create_desc_with_ori({-2}, dtype, format, {-2}, format);
  
  op.UpdateInputDesc("x", input_tensor);

  op.SetAttr("return_idx", true);
  op.SetAttr("return_counts", true);
  op.SetAttr("axis", 1000);
  Runtime2TestParam param{{"return_idx", "return_counts", "axis"}};
  EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  auto idx_desc = op.GetOutputDescByName("idx");
  auto count_desc = op.GetOutputDescByName("count");

  std::vector<int64_t> expected_idx_shape = {-2};
  std::vector<int64_t> expected_count_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_idx_shape);
  EXPECT_EQ(idx_desc.GetShape().GetDims(), expected_idx_shape);
  EXPECT_EQ(count_desc.GetShape().GetDims(), expected_idx_shape);
}

TEST_F(UniqueConsecutiveTest, unique_consecutive_infer_data_ype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("UniqueConsecutive"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("UniqueConsecutive")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_x = ge::DT_FLOAT;
    ge::DataType shape_dtype = ge::DT_INT32;
    ge::DataType y_datatype = ge::DT_FLOAT;
    ge::DataType idx_datatype = ge::DT_INT64;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(1)
                              .NodeIoNum(1, 3)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({{"return_idx", Ops::NN::AnyValue::CreateFrom<bool>(0)},
                                          {"return_counts", Ops::NN::AnyValue::CreateFrom<bool>(0)},
                                          {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                          {"out_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(9)}})
                              .InputDataTypes({&input_x})
                              .OutputDataTypes({&y_datatype, &idx_datatype, &idx_datatype})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), input_x);
    EXPECT_EQ(context->GetOutputDataType(0), y_datatype);
}

TEST_F(UniqueConsecutiveTest, unique_consecutive_infer_range_test0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("UniqueConsecutive"), nullptr);
    auto infer_shape_range_func = gert::OpImplRegistry::GetInstance().GetOpImpl("UniqueConsecutive")->infer_shape_range;

    gert::Shape input_x_range_min{0, 0, 0, 0};
    gert::Shape input_x_range_max{2, -1, 3, 8};
    gert::Shape null1{};
    gert::Shape null2{};
    gert::Shape null3{};
    gert::Shape null4{};
    gert::Shape null5{};
    gert::Shape null6{};

    gert::Range<gert::Shape> input_x_range(&input_x_range_min, &input_x_range_max);
    gert::Range<gert::Shape> output_y_shape_range(&null1, &null2);
    gert::Range<gert::Shape> output_idx_shape_range(&null3, &null4);
    gert::Range<gert::Shape> output_count_shape_range(&null5, &null6);

    gert::Range<gert::Shape> expect_output_0_shape_range(&input_x_range_min, &input_x_range_max);

    gert::Shape output_1_range_min{0};
    gert::Shape output_1_range_max{2};
    gert::Range<gert::Shape> expect_output_1_shape_range(&output_1_range_min, &output_1_range_max);

    auto context_holder = gert::InferShapeRangeContextFaker()
                              .IrInputNum(1)
                              .NodeIoNum(1, 3)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({{"return_idx", Ops::NN::AnyValue::CreateFrom<bool>(0)},
                                          {"return_counts", Ops::NN::AnyValue::CreateFrom<bool>(0)},
                                          {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                          {"out_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                              .InputShapeRanges({&input_x_range})
                              .OutputShapeRanges({&output_y_shape_range, &output_idx_shape_range, &output_count_shape_range})
                              .Build();

    auto context = context_holder.GetContext<gert::InferShapeRangeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_range_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(input_x_range_max));
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(input_x_range_min));

    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(1)->GetMax()), Ops::Base::ToString(output_1_range_max));
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(1)->GetMin()), Ops::Base::ToString(output_1_range_min));
}