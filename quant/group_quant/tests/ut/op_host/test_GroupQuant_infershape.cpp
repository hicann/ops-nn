/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "../../../op_graph/group_quant_proto.h"

using namespace ge;
using namespace op;

class GroupQuantProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupQuantProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupQuantProto TearDown" << std::endl;
    }
};

TEST_F(GroupQuantProto, inferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GroupQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT;
        ge::DataType input_scale_ref = ge::DT_FLOAT;
        ge::DataType input_group_index_ref = ge::DT_INT32;
        ge::DataType input_offset_ref = ge::DT_FLOAT;
        ge::DataType output_y_ref = ge::DT_INT8;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(1)
                .NodeIoNum(4, 1)
                .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({
                    {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                })
                .InputDataTypes({&input_x_ref, &input_scale_ref, &input_group_index_ref, &input_offset_ref})
                .OutputDataTypes({&output_y_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
    }
}

TEST_F(GroupQuantProto, infershape_case_1)
{
    // set input info
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupQuant")->infer_shape;

    int64_t dimS = 6;
    int64_t dimE = 4;
    int64_t dimH = 4;
    gert::StorageShape xShape = {{dimS, dimH}, {dimS, dimH}};
    gert::StorageShape scaleShape = {{dimE, dimH}, {dimE, dimH}};
    gert::StorageShape groupIndexShape = {
        {
            dimE,
        },
        {
            dimE,
        }};
    gert::StorageShape offsetShape = {
        {
            1,
        },
        {
            1,
        }};
    gert::StorageShape yShape = {{dimS, dimH}, {dimS, dimH}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({
                          {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                      })
                      .InputShapes({&xShape, &scaleShape, &groupIndexShape, &offsetShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[6, 4]");
}