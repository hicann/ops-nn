/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sparse_slice_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/sparse_slice_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class SparseSliceRt2UTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseSliceRt2UTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseSliceRt2UTest TearDown" << std::endl;
    }
};

TEST_F(SparseSliceRt2UTest, InferShape_succ_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSlice"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSlice")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape indices_shape = {{4, 2}, {4, 2}};
    gert::StorageShape values_shape = {{4}, {4}};
    gert::StorageShape shape_shape = {{4, 1}, {4, 1}};
    gert::StorageShape start_shape = {{1}, {1}};
    gert::StorageShape size_shape = {{1}, {1}};
    gert::StorageShape y_indices_shape = {{}, {}};
    gert::StorageShape y_values_shape = {{}, {}};
    gert::StorageShape y_shape_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInputNum(5)
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&indices_shape, &values_shape, &shape_shape, &start_shape, &size_shape})
                      .OutputShapes({&y_indices_shape, &y_values_shape, &y_shape_shape})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), -1);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(context->GetOutputShape(1)->GetDimNum(), 1);
    EXPECT_EQ(context->GetOutputShape(1)->GetDim(0), -1);
    EXPECT_EQ(context->GetOutputShape(2)->GetDimNum(), 2);
    EXPECT_EQ(context->GetOutputShape(2)->GetDim(0), 4);
    EXPECT_EQ(context->GetOutputShape(2)->GetDim(1), 1);
}

TEST_F(SparseSliceRt2UTest, InferDataType_success)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSlice"), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSlice")->infer_datatype;
    ASSERT_NE(infer_datatype_func, nullptr);
    ge::DataType indices_type = ge::DT_INT64;
    ge::DataType values_type = ge::DT_FLOAT;
    ge::DataType shape_type = ge::DT_INT64;
    ge::DataType sta_type = ge::DT_INT64;
    ge::DataType size_type = ge::DT_INT64;
    ge::DataType y_indices_type = ge::DT_UNDEFINED;
    ge::DataType y_values_type = ge::DT_UNDEFINED;
    ge::DataType y_shape_type = ge::DT_UNDEFINED;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(5, 3)
                              .IrInputNum(5)
                              .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&indices_type, &values_type, &shape_type, &sta_type, &size_type})
                              .OutputDataTypes({&y_indices_type, &y_values_type, &y_shape_type})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT64);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_INT64);
}

// TEST_F(SparseSliceRt2UTest, InferShapeRange_success)
// {
//     gert::Shape indices_min{0, 2};
//     gert::Shape indices_max{100, 2};
//     gert::Shape values_min{0};
//     gert::Shape values_max{100};
//     gert::Shape shape_min{2};
//     gert::Shape shape_max{2};
//     gert::Shape start_min{2};
//     gert::Shape start_max{2};
//     gert::Shape size_min{2};
//     gert::Shape size_max{2};

//     gert::Shape y_indices_min{0, 2};
//     gert::Shape y_indices_max{100, 2};
//     gert::Shape y_values_min{0};
//     gert::Shape y_values_max{100};
//     gert::Shape y_shape_min{2};
//     gert::Shape y_shape_max{2};

//     gert::Range<gert::Shape> range_indices(&indices_min, &indices_max);
//     gert::Range<gert::Shape> range_values(&values_min, &values_max);
//     gert::Range<gert::Shape> range_shape(&shape_min, &shape_max);
//     gert::Range<gert::Shape> range_start(&start_min, &start_max);
//     gert::Range<gert::Shape> range_size(&size_min, &size_max);

//     gert::Range<gert::Shape> range_y_indices(&y_indices_min, &y_indices_max);
//     gert::Range<gert::Shape> range_y_values(&y_values_min, &y_values_max);
//     gert::Range<gert::Shape> range_y_shape(&y_shape_min, &y_shape_max);

//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSlice"), nullptr);
//     auto shape_range_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSlice")->infer_shape_range;
//     ASSERT_NE(shape_range_func, nullptr);

//     ge::DataType indices_type = ge::DT_INT64;
//     ge::DataType values_type = ge::DT_FLOAT;
//     ge::DataType shape_type = ge::DT_INT64;
//     ge::DataType sta_type = ge::DT_INT64;
//     ge::DataType size_type = ge::DT_INT64;
//     ge::DataType y_indices_type = ge::DT_INT64;
//     ge::DataType y_values_type = ge::DT_FLOAT;
//     ge::DataType y_shape_type = ge::DT_INT64;
//     auto context_holder =
//         gert::InferShapeRangeContextFaker()
//             .NodeIoNum(5, 3)
//             .IrInputNum(5)
//             .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
//             .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//             .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
//             .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
//             .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
//             .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
//             .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//             .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
//             .InputShapeRanges({&range_indices, &range_values, &range_shape, &range_start, &range_size})
//             .OutputShapeRanges({&range_y_indices, &range_y_values, &range_y_shape})
//             .Build();
//     auto context = context_holder.GetContext<gert::InferShapeRangeContext>();
//     EXPECT_EQ(shape_range_func(context), ge::GRAPH_SUCCESS);

//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin(), &y_indices_min);
//     EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax(), &y_indices_max);
//     EXPECT_EQ(context->GetOutputShapeRange(1)->GetMin(), &y_values_min);
//     EXPECT_EQ(context->GetOutputShapeRange(1)->GetMax(), &y_values_max);
//     EXPECT_EQ(context->GetOutputShapeRange(2)->GetMin(), &y_shape_min);
//     EXPECT_EQ(context->GetOutputShapeRange(2)->GetMax(), &y_shape_max);
// }