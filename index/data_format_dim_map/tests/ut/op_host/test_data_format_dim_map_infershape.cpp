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
 * \file test_data_format_dim_map_infershape.cpp
 * \brief data_format_dim_map infershape UT test
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/data_format_dim_map_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class DataFormatDimMapInferShapeUTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DataFormatDimMapInferShapeUTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DataFormatDimMapInferShapeUTest TearDown" << std::endl; }
};

TEST_F(DataFormatDimMapInferShapeUTest, data_format_dim_map_infershape_1d)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{128}, {128}};
    gert::StorageShape out_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x_shape})
                      .OutputShapes({&out_shape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 128);
}

TEST_F(DataFormatDimMapInferShapeUTest, data_format_dim_map_infershape_2d)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{64, 128}, {64, 128}};
    gert::StorageShape out_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x_shape})
                      .OutputShapes({&out_shape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 64);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 128);
}

TEST_F(DataFormatDimMapInferShapeUTest, data_format_dim_map_infershape_4d)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    gert::StorageShape out_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x_shape})
                      .OutputShapes({&out_shape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 4);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 4);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 16);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(2), 4);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(3), 4);
}

TEST_F(DataFormatDimMapInferShapeUTest, data_format_dim_map_infershape_5d)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{2, 4, 8, 16, 32}, {2, 4, 8, 16, 32}};
    gert::StorageShape out_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x_shape})
                      .OutputShapes({&out_shape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 5);
}

TEST_F(DataFormatDimMapInferShapeUTest, data_format_dim_map_infershape_scalar)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{}, {}};
    gert::StorageShape out_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x_shape})
                      .OutputShapes({&out_shape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 0);
}

TEST_F(DataFormatDimMapInferShapeUTest, data_format_dim_map_infershape_int64)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{256, 512}, {256, 512}};
    gert::StorageShape out_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x_shape})
                      .OutputShapes({&out_shape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 256);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 512);
}

TEST_F(DataFormatDimMapInferShapeUTest, data_format_dim_map_infershape_large_tensor)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{1024, 2048}, {1024, 2048}};
    gert::StorageShape out_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x_shape})
                      .OutputShapes({&out_shape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 1024);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 2048);
}

TEST_F(DataFormatDimMapInferShapeUTest, data_format_dim_map_infershape_8d)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DataFormatDimMap")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}};
    gert::StorageShape out_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x_shape})
                      .OutputShapes({&out_shape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 8);
}
