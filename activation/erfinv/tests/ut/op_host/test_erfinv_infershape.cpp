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
 * \file test_erfinv_infershape.cpp
 * \brief erfinv infershape UT test
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/erfinv_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class ErfinvInferShapeUTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ErfinvInferShapeUTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ErfinvInferShapeUTest TearDown" << std::endl;
    }
};

TEST_F(ErfinvInferShapeUTest, erfinv_infershape_succ_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Erfinv"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Erfinv")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape self_shape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    gert::StorageShape out_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&self_shape})
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

TEST_F(ErfinvInferShapeUTest, erfinv_infershape_succ_2_1d)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Erfinv"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Erfinv")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape self_shape = {{128}, {128}};
    gert::StorageShape out_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&self_shape})
                      .OutputShapes({&out_shape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 128);
}

TEST_F(ErfinvInferShapeUTest, erfinv_infershape_succ_3_5d)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Erfinv"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Erfinv")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape self_shape = {{2, 4, 4, 4, 7}, {2, 4, 4, 4, 7}};
    gert::StorageShape out_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&self_shape})
                      .OutputShapes({&out_shape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 5);
}

TEST_F(ErfinvInferShapeUTest, erfinv_infershape_empty_tensor)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Erfinv"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Erfinv")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape self_shape = {{2, 0, 4}, {2, 0, 4}};
    gert::StorageShape out_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&self_shape})
                      .OutputShapes({&out_shape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 0);
}