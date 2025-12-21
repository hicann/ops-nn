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
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "runtime/infer_shape_range_context.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"

namespace ge
{
class GemmV3InferShape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "GemmV3 Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GemmV3 Proto Test TearDown" << std::endl;
    }
};

TEST_F(GemmV3InferShape, GemmV3_Test_TransA_False_And_TransB_False)
{
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmV3")->infer_shape;

    gert::StorageShape a_shape = {{32, 64}, {32, 64}};
    gert::StorageShape b_shape = {{64, 128}, {64, 128}};
    gert::StorageShape c_shape = {{32, 128}, {32, 128}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&a_shape, &b_shape, &c_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"beta", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"transpose_a", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_b", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    EXPECT_EQ(Ops::Base::ToString(*output), "[32, 128]");
}

TEST_F(GemmV3InferShape, GemmV3_Test_TransA_False_And_TransB_True)
{
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmV3")->infer_shape;

    gert::StorageShape a_shape = {{256, 1024}, {256, 1024}};
    gert::StorageShape b_shape = {{64, 1024}, {64, 1024}};
    gert::StorageShape c_shape = {{256, 64}, {256, 64}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&a_shape, &b_shape, &c_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"beta", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"transpose_a", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_b", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    EXPECT_EQ(Ops::Base::ToString(*output), "[256, 64]");
}

TEST_F(GemmV3InferShape, GemmV3_Test_TransA_True_And_TransB_False)
{
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmV3")->infer_shape;

    gert::StorageShape a_shape = {{128, 2048}, {128, 2048}};
    gert::StorageShape b_shape = {{128, 256}, {128, 2048}};
    gert::StorageShape c_shape = {{2048, 256}, {2048, 256}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&a_shape, &b_shape, &c_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"beta", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"transpose_a", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"transpose_b", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    EXPECT_EQ(Ops::Base::ToString(*output), "[2048, 256]");
}

TEST_F(GemmV3InferShape, GemmV3_Test_TransA_True_And_TransB_True)
{
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmV3")->infer_shape;

    gert::StorageShape a_shape = {{1024, 512}, {1024, 512}};
    gert::StorageShape b_shape = {{64, 1024}, {64, 1024}};
    gert::StorageShape c_shape = {{512, 64}, {512, 64}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&a_shape, &b_shape, &c_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"beta", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"transpose_a", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"transpose_b", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    EXPECT_EQ(Ops::Base::ToString(*output), "[512, 64]");
}

TEST_F(GemmV3InferShape, GemmV3_GemmV3_Test_Enable_Hf32_True)
{
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmV3")->infer_shape;

    gert::StorageShape a_shape = {{2048, 5120}, {2048, 5120}};
    gert::StorageShape b_shape = {{5120, 64}, {5120, 64}};
    gert::StorageShape c_shape = {{2048, 64}, {2048, 64}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&a_shape, &b_shape, &c_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"beta", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"transpose_a", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_b", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    EXPECT_EQ(Ops::Base::ToString(*output), "[2048, 64]");
}
}