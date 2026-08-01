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
 * \file test_group_norm_silu_quant_infershape.cpp
 * \brief GroupNormSiluQuant graph infershape / inferdatatype UT (faker-based, no proto dependency).
 *   y == x shape / DT_INT8; mean,rstd == (N, num_groups) / same dtype as x.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace ge;

namespace {
static std::vector<int64_t> ShapeToVec(const gert::Shape& s)
{
    std::vector<int64_t> v;
    for (size_t i = 0; i < s.GetDimNum(); i++) {
        v.push_back(s.GetDim(i));
    }
    return v;
}
} // namespace

class GroupNormSiluQuantInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupNormSiluQuantInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GroupNormSiluQuantInferShapeTest TearDown" << std::endl; }
};

TEST_F(GroupNormSiluQuantInferShapeTest, infershape_y_eq_x_mean_n_group)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNormSiluQuant");
    ASSERT_NE(opImpl, nullptr);
    auto infer_shape_func = opImpl->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::Shape x_shape = {4, 320, 16, 16};
    gert::Shape gamma_shape = {320};
    gert::Shape beta_shape = {320};
    gert::Shape quant_scale_shape = {320};
    gert::Shape y_shape = {};
    gert::Shape mean_shape = {};
    gert::Shape rstd_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 3)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape, &quant_scale_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                                  {"activate_silu", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(ShapeToVec(*context->GetOutputShape(0)), (std::vector<int64_t>{4, 320, 16, 16}));
    EXPECT_EQ(ShapeToVec(*context->GetOutputShape(1)), (std::vector<int64_t>{4, 32}));
    EXPECT_EQ(ShapeToVec(*context->GetOutputShape(2)), (std::vector<int64_t>{4, 32}));
}

// 动态 rank：x 为 -2(UNKNOWN_RANK) 时，三个输出都必须是未知秩。
// def 声明了 DynamicRankSupportFlag(true)，但此前 infershape 无对应分支——
// GetDim(0) 会取到 -2 标记值，把 mean/rstd 推成非法的 (-2, num_groups)。
TEST_F(GroupNormSiluQuantInferShapeTest, infershape_unknown_rank_x)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNormSiluQuant");
    ASSERT_NE(opImpl, nullptr);
    auto infer_shape_func = opImpl->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::Shape x_shape = {-2};
    gert::Shape gamma_shape = {320};
    gert::Shape beta_shape = {320};
    gert::Shape quant_scale_shape = {320};
    gert::Shape y_shape = {};
    gert::Shape mean_shape = {};
    gert::Shape rstd_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 3)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape, &quant_scale_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                                  {"activate_silu", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(ShapeToVec(*context->GetOutputShape(0)), (std::vector<int64_t>{-2}));
    EXPECT_EQ(ShapeToVec(*context->GetOutputShape(1)), (std::vector<int64_t>{-2}));
    EXPECT_EQ(ShapeToVec(*context->GetOutputShape(2)), (std::vector<int64_t>{-2}));
}

// 空 Tensor：任意维度为 0 都支持（判据在共用的 aclnn L2 的 self->IsEmpty()，
// A2 首版即有）。infershape 侧 y 与 x 同形、mean/rstd 仍为 (N, num_groups)。
// 逐轴各一例：N 轴、C 轴、HW 轴。
TEST_F(GroupNormSiluQuantInferShapeTest, infershape_empty_tensor_each_axis)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNormSiluQuant");
    ASSERT_NE(opImpl, nullptr);
    auto infer_shape_func = opImpl->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    const std::vector<std::vector<int64_t>> emptyCases = {
        {0, 320, 16, 16}, // N 轴为 0
        {4, 0, 16, 16},   // C 轴为 0
        {4, 320, 0, 16},  // HW 轴为 0
    };
    for (const auto& dims : emptyCases) {
        gert::Shape x_shape;
        for (auto d : dims) {
            x_shape.AppendDim(d);
        }
        gert::Shape gamma_shape = {320};
        gert::Shape beta_shape = {320};
        gert::Shape quant_scale_shape = {320};
        gert::Shape y_shape = {};
        gert::Shape mean_shape = {};
        gert::Shape rstd_shape = {};

        auto holder = gert::InferShapeContextFaker()
                          .NodeIoNum(4, 3)
                          .IrInstanceNum({1, 1, 1, 1})
                          .InputShapes({&x_shape, &gamma_shape, &beta_shape, &quant_scale_shape})
                          .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                                      {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                                      {"activate_silu", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                          .Build();

        auto context = holder.GetContext<gert::InferShapeContext>();
        ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(ShapeToVec(*context->GetOutputShape(0)), dims);
        EXPECT_EQ(ShapeToVec(*context->GetOutputShape(1)), (std::vector<int64_t>{dims[0], 32}));
        EXPECT_EQ(ShapeToVec(*context->GetOutputShape(2)), (std::vector<int64_t>{dims[0], 32}));
    }
}

TEST_F(GroupNormSiluQuantInferShapeTest, inferdatatype_y_int8_mean_same_as_x)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNormSiluQuant");
    ASSERT_NE(opImpl, nullptr);
    auto infer_dt_func = opImpl->infer_datatype;
    ASSERT_NE(infer_dt_func, nullptr);

    ge::DataType xDt = ge::DT_BF16;
    ge::DataType fDt = ge::DT_FLOAT;
    ge::DataType yOut = ge::DT_UNDEFINED;
    ge::DataType meanOut = ge::DT_UNDEFINED;
    ge::DataType rstdOut = ge::DT_UNDEFINED;

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(4, 3)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&xDt, &xDt, &xDt, &fDt})
                      .OutputDataTypes({&yOut, &meanOut, &rstdOut})
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(infer_dt_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_BF16);
}
