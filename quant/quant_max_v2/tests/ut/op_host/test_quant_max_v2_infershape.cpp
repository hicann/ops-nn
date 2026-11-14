/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_quant_max_v2_infershape.cpp
 * @brief Unit test for QuantMaxV2 infer shape / infer datatype
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "register/op_impl_registry.h"
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "ut_op_util.h"
#include "ut_op_common.h"

using namespace std;
using namespace ge;

namespace {
// IR 输入序：x / scale / group_list；IrInstanceNum 的 0 表示该 optional 输入未连接。
const std::vector<std::pair<std::string, Ops::NN::AnyValue>> kAttrs = {
    {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("round")},
    {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)},
    {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(15.0f)}};
} // namespace

class QuantMaxV2InferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

// 动态单张量（scale 缺席）：输出 y 与 x 同形，amax / y_scale 均为 [1]。
TEST_F(QuantMaxV2InferShapeTest, quant_max_v2_infershape_dynamic_tensor)
{
    std::string op_type("QuantMaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape({1024, 512}, {1024, 512});
    gert::StorageShape y_shape({1024, 512}, {1024, 512});
    gert::StorageShape amax_shape({1}, {1});
    gert::StorageShape y_scale_shape({1}, {1});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 0, 0}, {1, 1, 1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&y_shape, &amax_shape, &y_scale_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(kAttrs)
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(infer_shape_context, nullptr);
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    const gert::Shape* output_y_shape = infer_shape_context->GetOutputShape(0);
    ASSERT_NE(output_y_shape, nullptr);
    EXPECT_EQ(output_y_shape->GetDimNum(), 2);
    EXPECT_EQ(output_y_shape->GetDim(0), 1024);
    EXPECT_EQ(output_y_shape->GetDim(1), 512);
    ASSERT_NE(infer_shape_context->GetOutputShape(1), nullptr);
    EXPECT_EQ(infer_shape_context->GetOutputShape(1)->GetDim(0), 1);
    ASSERT_NE(infer_shape_context->GetOutputShape(2), nullptr);
    EXPECT_EQ(infer_shape_context->GetOutputShape(2)->GetDim(0), 1);
}

// 动态分组 G=48（scale 缺席）：amax / y_scale 均按 group_list 长度推导。
TEST_F(QuantMaxV2InferShapeTest, quant_max_v2_infershape_dynamic_group)
{
    std::string op_type("QuantMaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape({96, 256}, {96, 256});
    gert::StorageShape group_shape({48}, {48});
    gert::StorageShape y_shape({96, 256}, {96, 256});
    gert::StorageShape amax_shape({48}, {48});
    gert::StorageShape y_scale_shape({48}, {48});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 0, 1}, {1, 1, 1})
                      .InputShapes({&x_shape, &group_shape})
                      .OutputShapes({&y_shape, &amax_shape, &y_scale_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(kAttrs)
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(infer_shape_context, nullptr);
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    ASSERT_NE(infer_shape_context->GetOutputShape(1), nullptr);
    EXPECT_EQ(infer_shape_context->GetOutputShape(1)->GetDim(0), 48);
    ASSERT_NE(infer_shape_context->GetOutputShape(2), nullptr);
    EXPECT_EQ(infer_shape_context->GetOutputShape(2)->GetDim(0), 48);
}

// 静态单张量：y_scale 推导 [0]（静态不输出）。
TEST_F(QuantMaxV2InferShapeTest, quant_max_v2_infershape_static_tensor)
{
    std::string op_type("QuantMaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape({64, 128}, {64, 128});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({64, 128}, {64, 128});
    gert::StorageShape amax_shape({1}, {1});
    gert::StorageShape y_scale_shape({0}, {0});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 0}, {1, 1, 1})
                      .InputShapes({&x_shape, &scale_shape})
                      .OutputShapes({&y_shape, &amax_shape, &y_scale_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(kAttrs)
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(infer_shape_context, nullptr);
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    ASSERT_NE(infer_shape_context->GetOutputShape(2), nullptr);
    EXPECT_EQ(infer_shape_context->GetOutputShape(2)->GetDimNum(), 1);
    EXPECT_EQ(infer_shape_context->GetOutputShape(2)->GetDim(0), 0);
}

// 静态分组 G=4：amax=[4]、y_scale=[0]。
TEST_F(QuantMaxV2InferShapeTest, quant_max_v2_infershape_static_group)
{
    std::string op_type("QuantMaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape({96, 256}, {96, 256});
    gert::StorageShape scale_shape({4}, {4});
    gert::StorageShape group_shape({4}, {4});
    gert::StorageShape y_shape({96, 256}, {96, 256});
    gert::StorageShape amax_shape({4}, {4});
    gert::StorageShape y_scale_shape({0}, {0});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1}, {1, 1, 1})
                      .InputShapes({&x_shape, &scale_shape, &group_shape})
                      .OutputShapes({&y_shape, &amax_shape, &y_scale_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(kAttrs)
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(infer_shape_context, nullptr);
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    ASSERT_NE(infer_shape_context->GetOutputShape(1), nullptr);
    EXPECT_EQ(infer_shape_context->GetOutputShape(1)->GetDim(0), 4);
    ASSERT_NE(infer_shape_context->GetOutputShape(2), nullptr);
    EXPECT_EQ(infer_shape_context->GetOutputShape(2)->GetDim(0), 0);
}

// group_list 维数不为 1：失败。
TEST_F(QuantMaxV2InferShapeTest, quant_max_v2_infershape_group_rank_invalid)
{
    std::string op_type("QuantMaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape({8, 16}, {8, 16});
    gert::StorageShape group_shape({2, 2}, {2, 2}); // rank 2，非法
    gert::StorageShape y_shape({8, 16}, {8, 16});
    gert::StorageShape amax_shape({1}, {1});
    gert::StorageShape y_scale_shape({1}, {1});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 0, 1}, {1, 1, 1})
                      .InputShapes({&x_shape, &group_shape})
                      .OutputShapes({&y_shape, &amax_shape, &y_scale_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(kAttrs)
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(infer_shape_context, nullptr);
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_FAILED);
}

// group_list 数量超上限 384：失败。
TEST_F(QuantMaxV2InferShapeTest, quant_max_v2_infershape_group_count_invalid)
{
    std::string op_type("QuantMaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape({8, 16}, {8, 16});
    gert::StorageShape group_shape({385}, {385});
    gert::StorageShape y_shape({8, 16}, {8, 16});
    gert::StorageShape amax_shape({385}, {385});
    gert::StorageShape y_scale_shape({385}, {385});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 0, 1}, {1, 1, 1})
                      .InputShapes({&x_shape, &group_shape})
                      .OutputShapes({&y_shape, &amax_shape, &y_scale_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(kAttrs)
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(infer_shape_context, nullptr);
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_FAILED);
}

// 类型推导：y 由 dst_type 决定（该注册算子未链接进本 UT 目标时按缺省跳过，与仓内同类算子一致）。
TEST_F(QuantMaxV2InferShapeTest, quant_max_v2_inferdatatype_hifloat8)
{
    std::string op_type("QuantMaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    // InferDataType 注册单元在 op_graph（本 UT 目标未链接，nm 可验）：缺席时显式跳过，避免空断言冒充通过。
    if (infer_datatype_func == nullptr) {
        GTEST_SKIP() << "infer_datatype not linked in this UT target";
    }
    ge::DataType x_dtype = ge::DT_BF16;
    ge::DataType y_dtype = ge::DT_HIFLOAT8;
    ge::DataType amax_dtype = ge::DT_BF16;
    ge::DataType y_scale_dtype = ge::DT_FLOAT;

    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 3)
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&x_dtype, nullptr, nullptr})
                              .OutputDataTypes({&y_dtype, &amax_dtype, &y_scale_dtype})
                              .NodeAttrs(kAttrs)
                              .Build();

    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
}

// 类型推导：非法 dst_type 失败。
TEST_F(QuantMaxV2InferShapeTest, quant_max_v2_inferdatatype_invalid_dst_type)
{
    std::string op_type("QuantMaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    // InferDataType 注册单元在 op_graph（本 UT 目标未链接，nm 可验）：缺席时显式跳过，避免空断言冒充通过。
    if (infer_datatype_func == nullptr) {
        GTEST_SKIP() << "infer_datatype not linked in this UT target";
    }
    ge::DataType x_dtype = ge::DT_BF16;
    ge::DataType y_dtype = ge::DT_HIFLOAT8;
    ge::DataType amax_dtype = ge::DT_BF16;
    ge::DataType y_scale_dtype = ge::DT_FLOAT;

    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 3)
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&x_dtype, nullptr, nullptr})
                              .OutputDataTypes({&y_dtype, &amax_dtype, &y_scale_dtype})
                              .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("round")},
                                          {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(99)},
                                          {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(15.0f)}})
                              .Build();

    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_FAILED);
}
