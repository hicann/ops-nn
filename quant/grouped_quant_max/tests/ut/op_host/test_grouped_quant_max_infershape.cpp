/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_grouped_quant_max_infershape.cpp
 * @brief Unit test for GroupedQuantMax infer shape and datatype
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "ut_op_util.h"
#include "ut_op_common.h"

using namespace std;
using namespace ge;

class GroupedQuantMaxInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedQuantMaxInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedQuantMaxInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(GroupedQuantMaxInferShapeTest, grouped_quant_max_infershape_basic)
{
    // Basic infer shape: y = x shape, amax = [num_groups]
    std::string op_type("GroupedQuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape({6, 128}, {6, 128});
    gert::StorageShape scale_shape({3}, {3});
    gert::StorageShape group_list_shape({3}, {3});
    gert::StorageShape y_shape({6, 128}, {6, 128});
    gert::StorageShape amax_shape({3}, {3});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape, &group_list_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    const gert::Shape* output_y_shape = infer_shape_context->GetOutputShape(0);
    ASSERT_NE(output_y_shape, nullptr);
    EXPECT_EQ(output_y_shape->GetDimNum(), 2);
    EXPECT_EQ(output_y_shape->GetDim(0), 6);
    EXPECT_EQ(output_y_shape->GetDim(1), 128);

    const gert::Shape* output_amax_shape = infer_shape_context->GetOutputShape(1);
    ASSERT_NE(output_amax_shape, nullptr);
    EXPECT_EQ(output_amax_shape->GetDimNum(), 1);
    EXPECT_EQ(output_amax_shape->GetDim(0), 3);
}

TEST_F(GroupedQuantMaxInferShapeTest, grouped_quant_max_infershape_single_group)
{
    // Single group case: num_groups = 1
    std::string op_type("GroupedQuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape({1024, 512}, {1024, 512});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape group_list_shape({1}, {1});
    gert::StorageShape y_shape({1024, 512}, {1024, 512});
    gert::StorageShape amax_shape({1}, {1});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape, &group_list_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    const gert::Shape* output_y_shape = infer_shape_context->GetOutputShape(0);
    ASSERT_NE(output_y_shape, nullptr);
    EXPECT_EQ(output_y_shape->GetDimNum(), 2);
    EXPECT_EQ(output_y_shape->GetDim(0), 1024);
    EXPECT_EQ(output_y_shape->GetDim(1), 512);

    const gert::Shape* output_amax_shape = infer_shape_context->GetOutputShape(1);
    ASSERT_NE(output_amax_shape, nullptr);
    EXPECT_EQ(output_amax_shape->GetDimNum(), 1);
    EXPECT_EQ(output_amax_shape->GetDim(0), 1);
}

TEST_F(GroupedQuantMaxInferShapeTest, grouped_quant_max_infershape_multi_dim)
{
    // Multi-dimension input (4D)
    std::string op_type("GroupedQuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape({2, 4, 8, 16}, {2, 4, 8, 16});
    gert::StorageShape scale_shape({4}, {4});
    gert::StorageShape group_list_shape({4}, {4});
    gert::StorageShape y_shape({2, 4, 8, 16}, {2, 4, 8, 16});
    gert::StorageShape amax_shape({4}, {4});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape, &group_list_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    const gert::Shape* output_y_shape = infer_shape_context->GetOutputShape(0);
    ASSERT_NE(output_y_shape, nullptr);
    EXPECT_EQ(output_y_shape->GetDimNum(), 4);
    EXPECT_EQ(output_y_shape->GetDim(0), 2);
    EXPECT_EQ(output_y_shape->GetDim(1), 4);
    EXPECT_EQ(output_y_shape->GetDim(2), 8);
    EXPECT_EQ(output_y_shape->GetDim(3), 16);

    const gert::Shape* output_amax_shape = infer_shape_context->GetOutputShape(1);
    ASSERT_NE(output_amax_shape, nullptr);
    EXPECT_EQ(output_amax_shape->GetDimNum(), 1);
    EXPECT_EQ(output_amax_shape->GetDim(0), 4);
}

TEST_F(GroupedQuantMaxInferShapeTest, grouped_quant_max_infershape_8dim)
{
    // 8-dimension input (max supported rank)
    std::string op_type("GroupedQuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape({1, 2, 2, 2, 2, 2, 2, 2}, {1, 2, 2, 2, 2, 2, 2, 2});
    gert::StorageShape scale_shape({2}, {2});
    gert::StorageShape group_list_shape({2}, {2});
    gert::StorageShape y_shape({1, 2, 2, 2, 2, 2, 2, 2}, {1, 2, 2, 2, 2, 2, 2, 2});
    gert::StorageShape amax_shape({2}, {2});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape, &group_list_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    const gert::Shape* output_y_shape = infer_shape_context->GetOutputShape(0);
    ASSERT_NE(output_y_shape, nullptr);
    EXPECT_EQ(output_y_shape->GetDimNum(), 8);
}

TEST_F(GroupedQuantMaxInferShapeTest, grouped_quant_max_inferdatatype_fp32_fp8_e5m2)
{
    // FP32 input -> FP8_E5M2 output (dstType=35)
    std::string op_type("GroupedQuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_FLOAT;
        ge::DataType scale_dtype = ge::DT_FLOAT;
        ge::DataType group_list_dtype = ge::DT_INT64;
        ge::DataType y_dtype = ge::DT_FLOAT8_E5M2;
        ge::DataType amax_dtype = ge::DT_FLOAT;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype, &group_list_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    }
}

TEST_F(GroupedQuantMaxInferShapeTest, grouped_quant_max_inferdatatype_fp16_hifloat8)
{
    // FP16 input -> HIFLOAT8 output (dstType=34)
    std::string op_type("GroupedQuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_FLOAT16;
        ge::DataType scale_dtype = ge::DT_FLOAT;
        ge::DataType group_list_dtype = ge::DT_INT64;
        ge::DataType y_dtype = ge::DT_HIFLOAT8;
        ge::DataType amax_dtype = ge::DT_FLOAT16;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype, &group_list_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("round")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)}
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    }
}

TEST_F(GroupedQuantMaxInferShapeTest, grouped_quant_max_inferdatatype_bf16_fp8_e4m3fn)
{
    // BF16 input -> FP8_E4M3FN output (dstType=36)
    std::string op_type("GroupedQuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_BF16;
        ge::DataType scale_dtype = ge::DT_FLOAT;
        ge::DataType group_list_dtype = ge::DT_INT64;
        ge::DataType y_dtype = ge::DT_FLOAT8_E4M3FN;
        ge::DataType amax_dtype = ge::DT_BF16;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 2)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype, &group_list_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(36)}
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    }
}

TEST_F(GroupedQuantMaxInferShapeTest, grouped_quant_max_inferdatatype_invalid_x_dtype)
{
    // x dtype not in FLOAT/FLOAT16/BF16
    std::string op_type("GroupedQuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_INT32;
        ge::DataType scale_dtype = ge::DT_FLOAT;
        ge::DataType group_list_dtype = ge::DT_INT64;
        ge::DataType y_dtype = ge::DT_FLOAT8_E5M2;
        ge::DataType amax_dtype = ge::DT_INT32;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 2)
                                  .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype, &group_list_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(GroupedQuantMaxInferShapeTest, grouped_quant_max_inferdatatype_invalid_scale_dtype)
{
    // scale dtype != FLOAT
    std::string op_type("GroupedQuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_FLOAT;
        ge::DataType scale_dtype = ge::DT_FLOAT16;
        ge::DataType group_list_dtype = ge::DT_INT64;
        ge::DataType y_dtype = ge::DT_FLOAT8_E5M2;
        ge::DataType amax_dtype = ge::DT_FLOAT;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype, &group_list_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(GroupedQuantMaxInferShapeTest, grouped_quant_max_inferdatatype_invalid_grouplist_dtype)
{
    // groupList dtype != INT64
    std::string op_type("GroupedQuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_FLOAT;
        ge::DataType scale_dtype = ge::DT_FLOAT;
        ge::DataType group_list_dtype = ge::DT_INT32;
        ge::DataType y_dtype = ge::DT_FLOAT8_E5M2;
        ge::DataType amax_dtype = ge::DT_FLOAT;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype, &group_list_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_FAILED);
    }
}
