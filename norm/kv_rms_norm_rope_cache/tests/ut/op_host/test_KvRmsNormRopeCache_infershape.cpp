/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_interleave_grad_proto.h
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "common/utils/ut_op_common.h"
#include "experiment_ops.h"
#include "fusion_ops.h"

class KvRmsNormRopeCache : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvRmsNormRopeCache Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvRmsNormRopeCache Proto Test TearDown" << std::endl;
    }
};

TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_infershape)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("KvRmsNormRopeCache"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("KvRmsNormRopeCache")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape kv_shape = {{64, 1, 1, 576}, {64, 1, 1, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64}, {64}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{64, 288, 1, 64}, {64, 288, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{64, 288, 1, 512}, {64, 288, 1, 512}};
    gert::StorageShape k_rope_scale_shape = {{64}, {64}};
    gert::StorageShape ckv_scale_shape = {{512}, {512}};
    gert::StorageShape k_cache_shape_out = {{64, 288, 1, 64}, {64, 288, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{64, 288, 1, 512}, {64, 288, 1, 512}};
    gert::StorageShape k_rope_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape ckv_shape = {{64, 1, 1, 512}, {64, 1, 1, 512}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &ckv_scale_shape, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape, &ckv_shape})
                      .NodeAttrs(
                          {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", ge::AnyValue::CreateFrom<std::string>("PA_BNSD")},
                           {"is_output_kv", ge::AnyValue::CreateFrom<bool>(true)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    auto output2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    auto output3 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(3);

    ASSERT_EQ(ge::Shape2String(*output0), "[64, 288, 1, 64]");
    ASSERT_EQ(ge::Shape2String(*output1), "[64, 288, 1, 512]");
    ASSERT_EQ(ge::Shape2String(*output2), "[64, 1, 1, 64]");
    ASSERT_EQ(ge::Shape2String(*output3), "[64, 1, 1, 512]");
}

TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_infershape_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("KvRmsNormRopeCache"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("KvRmsNormRopeCache")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape kv_shape = {{-2}, {-2}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64}, {64}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{64, 288, 1, 64}, {64, 288, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{64, 288, 1, 512}, {64, 288, 1, 512}};
    gert::StorageShape k_rope_scale_shape = {{64}, {64}};
    gert::StorageShape ckv_scale_shape = {{512}, {512}};
    gert::StorageShape k_cache_shape_out = {{64, 288, 1, 64}, {64, 288, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{64, 288, 1, 512}, {64, 288, 1, 512}};
    gert::StorageShape k_rope_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape ckv_shape = {{64, 1, 1, 512}, {64, 1, 1, 512}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &ckv_scale_shape, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape, &ckv_shape})
                      .NodeAttrs(
                          {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", ge::AnyValue::CreateFrom<std::string>("PA_BNSD")},
                           {"is_output_kv", ge::AnyValue::CreateFrom<bool>(true)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    auto output2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    auto output3 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(3);

    ASSERT_EQ(ge::Shape2String(*output0), "[64, 288, 1, 64]");
    ASSERT_EQ(ge::Shape2String(*output1), "[64, 288, 1, 512]");
    ASSERT_EQ(ge::Shape2String(*output2), "[-1, -1, -1, 64]");
    ASSERT_EQ(ge::Shape2String(*output3), "[-1, -1, -1, 512]");
}

TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_infershape_error_dim)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("KvRmsNormRopeCache"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("KvRmsNormRopeCache")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape kv_shape = {{64, 1, 1, 1, 576}, {64, 1, 1, 1, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64}, {64}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{64, 288, 1, 64}, {64, 288, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{64, 288, 1, 512}, {64, 288, 1, 512}};
    gert::StorageShape k_rope_scale_shape = {{64}, {64}};
    gert::StorageShape ckv_scale_shape = {{512}, {512}};
    gert::StorageShape k_cache_shape_out = {{64, 288, 1, 64}, {64, 288, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{64, 288, 1, 512}, {64, 288, 1, 512}};
    gert::StorageShape k_rope_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape ckv_shape = {{64, 1, 1, 512}, {64, 1, 1, 512}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &ckv_scale_shape, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape, &ckv_shape})
                      .NodeAttrs(
                          {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", ge::AnyValue::CreateFrom<std::string>("PA_BNSD")},
                           {"is_output_kv", ge::AnyValue::CreateFrom<bool>(true)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_inferdtype_test1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("KvRmsNormRopeCache"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("KvRmsNormRopeCache")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType input_1 = ge::DT_INT64;
    ge::DataType input_2 = ge::DT_FLOAT;
    ge::DataType output_0 = ge::DT_FLOAT16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(11)
                              .NodeIoNum(11, 4)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs(
                                  {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                                   {"cache_mode", ge::AnyValue::CreateFrom<std::string>("PA_BNSD")},
                                   {"is_output_kv", ge::AnyValue::CreateFrom<bool>(true)}})
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes(
                                  {&input_0, &input_0, &input_0, &input_0, &input_1, &input_0, &input_0, &input_2,
                                   &input_2, &input_2, &input_2})
                              .OutputDataTypes({&output_0, &output_0, &output_0, &output_0})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT16);
}

TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_inferdtype_test2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("KvRmsNormRopeCache"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("KvRmsNormRopeCache")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_0 = ge::DT_BF16;
    ge::DataType input_1 = ge::DT_INT64;
    ge::DataType input_2 = ge::DT_FLOAT;
    ge::DataType output_0 = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(11)
                              .NodeIoNum(11, 4)
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(6, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs(
                                  {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                                   {"cache_mode", ge::AnyValue::CreateFrom<std::string>("PA_BNSD")},
                                   {"is_output_kv", ge::AnyValue::CreateFrom<bool>(true)}})
                              .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes(
                                  {&input_0, &input_0, &input_0, &input_0, &input_1, &input_0, &input_0, &input_2,
                                   &input_2, &input_2, &input_2})
                              .OutputDataTypes({&output_0, &output_0, &output_0, &output_0})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_BF16);
}

TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_inferdtype_test3)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("KvRmsNormRopeCache"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("KvRmsNormRopeCache")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType input_1 = ge::DT_INT64;
    ge::DataType input_2 = ge::DT_INT8;
    ge::DataType input_3 = ge::DT_FLOAT;
    ge::DataType output_0 = ge::DT_INT8;
    ge::DataType output_1 = ge::DT_FLOAT16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(11)
                              .NodeIoNum(11, 4)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(6, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs(
                                  {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                                   {"cache_mode", ge::AnyValue::CreateFrom<std::string>("PA_BNSD")},
                                   {"is_output_kv", ge::AnyValue::CreateFrom<bool>(true)}})
                              .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes(
                                  {&input_0, &input_0, &input_0, &input_0, &input_1, &input_2, &input_2, &input_3,
                                   &input_3, &input_3, &input_3})
                              .OutputDataTypes({&output_0, &output_0, &output_1, &output_1})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT16);
}

TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_inferdtype_test4)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("KvRmsNormRopeCache"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("KvRmsNormRopeCache")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_0 = ge::DT_BF16;
    ge::DataType input_1 = ge::DT_INT64;
    ge::DataType input_2 = ge::DT_INT8;
    ge::DataType input_3 = ge::DT_FLOAT;
    ge::DataType output_0 = ge::DT_INT8;
    ge::DataType output_1 = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(11)
                              .NodeIoNum(11, 4)
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(6, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs(
                                  {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                                   {"cache_mode", ge::AnyValue::CreateFrom<std::string>("PA_BNSD")},
                                   {"is_output_kv", ge::AnyValue::CreateFrom<bool>(true)}})
                              .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes(
                                  {&input_0, &input_0, &input_0, &input_0, &input_1, &input_2, &input_2, &input_3,
                                   &input_3, &input_3, &input_3})
                              .OutputDataTypes({&output_0, &output_0, &output_1, &output_1})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_BF16);
}