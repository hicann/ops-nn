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
 * \file test_repeat_interleave_grad_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_graph/repeat_interleave_grad_proto.h"
#include "infershape_test_util.h"
#include "infershape_case_executor.h"
#include "ut_op_common.h"

class RepeatInterleaveGrad : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RepeatInterleaveGrad SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RepeatInterleaveGrad TearDown" << std::endl; }
};

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_fp16_int32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{64}, {}};
    gert::StorageShape wShape = {{1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_fp16_int64)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{64}, {}};
    gert::StorageShape wShape = {{1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_fp32_int32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 16}, {2, 32, 16}};
    gert::StorageShape wShape = {{16}, {16}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_fp32_int64)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 16}, {2, 32, 16}};
    gert::StorageShape wShape = {{16}, {16}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_scalar_repeat_int32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 64, 16}, {2, 64, 16}};
    gert::StorageShape wShape = {{1}, {1}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_scalar_repeat_int64)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 64, 16}, {2, 64, 16}};
    gert::StorageShape wShape = {{1}, {1}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_negative_axis)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 64, 16}, {2, 64, 16}};
    gert::StorageShape wShape = {{1}, {1}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-2)}})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_axis0)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{4, 16, 8}, {4, 16, 8}};
    gert::StorageShape wShape = {{4}, {4}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_axis_last)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 4, 32}, {2, 4, 32}};
    gert::StorageShape wShape = {{1}, {1}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_4d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 4, 8, 16}, {2, 4, 8, 16}};
    gert::StorageShape wShape = {{8}, {8}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_bf16_int32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 16}, {2, 32, 16}};
    gert::StorageShape wShape = {{16}, {16}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_bf16_int64)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 16}, {2, 32, 16}};
    gert::StorageShape wShape = {{16}, {16}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_1d_input)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{32}, {32}};
    gert::StorageShape wShape = {{1}, {1}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test_large_batch)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{8, 64, 128}, {8, 64, 128}};
    gert::StorageShape wShape = {{64}, {64}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_inferdatatype_test_fp16)
{
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_datatype;

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferDataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_inferdatatype_test_fp32)
{
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_datatype;

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferDataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_inferdatatype_test_bf16)
{
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_datatype;

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferDataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
}

// const repeats (int32, scalar), 覆盖 GetOutputZeroAxisLen + DT_INT32 分支
TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_const_int32_scalar)
{
    int32_t repeatsData = {2};
    gert::InfershapeContextPara para("RepeatInterleaveGrad",
                                     {
                                         {{{2, 64, 16}, {2, 64, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &repeatsData},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 32, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// const repeats (int64, multi-dim), 覆盖 GetOutputZeroAxisLen + DT_INT64 分支
TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_const_int64_multi)
{
    int64_t repeatsData[16] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    gert::InfershapeContextPara para("RepeatInterleaveGrad",
                                     {
                                         {{{2, 32, 16}, {2, 32, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND, true, repeatsData},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 16, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 未知秩输入 {-2}
TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_unknown_rank)
{
    gert::InfershapeContextPara para("RepeatInterleaveGrad",
                                     {
                                         {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// axis 超出范围 → GRAPH_FAILED
TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_axis_out_of_range)
{
    int32_t repeatsData[16] = {2};
    gert::InfershapeContextPara para("RepeatInterleaveGrad",
                                     {
                                         {{{2, 32, 16}, {2, 32, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND, true, repeatsData},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(5)}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// repeats dtype 无效 → GRAPH_FAILED
TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_invalid_repeats_dtype)
{
    float repeatsData[16] = {0};
    gert::InfershapeContextPara para("RepeatInterleaveGrad",
                                     {
                                         {{{2, 32, 16}, {2, 32, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND, true, repeatsData},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
