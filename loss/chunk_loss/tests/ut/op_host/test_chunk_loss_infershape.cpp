/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "ut_op_common.h"
#include "ut_op_util.h"

using ut_util::ToVector;

class ChunkLossInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ChunkLossInferShape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ChunkLossInferShape TearDown" << std::endl; }
};

TEST_F(ChunkLossInferShape, chunk_loss_infershape_fp16_normal)
{
    gert::StorageShape hiddenShape = {{4, 8}, {4, 8}};
    gert::StorageShape weightShape = {{16, 8}, {16, 8}};
    gert::StorageShape labelsShape = {{4}, {4}};
    gert::StorageShape lossShape = {{}, {}};
    gert::StorageShape maxShape = {{1}, {1}};
    gert::StorageShape sumShape = {{1}, {1}};
    gert::StorageShape dHShape = {{1, 1}, {1, 1}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ChunkLoss")
                      .NodeIoNum(3, 4)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&hiddenShape, &weightShape, &labelsShape})
                      .OutputShapes({&lossShape, &maxShape, &sumShape, &dHShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(4.0f)},
                                  {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ChunkLoss")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(ToVector(*context->GetOutputShape(0)), (std::vector<int64_t>{}));
    EXPECT_EQ(ToVector(*context->GetOutputShape(1)), (std::vector<int64_t>{4}));
    EXPECT_EQ(ToVector(*context->GetOutputShape(2)), (std::vector<int64_t>{4}));
    EXPECT_EQ(ToVector(*context->GetOutputShape(3)), (std::vector<int64_t>{4, 8}));
}

TEST_F(ChunkLossInferShape, chunk_loss_infershape_bf16_normal)
{
    gert::StorageShape hiddenShape = {{128, 256}, {128, 256}};
    gert::StorageShape weightShape = {{512, 256}, {512, 256}};
    gert::StorageShape labelsShape = {{128}, {128}};
    gert::StorageShape lossShape = {{}, {}};
    gert::StorageShape maxShape = {{1}, {1}};
    gert::StorageShape sumShape = {{1}, {1}};
    gert::StorageShape dHShape = {{1, 1}, {1, 1}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ChunkLoss")
                      .NodeIoNum(3, 4)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&hiddenShape, &weightShape, &labelsShape})
                      .OutputShapes({&lossShape, &maxShape, &sumShape, &dHShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ChunkLoss")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(ToVector(*context->GetOutputShape(0)), (std::vector<int64_t>{}));
    EXPECT_EQ(ToVector(*context->GetOutputShape(1)), (std::vector<int64_t>{128}));
    EXPECT_EQ(ToVector(*context->GetOutputShape(2)), (std::vector<int64_t>{128}));
    EXPECT_EQ(ToVector(*context->GetOutputShape(3)), (std::vector<int64_t>{128, 256}));
}

TEST_F(ChunkLossInferShape, chunk_loss_infershape_invalid_rank)
{
    gert::StorageShape hiddenShape = {{4, 8, 1}, {4, 8, 1}};
    gert::StorageShape weightShape = {{16, 8}, {16, 8}};
    gert::StorageShape labelsShape = {{4}, {4}};
    gert::StorageShape lossShape = {{}, {}};
    gert::StorageShape maxShape = {{1}, {1}};
    gert::StorageShape sumShape = {{1}, {1}};
    gert::StorageShape dHShape = {{1, 1}, {1, 1}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ChunkLoss")
                      .NodeIoNum(3, 4)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&hiddenShape, &weightShape, &labelsShape})
                      .OutputShapes({&lossShape, &maxShape, &sumShape, &dHShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(4.0f)},
                                  {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ChunkLoss")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

TEST_F(ChunkLossInferShape, chunk_loss_infershape_m_mismatch)
{
    gert::StorageShape hiddenShape = {{4, 8}, {4, 8}};
    gert::StorageShape weightShape = {{16, 8}, {16, 8}};
    gert::StorageShape labelsShape = {{3}, {3}};
    gert::StorageShape lossShape = {{}, {}};
    gert::StorageShape maxShape = {{1}, {1}};
    gert::StorageShape sumShape = {{1}, {1}};
    gert::StorageShape dHShape = {{1, 1}, {1, 1}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ChunkLoss")
                      .NodeIoNum(3, 4)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&hiddenShape, &weightShape, &labelsShape})
                      .OutputShapes({&lossShape, &maxShape, &sumShape, &dHShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(4.0f)},
                                  {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ChunkLoss")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

TEST_F(ChunkLossInferShape, chunk_loss_infershape_h_mismatch)
{
    gert::StorageShape hiddenShape = {{4, 8}, {4, 8}};
    gert::StorageShape weightShape = {{16, 7}, {16, 7}};
    gert::StorageShape labelsShape = {{4}, {4}};
    gert::StorageShape lossShape = {{}, {}};
    gert::StorageShape maxShape = {{1}, {1}};
    gert::StorageShape sumShape = {{1}, {1}};
    gert::StorageShape dHShape = {{1, 1}, {1, 1}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ChunkLoss")
                      .NodeIoNum(3, 4)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&hiddenShape, &weightShape, &labelsShape})
                      .OutputShapes({&lossShape, &maxShape, &sumShape, &dHShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(4.0f)},
                                  {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ChunkLoss")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

TEST_F(ChunkLossInferShape, chunk_loss_infer_dtype_fp16)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ChunkLoss"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ChunkLoss")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType hiddenRef = ge::DT_FLOAT16;
    ge::DataType weightRef = ge::DT_FLOAT16;
    ge::DataType labelsRef = ge::DT_INT64;
    ge::DataType lossRef = ge::DT_FLOAT;
    ge::DataType maxRef = ge::DT_FLOAT;
    ge::DataType sumRef = ge::DT_FLOAT;
    ge::DataType dHRef = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 4)
                             .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&hiddenRef, &weightRef, &labelsRef})
                             .OutputDataTypes({&lossRef, &maxRef, &sumRef, &dHRef})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT16);
}

TEST_F(ChunkLossInferShape, chunk_loss_infer_dtype_bf16)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ChunkLoss"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ChunkLoss")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType hiddenRef = ge::DT_BF16;
    ge::DataType weightRef = ge::DT_BF16;
    ge::DataType labelsRef = ge::DT_INT64;
    ge::DataType lossRef = ge::DT_FLOAT;
    ge::DataType maxRef = ge::DT_FLOAT;
    ge::DataType sumRef = ge::DT_FLOAT;
    ge::DataType dHRef = ge::DT_BF16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 4)
                             .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&hiddenRef, &weightRef, &labelsRef})
                             .OutputDataTypes({&lossRef, &maxRef, &sumRef, &dHRef})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_BF16);
}
