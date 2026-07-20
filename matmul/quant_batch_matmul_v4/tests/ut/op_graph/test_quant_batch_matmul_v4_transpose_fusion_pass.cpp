/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <vector>
#include "ge/es_graph_builder.h"
#include "ge/ge_utils.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "../../../op_graph/fusion_pass/quant_batch_matmul_v4_transpose_fusion_pass.h"
#include "register/register_custom_pass.h"
#include "ut_op_util.h"

using namespace ut_util;
using namespace ge;
using namespace ops;

namespace {
constexpr int64_t DIM_128 = 128;
constexpr int64_t DIM_256 = 256;
constexpr int64_t DIM_64 = 64;
constexpr int64_t DIM_32 = 32;
constexpr int64_t DIM_5 = 5;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_MINUS_1 = -1;
constexpr int32_t BLOCK_SIZE = 32;

struct TransposeFusionResult {
    bool hasTranspose;
    bool hasQuantBmmv4;
    bool transposeX2Attr;
    std::vector<std::string> extraNodeTypes;
};

TransposeFusionResult CheckFusionResult(std::shared_ptr<Graph> graph)
{
    TransposeFusionResult result{false, false, false, {}};
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        std::string typeStr(type.GetString());
        if (typeStr == "Transpose" || typeStr == "TransposeD") {
            result.hasTranspose = true;
        } else if (typeStr == "QuantBatchMatmulV4") {
            result.hasQuantBmmv4 = true;
            node.GetAttr("transpose_x2", result.transposeX2Attr);
        } else if (typeStr != "Const" && typeStr != "Data") {
            result.extraNodeTypes.push_back(typeStr);
        }
    }
    return result;
}

es::EsTensorHolder CreateTranspose(es::EsGraphBuilder& builder, const es::EsTensorHolder& input)
{
    std::vector<int64_t> permShape{2};
    auto permTensor = builder.CreateConst(std::vector<int32_t>({1, 0}), permShape, DT_INT32, FORMAT_ND);
    return es::Transpose(input, permTensor);
}

static TensorDesc MakeTensorDesc(DataType dtype, const std::vector<int64_t>& dims)
{
    Shape shape(dims);
    TensorDesc desc;
    desc.SetDataType(dtype);
    desc.SetFormat(FORMAT_ND);
    desc.SetOriginFormat(FORMAT_ND);
    desc.SetShape(shape);
    desc.SetOriginShape(shape);
    return desc;
}

static void SetNodeOutputDesc(const es::EsTensorHolder& tensor, DataType dtype, const std::vector<int64_t>& dims)
{
    tensor.GetProducer()->UpdateOutputDesc(0, MakeTensorDesc(dtype, dims));
}

static void SetupShapesForTransposeFusion(std::shared_ptr<Graph>& graph,
                                          const std::vector<std::pair<DataType, std::vector<int64_t>>>& inputShapes)
{
    for (auto& node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        std::string typeStr(type.GetString());
        if (typeStr == "Data") {
            int64_t index = -1;
            node.GetAttr("index", index);
            if (index >= 0 && static_cast<size_t>(index) < inputShapes.size()) {
                auto desc = MakeTensorDesc(inputShapes[index].first, inputShapes[index].second);
                node.UpdateOutputDesc(0, desc);
                node.UpdateInputDesc(0, desc);
            }
        }
    }
    for (auto& node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        std::string typeStr(type.GetString());
        if (typeStr == "Transpose" || typeStr == "TransposeD") {
            auto peer = node.GetInDataNodesAndPortIndexs(0);
            if (peer.first) {
                TensorDesc inputDesc;
                peer.first->GetOutputDesc(peer.second, inputDesc);
                node.UpdateInputDesc(0, inputDesc);
                auto dims = inputDesc.GetShape().GetDims();
                if (dims.size() == 2) {
                    std::swap(dims[0], dims[1]);
                }
                node.UpdateOutputDesc(0, MakeTensorDesc(inputDesc.GetDataType(), dims));
            }
        } else if (typeStr == "Reshape") {
            auto peer = node.GetInDataNodesAndPortIndexs(0);
            if (peer.first) {
                TensorDesc inputDesc;
                peer.first->GetOutputDesc(peer.second, inputDesc);
                node.UpdateInputDesc(0, inputDesc);
                TensorDesc reshapeInputDesc;
                if (peer.first->GetInputDesc(0, reshapeInputDesc) == ge::GRAPH_SUCCESS) {
                    node.UpdateInputDesc(0, reshapeInputDesc);
                }
                node.UpdateOutputDesc(0, inputDesc);
            }
            auto shapePeer = node.GetInDataNodesAndPortIndexs(1);
            if (shapePeer.first) {
                TensorDesc shapeDesc;
                shapePeer.first->GetOutputDesc(shapePeer.second, shapeDesc);
                node.UpdateInputDesc(1, shapeDesc);
            }
        } else if (typeStr == "QuantBatchMatmulV4") {
            for (size_t i = 0; i < 10; ++i) {
                auto peer = node.GetInDataNodesAndPortIndexs(i);
                if (peer.first) {
                    TensorDesc peerOutDesc;
                    peer.first->GetOutputDesc(peer.second, peerOutDesc);
                    node.UpdateInputDesc(i, peerOutDesc);
                }
            }
        }
    }
}
} // namespace

class QuantBatchMatmulV4TransposeFusionPassTest : public testing::Test {
protected:
    class TestablePass : public ops::QuantBatchMatmulV4TransposeFusionPass {};
};

TEST_F(QuantBatchMatmulV4TransposeFusionPassTest, BasicTransposeFusion)
{
    auto graphBuilder = es::EsGraphBuilder("basic_transpose_fusion");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT8_E4M3FN, FORMAT_ND, {DIM_128, DIM_256});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT4_E2M1, FORMAT_ND, {DIM_64, DIM_256});
    auto x2Scale = graphBuilder.CreateInput(2, "x2_scale", DT_FLOAT, FORMAT_ND, {DIM_64, DIM_1});
    auto yScale = graphBuilder.CreateInput(3, "y_scale", DT_FLOAT, FORMAT_ND, {DIM_64});

    auto transposeX2 = CreateTranspose(graphBuilder, x2);
    auto transposeX2Scale = CreateTranspose(graphBuilder, x2Scale);
    auto quantBmmv4 = es::QuantBatchMatmulV4(x1, transposeX2, nullptr, nullptr, transposeX2Scale, yScale, nullptr,
                                             nullptr, nullptr, nullptr, DT_BF16, -1, false, false);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({quantBmmv4});
    ASSERT_NE(graph, nullptr);

    SetupShapesForTransposeFusion(graph, {{DT_FLOAT8_E4M3FN, {DIM_128, DIM_256}},
                                          {DT_FLOAT4_E2M1, {DIM_64, DIM_256}},
                                          {DT_FLOAT, {DIM_64, DIM_1}},
                                          {DT_FLOAT, {DIM_64}}});

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasQuantBmmv4);
}

TEST_F(QuantBatchMatmulV4TransposeFusionPassTest, TransposeFusionWithShapeValidation)
{
    auto graphBuilder = es::EsGraphBuilder("transpose_fusion_with_shape_check");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT8_E4M3FN, FORMAT_ND, {DIM_128, DIM_256});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT4_E2M1, FORMAT_ND, {DIM_64, DIM_256});
    auto x2Scale = graphBuilder.CreateInput(2, "x2_scale", DT_FLOAT, FORMAT_ND, {DIM_64});
    auto yScale = graphBuilder.CreateInput(3, "y_scale", DT_FLOAT, FORMAT_ND, {DIM_64});

    auto transposeX2 = CreateTranspose(graphBuilder, x2);
    std::vector<int64_t> permShape{1};
    auto permTensor = graphBuilder.CreateConst(std::vector<int32_t>({0}), permShape, DT_INT32, FORMAT_ND);
    auto transposeX2Scale = es::Transpose(x2Scale, permTensor);
    auto quantBmmv4 = es::QuantBatchMatmulV4(x1, transposeX2, nullptr, nullptr, transposeX2Scale, yScale, nullptr,
                                             nullptr, nullptr, nullptr, DT_BF16, -1, false, false, BLOCK_SIZE);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({quantBmmv4});
    ASSERT_NE(graph, nullptr);

    SetupShapesForTransposeFusion(graph, {{DT_FLOAT8_E4M3FN, {DIM_128, DIM_256}},
                                          {DT_FLOAT4_E2M1, {DIM_64, DIM_256}},
                                          {DT_FLOAT, {DIM_64}},
                                          {DT_FLOAT, {DIM_64}}});

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    bool found = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == "QuantBatchMatmulV4") {
            found = true;
            TensorDesc x2InputDesc;
            EXPECT_EQ(node.GetInputDesc(1, x2InputDesc), GRAPH_SUCCESS);
            Shape x2InputShape = x2InputDesc.GetShape();
            EXPECT_EQ(x2InputShape.GetDim(0), DIM_64);
            EXPECT_EQ(x2InputShape.GetDim(1), DIM_256);

            TensorDesc x2ScaleInputDesc;
            EXPECT_EQ(node.GetInputDesc(4, x2ScaleInputDesc), GRAPH_SUCCESS);
            Shape x2ScaleInputShape = x2ScaleInputDesc.GetShape();
            EXPECT_EQ(x2ScaleInputShape.GetDim(0), DIM_64);

            bool transposeX2Attr = false;
            EXPECT_EQ(node.GetAttr("transpose_x2", transposeX2Attr), GRAPH_SUCCESS);
            EXPECT_TRUE(transposeX2Attr);
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(QuantBatchMatmulV4TransposeFusionPassTest, DynamicShapeTransposeFusion)
{
    auto graphBuilder = es::EsGraphBuilder("dynamic_shape_transpose_fusion");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT8_E4M3FN, FORMAT_ND, {DIM_MINUS_1, DIM_MINUS_1});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT4_E2M1, FORMAT_ND, {DIM_MINUS_1, DIM_MINUS_1});
    auto x2Scale = graphBuilder.CreateInput(2, "x2_scale", DT_FLOAT16, FORMAT_ND, {DIM_MINUS_1});
    auto yScale = graphBuilder.CreateInput(3, "y_scale", DT_UINT64, FORMAT_ND, {DIM_MINUS_1});

    auto transposeX2 = CreateTranspose(graphBuilder, x2);
    auto transposeX2Scale = CreateTranspose(graphBuilder, x2Scale);
    auto quantBmmv4 = es::QuantBatchMatmulV4(x1, transposeX2, nullptr, nullptr, transposeX2Scale, yScale, nullptr,
                                             nullptr, nullptr, nullptr, DT_BF16, -1, false, false, BLOCK_SIZE);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({quantBmmv4});
    ASSERT_NE(graph, nullptr);

    SetupShapesForTransposeFusion(graph, {{DT_FLOAT8_E4M3FN, {DIM_MINUS_1, DIM_MINUS_1}},
                                          {DT_FLOAT4_E2M1, {DIM_MINUS_1, DIM_MINUS_1}},
                                          {DT_FLOAT16, {DIM_MINUS_1}},
                                          {DT_UINT64, {DIM_MINUS_1}}});

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasQuantBmmv4);
}

TEST_F(QuantBatchMatmulV4TransposeFusionPassTest, TransposeWithReshapeFusion)
{
    auto graphBuilder = es::EsGraphBuilder("transpose_with_reshape_fusion");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT8_E4M3FN, FORMAT_ND, {DIM_3, DIM_32});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT4_E2M1, FORMAT_ND, {DIM_5, DIM_32});
    auto x2Scale = graphBuilder.CreateInput(2, "x2_scale", DT_FLOAT16, FORMAT_ND, {DIM_5, DIM_1});
    auto yScale = graphBuilder.CreateInput(3, "y_scale", DT_UINT64, FORMAT_ND, {DIM_1, DIM_5});

    auto transposeX2 = CreateTranspose(graphBuilder, x2);
    std::vector<int64_t> permShape{2};
    auto reshapeTensor = graphBuilder.CreateConst(std::vector<int32_t>({1, 5}), permShape, DT_INT32, FORMAT_ND);
    auto reshapeX2Scale = es::Reshape(x2Scale, reshapeTensor);
    auto quantBmmv4 = es::QuantBatchMatmulV4(x1, transposeX2, nullptr, nullptr, reshapeX2Scale, yScale, nullptr,
                                             nullptr, nullptr, nullptr, DT_BF16, -1, false, false, BLOCK_SIZE);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({quantBmmv4});
    ASSERT_NE(graph, nullptr);

    SetupShapesForTransposeFusion(graph, {{DT_FLOAT8_E4M3FN, {DIM_3, DIM_32}},
                                          {DT_FLOAT4_E2M1, {DIM_5, DIM_32}},
                                          {DT_FLOAT16, {DIM_5, DIM_1}},
                                          {DT_UINT64, {DIM_1, DIM_5}}});

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasQuantBmmv4);
}

TEST_F(QuantBatchMatmulV4TransposeFusionPassTest, DynamicShapeWithPackReshapeFusion)
{
    auto graphBuilder = es::EsGraphBuilder("dynamic_shape_with_pack_reshape_fusion");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT8_E4M3FN, FORMAT_ND, {DIM_MINUS_1, DIM_MINUS_1});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT4_E2M1, FORMAT_ND, {DIM_MINUS_1, DIM_MINUS_1});
    auto x2Scale = graphBuilder.CreateInput(2, "x2_scale", DT_FLOAT16, FORMAT_ND, {DIM_1, DIM_MINUS_1});
    auto yScale = graphBuilder.CreateInput(3, "y_scale", DT_UINT64, FORMAT_ND, {DIM_MINUS_1});

    auto transposeX2 = CreateTranspose(graphBuilder, x2);
    auto shapeX2Scale = es::Shape(x2Scale);
    std::vector<int64_t> indicesShape{1};
    auto indicesTensor = graphBuilder.CreateConst(std::vector<int32_t>({1}), indicesShape, DT_INT32, FORMAT_ND);
    auto gatherX2Scale = es::Gather(shapeX2Scale, indicesTensor, 0);
    auto packX2Scale = es::Pack(std::vector<es::EsTensorHolder>({gatherX2Scale}));
    auto reshapeX2Scale = es::Reshape(x2Scale, packX2Scale);
    auto quantBmmv4 = es::QuantBatchMatmulV4(x1, transposeX2, nullptr, nullptr, reshapeX2Scale, yScale, nullptr,
                                             nullptr, nullptr, nullptr, DT_BF16, -1, false, false, BLOCK_SIZE);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({quantBmmv4});
    ASSERT_NE(graph, nullptr);

    SetupShapesForTransposeFusion(graph, {{DT_FLOAT8_E4M3FN, {DIM_MINUS_1, DIM_MINUS_1}},
                                          {DT_FLOAT4_E2M1, {DIM_MINUS_1, DIM_MINUS_1}},
                                          {DT_FLOAT16, {DIM_1, DIM_MINUS_1}},
                                          {DT_UINT64, {DIM_MINUS_1}}});

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    bool findTranspose = false;
    bool findReshape = false;
    bool findPack = false;
    bool findGather = false;
    bool findShape = false;
    bool findQuantBmmv4 = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        std::string typeStr(type.GetString());
        if (typeStr == "Transpose" || typeStr == "TransposeD") {
            findTranspose = true;
        } else if (typeStr == "Reshape") {
            findReshape = true;
        } else if (typeStr == "Pack") {
            findPack = true;
        } else if (typeStr == "Gather") {
            findGather = true;
        } else if (typeStr == "Shape") {
            findShape = true;
        } else if (typeStr == "QuantBatchMatmulV4") {
            findQuantBmmv4 = true;
        }
    }
    EXPECT_FALSE(findTranspose);
    EXPECT_FALSE(findReshape);
    EXPECT_FALSE(findPack);
    EXPECT_FALSE(findGather);
    EXPECT_FALSE(findShape);
    EXPECT_TRUE(findQuantBmmv4);
}

TEST_F(QuantBatchMatmulV4TransposeFusionPassTest, TransposeFusionWithBiasAndYScale)
{
    auto graphBuilder = es::EsGraphBuilder("transpose_fusion_with_bias_and_y_scale");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT8_E4M3FN, FORMAT_ND, {DIM_128, DIM_256});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT4_E2M1, FORMAT_ND, {DIM_64, DIM_256});
    auto bias = graphBuilder.CreateInput(2, "bias", DT_FLOAT, FORMAT_ND, {DIM_64});
    auto x2Scale = graphBuilder.CreateInput(3, "x2_scale", DT_FLOAT, FORMAT_ND, {DIM_64, DIM_1});
    auto yScale = graphBuilder.CreateInput(4, "y_scale", DT_FLOAT, FORMAT_ND, {DIM_64});

    auto transposeX2 = CreateTranspose(graphBuilder, x2);
    auto transposeX2Scale = CreateTranspose(graphBuilder, x2Scale);
    auto quantBmmv4 = es::QuantBatchMatmulV4(x1, transposeX2, bias, es::EsTensorHolder(), transposeX2Scale, yScale,
                                             es::EsTensorHolder(), es::EsTensorHolder(), es::EsTensorHolder(),
                                             es::EsTensorHolder(), DT_BF16, -1, false, false, BLOCK_SIZE);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({quantBmmv4});
    ASSERT_NE(graph, nullptr);

    SetupShapesForTransposeFusion(graph, {{DT_FLOAT8_E4M3FN, {DIM_128, DIM_256}},
                                          {DT_FLOAT4_E2M1, {DIM_64, DIM_256}},
                                          {DT_FLOAT, {DIM_64}},
                                          {DT_FLOAT, {DIM_64, DIM_1}},
                                          {DT_FLOAT, {DIM_64}}});

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasQuantBmmv4);
}

TEST_F(QuantBatchMatmulV4TransposeFusionPassTest, TransposeFusionWithX1Scale)
{
    auto graphBuilder = es::EsGraphBuilder("transpose_fusion_with_x1_scale");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT8_E4M3FN, FORMAT_ND, {DIM_128, DIM_256});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT4_E2M1, FORMAT_ND, {DIM_64, DIM_256});
    auto x1Scale = graphBuilder.CreateInput(2, "x1_scale", DT_FLOAT, FORMAT_ND, {DIM_128});
    auto x2Scale = graphBuilder.CreateInput(3, "x2_scale", DT_FLOAT, FORMAT_ND, {DIM_64, DIM_1});

    auto transposeX2 = CreateTranspose(graphBuilder, x2);
    auto transposeX2Scale = CreateTranspose(graphBuilder, x2Scale);
    auto quantBmmv4 = es::QuantBatchMatmulV4(
        x1, transposeX2, es::EsTensorHolder(), x1Scale, transposeX2Scale, es::EsTensorHolder(), es::EsTensorHolder(),
        es::EsTensorHolder(), es::EsTensorHolder(), es::EsTensorHolder(), DT_BF16, -1, false, false, BLOCK_SIZE);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({quantBmmv4});
    ASSERT_NE(graph, nullptr);

    SetupShapesForTransposeFusion(graph, {{DT_FLOAT8_E4M3FN, {DIM_128, DIM_256}},
                                          {DT_FLOAT4_E2M1, {DIM_64, DIM_256}},
                                          {DT_FLOAT, {DIM_128}},
                                          {DT_FLOAT, {DIM_64, DIM_1}}});

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    bool found = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == "QuantBatchMatmulV4") {
            found = true;
            auto x1ScaleNode = node.GetInDataNodesAndPortIndexs(3).first;
            EXPECT_NE(x1ScaleNode, nullptr);
            bool transposeX2Attr = false;
            node.GetAttr("transpose_x2", transposeX2Attr);
            EXPECT_TRUE(transposeX2Attr);
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(QuantBatchMatmulV4TransposeFusionPassTest, TransposeFusionWithAllOptionalInputs)
{
    auto graphBuilder = es::EsGraphBuilder("transpose_fusion_with_all_inputs");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT8_E4M3FN, FORMAT_ND, {DIM_128, DIM_256});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT4_E2M1, FORMAT_ND, {DIM_64, DIM_256});
    auto bias = graphBuilder.CreateInput(2, "bias", DT_FLOAT, FORMAT_ND, {DIM_64});
    auto x1Scale = graphBuilder.CreateInput(3, "x1_scale", DT_FLOAT, FORMAT_ND, {DIM_128});
    auto x2Scale = graphBuilder.CreateInput(4, "x2_scale", DT_FLOAT, FORMAT_ND, {DIM_64, DIM_1});
    auto yScale = graphBuilder.CreateInput(5, "y_scale", DT_FLOAT, FORMAT_ND, {DIM_64});

    auto transposeX2 = CreateTranspose(graphBuilder, x2);
    auto transposeX2Scale = CreateTranspose(graphBuilder, x2Scale);
    auto quantBmmv4 = es::QuantBatchMatmulV4(x1, transposeX2, bias, x1Scale, transposeX2Scale, yScale,
                                             es::EsTensorHolder(), es::EsTensorHolder(), es::EsTensorHolder(),
                                             es::EsTensorHolder(), DT_BF16, -1, false, false, BLOCK_SIZE);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({quantBmmv4});
    ASSERT_NE(graph, nullptr);

    SetupShapesForTransposeFusion(graph, {{DT_FLOAT8_E4M3FN, {DIM_128, DIM_256}},
                                          {DT_FLOAT4_E2M1, {DIM_64, DIM_256}},
                                          {DT_FLOAT, {DIM_64}},
                                          {DT_FLOAT, {DIM_128}},
                                          {DT_FLOAT, {DIM_64, DIM_1}},
                                          {DT_FLOAT, {DIM_64}}});

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasQuantBmmv4);
}

TEST_F(QuantBatchMatmulV4TransposeFusionPassTest, NoTransposeOnX2_ShouldNotFuse)
{
    auto graphBuilder = es::EsGraphBuilder("no_transpose_on_x2");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT8_E4M3FN, FORMAT_ND, {DIM_128, DIM_256});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT4_E2M1, FORMAT_ND, {DIM_64, DIM_256});
    auto x2Scale = graphBuilder.CreateInput(2, "x2_scale", DT_FLOAT, FORMAT_ND, {DIM_64, DIM_1});

    auto quantBmmv4 = es::QuantBatchMatmulV4(x1, x2, nullptr, nullptr, x2Scale, es::EsTensorHolder(), nullptr, nullptr,
                                             nullptr, nullptr, DT_BF16, -1, false, false, BLOCK_SIZE);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({quantBmmv4});
    ASSERT_NE(graph, nullptr);

    SetupShapesForTransposeFusion(
        graph,
        {{DT_FLOAT8_E4M3FN, {DIM_128, DIM_256}}, {DT_FLOAT4_E2M1, {DIM_64, DIM_256}}, {DT_FLOAT, {DIM_64, DIM_1}}});

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto result = CheckFusionResult(graph);
    EXPECT_TRUE(result.hasQuantBmmv4);
}

TEST_F(QuantBatchMatmulV4TransposeFusionPassTest, NonSimpleReshapeOnX2Scale_ShouldNotFuse)
{
    auto graphBuilder = es::EsGraphBuilder("non_simple_reshape_on_x2_scale");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT8_E4M3FN, FORMAT_ND, {DIM_128, DIM_256});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT4_E2M1, FORMAT_ND, {DIM_64, DIM_256});
    auto x2Scale = graphBuilder.CreateInput(2, "x2_scale", DT_FLOAT16, FORMAT_ND, {DIM_32, DIM_64});

    auto transposeX2 = CreateTranspose(graphBuilder, x2);
    std::vector<int64_t> reshapeShape{2};
    auto reshapeTensor = graphBuilder.CreateConst(std::vector<int32_t>({64, 32}), reshapeShape, DT_INT32, FORMAT_ND);
    auto reshapeX2Scale = es::Reshape(x2Scale, reshapeTensor);
    auto quantBmmv4 = es::QuantBatchMatmulV4(x1, transposeX2, nullptr, nullptr, reshapeX2Scale, es::EsTensorHolder(),
                                             nullptr, nullptr, nullptr, nullptr, DT_BF16, -1, false, false, BLOCK_SIZE);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({quantBmmv4});
    ASSERT_NE(graph, nullptr);

    SetupShapesForTransposeFusion(
        graph,
        {{DT_FLOAT8_E4M3FN, {DIM_128, DIM_256}}, {DT_FLOAT4_E2M1, {DIM_64, DIM_256}}, {DT_FLOAT16, {DIM_32, DIM_64}}});

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto result = CheckFusionResult(graph);
    EXPECT_TRUE(result.hasTranspose);
    EXPECT_TRUE(result.hasQuantBmmv4);
}

TEST_F(QuantBatchMatmulV4TransposeFusionPassTest, InvalidX1Dtype_ShouldNotFuse)
{
    auto graphBuilder = es::EsGraphBuilder("invalid_x1_dtype");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_INT8, FORMAT_ND, {DIM_128, DIM_256});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT4_E2M1, FORMAT_ND, {DIM_64, DIM_256});
    auto x2Scale = graphBuilder.CreateInput(2, "x2_scale", DT_FLOAT, FORMAT_ND, {DIM_64, DIM_1});

    auto transposeX2 = CreateTranspose(graphBuilder, x2);
    auto transposeX2Scale = CreateTranspose(graphBuilder, x2Scale);
    auto quantBmmv4 = es::QuantBatchMatmulV4(x1, transposeX2, nullptr, nullptr, transposeX2Scale, es::EsTensorHolder(),
                                             nullptr, nullptr, nullptr, nullptr, DT_BF16, -1, false, false, BLOCK_SIZE);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({quantBmmv4});
    ASSERT_NE(graph, nullptr);

    SetupShapesForTransposeFusion(
        graph, {{DT_INT8, {DIM_128, DIM_256}}, {DT_FLOAT4_E2M1, {DIM_64, DIM_256}}, {DT_FLOAT, {DIM_64, DIM_1}}});

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto result = CheckFusionResult(graph);
    EXPECT_TRUE(result.hasTranspose);
    EXPECT_TRUE(result.hasQuantBmmv4);
}

TEST_F(QuantBatchMatmulV4TransposeFusionPassTest, InvalidX2Dtype_ShouldNotFuse)
{
    auto graphBuilder = es::EsGraphBuilder("invalid_x2_dtype");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT8_E4M3FN, FORMAT_ND, {DIM_128, DIM_256});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto x2Scale = graphBuilder.CreateInput(2, "x2_scale", DT_FLOAT, FORMAT_ND, {DIM_64, DIM_1});

    auto transposeX2 = CreateTranspose(graphBuilder, x2);
    auto transposeX2Scale = CreateTranspose(graphBuilder, x2Scale);
    auto quantBmmv4 = es::QuantBatchMatmulV4(x1, transposeX2, nullptr, nullptr, transposeX2Scale, es::EsTensorHolder(),
                                             nullptr, nullptr, nullptr, nullptr, DT_BF16, -1, false, false, BLOCK_SIZE);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({quantBmmv4});
    ASSERT_NE(graph, nullptr);

    SetupShapesForTransposeFusion(
        graph, {{DT_FLOAT8_E4M3FN, {DIM_128, DIM_256}}, {DT_INT8, {DIM_64, DIM_256}}, {DT_FLOAT, {DIM_64, DIM_1}}});

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto result = CheckFusionResult(graph);
    EXPECT_TRUE(result.hasTranspose);
    EXPECT_TRUE(result.hasQuantBmmv4);
}

TEST_F(QuantBatchMatmulV4TransposeFusionPassTest, Non2DShape_ShouldNotFuse)
{
    auto graphBuilder = es::EsGraphBuilder("non_2d_shape");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT8_E4M3FN, FORMAT_ND, {DIM_1, DIM_128, DIM_256});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT4_E2M1, FORMAT_ND, {DIM_64, DIM_256});
    auto x2Scale = graphBuilder.CreateInput(2, "x2_scale", DT_FLOAT, FORMAT_ND, {DIM_64, DIM_1});

    auto transposeX2 = CreateTranspose(graphBuilder, x2);
    auto transposeX2Scale = CreateTranspose(graphBuilder, x2Scale);
    auto quantBmmv4 = es::QuantBatchMatmulV4(x1, transposeX2, nullptr, nullptr, transposeX2Scale, es::EsTensorHolder(),
                                             nullptr, nullptr, nullptr, nullptr, DT_BF16, -1, false, false, BLOCK_SIZE);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({quantBmmv4});
    ASSERT_NE(graph, nullptr);

    SetupShapesForTransposeFusion(graph, {{DT_FLOAT8_E4M3FN, {DIM_1, DIM_128, DIM_256}},
                                          {DT_FLOAT4_E2M1, {DIM_64, DIM_256}},
                                          {DT_FLOAT, {DIM_64, DIM_1}}});

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto result = CheckFusionResult(graph);
    EXPECT_TRUE(result.hasTranspose);
    EXPECT_TRUE(result.hasQuantBmmv4);
}
