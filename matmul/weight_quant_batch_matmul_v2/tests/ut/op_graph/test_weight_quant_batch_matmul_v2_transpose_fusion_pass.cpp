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
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "ge/ge_utils.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "es_Transpose.h"
#include "es_Reshape.h"
#include "../../../op_graph/fusion_pass/weight_quant_batch_matmul_v2_transpose_fusion_pass.h"
#include "register/register_custom_pass.h"
#include "ut_op_util.h"

using namespace ut_util;
using namespace ge;
using namespace fe;
using namespace ops;

namespace {
constexpr int64_t DIM_128 = 128;
constexpr int64_t DIM_256 = 256;
constexpr int64_t DIM_64 = 64;
constexpr int64_t DIM_32 = 32;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_MINUS_1 = -1;

struct TransposeFusionResult {
    bool hasTranspose;
    bool hasReshape;
    bool hasWeightQuant;
    bool transposeXAttr;
    bool transposeWeightAttr;
};

TransposeFusionResult CheckFusionResult(std::shared_ptr<Graph> graph)
{
    TransposeFusionResult result{false, false, false, false, false};
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) != GRAPH_SUCCESS) {
            continue;
        }
        std::string typeStr(type.GetString());
        if (typeStr == "Transpose" || typeStr == "TransposeD") {
            result.hasTranspose = true;
        } else if (typeStr == "Reshape") {
            result.hasReshape = true;
        } else if (typeStr == "WeightQuantBatchMatmulV2") {
            result.hasWeightQuant = true;
            if (node.GetAttr("transpose_x", result.transposeXAttr) != GRAPH_SUCCESS) {
                result.transposeXAttr = false;
            }
            if (node.GetAttr("transpose_weight", result.transposeWeightAttr) != GRAPH_SUCCESS) {
                result.transposeWeightAttr = false;
            }
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

static TensorDesc MakeTD(const std::vector<int64_t>& shape, DataType dtype)
{
    TensorDesc desc;
    desc.SetDataType(dtype);
    desc.SetShape(Shape(shape));
    desc.SetFormat(FORMAT_ND);
    desc.SetOriginShape(Shape(shape));
    desc.SetOriginFormat(FORMAT_ND);
    return desc;
}

static std::vector<int64_t> SwapLastTwo(const std::vector<int64_t>& s)
{
    auto r = s;
    if (r.size() >= 2) {
        std::swap(r[r.size() - 1], r[r.size() - 2]);
    }
    return r;
}

// Set desc on a Data node's output
static void SetDataDesc(const es::EsTensorHolder& data, const std::vector<int64_t>& shape, DataType dtype)
{
    auto desc = MakeTD(shape, dtype);
    data.GetProducer()->UpdateOutputDesc(0, desc);
}

// Set desc on a Transpose node's input and output
static void SetTransposeDescs(const es::EsTensorHolder& trans, const std::vector<int64_t>& inShape, DataType dtype)
{
    auto* node = trans.GetProducer();
    auto inDesc = MakeTD(inShape, dtype);
    auto outDesc = MakeTD(SwapLastTwo(inShape), dtype);
    node->UpdateInputDesc(0, inDesc);
    node->UpdateOutputDesc(0, outDesc);
}

// Set desc on a Reshape node's input and output
static void SetReshapeDescs(const es::EsTensorHolder& reshape, const std::vector<int64_t>& inShape, DataType dtype,
                            const std::vector<int64_t>& outShape)
{
    auto* node = reshape.GetProducer();
    auto inDesc = MakeTD(inShape, dtype);
    auto outDesc = MakeTD(outShape, dtype);
    node->UpdateInputDesc(0, inDesc);
    node->UpdateOutputDesc(0, outDesc);
}

// Set all input descs and output desc on WQBMMv2
static void SetWqbmmv2Descs(const es::EsTensorHolder& wq, const std::vector<int64_t>& xInShape, DataType xDtype,
                            const std::vector<int64_t>& wInShape, DataType wDtype, const std::vector<int64_t>& sInShape,
                            DataType sDtype, const std::vector<int64_t>& oInShape, DataType oDtype, bool hasOffset,
                            const std::vector<int64_t>& bInShape, DataType bDtype, bool hasBias,
                            const std::vector<int64_t>& outShape, DataType outDtype)
{
    auto* node = wq.GetProducer();
    node->UpdateInputDesc(0, MakeTD(xInShape, xDtype));
    node->UpdateInputDesc(1, MakeTD(wInShape, wDtype));
    node->UpdateInputDesc(2, MakeTD(sInShape, sDtype));
    if (hasOffset) {
        node->UpdateInputDesc(3, MakeTD(oInShape, oDtype));
    }
    if (hasBias) {
        node->UpdateInputDesc(6, MakeTD(bInShape, bDtype));
    }
    node->UpdateOutputDesc(0, MakeTD(outShape, outDtype));
}

// Set transpose attrs on all WQBMMv2 nodes in the built graph
static void SetWqbmmv2Attrs(const std::shared_ptr<Graph>& graph, bool transposeX, bool transposeWeight)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS &&
            std::string(nodeType.GetString()) == "WeightQuantBatchMatmulV2") {
            node.SetAttr("transpose_x", transposeX);
            node.SetAttr("transpose_weight", transposeWeight);
        }
    }
}

// Set transpose attrs on the WQBMMv2 EsTensorHolder's producer node
static void SetWqbmmv2Attrs(const es::EsTensorHolder& wq, bool transposeX, bool transposeWeight)
{
    auto* node = wq.GetProducer();
    node->SetAttr("transpose_x", transposeX);
    node->SetAttr("transpose_weight", transposeWeight);
}
} // namespace

class WeightQuantBatchMatmulV2TransposeFusionPassTest : public testing::Test {
protected:
    class TestablePass : public ops::WeightQuantBatchMatmulV2TransposeFusionPass {};

    static void SetUpTestCase()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_l0c2out"] = {"float16"};
        optiCompilationInfo.soc_version = "soc_version";
        fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static void TearDownTestCase() { fe::PlatformInfoManager::Instance().platform_info_map_.clear(); }
};

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, BasicXTransposeFusion)
{
    auto gb = es::EsGraphBuilder("basic_x_transpose_fusion");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});

    auto tx = CreateTranspose(gb, x);
    auto wq = es::WeightQuantBatchMatmulV2(tx, w, s, o);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(graph, false, false);

    // Set descriptions manually
    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(tx, {DIM_128, DIM_256}, DT_BF16); // input=[128,256] -> output=[256,128]
    // WQBMMv2: x_in=[256,128](transposed), w_in=[64,256], s_in=[64,1], o_in=[64,1], out=[128,64]
    SetWqbmmv2Descs(wq, {DIM_256, DIM_128}, DT_BF16, {DIM_64, DIM_256}, DT_INT8, {DIM_64, DIM_1}, DT_BF16,
                    {DIM_64, DIM_1}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeXAttr);
    EXPECT_FALSE(result.transposeWeightAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, WeightTransposeFusion)
{
    auto gb = es::EsGraphBuilder("weight_transpose_fusion");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});

    auto tw = CreateTranspose(gb, w);
    auto ts = CreateTranspose(gb, s);
    auto to = CreateTranspose(gb, o);
    auto wq = es::WeightQuantBatchMatmulV2(x, tw, ts, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(tw, {DIM_64, DIM_256}, DT_INT8); // w: [64,256] -> [256,64]
    SetTransposeDescs(ts, {DIM_64, DIM_1}, DT_BF16);   // s: [64,1] -> [1,64]
    SetTransposeDescs(to, {DIM_64, DIM_1}, DT_BF16);   // o: [64,1] -> [1,64]
    SetWqbmmv2Descs(wq, {DIM_128, DIM_256}, DT_BF16, {DIM_256, DIM_64}, DT_INT8, {DIM_1, DIM_64}, DT_BF16,
                    {DIM_1, DIM_64}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_FALSE(result.transposeXAttr);
    EXPECT_TRUE(result.transposeWeightAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, XAndWeightTransposeFusion)
{
    auto gb = es::EsGraphBuilder("x_and_weight_transpose_fusion");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});

    auto tx = CreateTranspose(gb, x);
    auto tw = CreateTranspose(gb, w);
    auto ts = CreateTranspose(gb, s);
    auto to = CreateTranspose(gb, o);
    auto wq = es::WeightQuantBatchMatmulV2(tx, tw, ts, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(tx, {DIM_128, DIM_256}, DT_BF16); // x: [128,256] -> [256,128]
    SetTransposeDescs(tw, {DIM_64, DIM_256}, DT_INT8);  // w: [64,256] -> [256,64]
    SetTransposeDescs(ts, {DIM_64, DIM_1}, DT_BF16);    // s: [64,1] -> [1,64]
    SetTransposeDescs(to, {DIM_64, DIM_1}, DT_BF16);    // o: [64,1] -> [1,64]
    SetWqbmmv2Descs(wq, {DIM_256, DIM_128}, DT_BF16, {DIM_256, DIM_64}, DT_INT8, {DIM_1, DIM_64}, DT_BF16,
                    {DIM_1, DIM_64}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeXAttr);
    EXPECT_TRUE(result.transposeWeightAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, WeightTransposeWithReshapeFusion)
{
    auto gb = es::EsGraphBuilder("weight_transpose_with_reshape_fusion");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_32, DIM_128});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_256, DIM_128});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_256, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_256, DIM_1});

    auto tw = CreateTranspose(gb, w);

    std::vector<int64_t> reshapeShapeScale{2};
    auto reshapeTensorScale = gb.CreateConst(std::vector<int32_t>({1, 256}), reshapeShapeScale, DT_INT32, FORMAT_ND);
    auto rs = es::Reshape(s, reshapeTensorScale);

    std::vector<int64_t> reshapeShapeOffset{2};
    auto reshapeTensorOffset = gb.CreateConst(std::vector<int32_t>({1, 256}), reshapeShapeOffset, DT_INT32, FORMAT_ND);
    auto ro = es::Reshape(o, reshapeTensorOffset);

    auto wq = es::WeightQuantBatchMatmulV2(x, tw, rs, ro);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_32, DIM_128}, DT_BF16);
    SetDataDesc(w, {DIM_256, DIM_128}, DT_INT8);
    SetDataDesc(s, {DIM_256, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_256, DIM_1}, DT_BF16);
    SetTransposeDescs(tw, {DIM_256, DIM_128}, DT_INT8);               // w: [256,128] -> [128,256]
    SetReshapeDescs(rs, {DIM_256, DIM_1}, DT_BF16, {DIM_1, DIM_256}); // s: [256,1] -> [1,256]
    SetReshapeDescs(ro, {DIM_256, DIM_1}, DT_BF16, {DIM_1, DIM_256}); // o: [256,1] -> [1,256]
    SetWqbmmv2Descs(wq, {DIM_32, DIM_128}, DT_BF16, {DIM_128, DIM_256}, DT_INT8, {DIM_1, DIM_256}, DT_BF16,
                    {DIM_1, DIM_256}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_32, DIM_256}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_FALSE(result.hasReshape);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeWeightAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, WeightTransposeWithoutOffset)
{
    auto gb = es::EsGraphBuilder("weight_transpose_without_offset");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});

    auto tw = CreateTranspose(gb, w);
    auto ts = CreateTranspose(gb, s);
    auto wq = es::WeightQuantBatchMatmulV2(x, tw, ts);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(tw, {DIM_64, DIM_256}, DT_INT8);
    SetTransposeDescs(ts, {DIM_64, DIM_1}, DT_BF16);
    SetWqbmmv2Descs(wq, {DIM_128, DIM_256}, DT_BF16, {DIM_256, DIM_64}, DT_INT8, {DIM_1, DIM_64}, DT_BF16, {}, DT_BF16,
                    false, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeWeightAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, WeightTransposeWithBias)
{
    auto gb = es::EsGraphBuilder("weight_transpose_with_bias");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto b = gb.CreateInput(3, "bias", DT_FLOAT, FORMAT_ND, {DIM_64});

    auto tw = CreateTranspose(gb, w);
    auto ts = CreateTranspose(gb, s);
    auto wq = es::WeightQuantBatchMatmulV2(x, tw, ts, nullptr, nullptr, nullptr, b);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(b, {DIM_64}, DT_FLOAT);
    SetTransposeDescs(tw, {DIM_64, DIM_256}, DT_INT8);
    SetTransposeDescs(ts, {DIM_64, DIM_1}, DT_BF16);
    SetWqbmmv2Descs(wq, {DIM_128, DIM_256}, DT_BF16, {DIM_256, DIM_64}, DT_INT8, {DIM_1, DIM_64}, DT_BF16, {}, DT_BF16,
                    false, {DIM_64}, DT_FLOAT, true, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeWeightAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, NoTransposeNoFusion)
{
    auto gb = es::EsGraphBuilder("no_transpose_no_fusion");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});

    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, o);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_64, DIM_1}, DT_BF16);
    SetWqbmmv2Descs(wq, {DIM_128, DIM_256}, DT_BF16, {DIM_64, DIM_256}, DT_INT8, {DIM_64, DIM_1}, DT_BF16,
                    {DIM_64, DIM_1}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto result = CheckFusionResult(graph);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_FALSE(result.transposeXAttr);
    EXPECT_FALSE(result.transposeWeightAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, DynamicShapeTransposeFusion)
{
    auto gb = es::EsGraphBuilder("dynamic_shape_transpose_fusion");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_MINUS_1, DIM_MINUS_1});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_MINUS_1, DIM_MINUS_1});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_MINUS_1, DIM_MINUS_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_MINUS_1, DIM_MINUS_1});

    auto tx = CreateTranspose(gb, x);
    auto tw = CreateTranspose(gb, w);
    auto ts = CreateTranspose(gb, s);
    auto to = CreateTranspose(gb, o);
    auto wq = es::WeightQuantBatchMatmulV2(tx, tw, ts, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    // Dynamic shape: all dims are -1, swap still gives [-1, -1]
    SetDataDesc(x, {DIM_MINUS_1, DIM_MINUS_1}, DT_BF16);
    SetDataDesc(w, {DIM_MINUS_1, DIM_MINUS_1}, DT_INT8);
    SetDataDesc(s, {DIM_MINUS_1, DIM_MINUS_1}, DT_BF16);
    SetDataDesc(o, {DIM_MINUS_1, DIM_MINUS_1}, DT_BF16);
    SetTransposeDescs(tx, {DIM_MINUS_1, DIM_MINUS_1}, DT_BF16);
    SetTransposeDescs(tw, {DIM_MINUS_1, DIM_MINUS_1}, DT_INT8);
    SetTransposeDescs(ts, {DIM_MINUS_1, DIM_MINUS_1}, DT_BF16);
    SetTransposeDescs(to, {DIM_MINUS_1, DIM_MINUS_1}, DT_BF16);
    SetWqbmmv2Descs(wq, {DIM_MINUS_1, DIM_MINUS_1}, DT_BF16, {DIM_MINUS_1, DIM_MINUS_1}, DT_INT8,
                    {DIM_MINUS_1, DIM_MINUS_1}, DT_BF16, {DIM_MINUS_1, DIM_MINUS_1}, DT_BF16, true, {}, DT_FLOAT, false,
                    {DIM_MINUS_1, DIM_MINUS_1}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeXAttr);
    EXPECT_TRUE(result.transposeWeightAttr);
}

// Test: transpose_weight already true -> still flips to false (matching original logic)
TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, TransposeWeightAlreadyTrue)
{
    auto gb = es::EsGraphBuilder("transpose_weight_already_true");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});

    auto tx = CreateTranspose(gb, x);
    auto tw = CreateTranspose(gb, w);
    auto ts = CreateTranspose(gb, s);
    auto to = CreateTranspose(gb, o);
    auto wq = es::WeightQuantBatchMatmulV2(tx, tw, ts, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, true);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(tx, {DIM_128, DIM_256}, DT_BF16); // x: [128,256] -> [256,128]
    SetTransposeDescs(tw, {DIM_64, DIM_256}, DT_INT8);  // w: [64,256] -> [256,64]
    SetTransposeDescs(ts, {DIM_64, DIM_1}, DT_BF16);    // s: [64,1] -> [1,64]
    SetTransposeDescs(to, {DIM_64, DIM_1}, DT_BF16);    // o: [64,1] -> [1,64]
    SetWqbmmv2Descs(wq, {DIM_256, DIM_128}, DT_BF16, {DIM_256, DIM_64}, DT_INT8, {DIM_1, DIM_64}, DT_BF16,
                    {DIM_1, DIM_64}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    // All transposes should be fused (removed), both attrs flipped
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeXAttr);       // false -> true
    EXPECT_FALSE(result.transposeWeightAttr); // true -> false
}

// Test: transpose_x already true -> still flips to false (matching original logic)
TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, TransposeXAlreadyTrue)
{
    auto gb = es::EsGraphBuilder("transpose_x_already_true");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});

    auto tx = CreateTranspose(gb, x);
    auto tw = CreateTranspose(gb, w);
    auto ts = CreateTranspose(gb, s);
    auto to = CreateTranspose(gb, o);
    auto wq = es::WeightQuantBatchMatmulV2(tx, tw, ts, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, true, false);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(tx, {DIM_128, DIM_256}, DT_BF16); // x: [128,256] -> [256,128]
    SetTransposeDescs(tw, {DIM_64, DIM_256}, DT_INT8);  // w: [64,256] -> [256,64]
    SetTransposeDescs(ts, {DIM_64, DIM_1}, DT_BF16);    // s: [64,1] -> [1,64]
    SetTransposeDescs(to, {DIM_64, DIM_1}, DT_BF16);    // o: [64,1] -> [1,64]
    SetWqbmmv2Descs(wq, {DIM_256, DIM_128}, DT_BF16, {DIM_256, DIM_64}, DT_INT8, {DIM_1, DIM_64}, DT_BF16,
                    {DIM_1, DIM_64}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    // All transposes should be fused (removed), both attrs flipped
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_FALSE(result.transposeXAttr);     // true -> false
    EXPECT_TRUE(result.transposeWeightAttr); // false -> true
}

// Test: non-equivalent reshape on scale (e.g., [256,2]->[128,4]) aborts entire fusion
TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, ReshapeNotEquivalent)
{
    auto gb = es::EsGraphBuilder("reshape_not_equivalent");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_32, DIM_128});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_256, DIM_128});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_256, 2});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_256, DIM_1});

    auto tw = CreateTranspose(gb, w);

    // Non-equivalent reshape on scale: [256,2] -> [128,4], neither dim is 1
    std::vector<int64_t> reshapeShapeScale{2};
    auto reshapeTensorScale = gb.CreateConst(std::vector<int32_t>({128, 4}), reshapeShapeScale, DT_INT32, FORMAT_ND);
    auto rs = es::Reshape(s, reshapeTensorScale);

    auto to = CreateTranspose(gb, o);

    auto wq = es::WeightQuantBatchMatmulV2(x, tw, rs, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_32, DIM_128}, DT_BF16);
    SetDataDesc(w, {DIM_256, DIM_128}, DT_INT8);
    SetDataDesc(s, {DIM_256, 2}, DT_BF16);
    SetDataDesc(o, {DIM_256, DIM_1}, DT_BF16);
    SetTransposeDescs(tw, {DIM_256, DIM_128}, DT_INT8);   // w: [256,128] -> [128,256]
    SetReshapeDescs(rs, {DIM_256, 2}, DT_BF16, {128, 4}); // s: [256,2] -> [128,4] non-simple
    SetTransposeDescs(to, {DIM_256, DIM_1}, DT_BF16);     // o: [256,1] -> [1,256]
    SetWqbmmv2Descs(wq, {DIM_32, DIM_128}, DT_BF16, {DIM_128, DIM_256}, DT_INT8, {128, 4}, DT_BF16, {DIM_1, DIM_256},
                    DT_BF16, true, {}, DT_FLOAT, false, {DIM_32, DIM_256}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto result = CheckFusionResult(graph);
    // Non-simple reshape aborts entire fusion: all nodes remain
    EXPECT_TRUE(result.hasReshape);
    EXPECT_TRUE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_FALSE(result.transposeWeightAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, ReshapeInputDim1OutputNot)
{
    auto gb = es::EsGraphBuilder("reshape_input_dim1_output_not");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_32, DIM_128});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_256, DIM_128});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_256, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_256, DIM_1});

    auto tw = CreateTranspose(gb, w);

    std::vector<int64_t> reshapeShapeScale{2};
    auto reshapeTensorScale = gb.CreateConst(std::vector<int32_t>({128, 2}), reshapeShapeScale, DT_INT32, FORMAT_ND);
    auto rs = es::Reshape(s, reshapeTensorScale);

    auto to = CreateTranspose(gb, o);

    auto wq = es::WeightQuantBatchMatmulV2(x, tw, rs, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_32, DIM_128}, DT_BF16);
    SetDataDesc(w, {DIM_256, DIM_128}, DT_INT8);
    SetDataDesc(s, {DIM_256, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_256, DIM_1}, DT_BF16);
    SetTransposeDescs(tw, {DIM_256, DIM_128}, DT_INT8);
    SetReshapeDescs(rs, {DIM_256, DIM_1}, DT_BF16, {128, 2});
    SetTransposeDescs(to, {DIM_256, DIM_1}, DT_BF16);
    SetWqbmmv2Descs(wq, {DIM_32, DIM_128}, DT_BF16, {DIM_128, DIM_256}, DT_INT8, {128, 2}, DT_BF16, {DIM_1, DIM_256},
                    DT_BF16, true, {}, DT_FLOAT, false, {DIM_32, DIM_256}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    // Input has dim 1 -> simple reshape (matching original: only checks input shape) -> all fused
    EXPECT_FALSE(result.hasReshape);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeWeightAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, ReshapeOutputDim1InputNot)
{
    auto gb = es::EsGraphBuilder("reshape_output_dim1_input_not");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_32, DIM_128});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_256, DIM_128});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_256, 2});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_256, DIM_1});

    auto tw = CreateTranspose(gb, w);

    std::vector<int64_t> reshapeShapeScale{2};
    auto reshapeTensorScale = gb.CreateConst(std::vector<int32_t>({128, 1}), reshapeShapeScale, DT_INT32, FORMAT_ND);
    auto rs = es::Reshape(s, reshapeTensorScale);

    auto to = CreateTranspose(gb, o);

    auto wq = es::WeightQuantBatchMatmulV2(x, tw, rs, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_32, DIM_128}, DT_BF16);
    SetDataDesc(w, {DIM_256, DIM_128}, DT_INT8);
    SetDataDesc(s, {DIM_256, 2}, DT_BF16);
    SetDataDesc(o, {DIM_256, DIM_1}, DT_BF16);
    SetTransposeDescs(tw, {DIM_256, DIM_128}, DT_INT8);
    SetReshapeDescs(rs, {DIM_256, 2}, DT_BF16, {128, DIM_1});
    SetTransposeDescs(to, {DIM_256, DIM_1}, DT_BF16);
    SetWqbmmv2Descs(wq, {DIM_32, DIM_128}, DT_BF16, {DIM_128, DIM_256}, DT_INT8, {128, DIM_1}, DT_BF16,
                    {DIM_1, DIM_256}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_32, DIM_256}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto result = CheckFusionResult(graph);
    // Input [256,2] has no dim of 1 -> not simple -> abort entire fusion, all nodes remain
    EXPECT_TRUE(result.hasReshape);
    EXPECT_TRUE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_FALSE(result.transposeWeightAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, ScaleReshapeEquivalentOffsetTranspose)
{
    auto gb = es::EsGraphBuilder("scale_reshape_equivalent_offset_transpose");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_32, DIM_128});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_256, DIM_128});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_256, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_256, DIM_1});

    auto tw = CreateTranspose(gb, w);

    std::vector<int64_t> reshapeShapeScale{2};
    auto reshapeTensorScale = gb.CreateConst(std::vector<int32_t>({1, 256}), reshapeShapeScale, DT_INT32, FORMAT_ND);
    auto rs = es::Reshape(s, reshapeTensorScale);

    auto to = CreateTranspose(gb, o);

    auto wq = es::WeightQuantBatchMatmulV2(x, tw, rs, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_32, DIM_128}, DT_BF16);
    SetDataDesc(w, {DIM_256, DIM_128}, DT_INT8);
    SetDataDesc(s, {DIM_256, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_256, DIM_1}, DT_BF16);
    SetTransposeDescs(tw, {DIM_256, DIM_128}, DT_INT8);
    SetReshapeDescs(rs, {DIM_256, DIM_1}, DT_BF16, {DIM_1, DIM_256});
    SetTransposeDescs(to, {DIM_256, DIM_1}, DT_BF16);
    SetWqbmmv2Descs(wq, {DIM_32, DIM_128}, DT_BF16, {DIM_128, DIM_256}, DT_INT8, {DIM_1, DIM_256}, DT_BF16,
                    {DIM_1, DIM_256}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_32, DIM_256}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasReshape);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeWeightAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, WeightTransposeDirectScaleOffset)
{
    auto gb = es::EsGraphBuilder("weight_transpose_direct_scale_offset");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});

    auto tw = CreateTranspose(gb, w);
    auto wq = es::WeightQuantBatchMatmulV2(x, tw, s, o);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(tw, {DIM_64, DIM_256}, DT_INT8);
    SetWqbmmv2Descs(wq, {DIM_128, DIM_256}, DT_BF16, {DIM_256, DIM_64}, DT_INT8, {DIM_64, DIM_1}, DT_BF16,
                    {DIM_64, DIM_1}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeWeightAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, InvalidXInt8Dtype)
{
    auto gb = es::EsGraphBuilder("invalid_x_int8_dtype");
    auto x = gb.CreateInput(0, "x", DT_INT8, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});

    auto tx = CreateTranspose(gb, x);
    auto wq = es::WeightQuantBatchMatmulV2(tx, w, s, o);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_INT8);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(tx, {DIM_128, DIM_256}, DT_INT8);
    SetWqbmmv2Descs(wq, {DIM_256, DIM_128}, DT_INT8, {DIM_64, DIM_256}, DT_INT8, {DIM_64, DIM_1}, DT_BF16,
                    {DIM_64, DIM_1}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto result = CheckFusionResult(graph);
    EXPECT_TRUE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_FALSE(result.transposeXAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, InvalidXShape3D)
{
    auto gb = es::EsGraphBuilder("invalid_x_shape_3d");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {4, DIM_32, DIM_64});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_64});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});

    auto tx = CreateTranspose(gb, x);
    auto wq = es::WeightQuantBatchMatmulV2(tx, w, s, o);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {4, DIM_32, DIM_64}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_64}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(tx, {4, DIM_32, DIM_64}, DT_BF16);
    SetWqbmmv2Descs(wq, {4, DIM_64, DIM_32}, DT_BF16, {DIM_64, DIM_64}, DT_INT8, {DIM_64, DIM_1}, DT_BF16,
                    {DIM_64, DIM_1}, DT_BF16, true, {}, DT_FLOAT, false, {4, DIM_32, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto result = CheckFusionResult(graph);
    EXPECT_TRUE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_FALSE(result.transposeXAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, BothTransposeAttrsAlreadyTrue)
{
    auto gb = es::EsGraphBuilder("both_transpose_attrs_already_true");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});

    auto tx = CreateTranspose(gb, x);
    auto tw = CreateTranspose(gb, w);
    auto ts = CreateTranspose(gb, s);
    auto to = CreateTranspose(gb, o);
    auto wq = es::WeightQuantBatchMatmulV2(tx, tw, ts, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, true, true);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(tx, {DIM_128, DIM_256}, DT_BF16);
    SetTransposeDescs(tw, {DIM_64, DIM_256}, DT_INT8);
    SetTransposeDescs(ts, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(to, {DIM_64, DIM_1}, DT_BF16);
    SetWqbmmv2Descs(wq, {DIM_256, DIM_128}, DT_BF16, {DIM_256, DIM_64}, DT_INT8, {DIM_1, DIM_64}, DT_BF16,
                    {DIM_1, DIM_64}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    // Both transposes fused, both attrs flipped to false
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_FALSE(result.transposeXAttr);      // true -> false
    EXPECT_FALSE(result.transposeWeightAttr); // true -> false
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, Float16DtypeFusion)
{
    auto gb = es::EsGraphBuilder("float16_dtype_fusion");
    auto x = gb.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_FLOAT16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_FLOAT16, FORMAT_ND, {DIM_64, DIM_1});

    auto tx = CreateTranspose(gb, x);
    auto wq = es::WeightQuantBatchMatmulV2(tx, w, s, o);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_FLOAT16);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_FLOAT16);
    SetDataDesc(o, {DIM_64, DIM_1}, DT_FLOAT16);
    SetTransposeDescs(tx, {DIM_128, DIM_256}, DT_FLOAT16);
    SetWqbmmv2Descs(wq, {DIM_256, DIM_128}, DT_FLOAT16, {DIM_64, DIM_256}, DT_INT8, {DIM_64, DIM_1}, DT_FLOAT16,
                    {DIM_64, DIM_1}, DT_FLOAT16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_FLOAT16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeXAttr);
    EXPECT_FALSE(result.transposeWeightAttr);
}

class WeightQuantBatchMatmulV2TransposeNZFusionPassTest : public testing::Test {
protected:
    class TestableNZPass : public ops::WeightQuantBatchMatmulV2TransposeNZFusionPass {};

    static void SetUpTestCase()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_ub2out"] = {"float16"};
        optiCompilationInfo.soc_version = "soc_version";
        fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static void TearDownTestCase() { fe::PlatformInfoManager::Instance().platform_info_map_.clear(); }
};

TEST_F(WeightQuantBatchMatmulV2TransposeNZFusionPassTest, NZWeightTransposeWithScaleOffset)
{
    auto gb = es::EsGraphBuilder("nz_weight_transpose_with_scale_offset");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});

    auto tw = CreateTranspose(gb, w);
    auto ts = CreateTranspose(gb, s);
    auto to = CreateTranspose(gb, o);
    auto wq = es::WeightQuantBatchMatmulV2(x, tw, ts, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(graph, false, false);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(tw, {DIM_64, DIM_256}, DT_INT8);
    SetTransposeDescs(ts, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(to, {DIM_64, DIM_1}, DT_BF16);
    SetWqbmmv2Descs(wq, {DIM_128, DIM_256}, DT_BF16, {DIM_256, DIM_64}, DT_INT8, {DIM_1, DIM_64}, DT_BF16,
                    {DIM_1, DIM_64}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestableNZPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
}

static int CountNodesByType(const std::shared_ptr<Graph>& graph, const std::string& type)
{
    int count = 0;
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && std::string(nodeType.GetString()) == type) {
            count++;
        }
    }
    return count;
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, SharedTransposeWeightOffset)
{
    auto gb = es::EsGraphBuilder("shared_transpose_weight_offset");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_128});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_64, DIM_128});

    auto tw = CreateTranspose(gb, w);
    auto to = CreateTranspose(gb, o);
    auto wq = es::WeightQuantBatchMatmulV2(x, tw, s, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_128}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_64, DIM_128}, DT_BF16);
    SetTransposeDescs(tw, {DIM_64, DIM_128}, DT_INT8);
    SetTransposeDescs(to, {DIM_64, DIM_128}, DT_BF16);
    SetWqbmmv2Descs(wq, {DIM_128, DIM_256}, DT_BF16, {DIM_128, DIM_64}, DT_INT8, {DIM_64, DIM_1}, DT_BF16,
                    {DIM_128, DIM_64}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeWeightAttr);
    EXPECT_EQ(CountNodesByType(graph, "Transpose"), 0);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, SharedTransposeWeightScale)
{
    auto gb = es::EsGraphBuilder("shared_transpose_weight_scale");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_128});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_128, DIM_128});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_128, DIM_128});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_128, DIM_1});

    auto tw = CreateTranspose(gb, w);
    auto ts = CreateTranspose(gb, s);
    auto to = CreateTranspose(gb, o);
    auto wq = es::WeightQuantBatchMatmulV2(x, tw, ts, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_128, DIM_128}, DT_BF16);
    SetDataDesc(w, {DIM_128, DIM_128}, DT_INT8);
    SetDataDesc(s, {DIM_128, DIM_128}, DT_BF16);
    SetDataDesc(o, {DIM_128, DIM_1}, DT_BF16);
    SetTransposeDescs(tw, {DIM_128, DIM_128}, DT_INT8);
    SetTransposeDescs(ts, {DIM_128, DIM_128}, DT_BF16);
    SetTransposeDescs(to, {DIM_128, DIM_1}, DT_BF16);
    SetWqbmmv2Descs(wq, {DIM_128, DIM_128}, DT_BF16, {DIM_128, DIM_128}, DT_INT8, {DIM_128, DIM_128}, DT_BF16,
                    {DIM_1, DIM_128}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_128}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodesByType(graph, "Transpose"), 0);
    auto result = CheckFusionResult(graph);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeWeightAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, XTransposePlusSharedWeightOffset)
{
    auto gb = es::EsGraphBuilder("x_transpose_plus_shared_weight_offset");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_128});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_64, DIM_128});

    auto tx = CreateTranspose(gb, x);
    auto tw = CreateTranspose(gb, w);
    auto to = CreateTranspose(gb, o);
    auto wq = es::WeightQuantBatchMatmulV2(tx, tw, s, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_128}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_64, DIM_128}, DT_BF16);
    SetTransposeDescs(tx, {DIM_128, DIM_256}, DT_BF16);
    SetTransposeDescs(tw, {DIM_64, DIM_128}, DT_INT8);
    SetTransposeDescs(to, {DIM_64, DIM_128}, DT_BF16);
    SetWqbmmv2Descs(wq, {DIM_256, DIM_128}, DT_BF16, {DIM_128, DIM_64}, DT_INT8, {DIM_64, DIM_1}, DT_BF16,
                    {DIM_128, DIM_64}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodesByType(graph, "Transpose"), 0);
    auto result = CheckFusionResult(graph);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeXAttr);
    EXPECT_TRUE(result.transposeWeightAttr);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, MultipleWqbmmv2Nodes)
{
    auto gb = es::EsGraphBuilder("multiple_wqbmmv2_nodes");
    auto x1 = gb.CreateInput(0, "x1", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w1 = gb.CreateInput(1, "w1", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s1 = gb.CreateInput(2, "s1", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto x2 = gb.CreateInput(3, "x2", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w2 = gb.CreateInput(4, "w2", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s2 = gb.CreateInput(5, "s2", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});

    auto tx1 = CreateTranspose(gb, x1);
    auto tw1 = CreateTranspose(gb, w1);
    auto wq1 = es::WeightQuantBatchMatmulV2(tx1, tw1, s1);
    auto tx2 = CreateTranspose(gb, x2);
    auto tw2 = CreateTranspose(gb, w2);
    auto wq2 = es::WeightQuantBatchMatmulV2(tx2, tw2, s2);

    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq1, wq2});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq1, false, false);
    SetWqbmmv2Attrs(wq2, false, false);

    SetDataDesc(x1, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w1, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s1, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(x2, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w2, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s2, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(tx1, {DIM_128, DIM_256}, DT_BF16);
    SetTransposeDescs(tw1, {DIM_64, DIM_256}, DT_INT8);
    SetTransposeDescs(tx2, {DIM_128, DIM_256}, DT_BF16);
    SetTransposeDescs(tw2, {DIM_64, DIM_256}, DT_INT8);
    SetWqbmmv2Descs(wq1, {DIM_256, DIM_128}, DT_BF16, {DIM_256, DIM_64}, DT_INT8, {DIM_64, DIM_1}, DT_BF16, {}, DT_BF16,
                    false, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);
    SetWqbmmv2Descs(wq2, {DIM_256, DIM_128}, DT_BF16, {DIM_256, DIM_64}, DT_INT8, {DIM_64, DIM_1}, DT_BF16, {}, DT_BF16,
                    false, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodesByType(graph, "Transpose"), 0);
    EXPECT_EQ(CountNodesByType(graph, "WeightQuantBatchMatmulV2"), 2);
}

static GNode FindWqbmmv2(const std::shared_ptr<Graph>& graph)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == GRAPH_SUCCESS && std::string(type.GetString()) == "WeightQuantBatchMatmulV2") {
            return node;
        }
    }
    return GNode();
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, WeightTransposeInputDescUpdated)
{
    auto gb = es::EsGraphBuilder("weight_transpose_input_desc_updated");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});

    auto tw = CreateTranspose(gb, w);
    auto ts = CreateTranspose(gb, s);
    auto to = CreateTranspose(gb, o);
    auto wq = es::WeightQuantBatchMatmulV2(x, tw, ts, to);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(tw, {DIM_64, DIM_256}, DT_INT8);
    SetTransposeDescs(ts, {DIM_64, DIM_1}, DT_BF16);
    SetTransposeDescs(to, {DIM_64, DIM_1}, DT_BF16);
    SetWqbmmv2Descs(wq, {DIM_128, DIM_256}, DT_BF16, {DIM_256, DIM_64}, DT_INT8, {DIM_1, DIM_64}, DT_BF16,
                    {DIM_1, DIM_64}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto wqNode = FindWqbmmv2(graph);
    AscendString wqType;
    ASSERT_EQ(wqNode.GetType(wqType), GRAPH_SUCCESS);

    bool transW = false;
    EXPECT_EQ(wqNode.GetAttr("transpose_weight", transW), GRAPH_SUCCESS);
    EXPECT_TRUE(transW);

    TensorDesc wDesc;
    EXPECT_EQ(wqNode.GetInputDesc(1, wDesc), GRAPH_SUCCESS);
    auto wShape = wDesc.GetShape();
    EXPECT_EQ(wShape.GetDimNum(), 2);
    EXPECT_EQ(wShape.GetDim(0), DIM_64);
    EXPECT_EQ(wShape.GetDim(1), DIM_256);
}

TEST_F(WeightQuantBatchMatmulV2TransposeFusionPassTest, ScaleTransposeNoOffsetTranspose)
{
    auto gb = es::EsGraphBuilder("scale_transpose_no_offset_transpose");
    auto x = gb.CreateInput(0, "x", DT_BF16, FORMAT_ND, {DIM_128, DIM_256});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_64, DIM_256});
    auto s = gb.CreateInput(2, "scale", DT_BF16, FORMAT_ND, {DIM_64, DIM_1});
    auto o = gb.CreateInput(3, "offset", DT_BF16, FORMAT_ND, {DIM_1, DIM_64});

    auto tw = CreateTranspose(gb, w);
    auto ts = CreateTranspose(gb, s);
    auto wq = es::WeightQuantBatchMatmulV2(x, tw, ts, o);
    std::shared_ptr<Graph> graph = gb.BuildAndReset({wq});
    ASSERT_NE(graph, nullptr);
    SetWqbmmv2Attrs(wq, false, false);

    SetDataDesc(x, {DIM_128, DIM_256}, DT_BF16);
    SetDataDesc(w, {DIM_64, DIM_256}, DT_INT8);
    SetDataDesc(s, {DIM_64, DIM_1}, DT_BF16);
    SetDataDesc(o, {DIM_1, DIM_64}, DT_BF16);
    SetTransposeDescs(tw, {DIM_64, DIM_256}, DT_INT8);
    SetTransposeDescs(ts, {DIM_64, DIM_1}, DT_BF16);
    SetWqbmmv2Descs(wq, {DIM_128, DIM_256}, DT_BF16, {DIM_256, DIM_64}, DT_INT8, {DIM_1, DIM_64}, DT_BF16,
                    {DIM_1, DIM_64}, DT_BF16, true, {}, DT_FLOAT, false, {DIM_128, DIM_64}, DT_BF16);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto result = CheckFusionResult(graph);
    EXPECT_FALSE(result.hasTranspose);
    EXPECT_TRUE(result.hasWeightQuant);
    EXPECT_TRUE(result.transposeWeightAttr);
}
