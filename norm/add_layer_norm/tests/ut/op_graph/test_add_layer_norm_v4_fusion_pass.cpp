/*
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "../../../op_graph/fusion_pass/add_layer_norm_v4_fusion_pass.h"
#include "register/register_custom_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace ops;

class AddLayerNormV4FusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // set version
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static void InferShapeForTest(
        DataType dtype,
        Shape& shape_x, Shape& shape_gamma, Shape& shape_bias, Shape& shape_normalized_shape,
        es::EsTensorHolder& x1, es::EsTensorHolder& x2, es::EsTensorHolder& bias, es::EsTensorHolder& normalized_shape,
        es::EsTensorHolder& add1, es::EsTensorHolder& add2, es::EsTensorHolder& cast1,
        es::LayerNormV4Output& layernorm, es::EsTensorHolder& cast2)
    {
        //x1
        TensorDesc x1_output_desc;
        x1.GetProducer()->GetOutputDesc(0, x1_output_desc);
        x1_output_desc.SetDataType(dtype);
        x1_output_desc.SetShape(shape_x);
        x1.GetProducer()->UpdateOutputDesc(0, x1_output_desc);
        //x2
        TensorDesc x2_output_desc;
        x2.GetProducer()->GetOutputDesc(0, x2_output_desc);
        x2_output_desc.SetDataType(dtype);
        x2_output_desc.SetShape(shape_x);
        x2.GetProducer()->UpdateOutputDesc(0, x2_output_desc);
        //bias
        TensorDesc bias_output_desc;
        bias.GetProducer()->GetOutputDesc(0, bias_output_desc);
        bias_output_desc.SetDataType(dtype);
        bias_output_desc.SetShape(shape_bias);
        bias.GetProducer()->UpdateOutputDesc(0, bias_output_desc);
        //normalized_shape
        TensorDesc normalized_shape_output_desc;
        normalized_shape.GetProducer()->GetOutputDesc(0, normalized_shape_output_desc);
        normalized_shape_output_desc.SetDataType(DT_INT32);
        normalized_shape_output_desc.SetShape(shape_normalized_shape);
        normalized_shape.GetProducer()->UpdateOutputDesc(0, normalized_shape_output_desc);
        //add1
        TensorDesc add1_input_0_desc;
        add1.GetProducer()->GetInputDesc(0, add1_input_0_desc);
        add1_input_0_desc.SetDataType(dtype);
        add1_input_0_desc.SetShape(shape_x);
        add1.GetProducer()->UpdateInputDesc(0, add1_input_0_desc);
        TensorDesc add1_input_1_desc;
        add1.GetProducer()->GetInputDesc(1, add1_input_1_desc);
        add1_input_1_desc.SetDataType(dtype);
        add1_input_1_desc.SetShape(shape_bias);
        add1.GetProducer()->UpdateInputDesc(1, add1_input_1_desc);
        TensorDesc add1_output_desc;
        add1.GetProducer()->UpdateOutputDesc(0, add1_output_desc);
        add1_output_desc.SetDataType(dtype);
        add1_output_desc.SetShape(shape_x);
        add1.GetProducer()->UpdateOutputDesc(0, add1_output_desc);
        //add2
        TensorDesc add2_input_0_desc;
        add2.GetProducer()->GetInputDesc(0, add2_input_0_desc);
        add2_input_0_desc.SetDataType(dtype);
        add2_input_0_desc.SetShape(shape_x);
        add2.GetProducer()->UpdateInputDesc(0, add2_input_0_desc);
        TensorDesc add2_input_1_desc;
        add2.GetProducer()->GetInputDesc(1, add2_input_1_desc);
        add2_input_1_desc.SetDataType(dtype);
        add2_input_1_desc.SetShape(shape_x);
        add2.GetProducer()->UpdateInputDesc(1, add2_input_1_desc);
        TensorDesc add2_output_desc;
        add2.GetProducer()->UpdateOutputDesc(0, add2_output_desc);
        add2_output_desc.SetDataType(dtype);
        add2_output_desc.SetShape(shape_x);
        add2.GetProducer()->UpdateOutputDesc(0, add2_output_desc);
        //cast1
        TensorDesc cast1_input_0_desc;
        cast1.GetProducer()->GetInputDesc(0, cast1_input_0_desc);
        cast1_input_0_desc.SetDataType(dtype);
        cast1_input_0_desc.SetShape(shape_x);
        cast1.GetProducer()->UpdateInputDesc(0, cast1_input_0_desc);
        TensorDesc cast1_output_desc;
        cast1.GetProducer()->UpdateOutputDesc(0, cast1_output_desc);
        cast1_output_desc.SetDataType(DT_FLOAT);
        cast1_output_desc.SetShape(shape_x);
        cast1.GetProducer()->UpdateOutputDesc(0, cast1_output_desc);
        //layernorm
        TensorDesc layernorm_input_0_desc;
        layernorm.y.GetProducer()->GetInputDesc(0, layernorm_input_0_desc);
        layernorm_input_0_desc.SetDataType(dtype);
        layernorm_input_0_desc.SetShape(shape_x);
        layernorm.y.GetProducer()->UpdateInputDesc(0, layernorm_input_0_desc);
        TensorDesc layernorm_input_1_desc;
        layernorm.y.GetProducer()->GetInputDesc(1, layernorm_input_1_desc);
        layernorm_input_1_desc.SetDataType(DT_INT32);
        layernorm_input_1_desc.SetShape(shape_normalized_shape);
        layernorm_input_1_desc.SetFormat(FORMAT_NCHW);
        layernorm.y.GetProducer()->UpdateInputDesc(1, layernorm_input_1_desc);
        TensorDesc layernorm_output_0_desc;
        layernorm.y.GetProducer()->GetOutputDesc(0, layernorm_output_0_desc);
        layernorm_output_0_desc.SetDataType(DT_FLOAT);
        layernorm_output_0_desc.SetShape(shape_x);
        layernorm.y.GetProducer()->UpdateOutputDesc(0, layernorm_output_0_desc);
        //cast2
        TensorDesc cast2_input_0_desc;
        cast2.GetProducer()->GetInputDesc(0, cast2_input_0_desc);
        cast2_input_0_desc.SetDataType(DT_FLOAT);
        cast2_input_0_desc.SetShape(shape_x);
        cast2.GetProducer()->UpdateInputDesc(0, cast2_input_0_desc);
        TensorDesc cast2_output_desc;
        cast1.GetProducer()->UpdateOutputDesc(0, cast2_output_desc);
        cast2_output_desc.SetDataType(dtype);
        cast2_output_desc.SetShape(shape_x);
        cast1.GetProducer()->UpdateOutputDesc(0, cast2_output_desc);
    }

    bool IsAddLayerNormInputRight(
        GNode& node, Shape& shape_x, Shape& shape_gamma, Shape& shape_bias, DataType dtype, bool no_gamma, bool no_beta)
    {
        TensorDesc x1_desc;
        TensorDesc x2_desc;
        TensorDesc gamma_desc; //fill
        TensorDesc beta_desc;  //fill
        TensorDesc bias_desc;

        node.GetInputDesc(0, x1_desc);
        node.GetInputDesc(1, x2_desc);
        node.GetInputDesc(2, gamma_desc); //fill
        node.GetInputDesc(3, beta_desc);  //fill
        node.GetInputDesc(4, bias_desc);

        if (x1_desc.GetDataType() != x2_desc.GetDataType() ||
            x1_desc.GetDataType() != bias_desc.GetDataType() ||
            x1_desc.GetDataType() != dtype) {
            return false;
        }
        if (gamma_desc.GetDataType() != beta_desc.GetDataType() ||
            gamma_desc.GetDataType() != DT_FLOAT) {
            return false;
        }
        if (x1_desc.GetShape().GetShapeSize() != shape_x.GetShapeSize() ||
            bias_desc.GetShape().GetShapeSize() != shape_bias.GetShapeSize() ||
            x1_desc.GetShape().GetShapeSize() != x2_desc.GetShape().GetShapeSize()) {
            return false;
        }
        if (no_gamma) {
            if (no_beta) {
                return true; //nogamma & nobeta
            }
            if (beta_desc.GetShape().GetShapeSize() != shape_gamma.GetShapeSize() ||
                beta_desc.GetDataType() != DT_FLOAT) {
                return false; //nogamma
            }
        } else {
            if (gamma_desc.GetShape().GetShapeSize() != shape_gamma.GetShapeSize() ||
                gamma_desc.GetDataType() != DT_FLOAT) {
                return false;
            }
            if (!no_beta && gamma_desc.GetShape().GetShapeSize() != beta_desc.GetShape().GetShapeSize()) {
                return false;
            }
        }
        return true;
    }
};

//场景一
TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_95_FP16_OK)
{
    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape(dims_normalized);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test1_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);
    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);

    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(
                node, shape_x, shape_gamma, shape_bias, DT_FLOAT16, false, false)) {
            findAddLayerNorm = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm, true);
    EXPECT_EQ(node_count, 8);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_95_BF16_OK)
{
    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape(dims_normalized);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test2_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_BF16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_BF16);

    InferShapeForTest(
        DT_BF16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);
    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(
                node, shape_x, shape_gamma, shape_bias, DT_BF16, false, false)) {
            findAddLayerNorm = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm, true);
    EXPECT_EQ(node_count, 8);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_95_FP16_noBeta_OK)
{
    std::vector<int64_t> dims_x{16, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape(dims_normalized);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test3_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);
    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    bool findFill = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(
                node, shape_x, shape_gamma, shape_bias, DT_FLOAT16, false, true)) {
            findAddLayerNorm = true;
        }
        if (type == "Fill") {
            findFill = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm && findFill, true);
    EXPECT_EQ(node_count, 9);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_95_FP16_noGammaBeta_OK)
{
    std::vector<int64_t> dims_x{16, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape(dims_normalized);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test4_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    bool findFill = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(
                node, shape_x, shape_gamma, shape_bias, DT_FLOAT16, true, true)) {
            findAddLayerNorm = true;
        }
        if (type == "Fill") {
            findFill = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm && findFill, true);
    EXPECT_EQ(node_count, 10);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_95_FP16_noGamma_OK)
{
    std::vector<int64_t> dims_x{16, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape(dims_normalized);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test5_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto beta = graph_builder.CreateInput(4, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, nullptr, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    bool findFill = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(
                node, shape_x, shape_gamma, shape_bias, DT_FLOAT16, true, false)) {
            findAddLayerNorm = true;
        }
        if (type == "Fill") {
            findFill = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm && findFill, true);
    EXPECT_EQ(node_count, 9);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_platform_not_95_Fail)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_93";
    optiCompilationInfo.soc_version = "Ascend910_93";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape(dims_normalized);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test6_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);
    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_BegiNnormAxis_wrong_Fail)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    std::vector<int64_t> dims_normalized{2};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({2});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test7_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{2,128}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);
    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_AddInputShapeIsDynamic_Fail)
{
    std::vector<int64_t> dims_x{-1, 128, 128, -1};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test8_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);
    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_AddInputIsScaler_Fail)
{
    std::vector<int64_t> dims_x;
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test9_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);
    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_Add1InputsShapeLastDimNotSame_Fail)
{
    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 128};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test10_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);
    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_BiasDimBiggerThanXDim_Fail)
{
    std::vector<int64_t> dims_x{2, 32, 2};
    std::vector<int64_t> dims_gamma{2};
    std::vector<int64_t> dims_bias{2, 2, 2, 2};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test11_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{2}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);
    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_Add1InputsDtypeInvaild_Fail)
{
    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test12_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);
    //x1
    TensorDesc x1_output_desc;
    x1.GetProducer()->GetOutputDesc(0, x1_output_desc);
    x1_output_desc.SetDataType(DT_FLOAT16);
    x1_output_desc.SetShape(shape_x);
    x1.GetProducer()->UpdateOutputDesc(0, x1_output_desc);
    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //add2
    TensorDesc add2_input_0_desc;
    add2.GetProducer()->GetInputDesc(0, add2_input_0_desc);
    add2_input_0_desc.SetDataType(DT_FLOAT16);
    add2_input_0_desc.SetShape(shape_x);
    add2.GetProducer()->UpdateInputDesc(0, add2_input_0_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_Add1InputsDtypeNotSame_Fail)
{
    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test13_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_BF16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);

    //bias
    TensorDesc bias_output_desc;
    bias.GetProducer()->GetOutputDesc(0, bias_output_desc);
    bias_output_desc.SetDataType(DT_BF16);
    bias_output_desc.SetShape(shape_bias);
    bias.GetProducer()->UpdateOutputDesc(0, bias_output_desc);
    //add1
    TensorDesc add1_input_1_desc;
    add1.GetProducer()->GetInputDesc(1, add1_input_1_desc);
    add1_input_1_desc.SetDataType(DT_BF16);
    add1_input_1_desc.SetShape(shape_bias);
    add1.GetProducer()->UpdateInputDesc(1, add1_input_1_desc);
    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_x1DtypeNotSameWithx2Dtype_Fail)
{
    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test14_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);
    //x1
    TensorDesc x1_output_desc;
    x1.GetProducer()->GetOutputDesc(0, x1_output_desc);
    x1_output_desc.SetDataType(DT_BF16);
    x1_output_desc.SetShape(shape_x);
    x1.GetProducer()->UpdateOutputDesc(0, x1_output_desc);
    //add2
    TensorDesc add2_input_0_desc;
    add2.GetProducer()->GetInputDesc(0, add2_input_0_desc);
    add2_input_0_desc.SetDataType(DT_BF16);
    add2_input_0_desc.SetShape(shape_x);
    add2.GetProducer()->UpdateInputDesc(0, add2_input_0_desc);
    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_x1ShapeNotSameWithx2Shape_Fail)
{
    std::vector<int64_t> dims_x1{1, 128, 128, 256};
    std::vector<int64_t> dims_x2{1, 128, 64, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x1(dims_x1);
    Shape shape_x2(dims_x2);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test15_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x1.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x2.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x2, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);
    //x1
    TensorDesc x1_output_desc;
    x1.GetProducer()->GetOutputDesc(0, x1_output_desc);
    x1_output_desc.SetDataType(DT_FLOAT16);
    x1_output_desc.SetShape(shape_x1);
    x1.GetProducer()->UpdateOutputDesc(0, x1_output_desc);
    //add2
    TensorDesc add2_input_0_desc;
    add2.GetProducer()->GetInputDesc(0, add2_input_0_desc);
    add2_input_0_desc.SetDataType(DT_FLOAT16);
    add2_input_0_desc.SetShape(shape_x1);
    add2.GetProducer()->UpdateInputDesc(0, add2_input_0_desc);
    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_Cast1HaveCtrlEdge_Fail)
{
    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test16_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);

    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    graph->AddControlEdge(*add2.GetProducer(), *cast1.GetProducer());
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

//场景二
TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_S2_95_FP16_OK)
{
    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{128, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test17_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto add3 = es::Add(cast1, cast1);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);

    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(
                node, shape_x, shape_gamma, shape_bias, DT_FLOAT16, false, false)) {
            findAddLayerNorm = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm, true);
    EXPECT_EQ(node_count, 10);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_S2_95_BF16_OK)
{
    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test18_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_BF16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto add3 = es::Add(cast1, cast1);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_BF16);

    InferShapeForTest(
        DT_BF16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);

    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(
                node, shape_x, shape_gamma, shape_bias, DT_BF16, false, false)) {
            findAddLayerNorm = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm, true);
    EXPECT_EQ(node_count, 10);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_S2_95_FP16_noBeta_OK)
{
    std::vector<int64_t> dims_x{16, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test19_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto add3 = es::Add(cast1, cast1);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);

    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    bool findFill = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(
                node, shape_x, shape_gamma, shape_bias, DT_FLOAT16, false, true)) {
            findAddLayerNorm = true;
        }
        if (type == "Fill") {
            findFill = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm && findFill, true);
    EXPECT_EQ(node_count, 11);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_S2_95_FP16_noGammaBeta_OK)
{
    std::vector<int64_t> dims_x{16, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test20_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());

    //auto normalized_shape = graph_builder.CreateConst(std::vector<int64_t>{256}, dims_normalized);
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto add3 = es::Add(cast1, cast1);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    bool findFill = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(
                node, shape_x, shape_gamma, shape_bias, DT_FLOAT16, true, true)) {
            findAddLayerNorm = true;
        }
        if (type == "Fill") {
            findFill = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm && findFill, true);
    EXPECT_EQ(node_count, 12);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_95_S2_FP16_noGamma_OK)
{
    std::vector<int64_t> dims_x{16, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test21_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_NCHW, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_NCHW, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_NCHW, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_NCHW, shape_normalized_shape.GetDims());
    auto beta = graph_builder.CreateInput(4, "beta", DT_FLOAT, FORMAT_NCHW, shape_gamma.GetDims());

    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto add3 = es::Add(cast1, cast1);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, nullptr, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);

    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    bool findFill = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(
                node, shape_x, shape_gamma, shape_bias, DT_FLOAT16, true, false)) {
            findAddLayerNorm = true;
        }
        if (type == "Fill") {
            findFill = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm && findFill, true);
    EXPECT_EQ(node_count, 11);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_MultiQuoteAdd2_95_FP16_OK)
{
    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{128, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test22_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto multiquote = es::Add(add2, add2);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);

    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(
                node, shape_x, shape_gamma, shape_bias, DT_FLOAT16, false, false)) {
            findAddLayerNorm = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm, true);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_ReverseAddOrder_95_FP16_OK)
{
    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{128, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test23_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    auto add1 = bias + x2;
    auto add2 = add1 + x1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);

    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(
                node, shape_x, shape_gamma, shape_bias, DT_FLOAT16, false, false)) {
            findAddLayerNorm = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm, true);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_AddLayerNormFirstOutputMultiQuote_95_FP16_Fail)
{
    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{128, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test24_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);
    auto multiquote_y = es::Add(layernorm.y, layernorm.y);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);

    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormV4FusionPassTest, add_layer_normV4_fusion_Add1MultiQuote_95_FP16_Fail)
{
    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{128, 256};
    std::vector<int64_t> dims_normalized{1};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);
    Shape shape_normalized_shape({1});

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_v4_fusion_test25_graph");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto bias = graph_builder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        3, "normalized_shape", DT_INT32, FORMAT_ND, shape_normalized_shape.GetDims());
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(5, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());

    auto add1 = x2 + bias;
    auto multiquote_add1 = es::Add(add1, add1);
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, shape_normalized_shape,
        x1, x2, bias, normalized_shape, add1, add2, cast1, layernorm, cast2);

    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    //beta
    TensorDesc beta_output_desc;
    beta.GetProducer()->GetOutputDesc(0, beta_output_desc);
    beta_output_desc.SetDataType(DT_FLOAT);
    beta_output_desc.SetShape(shape_gamma);
    beta.GetProducer()->UpdateOutputDesc(0, beta_output_desc);
    //layernorm
    TensorDesc layernorm_input_2_desc;
    layernorm.y.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
    layernorm_input_2_desc.SetDataType(DT_FLOAT);
    layernorm_input_2_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
    TensorDesc layernorm_input_3_desc;
    layernorm.y.GetProducer()->GetInputDesc(3, layernorm_input_3_desc);
    layernorm_input_3_desc.SetDataType(DT_FLOAT);
    layernorm_input_3_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(3, layernorm_input_3_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormV4FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}