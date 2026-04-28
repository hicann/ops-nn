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
#include "../../../op_graph/fusion_pass/add_layer_norm_fusion_pass.h"
#include "register/register_custom_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace ops;

class AddLayerNormFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // set version
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static void InferShapeForTest(
        DataType dtype, Shape& shape_x, Shape& shape_gamma, Shape& shape_bias,
        es::EsTensorHolder& x1, es::EsTensorHolder& x2, es::EsTensorHolder& bias, es::EsTensorHolder& gamma,
        es::EsTensorHolder& beta, es::EsTensorHolder& add1, es::EsTensorHolder& add2, es::EsTensorHolder& cast1,
        es::EsTensorHolder& layernorm, es::EsTensorHolder& cast2)
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
        add1.GetProducer()->GetOutputDesc(0, add1_output_desc);
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
        add2.GetProducer()->GetOutputDesc(0, add2_output_desc);
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
        cast1.GetProducer()->GetOutputDesc(0, cast1_output_desc);
        cast1_output_desc.SetDataType(DT_FLOAT);
        cast1_output_desc.SetShape(shape_x);
        cast1.GetProducer()->UpdateOutputDesc(0, cast1_output_desc);
        //layernorm
        TensorDesc layernorm_input_0_desc;
        layernorm.GetProducer()->GetInputDesc(0, layernorm_input_0_desc);
        layernorm_input_0_desc.SetDataType(DT_FLOAT);
        layernorm_input_0_desc.SetShape(shape_x);
        layernorm.GetProducer()->UpdateInputDesc(0, layernorm_input_0_desc);
        TensorDesc layernorm_input_1_desc;
        layernorm.GetProducer()->GetInputDesc(1, layernorm_input_1_desc);
        layernorm_input_1_desc.SetDataType(DT_FLOAT);
        layernorm_input_1_desc.SetShape(shape_gamma);
        layernorm.GetProducer()->UpdateInputDesc(1, layernorm_input_1_desc);
        TensorDesc layernorm_input_2_desc;
        layernorm.GetProducer()->GetInputDesc(2, layernorm_input_2_desc);
        layernorm_input_2_desc.SetDataType(DT_FLOAT);
        layernorm_input_2_desc.SetShape(shape_gamma);
        layernorm.GetProducer()->UpdateInputDesc(2, layernorm_input_2_desc);
        TensorDesc layernorm_output_0_desc;
        layernorm.GetProducer()->GetOutputDesc(0, layernorm_output_0_desc);
        layernorm_output_0_desc.SetDataType(DT_FLOAT);
        layernorm_output_0_desc.SetShape(shape_x);
        layernorm.GetProducer()->UpdateOutputDesc(0, layernorm_output_0_desc);
        //cast2
        TensorDesc cast2_input_0_desc;
        cast2.GetProducer()->GetInputDesc(0, cast2_input_0_desc);
        cast2_input_0_desc.SetDataType(DT_FLOAT);
        cast2_input_0_desc.SetShape(shape_x);
        cast2.GetProducer()->UpdateInputDesc(0, cast2_input_0_desc);
        TensorDesc cast2_output_desc;
        cast1.GetProducer()->GetOutputDesc(0, cast2_output_desc);
        cast2_output_desc.SetDataType(dtype);
        cast2_output_desc.SetShape(shape_x);
        cast1.GetProducer()->UpdateOutputDesc(0, cast2_output_desc);
    }

    bool IsAddLayerNormInputRight(GNode& node, Shape& shape_x, Shape& shape_gamma, Shape& shape_bias, DataType dtype)
    {
        TensorDesc x1_desc;
        TensorDesc x2_desc;
        TensorDesc gamma_desc;
        TensorDesc beta_desc;
        TensorDesc bias_desc;

        node.GetInputDesc(0, x1_desc);
        node.GetInputDesc(1, x2_desc);
        node.GetInputDesc(2, gamma_desc);
        node.GetInputDesc(3, beta_desc);
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
            gamma_desc.GetShape().GetShapeSize() != shape_gamma.GetShapeSize() ||
            bias_desc.GetShape().GetShapeSize() != shape_bias.GetShapeSize() ||
            x1_desc.GetShape().GetShapeSize() != x2_desc.GetShape().GetShapeSize() ||
            gamma_desc.GetShape().GetShapeSize() != beta_desc.GetShape().GetShapeSize()) {
            return false;
        }
        return true;
    }
};

TEST_F(AddLayerNormFusionPassTest, add_layer_norm_fusion_93_fp_OK)
{
    std::vector<int64_t> dims_x{1, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{256};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());

    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNorm(cast1, gamma, beta, 2, 0, 1e-5);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.variance});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);

    bool findAddLayerNorm = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(node, shape_x, shape_gamma, shape_bias, DT_FLOAT16)) {
            findAddLayerNorm = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm, true);
    EXPECT_EQ(node_count, 7);
}

TEST_F(AddLayerNormFusionPassTest, add_layer_normV3_fusion_93_bf16_OK)
{
    std::vector<int64_t> dims_x{2, 8, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_BF16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV3(cast1, gamma, beta, -1, 0, 1e-5);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_BF16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(node, shape_x, shape_gamma, shape_bias, DT_BF16)) {
            findAddLayerNorm = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm, true);
    EXPECT_EQ(node_count, 7);
}

TEST_F(AddLayerNormFusionPassTest, add_layer_norm_fusion_95_fp_OK)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> dims_x{1, 128, 1024};
    std::vector<int64_t> dims_gamma{1024};
    std::vector<int64_t> dims_bias{64, 1024};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNorm(cast1, gamma, beta, -1, 0, 1e-5);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.variance});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(node, shape_x, shape_gamma, shape_bias, DT_FLOAT16)) {
            findAddLayerNorm = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm, true);
    EXPECT_EQ(node_count, 7);
}

TEST_F(AddLayerNormFusionPassTest, add_layer_normV3_fusion_xLastDimNotEqualBias_Fail)
{
    std::vector<int64_t> dims_x{1, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{128};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV3(cast1, gamma, beta, 2, 0, 1e-5);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormFusionPassTest, add_layer_normV3_fusion_xDTypeAndBiasDTypeNotSame_Fail)
{
    std::vector<int64_t> dims_x{1, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{256};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_BF16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV3(cast1, gamma, beta, -1, 0, 1e-5);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

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

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormFusionPassTest, add_layer_norm_fusion_x1DTypeAndx2DTypeNotSame_Fail)
{
    std::vector<int64_t> dims_x{1, 2, 128, 128};
    std::vector<int64_t> dims_gamma{128};
    std::vector<int64_t> dims_bias{2, 128};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNorm(cast1, gamma, beta, 3, 0, 1e-5);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

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

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.variance});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormFusionPassTest, add_layer_norm_fusion_GammaDTypeNotfp32_Fail)
{
    std::vector<int64_t> dims_x{1, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{256};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT16, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT16, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNorm(cast1, gamma, beta, 2, 0, 1e-5);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    //gamma
    TensorDesc gamma_output_desc;
    gamma.GetProducer()->GetOutputDesc(0, gamma_output_desc);
    gamma_output_desc.SetDataType(DT_FLOAT16);
    gamma_output_desc.SetShape(shape_gamma);
    gamma.GetProducer()->UpdateOutputDesc(0, gamma_output_desc);
    TensorDesc layernorm_input_1_desc;
    layernorm.y.GetProducer()->GetInputDesc(1, layernorm_input_1_desc);
    layernorm_input_1_desc.SetDataType(DT_FLOAT16);
    layernorm_input_1_desc.SetShape(shape_gamma);
    layernorm.y.GetProducer()->UpdateInputDesc(1, layernorm_input_1_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.variance});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormFusionPassTest, add_layer_norm_fusion_cast1HaveCtlEdge_Fail)
{
    std::vector<int64_t> dims_x{1, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{256};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNorm(cast1, gamma, beta, 2, 0, 1e-5);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.variance});
    graph->AddControlEdge(*add2.GetProducer(), *cast1.GetProducer());
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormFusionPassTest, add_layer_norm_fusion_AddInputShapeIsDynamic_Fail)
{
    std::vector<int64_t> dims_x{-1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV3(cast1, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormFusionPassTest, add_layer_norm_fusion_AddInputIsScaler_Fail)
{
    std::vector<int64_t> dims_x;
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV3(cast1, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormFusionPassTest, add_layer_normV4_fusion_BegiNnormAxis_wrong_Fail)
{
    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto normalized_shape = graph_builder.CreateInput(
        5, "normalized_shape", DT_INT32, FORMAT_ND, shape_gamma.GetDims());
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV3(cast1, normalized_shape, gamma, 2);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddLayerNormFusionPassTest, add_layer_norm_fusion_PlatformNotRight_Fail)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> dims_x{1, 128, 128, 256};
    std::vector<int64_t> dims_gamma{256};
    std::vector<int64_t> dims_bias{2, 256};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto layernorm = es::LayerNormV3(cast1, gamma, beta);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

//场景2
TEST_F(AddLayerNormFusionPassTest, for2_add_layer_norm_fusion_95_fp_OK)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> dims_x{1, 128, 1024};
    std::vector<int64_t> dims_gamma{1024};
    std::vector<int64_t> dims_bias{64, 1024};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_FLOAT16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto add3 = es::Add(cast1, cast1);
    auto layernorm = es::LayerNorm(cast1, gamma, beta, -1, 0, 1e-5);
    auto cast2 = es::Cast(layernorm.y, DT_FLOAT16);

    InferShapeForTest(
        DT_FLOAT16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.variance, add3});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(node, shape_x, shape_gamma, shape_bias, DT_FLOAT16)) {
            findAddLayerNorm = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm, true);
    EXPECT_EQ(node_count, 9);
}

TEST_F(AddLayerNormFusionPassTest, for2_add_layer_normV3_fusion_93_bf_OK)
{
    std::vector<int64_t> dims_x{1, 128, 1024};
    std::vector<int64_t> dims_gamma{1024};
    std::vector<int64_t> dims_bias{1024};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_BF16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto add3 = es::Add(cast1, cast1);
    auto layernorm = es::LayerNormV3(cast1, gamma, beta, -1, 0, 1e-5);
    auto cast2 = es::Cast(layernorm.y, DT_BF16);

    InferShapeForTest(
        DT_BF16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd, add3});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(node, shape_x, shape_gamma, shape_bias, DT_BF16)) {
            findAddLayerNorm = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm, true);
    EXPECT_EQ(node_count, 9);
}

TEST_F(AddLayerNormFusionPassTest, for2_add_layer_normV3_fusion_MultiQuoteAdd2_93_bf_OK)
{
    std::vector<int64_t> dims_x{1, 128, 1024};
    std::vector<int64_t> dims_gamma{1024};
    std::vector<int64_t> dims_bias{1024};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_BF16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto multiquote_add2 = es::Add(add2, add2);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto add3 = es::Add(cast1, cast1);
    auto layernorm = es::LayerNormV3(cast1, gamma, beta, -1, 0, 1e-5);
    auto cast2 = es::Cast(layernorm.y, DT_BF16);

    InferShapeForTest(
        DT_BF16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd, add3});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(node, shape_x, shape_gamma, shape_bias, DT_BF16)) {
            findAddLayerNorm = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm, true);
}

TEST_F(AddLayerNormFusionPassTest, for2_add_layer_normV3_fusion_ReverseAddOrder_93_bf_OK)
{
    std::vector<int64_t> dims_x{1, 128, 1024};
    std::vector<int64_t> dims_gamma{1024};
    std::vector<int64_t> dims_bias{1024};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_BF16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = es::Add(bias, x2);
    auto add2 = es::Add(add1, x1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto add3 = es::Add(cast1, cast1);
    auto layernorm = es::LayerNormV3(cast1, gamma, beta, -1, 0, 1e-5);
    auto cast2 = es::Cast(layernorm.y, DT_BF16);

    InferShapeForTest(
        DT_BF16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd, add3});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    bool findAddLayerNorm = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "AddLayerNorm" && IsAddLayerNormInputRight(node, shape_x, shape_gamma, shape_bias, DT_BF16)) {
            findAddLayerNorm = true;
        }
    }
    EXPECT_EQ(findAddLayerNorm, true);
}

TEST_F(AddLayerNormFusionPassTest, for2_add_layer_normV3_fusion_AddLayerNormFirstOutputMultiQuote_95_FP16_Fail)
{
    std::vector<int64_t> dims_x{1, 128, 1024};
    std::vector<int64_t> dims_gamma{1024};
    std::vector<int64_t> dims_bias{1024};
    Shape shape_x(dims_x);
    Shape shape_gamma(dims_gamma);
    Shape shape_bias(dims_bias);

    auto graph_builder = es::EsGraphBuilder("add_layer_norm_fusion_third_patten");
    auto x1 = graph_builder.CreateInput(0, "x1", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graph_builder.CreateInput(1, "x2", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto gamma = graph_builder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto beta = graph_builder.CreateInput(3, "beta", DT_FLOAT, FORMAT_ND, shape_gamma.GetDims());
    auto bias = graph_builder.CreateInput(4, "bias", DT_BF16, FORMAT_ND, shape_bias.GetDims());
    auto add1 = es::Add(x2, bias);
    auto add2 = es::Add(x1, add1);
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto add3 = es::Add(cast1, cast1);
    auto layernorm = es::LayerNormV3(cast1, gamma, beta, -1, 0, 1e-5);
    auto cast2 = es::Cast(layernorm.y, DT_BF16);
    auto multiquote_y = es::Add(layernorm.y, layernorm.y);

    InferShapeForTest(
        DT_BF16, shape_x, shape_gamma, shape_bias, x1, x2, bias, gamma, beta,
        add1, add2, cast1, layernorm.y, cast2);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({cast2, layernorm.mean, layernorm.rstd, add3});
    CustomPassContext pass_contex;
    ops::AddLayerNormFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}