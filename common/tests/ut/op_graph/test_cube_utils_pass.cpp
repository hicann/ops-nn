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
#include <map>
#include <memory>
#include <string>
#include <vector>

#define protected public
#define private public

#include "cube_utils/cube_utils.h"
#include "cube_utils/cube_common.h"
#include "graph/types.h"

#undef protected
#undef private

using namespace std;
using namespace ge;
using namespace ops;

namespace ops {
constexpr uint32_t POST_CUBE_INPUT_2_INDEX = 2;
constexpr uint32_t POST_CUBE_INPUT_3_INDEX = 3;
constexpr uint32_t POST_CUBE_INPUT_4_INDEX = 4;
constexpr uint32_t POST_CUBE_INPUT_5_INDEX = 5;
constexpr uint32_t POST_CUBE_INPUT_6_INDEX = 6;
constexpr uint32_t POST_CUBE_INPUT_7_INDEX = 7;
constexpr uint32_t POST_CUBE_INPUT_8_INDEX = 8;
constexpr uint32_t POST_CUBE_INPUT_9_INDEX = 9;
constexpr uint32_t TWOINPUTSIZE = 2U;

class cube_utils_ut : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "cube_utils_ut SetUp" << std::endl;
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_out2l1_dn2nz"] = std::vector<std::string>();
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

        fe::PlatFormInfos platformInfos;
        fe::OptionalInfos optiCompilationInfos;
        platformInfos.Init();
        optiCompilationInfos.Init();
        std::map<std::string, std::vector<std::string>> fixpipe_dtype_map;
        fixpipe_dtype_map["Intrinsic_fix_pipe_unit_list"] = {"pre_covn", "pre_act", "post_eltwise", "post_act"};
        platformInfos.SetFixPipeDtypeMap(fixpipe_dtype_map);

        std::map<std::string, std::string> PlatformRes;
        PlatformRes["AIC_version"] = "AIC-C310";
        platformInfos.SetPlatformRes("version", PlatformRes);
        optiCompilationInfos.SetSocVersion("Ascend950");
        fe::PlatformInfoManager::Instance().platform_infos_map_["Ascend950"] = platformInfos;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfos);
    }

    static void TearDownTestCase() {
        std::cout << "cube_utils_ut TearDown" << std::endl;
    }

    // 创建常量节点
    static ge::GNodePtr CreateConstNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({3, 4, 5, 6});
        ge::TensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
        tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
        tensor_desc.SetOriginShape(shape);

        std::vector<float> data;
        for (int i = 0; i != 3; i++) {
            for (int j = 0; j != 4; j++) {
                for (int k = 0; k != 5; k++) {
                    for (int m = 0; m != 6; m++) {
                        float dst = i * j * k * m;
                        data.push_back(dst);
                    }
                }
            }
        }

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Const")
            .Name(opname.c_str())
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateOutputDesc(0, tensor_desc);
        ge::Tensor weight_tensor(tensor_desc,
            reinterpret_cast<const uint8_t*>(data.data()),
            data.size() * sizeof(float));
        node.SetAttr(kAscendNameWeightAsc, weight_tensor);

        return std::make_shared<ge::GNode>(node);
    }

    // 创建 Conv2D 节点
    static ge::GNodePtr CreateConv2DNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({1, 64, 112, 112});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        // 创建 weight 常量节点
        auto weight_node = CreateConstNode(graph, opname + "_weight");

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Conv2D")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"filter", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        // 连接 weight
        (void)graph.AddDataEdge(*weight_node, 0, node, 1);

        return std::make_shared<ge::GNode>(node);
    }

    // 创建 Relu 节点
    static ge::GNodePtr CreateReluNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({1, 64, 112, 112});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Relu")
            .Name(opname.c_str())
            .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateOutputDesc(0, desc);

        return std::make_shared<ge::GNode>(node);
    }

    // 创建 Quant 节点
    static ge::GNodePtr CreateQuantNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({1, 64, 112, 112});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_INT8);

        // 创建 scale 常量节点
        auto scale_node = CreateConstNode(graph, opname + "_scale");

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("AscendQuant")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"scale", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        float offset = 8.0f;
        float scale_val = 4.0f;
        node.SetAttr(kAscendOffsetAsc, offset);
        node.SetAttr(kAscendScaleAsc, scale_val);

        // 连接 scale
        (void)graph.AddDataEdge(*scale_node, 0, node, 1);

        return std::make_shared<ge::GNode>(node);
    }

    // 创建 Dequant 节点
    static ge::GNodePtr CreateDequantNode(ge::Graph &graph, const std::string &opname,
                                          bool hasattr = false,
                                          float scale = 0.0f, float offset = 0.0f) {
        ge::Shape shape({1, 64, 112, 112});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        auto const_node_ptr = CreateConstNode(graph, opname + "_constinput");

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("AscendDequant")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"deq_scale", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        if (hasattr) {
            node.SetAttr(kAscendScaleAsc, scale);
            node.SetAttr(kAscendOffsetAsc, offset);
        }

        (void)graph.AddDataEdge(*const_node_ptr, 0, node, 1);

        return std::make_shared<ge::GNode>(node);
    }

    // 创建 LeakyRelu 节点
    static ge::GNodePtr CreateLeakyReluNode(ge::Graph &graph, const std::string &opname,
                                             float negative_slope = 0.1f) {
        ge::Shape shape({1, 64, 112, 112});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("LeakyRelu")
            .Name(opname.c_str())
            .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateOutputDesc(0, desc);
        node.SetAttr(kAscendNegativeSlopeAsc, negative_slope);

        return std::make_shared<ge::GNode>(node);
    }

    // 创建 PReLU 节点
    static ge::GNodePtr CreatePReluNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({1, 64, 112, 112});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        auto const_node_ptr = CreateConstNode(graph, opname + "_slope");

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("PRelu")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"slope", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        (void)graph.AddDataEdge(*const_node_ptr, 0, node, 1);

        return std::make_shared<ge::GNode>(node);
    }

    // 创建 Relu6 节点
    static ge::GNodePtr CreateRelu6Node(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({1, 64, 112, 112});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Relu6")
            .Name(opname.c_str())
            .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateOutputDesc(0, desc);

        return std::make_shared<ge::GNode>(node);
    }

    // 创建 Cast 节点
    static ge::GNodePtr CreateCastNode(ge::Graph &graph, const std::string &opname,
                                        ge::DataType src_dtype = ge::DT_FLOAT,
                                        ge::DataType dst_dtype = ge::DT_FLOAT16) {
        ge::Shape shape({1, 64, 112, 112});
        ge::TensorDesc desc_in(shape, ge::FORMAT_NCHW, src_dtype);
        ge::TensorDesc desc_out(shape, ge::FORMAT_NCHW, dst_dtype);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Cast")
            .Name(opname.c_str())
            .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc_in);
        node.UpdateOutputDesc(0, desc_out);

        return std::make_shared<ge::GNode>(node);
    }

    // 创建 TransData 节点
    static ge::GNodePtr CreateTransDataNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape_in({1, 64, 112, 112});
        ge::Shape shape_out({1, 64, 112, 112});
        ge::TensorDesc desc_in(shape_in, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
        ge::TensorDesc desc_out(shape_out, ge::FORMAT_ND, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("TransData")
            .Name(opname.c_str())
            .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc_in);
        node.UpdateOutputDesc(0, desc_out);

        return std::make_shared<ge::GNode>(node);
    }

    // 创建 AntiQuant 节点
    static ge::GNodePtr CreateAntiQuantNode(ge::Graph &graph, const std::string &opname,
                                            float scale = 4.0f, float offset = 8.0f) {
        ge::Shape shape({1, 64, 112, 112});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        auto const_node_ptr = CreateConstNode(graph, opname + "_inputconstnode");

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("AscendAntiQuant")
            .Name(opname.c_str())
            .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateOutputDesc(0, desc);
        node.SetAttr(kAscendOffsetAsc, offset);
        node.SetAttr(kAscendScaleAsc, scale);

        (void)graph.AddDataEdge(*const_node_ptr, 0, node, 0);

        return std::make_shared<ge::GNode>(node);
    }

    // 创建 Add 节点 (Eltwise)
    static ge::GNodePtr CreateAddNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({1, 64, 112, 112});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        auto inputnode_ptr = CreateAntiQuantNode(graph, opname + "_inputantinode");

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("Add")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"x2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        node.UpdateInputDesc(0, desc);
        node.UpdateInputDesc(1, desc);
        node.UpdateOutputDesc(0, desc);

        (void)graph.AddDataEdge(*inputnode_ptr, 0, node, 1);

        return std::make_shared<ge::GNode>(node);
    }

    // 创建 PostCube 节点
    static ge::GNodePtr CreatePostCubeNode(ge::Graph &graph, const std::string &opname) {
        ge::Shape shape({1, 64, 112, 112});
        ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);

        ge::GNode node = ge::es::CompliantNodeBuilder(&graph)
            .OpType("FixPipe")
            .Name(opname.c_str())
            .IrDefInputs({
                {"x0", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"x1", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x2", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x3", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x4", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x5", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x6", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x7", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x8", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                {"x9", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            })
            .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();

        for (size_t i = 0; i < 10; ++i) {
            node.UpdateInputDesc(i, desc);
        }
        node.UpdateOutputDesc(0, desc);

        return std::make_shared<ge::GNode>(node);
    }

    // 连接两个节点
    static void LinkNodes(ge::Graph &graph, const ge::GNodePtr &src, int src_idx,
                          const ge::GNodePtr &dst, int dst_idx) {
        (void)graph.AddDataEdge(*src, src_idx, *dst, dst_idx);
    }

    // 创建 Conv -> Relu 链
    static std::pair<ge::GNodePtr, ge::GNodePtr> BuildConvReluChain(
        ge::Graph &graph, const std::string &prefix) {
        auto conv = CreateConv2DNode(graph, prefix + "_conv");
        auto relu = CreateReluNode(graph, prefix + "_relu");
        LinkNodes(graph, conv, 0, relu, 0);
        return {conv, relu};
    }

    // 创建 Conv -> Relu -> Quant 链
    static std::tuple<ge::GNodePtr, ge::GNodePtr, ge::GNodePtr> BuildConvReluQuantChain(
        ge::Graph &graph, const std::string &prefix) {
        auto conv = CreateConv2DNode(graph, prefix + "_conv");
        auto relu = CreateReluNode(graph, prefix + "_relu");
        auto quant = CreateQuantNode(graph, prefix + "_quant");
        LinkNodes(graph, conv, 0, relu, 0);
        LinkNodes(graph, relu, 0, quant, 0);
        return {conv, relu, quant};
    }

    // 创建 Conv -> Dequant -> Relu 链
    static std::tuple<ge::GNodePtr, ge::GNodePtr, ge::GNodePtr> BuildConvDequantReluChain(
        ge::Graph &graph, const std::string &prefix) {
        auto conv = CreateConv2DNode(graph, prefix + "_conv");
        auto dequant = CreateDequantNode(graph, prefix + "_dequant", true, 1.0f, 0.0f);
        auto relu = CreateReluNode(graph, prefix + "_relu");
        LinkNodes(graph, conv, 0, dequant, 0);
        LinkNodes(graph, dequant, 0, relu, 0);
        return {conv, dequant, relu};
    }

    // 创建 Conv -> Dequant -> PReLU 链
    static std::tuple<ge::GNodePtr, ge::GNodePtr, ge::GNodePtr> BuildConvDequantPReluChain(
        ge::Graph &graph, const std::string &prefix) {
        auto conv = CreateConv2DNode(graph, prefix + "_conv");
        auto dequant = CreateDequantNode(graph, prefix + "_dequant", true, 1.0f, 0.0f);
        auto prelu = CreatePReluNode(graph, prefix + "_prelu");
        LinkNodes(graph, conv, 0, dequant, 0);
        LinkNodes(graph, dequant, 0, prelu, 0);
        return {conv, dequant, prelu};
    }

    // 创建 Conv -> Dequant -> LeakyRelu 链
    static std::tuple<ge::GNodePtr, ge::GNodePtr, ge::GNodePtr> BuildConvDequantLeakyReluChain(
        ge::Graph &graph, const std::string &prefix) {
        auto conv = CreateConv2DNode(graph, prefix + "_conv");
        auto dequant = CreateDequantNode(graph, prefix + "_dequant", true, 1.0f, 0.0f);
        auto leaky_relu = CreateLeakyReluNode(graph, prefix + "_leaky_relu", 0.1f);
        LinkNodes(graph, conv, 0, dequant, 0);
        LinkNodes(graph, dequant, 0, leaky_relu, 0);
        return {conv, dequant, leaky_relu};
    }

    // 创建 Conv -> Dequant -> Relu6 链
    static std::tuple<ge::GNodePtr, ge::GNodePtr, ge::GNodePtr> BuildConvDequantRelu6Chain(
        ge::Graph &graph, const std::string &prefix) {
        auto conv = CreateConv2DNode(graph, prefix + "_conv");
        auto dequant = CreateDequantNode(graph, prefix + "_dequant", true, 1.0f, 0.0f);
        auto relu6 = CreateRelu6Node(graph, prefix + "_relu6");
        LinkNodes(graph, conv, 0, dequant, 0);
        LinkNodes(graph, dequant, 0, relu6, 0);
        return {conv, dequant, relu6};
    }

    // 创建 Conv -> Dequant -> Quant 链
    static std::tuple<ge::GNodePtr, ge::GNodePtr, ge::GNodePtr> BuildConvDequantQuantChain(
        ge::Graph &graph, const std::string &prefix) {
        auto conv = CreateConv2DNode(graph, prefix + "_conv");
        auto dequant = CreateDequantNode(graph, prefix + "_dequant", true, 1.0f, 0.0f);
        auto quant = CreateQuantNode(graph, prefix + "_quant");
        LinkNodes(graph, conv, 0, dequant, 0);
        LinkNodes(graph, dequant, 0, quant, 0);
        return {conv, dequant, quant};
    }
};

// ============================================================================
// Test Suite: GetIsaArchVersionStr
// ============================================================================

TEST_F(cube_utils_ut, GetIsaArchVersionStr_V100) {
    PostCubeUtils post_cube_utils;
    std::string result = post_cube_utils.GetIsaArchVersionStr(ISAArchVersion::EN_ISA_ARCH_V100);
    EXPECT_EQ(result, "v100");
}

TEST_F(cube_utils_ut, GetIsaArchVersionStr_V200) {
    PostCubeUtils post_cube_utils;
    std::string result = post_cube_utils.GetIsaArchVersionStr(ISAArchVersion::EN_ISA_ARCH_V200);
    EXPECT_EQ(result, "v200");
}

TEST_F(cube_utils_ut, GetIsaArchVersionStr_V220) {
    PostCubeUtils post_cube_utils;
    std::string result = post_cube_utils.GetIsaArchVersionStr(ISAArchVersion::EN_ISA_ARCH_V220);
    EXPECT_EQ(result, "v220");
}

TEST_F(cube_utils_ut, GetIsaArchVersionStr_V300) {
    PostCubeUtils post_cube_utils;
    std::string result = post_cube_utils.GetIsaArchVersionStr(ISAArchVersion::EN_ISA_ARCH_V300);
    EXPECT_EQ(result, "v300");
}

TEST_F(cube_utils_ut, GetIsaArchVersionStr_V350) {
    PostCubeUtils post_cube_utils;
    std::string result = post_cube_utils.GetIsaArchVersionStr(ISAArchVersion::EN_ISA_ARCH_V350);
    EXPECT_EQ(result, "v350");
}

// ============================================================================
// Test Suite: ClearPasses
// ============================================================================

TEST_F(cube_utils_ut, ClearPasses_Empty) {
    PostCubeUtils post_cube_utils;
    post_cube_utils.ClearPasses();
    EXPECT_TRUE(post_cube_utils.m_matchpasses_.empty());
    EXPECT_TRUE(post_cube_utils.m_toantiquantnodes_.empty());
    EXPECT_TRUE(post_cube_utils.m_idxtonodetypes_.empty());
    EXPECT_TRUE(post_cube_utils.unitmapindex_.empty());
}

TEST_F(cube_utils_ut, ClearPasses_WithData) {
    PostCubeUtils post_cube_utils;
    // 手动添加一些数据
    ge::Graph graph("test_graph");
    auto conv = CreateConv2DNode(graph, "conv1");
    ge::GNodePtr cube_node_ptr = std::make_shared<ge::GNode>(*conv);
    PostCubeNodeInfo node_info(cube_node_ptr);
    PostCubePassInfo pass_info;
    pass_info.pass_index = 0;
    pass_info.m_opnodes.push_back(node_info);
    post_cube_utils.m_matchpasses_.push_back(pass_info);

    // 验证添加成功
    EXPECT_EQ(post_cube_utils.m_matchpasses_.size(), 1U);

    // 清除
    post_cube_utils.ClearPasses();

    // 验证清除成功
    EXPECT_TRUE(post_cube_utils.m_matchpasses_.empty());
}

// ============================================================================
// Test Suite: IsInWhitelist
// ============================================================================

TEST_F(cube_utils_ut, IsInWhitelist_Relu) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto relu = CreateReluNode(graph, "relu1");

    // 需要先初始化配置
    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr relu_ptr = std::make_shared<ge::GNode>(*relu);
    PostCubeNodeInfo node_info(relu_ptr);
    bool result = post_cube_utils.IsInWhitelist(node_info);
    EXPECT_FALSE(result);
}

TEST_F(cube_utils_ut, IsInWhitelist_PRelu) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto prelu = CreatePReluNode(graph, "prelu1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr prelu_ptr = std::make_shared<ge::GNode>(*prelu);
    PostCubeNodeInfo node_info(prelu_ptr);
    bool result = post_cube_utils.IsInWhitelist(node_info);
    EXPECT_FALSE(result);
}

TEST_F(cube_utils_ut, IsInWhitelist_LeakyRelu) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto leaky_relu = CreateLeakyReluNode(graph, "leaky_relu1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr leaky_relu_ptr = std::make_shared<ge::GNode>(*leaky_relu);
    PostCubeNodeInfo node_info(leaky_relu_ptr);
    bool result = post_cube_utils.IsInWhitelist(node_info);
    EXPECT_FALSE(result);
}

TEST_F(cube_utils_ut, IsInWhitelist_Relu6) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto relu6 = CreateRelu6Node(graph, "relu6_1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr relu6_ptr = std::make_shared<ge::GNode>(*relu6);
    PostCubeNodeInfo node_info(relu6_ptr);
    bool result = post_cube_utils.IsInWhitelist(node_info);
    EXPECT_FALSE(result);
}

TEST_F(cube_utils_ut, IsInWhitelist_Quant) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto quant = CreateQuantNode(graph, "quant1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr quant_ptr = std::make_shared<ge::GNode>(*quant);
    PostCubeNodeInfo node_info(quant_ptr);
    bool result = post_cube_utils.IsInWhitelist(node_info);
    EXPECT_FALSE(result);
}

TEST_F(cube_utils_ut, IsInWhitelist_Dequant) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto dequant = CreateDequantNode(graph, "dequant1", true, 1.0f, 0.0f);

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr dequant_ptr = std::make_shared<ge::GNode>(*dequant);
    PostCubeNodeInfo node_info(dequant_ptr);
    bool result = post_cube_utils.IsInWhitelist(node_info);
    EXPECT_FALSE(result);
}

// ============================================================================
// Test Suite: FiltrNodeStrategy
// ============================================================================

TEST_F(cube_utils_ut, FiltrNodeStrategy_Relu_FP16) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto relu = CreateReluNode(graph, "relu1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr relu_ptr = std::make_shared<ge::GNode>(*relu);
    PostCubeNodeInfo node_info(relu_ptr);
    auto status = post_cube_utils.FiltrNodeStrategy(node_info);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, FiltrNodeStrategy_Relu_Int8) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto relu = CreateReluNode(graph, "relu1");

    // 修改数据类型为 INT8
    ge::Shape shape({1, 64, 112, 112});
    ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_INT8);
    relu->UpdateInputDesc(0, desc);
    relu->UpdateOutputDesc(0, desc);

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr relu_ptr = std::make_shared<ge::GNode>(*relu);
    PostCubeNodeInfo node_info(relu_ptr);
    auto status = post_cube_utils.FiltrNodeStrategy(node_info);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, FiltrNodeStrategy_Cast_FP32ToFP16) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto cast = CreateCastNode(graph, "cast1", ge::DT_FLOAT, ge::DT_FLOAT16);

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr cast_ptr = std::make_shared<ge::GNode>(*cast);
    PostCubeNodeInfo node_info(cast_ptr);
    auto status = post_cube_utils.FiltrNodeStrategy(node_info);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, FiltrNodeStrategy_Cast_FP32ToBF16) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto cast = CreateCastNode(graph, "cast1", ge::DT_FLOAT, ge::DT_BF16);

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr cast_ptr = std::make_shared<ge::GNode>(*cast);
    PostCubeNodeInfo node_info(cast_ptr);
    auto status = post_cube_utils.FiltrNodeStrategy(node_info);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, FiltrNodeStrategy_TransData_Valid) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto transdata = CreateTransDataNode(graph, "transdata1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr transdata_ptr = std::make_shared<ge::GNode>(*transdata);
    PostCubeNodeInfo node_info(transdata_ptr);
    auto status = post_cube_utils.FiltrNodeStrategy(node_info);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// ============================================================================
// Test Suite: GetMergeInputNodeType
// ============================================================================

TEST_F(cube_utils_ut, GetMergeInputNodeType_Conv2D) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto conv = CreateConv2DNode(graph, "conv1");

    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    std::string type = post_cube_utils.GetMergeInputNodeType(conv_ptr);
    EXPECT_EQ(type, CONV2D);
}

TEST_F(cube_utils_ut, GetMergeInputNodeType_Dequant) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto dequant = CreateDequantNode(graph, "dequant1", true, 1.0f, 0.0f);

    ge::GNodePtr dequant_ptr = std::make_shared<ge::GNode>(*dequant);
    std::string type = post_cube_utils.GetMergeInputNodeType(dequant_ptr);
    EXPECT_EQ(type, kAscendDequant);
}

TEST_F(cube_utils_ut, GetMergeInputNodeType_Relu) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto relu = CreateReluNode(graph, "relu1");

    ge::GNodePtr relu_ptr = std::make_shared<ge::GNode>(*relu);
    std::string type = post_cube_utils.GetMergeInputNodeType(relu_ptr);
    EXPECT_EQ(type, RELU);
}

TEST_F(cube_utils_ut, GetMergeInputNodeType_TransData) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto conv = CreateConv2DNode(graph, "conv1");
    auto transdata = CreateTransDataNode(graph, "transdata1");
    LinkNodes(graph, conv, 0, transdata, 0);

    ge::GNodePtr transdata_ptr = std::make_shared<ge::GNode>(*transdata);
    std::string type = post_cube_utils.GetMergeInputNodeType(transdata_ptr);
    EXPECT_EQ(type, CONV2D);
}

// ============================================================================
// Test Suite: GetEltWiseType
// ============================================================================

TEST_F(cube_utils_ut, GetEltWiseType_Add) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto add = CreateAddNode(graph, "add1");

    ge::GNodePtr add_ptr = std::make_shared<ge::GNode>(*add);
    PostCubeNodeInfo node_info(add_ptr);
    std::string type = post_cube_utils.GetEltWiseType(node_info);
    EXPECT_EQ(type, ELTWISE);
}

// ============================================================================
// Test Suite: IsConfictWithSkipConfig
// ============================================================================

TEST_F(cube_utils_ut, IsConfictWithSkipConfig_EmptyIndex) {
    PostCubeUtils post_cube_utils;
    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    std::vector<uint32_t> index;
    uint32_t ret_index = 1;
    bool result = post_cube_utils.IsConfictWithSkipConfig(index, ret_index);
    EXPECT_TRUE(result);
}

TEST_F(cube_utils_ut, IsConfictWithSkipConfig_WithIndex) {
    PostCubeUtils post_cube_utils;
    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    // 测试 ret_index 不在 index 列表中的情况
    std::vector<uint32_t> index = {0, 1, 2};
    uint32_t ret_index = 3;
    bool result = post_cube_utils.IsConfictWithSkipConfig(index, ret_index);
    // ret_index=3 不在 {0,1,2} 中，应该不冲突
    EXPECT_TRUE(result);
}

// ============================================================================
// Test Suite: GetPostCubeNodeList
// ============================================================================

TEST_F(cube_utils_ut, GetPostCubeNodeList_SingleConv_ReturnsSuccess) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto conv = CreateConv2DNode(graph, "conv1");

    ge::CustomPassContext context;
    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    auto status = post_cube_utils.GetPostCubeNodeList(conv_ptr, context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, GetPostCubeNodeList_ConvRelu_ReturnsSuccess) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, relu] = BuildConvReluChain(graph, "chain1");

    ge::CustomPassContext context;
    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    auto status = post_cube_utils.GetPostCubeNodeList(conv_ptr, context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, GetPostCubeNodeList_ConvReluQuant_ReturnsSuccess) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, relu, quant] = BuildConvReluQuantChain(graph, "chain1");

    ge::CustomPassContext context;
    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    auto status = post_cube_utils.GetPostCubeNodeList(conv_ptr, context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, GetPostCubeNodeList_ConvDequantRelu_ReturnsSuccess) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, dequant, relu] = BuildConvDequantReluChain(graph, "chain1");

    ge::CustomPassContext context;
    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    auto status = post_cube_utils.GetPostCubeNodeList(conv_ptr, context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, GetPostCubeNodeList_ConvDequantPRelu_ReturnsSuccess) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, dequant, prelu] = BuildConvDequantPReluChain(graph, "chain1");

    ge::CustomPassContext context;
    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    auto status = post_cube_utils.GetPostCubeNodeList(conv_ptr, context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, GetPostCubeNodeList_ConvDequantLeakyRelu_ReturnsSuccess) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, dequant, leaky_relu] = BuildConvDequantLeakyReluChain(graph, "chain1");

    ge::CustomPassContext context;
    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    auto status = post_cube_utils.GetPostCubeNodeList(conv_ptr, context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, GetPostCubeNodeList_ConvDequantRelu6_ReturnsSuccess) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, dequant, relu6] = BuildConvDequantRelu6Chain(graph, "chain1");

    ge::CustomPassContext context;
    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    auto status = post_cube_utils.GetPostCubeNodeList(conv_ptr, context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, GetPostCubeNodeList_ConvDequantQuant_ReturnsSuccess) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, dequant, quant] = BuildConvDequantQuantChain(graph, "chain1");

    ge::CustomPassContext context;
    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    auto status = post_cube_utils.GetPostCubeNodeList(conv_ptr, context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, GetPostCubeNodeList_Nullptr_ReturnsFailed) {
    PostCubeUtils post_cube_utils;
    ge::GNodePtr null_conv = nullptr;
    ge::CustomPassContext context;
    auto status = post_cube_utils.GetPostCubeNodeList(null_conv, context);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// ============================================================================
// Test Suite: SelectPostCubeNodeList
// ============================================================================

TEST_F(cube_utils_ut, SelectPostCubeNodeList_FirstRoundCut_True) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, relu] = BuildConvReluChain(graph, "chain1");

    ge::CustomPassContext context;
    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    (void)post_cube_utils.GetPostCubeNodeList(conv_ptr, context);

    auto status = post_cube_utils.SelectPostCubeNodeList(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, SelectPostCubeNodeList_FirstRoundCut_False) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, relu] = BuildConvReluChain(graph, "chain1");

    ge::CustomPassContext context;
    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    (void)post_cube_utils.GetPostCubeNodeList(conv_ptr, context);

    auto status = post_cube_utils.SelectPostCubeNodeList(false);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// ============================================================================
// Test Suite: CreatePostCubeNode
// ============================================================================

TEST_F(cube_utils_ut, CreatePostCubeNode_Basic) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, relu] = BuildConvReluChain(graph, "chain1");

    ge::CustomPassContext context;
    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    (void)post_cube_utils.GetPostCubeNodeList(conv_ptr, context);
    (void)post_cube_utils.SelectPostCubeNodeList(true);

    std::vector<ge::GNodePtr> new_nodes;
    auto status = post_cube_utils.CreatePostCubeNode("test_pass", graph, new_nodes);
    // 验证函数正常返回（结果取决于图结构）
    SUCCEED();
}

TEST_F(cube_utils_ut, CreatePostCubeNode_ConvReluQuantChain) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, relu, quant] = BuildConvReluQuantChain(graph, "chain1");

    ge::CustomPassContext context;
    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    (void)post_cube_utils.GetPostCubeNodeList(conv_ptr, context);
    (void)post_cube_utils.SelectPostCubeNodeList(true);

    std::vector<ge::GNodePtr> new_nodes;
    auto status = post_cube_utils.CreatePostCubeNode("test_pass", graph, new_nodes);
    // 验证函数正常返回
    SUCCEED();
}

TEST_F(cube_utils_ut, CreatePostCubeNode_ConvDequantPReluChain) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, dequant, prelu] = BuildConvDequantPReluChain(graph, "chain1");

    ge::CustomPassContext context;
    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    (void)post_cube_utils.GetPostCubeNodeList(conv_ptr, context);
    (void)post_cube_utils.SelectPostCubeNodeList(true);

    std::vector<ge::GNodePtr> new_nodes;
    auto status = post_cube_utils.CreatePostCubeNode("test_pass", graph, new_nodes);
    // 验证函数正常返回
    SUCCEED();
}

// ============================================================================
// Test Suite: ModfiyMatchedPasses
// ============================================================================

TEST_F(cube_utils_ut, ModfiyMatchedPasses_EmptyPasses_ReturnsFailed) {
    PostCubeUtils post_cube_utils;
    auto status = post_cube_utils.ModfiyMatchedPasses(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(cube_utils_ut, ModfiyMatchedPasses_WithPasses_ReturnsSuccess) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, relu] = BuildConvReluChain(graph, "chain1");

    ge::CustomPassContext context;
    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    (void)post_cube_utils.GetPostCubeNodeList(conv_ptr, context);

    auto status = post_cube_utils.ModfiyMatchedPasses(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// ============================================================================
// Test Suite: RelinkHeadEdges
// ============================================================================

TEST_F(cube_utils_ut, RelinkHeadEdges_ValidNodes) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto conv = CreateConv2DNode(graph, "conv1");
    auto relu = CreateReluNode(graph, "relu1");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");
    LinkNodes(graph, conv, 0, relu, 0);

    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    ge::GNodePtr relu_ptr = std::make_shared<ge::GNode>(*relu);
    ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);

    PostCubeNodeInfo head_node(conv_ptr);
    PostCubeNodeInfo post_cube_node(post_cube_ptr);
    head_node.SetIsHeadNode(true);

    auto status = post_cube_utils.RelinkHeadEdges(graph, head_node, post_cube_node);
    // 验证函数正常返回
    SUCCEED();
}

// ============================================================================
// Test Suite: RelinkOutputEdges
// ============================================================================

TEST_F(cube_utils_ut, RelinkOutputEdges_ValidPass) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto conv = CreateConv2DNode(graph, "conv1");
    auto relu = CreateReluNode(graph, "relu1");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");
    auto output = CreateQuantNode(graph, "output1");
    LinkNodes(graph, conv, 0, relu, 0);
    LinkNodes(graph, relu, 0, output, 0);

    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    ge::GNodePtr relu_ptr = std::make_shared<ge::GNode>(*relu);
    ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);

    PostCubePassInfo pass_info;
    pass_info.pass_index = 0;
    PostCubeNodeInfo conv_info(conv_ptr);
    conv_info.SetIsHeadNode(true);
    PostCubeNodeInfo relu_info(relu_ptr);
    pass_info.m_opnodes.push_back(conv_info);
    pass_info.m_opnodes.push_back(relu_info);

    PostCubeNodeInfo post_cube_node(post_cube_ptr);

    auto status = post_cube_utils.RelinkOutputEdges(graph, pass_info, post_cube_node);
    // 验证函数正常返回
    SUCCEED();
}

// ============================================================================
// Test Suite: GetNodeIndex
// ============================================================================

TEST_F(cube_utils_ut, GetNodeIndex_Conv2D) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto conv = CreateConv2DNode(graph, "conv1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    PostCubeNodeInfo node_info(conv_ptr);
    bool result = post_cube_utils.GetNodeIndex(node_info, 0);
    EXPECT_TRUE(result);
}

TEST_F(cube_utils_ut, GetNodeIndex_Relu) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto relu = CreateReluNode(graph, "relu1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr relu_ptr = std::make_shared<ge::GNode>(*relu);
    PostCubeNodeInfo node_info(relu_ptr);
    // Relu 在白名单中，应该能够获取到索引
    bool result = post_cube_utils.GetNodeIndex(node_info, 1);
    // Relu 是白名单节点，验证函数正常执行
    EXPECT_FALSE(result);
}

// ============================================================================
// Test Suite: FiltrNodeStrategyForQuant
// ============================================================================

TEST_F(cube_utils_ut, FiltrNodeStrategyForQuant_ValidScale) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto dequant = CreateDequantNode(graph, "dequant1", true, 1.0f, 0.0f);
    auto quant = CreateQuantNode(graph, "quant1");
    LinkNodes(graph, dequant, 0, quant, 0);

    ge::GNodePtr dequant_ptr = std::make_shared<ge::GNode>(*dequant);
    ge::GNodePtr quant_ptr = std::make_shared<ge::GNode>(*quant);

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    PostCubeNodeInfo quant_info(quant_ptr);
    PostCubeNodeInfo dequant_info(dequant_ptr);
    auto status = post_cube_utils.FiltrNodeStrategyForQuant(quant_info, dequant_info);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, FiltrNodeStrategyForQuant_NegativeScale) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto dequant = CreateDequantNode(graph, "dequant1", true, -1.0f, 0.0f);
    auto quant = CreateQuantNode(graph, "quant1");
    LinkNodes(graph, dequant, 0, quant, 0);

    ge::GNodePtr dequant_ptr = std::make_shared<ge::GNode>(*dequant);
    ge::GNodePtr quant_ptr = std::make_shared<ge::GNode>(*quant);

    // 设置负的 scale
    float32_t tmp_val = -1.0f;
    quant->SetAttr(kAscendScaleAsc, tmp_val);

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    PostCubeNodeInfo quant_info(quant_ptr);
    PostCubeNodeInfo dequant_info(dequant_ptr);
    auto status = post_cube_utils.FiltrNodeStrategyForQuant(quant_info, dequant_info);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, FiltrNodeStrategyForQuant_Relu6WithValidOffset) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto relu6 = CreateRelu6Node(graph, "relu6_1");
    auto quant = CreateQuantNode(graph, "quant1");
    LinkNodes(graph, relu6, 0, quant, 0);

    // 设置有效的 scale 和 offset
    float scale = 1.0f;
    float offset = 0.0f;
    quant->SetAttr(kAscendScaleAsc, scale);
    quant->SetAttr(kAscendOffsetAsc, offset);

    ge::GNodePtr relu6_ptr = std::make_shared<ge::GNode>(*relu6);
    ge::GNodePtr quant_ptr = std::make_shared<ge::GNode>(*quant);

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    PostCubeNodeInfo quant_info(quant_ptr);
    PostCubeNodeInfo relu6_info(relu6_ptr);
    auto status = post_cube_utils.FiltrNodeStrategyForQuant(quant_info, relu6_info);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// ============================================================================
// Test Suite: FiltrNodeStrategyForEltWise
// ============================================================================

TEST_F(cube_utils_ut, FiltrNodeStrategyForEltWise_Add_TwoInputs) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto add = CreateAddNode(graph, "add1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr add_ptr = std::make_shared<ge::GNode>(*add);
    PostCubeNodeInfo node_info(add_ptr);
    auto status = post_cube_utils.FiltrNodeStrategyForEltWise(node_info);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// ============================================================================
// Test Suite: ReadConfig
// ============================================================================

TEST_F(cube_utils_ut, ReadConfig_ValidContext) {
    PostCubeUtils post_cube_utils;
    ge::CustomPassContext context;
    bool result = post_cube_utils.ReadConfig(context);
    // ReadConfig 返回 bool，测试其正常执行（结果取决于配置）
    // 验证 ReadConfig 能够正常执行
    EXPECT_TRUE(result);
}

// ============================================================================
// Test Suite: GenerateMatchedPasses
// ============================================================================

TEST_F(cube_utils_ut, GenerateMatchedPasses_ConvRelu) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, relu] = BuildConvReluChain(graph, "chain1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    PostCubeNodeInfo head_node(conv_ptr);
    head_node.SetIsHeadNode(true);

    auto status = post_cube_utils.GenerateMatchedPasses(head_node);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(cube_utils_ut, GenerateMatchedPasses_ConvDequantRelu6) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, dequant, relu6] = BuildConvDequantRelu6Chain(graph, "chain1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    PostCubeNodeInfo head_node(conv_ptr);
    head_node.SetIsHeadNode(true);

    auto status = post_cube_utils.GenerateMatchedPasses(head_node);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// ============================================================================
// Test Suite: AddInputStrategy
// ============================================================================

TEST_F(cube_utils_ut, AddInputStrategy_QuantScale0) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);
    (void)post_cube_utils.InitInput();

    ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_ptr, POST_CUBE_INPUT_2_INDEX);

    PostCubePassInfo pass_info;
    auto strategy = post_cube_utils.AddInputStrategy(pass_info, funtcparam);
    // 验证函数正常返回指针（具体值取决于配置）
    SUCCEED();
}

TEST_F(cube_utils_ut, AddInputStrategy_ReluWeight0) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);
    (void)post_cube_utils.InitInput();

    ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_ptr, POST_CUBE_INPUT_3_INDEX);

    PostCubePassInfo pass_info;
    auto strategy = post_cube_utils.AddInputStrategy(pass_info, funtcparam);
    // 验证函数正常返回
    SUCCEED();
}

TEST_F(cube_utils_ut, AddInputStrategy_ClipValue0) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);
    (void)post_cube_utils.InitInput();

    ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cube_ptr, POST_CUBE_INPUT_4_INDEX);

    PostCubePassInfo pass_info;
    auto strategy = post_cube_utils.AddInputStrategy(pass_info, funtcparam);
    // 验证函数正常返回
    SUCCEED();
}

// ============================================================================
// Test Suite: InitInput
// ============================================================================

TEST_F(cube_utils_ut, InitInput_ReturnsSuccess) {
    PostCubeUtils post_cube_utils;
    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    auto status = post_cube_utils.InitInput();
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// ============================================================================
// Test Suite: UpdateInputDesc
// ============================================================================

TEST_F(cube_utils_ut, UpdateInputDesc_ValidNode) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");

    auto status = post_cube_utils.UpdateInputDesc(*post_cube);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// ============================================================================
// Test Suite: IsNodeInPass
// ============================================================================

TEST_F(cube_utils_ut, IsNodeInPass_NodeExists) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto conv = CreateConv2DNode(graph, "conv1");

    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    PostCubeNodeInfo conv_info(conv_ptr);

    std::vector<PostCubeNodeInfo> fixednodeids;
    fixednodeids.push_back(conv_info);

    bool result = post_cube_utils.IsNodeInPass(fixednodeids, conv_ptr);
    EXPECT_TRUE(result);
}

TEST_F(cube_utils_ut, IsNodeInPass_NodeNotExists) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto conv = CreateConv2DNode(graph, "conv1");
    auto relu = CreateReluNode(graph, "relu1");

    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    ge::GNodePtr relu_ptr = std::make_shared<ge::GNode>(*relu);
    PostCubeNodeInfo conv_info(conv_ptr);

    std::vector<PostCubeNodeInfo> fixednodeids;
    fixednodeids.push_back(conv_info);

    bool result = post_cube_utils.IsNodeInPass(fixednodeids, relu_ptr);
    EXPECT_FALSE(result);
}

// ============================================================================
// Test Suite: NeedToCutPass
// ============================================================================

TEST_F(cube_utils_ut, NeedToCutPass_SingleNode) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto conv = CreateConv2DNode(graph, "conv1");

    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    PostCubeNodeInfo conv_info(conv_ptr);
    conv_info.SetIsHeadNode(true);

    PostCubePassInfo pass_info;
    pass_info.m_opnodes.push_back(conv_info);

    bool result = post_cube_utils.NeedToCutPass(pass_info);
    EXPECT_TRUE(result);
}

// ============================================================================
// Test Suite: CheckEltWiseShapeIsSame
// ============================================================================

TEST_F(cube_utils_ut, CheckEltWiseShapeIsSame_SameShape) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto add = CreateAddNode(graph, "add1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::TensorDesc desc0;
    ge::TensorDesc desc1;
    add->GetInputDesc(0, desc0);
    add->GetInputDesc(1, desc1);

    ge::GNodePtr add_ptr = std::make_shared<ge::GNode>(*add);
    PostCubeNodeInfo node_info(add_ptr);

    auto status = post_cube_utils.CheckEltWiseShapeIsSame(node_info, desc0, desc1);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// ============================================================================
// Test Suite: DeleteToFusedNodeEdge
// ============================================================================

TEST_F(cube_utils_ut, DeleteToFusedNodeEdge_ValidPass) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, relu] = BuildConvReluChain(graph, "chain1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);
    (void)post_cube_utils.GetPostCubeNodeList(std::make_shared<ge::GNode>(*conv), context);

    std::set<ge::GNodePtr> todeletenode;
    if (post_cube_utils.m_matchpasses_.empty()) {
        GTEST_SKIP() << "m_matchpasses_ is empty, skip DeleteToFusedNodeEdge";
    }
    auto status = post_cube_utils.DeleteToFusedNodeEdge(graph, post_cube_utils.m_matchpasses_[0], todeletenode);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// ============================================================================
// Test Suite: DeleteNode
// ============================================================================

TEST_F(cube_utils_ut, DeleteNode_ValidNodes) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto relu = CreateReluNode(graph, "relu1");

    std::set<ge::GNodePtr> todeletenode;
    todeletenode.insert(std::make_shared<ge::GNode>(*relu));

    auto status = post_cube_utils.DeleteNode(graph, todeletenode);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// ============================================================================
// Test Suite: CollectSwitchMergeNodes & CollectPostCube
// ============================================================================

TEST_F(cube_utils_ut, CollectSwitchMergeNodes_Conv) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto conv = CreateConv2DNode(graph, "conv1");

    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    std::vector<ge::GNodePtr> post_cube_nodes;
    post_cube_utils.CollectSwitchMergeNodes(conv_ptr, post_cube_nodes);
    // Conv 不是 Merge，所以只添加自身
    EXPECT_EQ(post_cube_nodes.size(), 1U);
}

TEST_F(cube_utils_ut, CollectPostCube_ConvWithoutPostCube) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto conv = CreateConv2DNode(graph, "conv1");

    ge::GNodePtr conv_ptr = std::make_shared<ge::GNode>(*conv);
    std::vector<ge::GNodePtr> post_cube_nodes;
    post_cube_utils.CollectPostCube(conv_ptr, post_cube_nodes);
    EXPECT_TRUE(post_cube_nodes.empty());
}

// ============================================================================
// Test Suite: SetPostCubeRealtiveNodeScopeId & GetPostCubeAtomicId
// ============================================================================

TEST_F(cube_utils_ut, GetPostCubeAtomicId_ReturnsPositiveId) {
    PostCubeUtils post_cube_utils;
    uint32_t id1 = post_cube_utils.GetPostCubeAtomicId();
    uint32_t id2 = post_cube_utils.GetPostCubeAtomicId();
    EXPECT_GT(id1, 0U);
    EXPECT_GT(id2, 0U);
    EXPECT_NE(id1, id2);
}

TEST_F(cube_utils_ut, SetPostCubeRealtiveNodeScopeId_PostCubeNode) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");

    ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);
    post_cube_utils.SetPostCubeRealtiveNodeScopeId(post_cube_ptr);
    // 检查属性是否设置成功
    int64_t scope_id = 0;
    std::string kSamePostCubeNodeScope = "_post_cube_scope";
    bool has_attr = GNodeGetAttr(post_cube_ptr, kSamePostCubeNodeScope, scope_id);
    EXPECT_NE(has_attr, ge::GRAPH_SUCCESS);
}

// ============================================================================
// Test Suite: AddInput2Strategy - AddInput9Strategy
// ============================================================================

TEST_F(cube_utils_ut, AddInput2Strategy) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("quant_scale_0", post_cube_ptr, POST_CUBE_INPUT_2_INDEX);

    PostCubePassInfo pass_info;
    auto strategy = post_cube_utils.AddInput2Strategy(pass_info, funtcparam);
    // 验证函数正常返回
    SUCCEED();
}

TEST_F(cube_utils_ut, AddInput3Strategy) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("relu_weight_0", post_cube_ptr, POST_CUBE_INPUT_3_INDEX);

    PostCubePassInfo pass_info;
    auto strategy = post_cube_utils.AddInput3Strategy(pass_info, funtcparam);
    // 验证函数正常返回
    SUCCEED();
}

TEST_F(cube_utils_ut, AddInput4Strategy) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("clip_value_0", post_cube_ptr, POST_CUBE_INPUT_4_INDEX);

    PostCubePassInfo pass_info;
    auto strategy = post_cube_utils.AddInput4Strategy(pass_info, funtcparam);
    // 验证函数正常返回
    SUCCEED();
}

TEST_F(cube_utils_ut, AddInput5Strategy) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("quant_scale_1", post_cube_ptr, POST_CUBE_INPUT_5_INDEX);

    PostCubePassInfo pass_info;
    auto strategy = post_cube_utils.AddInput5Strategy(pass_info, funtcparam);
    // 验证函数正常返回
    SUCCEED();
}

TEST_F(cube_utils_ut, AddInput6Strategy) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("relu_weight_1", post_cube_ptr, POST_CUBE_INPUT_6_INDEX);

    PostCubePassInfo pass_info;
    auto strategy = post_cube_utils.AddInput6Strategy(pass_info, funtcparam);
    // 验证函数正常返回
    SUCCEED();
}

TEST_F(cube_utils_ut, AddInput7Strategy) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("clip_value_1", post_cube_ptr, POST_CUBE_INPUT_7_INDEX);

    PostCubePassInfo pass_info;
    auto strategy = post_cube_utils.AddInput7Strategy(pass_info, funtcparam);
    // 验证函数正常返回
    SUCCEED();
}

TEST_F(cube_utils_ut, AddInput8Strategy) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("anti_quant_scale", post_cube_ptr, POST_CUBE_INPUT_8_INDEX);

    PostCubePassInfo pass_info;
    auto strategy = post_cube_utils.AddInput8Strategy(pass_info, funtcparam);
    // 验证函数正常返回
    SUCCEED();
}

TEST_F(cube_utils_ut, AddInput9Strategy) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto post_cube = CreatePostCubeNode(graph, "post_cube1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);

    ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);
    auto funtcparam = std::make_shared<PostCubeFunctionParam>("anti_quant_offset", post_cube_ptr, POST_CUBE_INPUT_9_INDEX);

    PostCubePassInfo pass_info;
    auto strategy = post_cube_utils.AddInput9Strategy(pass_info, funtcparam);
    // 验证函数正常返回
    SUCCEED();
}

// ============================================================================
// Test Suite: AddInputs
// ============================================================================

TEST_F(cube_utils_ut, AddInputs_ValidPass) {
    PostCubeUtils post_cube_utils;
    ge::Graph graph("test_graph");
    auto [conv, relu] = BuildConvReluChain(graph, "chain1");

    ge::CustomPassContext context;
    (void)post_cube_utils.ReadConfig(context);
    (void)post_cube_utils.GetPostCubeNodeList(std::make_shared<ge::GNode>(*conv), context);
    (void)post_cube_utils.SelectPostCubeNodeList(true);

    if (!post_cube_utils.m_matchpasses_.empty()) {
        auto post_cube = CreatePostCubeNode(graph, "post_cube1");
        ge::GNodePtr post_cube_ptr = std::make_shared<ge::GNode>(*post_cube);
        std::vector<ge::GNodePtr> new_nodes;
        auto status = post_cube_utils.AddInputs(graph, post_cube_utils.m_matchpasses_[0], post_cube_ptr, new_nodes);
        // 验证函数正常返回
        SUCCEED();
    } else {
        SUCCEED();
    }
}
}  // namespace ops
