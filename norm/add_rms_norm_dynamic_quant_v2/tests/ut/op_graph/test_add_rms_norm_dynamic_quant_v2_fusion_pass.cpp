/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "compliant_node_builder.h"
#include "register/register_custom_pass.h"
#include "external/ge_common/ge_api_types.h"
#include "../../../op_graph/fusion_pass/add_rms_norm_dynamic_quant_v2_fusion_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace ops;

namespace {
struct AddRmsNormBuildResult {
    es::EsTensorHolder y;
    es::EsTensorHolder rstd;
    es::EsTensorHolder x;
};

AddRmsNormBuildResult BuildAddRmsNormNode(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x1,
                                          const es::EsTensorHolder& x2, const es::EsTensorHolder& gamma)
{
    static int counter = 0;
    const std::string name = "AddRmsNorm_" + std::to_string(counter++);
    auto graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("AddRmsNorm")
                     .Name(name.c_str())
                     .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"gamma", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"rstd", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"x", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs(
                         {{"epsilon", es::CompliantNodeBuilder::kEsAttrOptional, "Float", es::CreateFrom(1e-6f)}})
                     .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x1.GetProducer(), x1.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *x2.GetProducer(), x2.GetProducerOutIndex(), node, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *gamma.GetProducer(), gamma.GetProducerOutIndex(), node, 2);
    return {es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0)),
            es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 1)),
            es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 2))};
}

struct DynamicQuantBuildResult {
    es::EsTensorHolder y;
    es::EsTensorHolder scale;
};

DynamicQuantBuildResult BuildDynamicQuantNode(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x,
                                              const es::EsTensorHolder& smoothScales, int64_t dstType)
{
    static int counter = 0;
    const std::string name = "DynamicQuant_" + std::to_string(counter++);
    auto graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("DynamicQuant")
                     .Name(name.c_str())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"smooth_scales", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"group_index", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"scale", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs(
                         {{"dst_type", es::CompliantNodeBuilder::kEsAttrOptional, "Int", es::CreateFrom(dstType)}})
                     .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *smoothScales.GetProducer(), smoothScales.GetProducerOutIndex(), node, 1);
    return {es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0)),
            es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 1))};
}
} // namespace

class AddRmsNormDynamicQuantV2FusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 48;
        platformInfo.str_info.short_soc_version = "Ascend910B";
        optiCompilationInfo.soc_version = "Ascend910B";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUp() override
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 48;
        platformInfo.str_info.short_soc_version = "Ascend910B";
        optiCompilationInfo.soc_version = "Ascend910B";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static void SetInputNodeOutputDesc(es::EsTensorHolder& input, const Shape& shape, DataType dtype)
    {
        TensorDesc desc;
        input.GetProducer()->GetOutputDesc(0, desc);
        desc.SetShape(shape);
        desc.SetDataType(dtype);
        desc.SetFormat(FORMAT_ND);
        input.GetProducer()->UpdateOutputDesc(0, desc);
    }

    static void SetAddRmsNormDescs(GNode& node, const Shape& xShape, const Shape& gammaShape, DataType dtype)
    {
        for (int i = 0; i < 2; ++i) {
            TensorDesc inputDesc;
            node.GetInputDesc(i, inputDesc);
            inputDesc.SetShape(xShape);
            inputDesc.SetDataType(dtype);
            inputDesc.SetFormat(FORMAT_ND);
            node.UpdateInputDesc(i, inputDesc);
        }
        TensorDesc gammaDesc;
        node.GetInputDesc(2, gammaDesc);
        gammaDesc.SetShape(gammaShape);
        gammaDesc.SetDataType(dtype);
        gammaDesc.SetFormat(FORMAT_ND);
        node.UpdateInputDesc(2, gammaDesc);

        TensorDesc yDesc;
        node.GetOutputDesc(0, yDesc);
        yDesc.SetShape(xShape);
        yDesc.SetDataType(dtype);
        yDesc.SetFormat(FORMAT_ND);
        node.UpdateOutputDesc(0, yDesc);

        TensorDesc rstdDesc;
        node.GetOutputDesc(1, rstdDesc);
        rstdDesc.SetShape(xShape);
        rstdDesc.SetDataType(dtype);
        rstdDesc.SetFormat(FORMAT_ND);
        node.UpdateOutputDesc(1, rstdDesc);

        TensorDesc xOutDesc;
        node.GetOutputDesc(2, xOutDesc);
        xOutDesc.SetShape(xShape);
        xOutDesc.SetDataType(dtype);
        xOutDesc.SetFormat(FORMAT_ND);
        node.UpdateOutputDesc(2, xOutDesc);
    }

    static void SetCastDescs(GNode& node, const Shape& xShape, DataType srcDtype)
    {
        TensorDesc inputDesc;
        node.GetInputDesc(0, inputDesc);
        inputDesc.SetShape(xShape);
        inputDesc.SetDataType(srcDtype);
        inputDesc.SetFormat(FORMAT_ND);
        node.UpdateInputDesc(0, inputDesc);

        TensorDesc outputDesc;
        node.GetOutputDesc(0, outputDesc);
        outputDesc.SetShape(xShape);
        outputDesc.SetDataType(DT_FLOAT);
        outputDesc.SetFormat(FORMAT_ND);
        node.UpdateOutputDesc(0, outputDesc);
    }

    static void SetDynamicQuantDescs(GNode& node, const Shape& xShape, DataType srcDtype, bool hasSmooth,
                                     const Shape& smoothShape, DataType quantDtype = DT_INT8)
    {
        TensorDesc inputDesc;
        node.GetInputDesc(0, inputDesc);
        inputDesc.SetShape(xShape);
        inputDesc.SetDataType(srcDtype);
        inputDesc.SetFormat(FORMAT_ND);
        node.UpdateInputDesc(0, inputDesc);

        if (hasSmooth) {
            TensorDesc smoothDesc;
            node.GetInputDesc(1, smoothDesc);
            smoothDesc.SetShape(smoothShape);
            smoothDesc.SetDataType(srcDtype);
            smoothDesc.SetFormat(FORMAT_ND);
            node.UpdateInputDesc(1, smoothDesc);
        }

        TensorDesc yDesc;
        node.GetOutputDesc(0, yDesc);
        yDesc.SetShape(xShape);
        yDesc.SetDataType(quantDtype);
        yDesc.SetFormat(FORMAT_ND);
        node.UpdateOutputDesc(0, yDesc);

        int64_t dstType = static_cast<int64_t>(quantDtype);
        node.SetAttr("dst_type", dstType);

        std::vector<int64_t> scaleDims;
        if (xShape.GetDimNum() > 0) {
            scaleDims.push_back(xShape.GetDim(0));
            scaleDims.push_back(1);
        } else {
            scaleDims.push_back(1);
        }
        TensorDesc scaleDesc;
        node.GetOutputDesc(1, scaleDesc);
        scaleDesc.SetShape(Shape(scaleDims));
        scaleDesc.SetDataType(DT_FLOAT);
        scaleDesc.SetFormat(FORMAT_ND);
        node.UpdateOutputDesc(1, scaleDesc);
    }

    static bool IsFusedNodeValid(GNode& node, const Shape& xShape, const Shape& gammaShape, DataType dtype,
                                 DataType quantDtype, bool hasSmooth1, bool hasSmooth2)
    {
        TensorDesc x1Desc;
        TensorDesc x2Desc;
        TensorDesc gammaDesc;
        node.GetInputDesc(0, x1Desc);
        node.GetInputDesc(1, x2Desc);
        node.GetInputDesc(2, gammaDesc);
        if (x1Desc.GetDataType() != dtype || x2Desc.GetDataType() != dtype || gammaDesc.GetDataType() != dtype) {
            return false;
        }
        if (x1Desc.GetShape().GetShapeSize() != xShape.GetShapeSize() ||
            x2Desc.GetShape().GetShapeSize() != xShape.GetShapeSize() ||
            gammaDesc.GetShape().GetShapeSize() != gammaShape.GetShapeSize()) {
            return false;
        }
        if (hasSmooth1) {
            TensorDesc smoothDesc;
            if (node.GetInputDesc(3, smoothDesc) != SUCCESS) {
                return false;
            }
            if (smoothDesc.GetDataType() != dtype) {
                return false;
            }
        }
        if (hasSmooth2) {
            TensorDesc smoothDesc;
            if (node.GetInputDesc(4, smoothDesc) != SUCCESS) {
                return false;
            }
            if (smoothDesc.GetDataType() != dtype) {
                return false;
            }
        }

        TensorDesc y1Desc;
        node.GetOutputDesc(0, y1Desc);
        if (y1Desc.GetDataType() != quantDtype) {
            return false;
        }
        if (hasSmooth2) {
            TensorDesc y2Desc;
            node.GetOutputDesc(1, y2Desc);
            if (y2Desc.GetDataType() != quantDtype) {
                return false;
            }
            return true;
        }
        return true;
    }
};

TEST_F(AddRmsNormDynamicQuantV2FusionPassTest, patternTest)
{
    ops::AddRmsNormDynamicQuantV2FusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(AddRmsNormDynamicQuantV2FusionPassTest, fusionFp16SuccessWithSingleSmooths)
{
    std::vector<int64_t> dimsX{1, 128, 1024};
    std::vector<int64_t> dimsGamma{1024};
    Shape shapeX(dimsX);
    Shape shapeGamma(dimsGamma);
    Shape shapeSmooth(dimsGamma);

    auto graphBuilder = es::EsGraphBuilder("addrmsnorm_dynamic_quant_v2_fp16_test");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto gamma = graphBuilder.CreateInput(2, "gamma", DT_FLOAT16, FORMAT_ND, shapeGamma.GetDims());
    auto smooth1 = graphBuilder.CreateInput(3, "smooth1", DT_FLOAT16, FORMAT_ND, shapeSmooth.GetDims());
    auto addRmsNormOut = BuildAddRmsNormNode(graphBuilder, x1, x2, gamma);
    auto castY = es::Cast(addRmsNormOut.y, DT_FLOAT);
    auto dynamicQuantOut1 = BuildDynamicQuantNode(graphBuilder, addRmsNormOut.y, smooth1,
                                                  static_cast<int64_t>(DT_INT8));

    SetInputNodeOutputDesc(x1, shapeX, DT_FLOAT16);
    SetInputNodeOutputDesc(x2, shapeX, DT_FLOAT16);
    SetInputNodeOutputDesc(gamma, shapeGamma, DT_FLOAT16);
    SetInputNodeOutputDesc(smooth1, shapeSmooth, DT_FLOAT16);
    SetAddRmsNormDescs(*addRmsNormOut.y.GetProducer(), shapeX, shapeGamma, DT_FLOAT16);
    SetCastDescs(*castY.GetProducer(), shapeX, DT_FLOAT16);
    SetDynamicQuantDescs(*dynamicQuantOut1.y.GetProducer(), shapeX, DT_FLOAT16, true, shapeSmooth, DT_INT8);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {dynamicQuantOut1.y, castY, addRmsNormOut.y, addRmsNormOut.x, dynamicQuantOut1.scale});
    CustomPassContext passContext;
    ops::AddRmsNormDynamicQuantV2FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    bool findFused = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "AddRmsNormDynamicQuantV2" &&
            IsFusedNodeValid(node, shapeX, shapeGamma, DT_FLOAT16, DT_INT8, true, false)) {
            findFused = true;
        }
    }
    EXPECT_EQ(findFused, true);
}

TEST_F(AddRmsNormDynamicQuantV2FusionPassTest, fusionBf16SuccessWithSingleSmooth)
{
    std::vector<int64_t> dimsX{2, 64, 512};
    std::vector<int64_t> dimsGamma{512};
    Shape shapeX(dimsX);
    Shape shapeGamma(dimsGamma);
    Shape shapeSmooth(dimsGamma);

    auto graphBuilder = es::EsGraphBuilder("addrmsnorm_dynamic_quant_v2_bf16_smooth_test");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_BF16, FORMAT_ND, shapeX.GetDims());
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_BF16, FORMAT_ND, shapeX.GetDims());
    auto gamma = graphBuilder.CreateInput(2, "gamma", DT_BF16, FORMAT_ND, shapeGamma.GetDims());
    auto smooth1 = graphBuilder.CreateInput(3, "smooth1", DT_BF16, FORMAT_ND, shapeSmooth.GetDims());
    auto addRmsNormOut = BuildAddRmsNormNode(graphBuilder, x1, x2, gamma);
    auto castY = es::Cast(addRmsNormOut.y, DT_FLOAT);
    auto dynamicQuantOut1 = BuildDynamicQuantNode(graphBuilder, addRmsNormOut.y, smooth1,
                                                  static_cast<int64_t>(DT_INT8));

    SetInputNodeOutputDesc(x1, shapeX, DT_BF16);
    SetInputNodeOutputDesc(x2, shapeX, DT_BF16);
    SetInputNodeOutputDesc(gamma, shapeGamma, DT_BF16);
    SetInputNodeOutputDesc(smooth1, shapeSmooth, DT_BF16);
    SetAddRmsNormDescs(*addRmsNormOut.y.GetProducer(), shapeX, shapeGamma, DT_BF16);
    SetCastDescs(*castY.GetProducer(), shapeX, DT_BF16);
    SetDynamicQuantDescs(*dynamicQuantOut1.y.GetProducer(), shapeX, DT_BF16, true, shapeSmooth, DT_INT8);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {dynamicQuantOut1.y, castY, addRmsNormOut.y, addRmsNormOut.x, dynamicQuantOut1.scale});
    CustomPassContext passContext;
    ops::AddRmsNormDynamicQuantV2FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    bool findFused = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "AddRmsNormDynamicQuantV2" &&
            IsFusedNodeValid(node, shapeX, shapeGamma, DT_BF16, DT_INT8, true, false)) {
            findFused = true;
        }
    }
    EXPECT_EQ(findFused, true);
}

TEST_F(AddRmsNormDynamicQuantV2FusionPassTest, fusionFp16Int4SuccessWithInt4)
{
    std::vector<int64_t> dimsX{1, 32, 256};
    std::vector<int64_t> dimsGamma{256};
    Shape shapeX(dimsX);
    Shape shapeGamma(dimsGamma);
    Shape shapeSmooth(dimsGamma);

    auto graphBuilder = es::EsGraphBuilder("addrmsnorm_dynamic_quant_v2_int4_test");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto gamma = graphBuilder.CreateInput(2, "gamma", DT_FLOAT16, FORMAT_ND, shapeGamma.GetDims());
    auto smooth1 = graphBuilder.CreateInput(3, "smooth1", DT_FLOAT16, FORMAT_ND, shapeSmooth.GetDims());
    auto addRmsNormOut = BuildAddRmsNormNode(graphBuilder, x1, x2, gamma);
    auto castY = es::Cast(addRmsNormOut.y, DT_FLOAT);
    auto dynamicQuantOut1 = BuildDynamicQuantNode(graphBuilder, addRmsNormOut.y, smooth1,
                                                  static_cast<int64_t>(DT_INT4));

    SetInputNodeOutputDesc(x1, shapeX, DT_FLOAT16);
    SetInputNodeOutputDesc(x2, shapeX, DT_FLOAT16);
    SetInputNodeOutputDesc(gamma, shapeGamma, DT_FLOAT16);
    SetInputNodeOutputDesc(smooth1, shapeSmooth, DT_FLOAT16);
    SetAddRmsNormDescs(*addRmsNormOut.y.GetProducer(), shapeX, shapeGamma, DT_FLOAT16);
    SetCastDescs(*castY.GetProducer(), shapeX, DT_FLOAT16);
    SetDynamicQuantDescs(*dynamicQuantOut1.y.GetProducer(), shapeX, DT_FLOAT16, true, shapeSmooth, DT_INT4);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {dynamicQuantOut1.y, castY, addRmsNormOut.y, addRmsNormOut.x, dynamicQuantOut1.scale});
    CustomPassContext passContext;
    ops::AddRmsNormDynamicQuantV2FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    bool findFused = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "AddRmsNormDynamicQuantV2" &&
            IsFusedNodeValid(node, shapeX, shapeGamma, DT_FLOAT16, DT_INT4, true, false)) {
            findFused = true;
        }
    }
    EXPECT_EQ(findFused, true);
}

TEST_F(AddRmsNormDynamicQuantV2FusionPassTest, fusion950Success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> dimsX{1, 64, 256};
    std::vector<int64_t> dimsGamma{256};
    Shape shapeX(dimsX);
    Shape shapeGamma(dimsGamma);
    Shape shapeSmooth(dimsGamma);

    auto graphBuilder = es::EsGraphBuilder("addrmsnorm_dynamic_quant_v2_950_test");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto gamma = graphBuilder.CreateInput(2, "gamma", DT_FLOAT16, FORMAT_ND, shapeGamma.GetDims());
    auto smooth1 = graphBuilder.CreateInput(3, "smooth1", DT_FLOAT16, FORMAT_ND, shapeSmooth.GetDims());
    auto addRmsNormOut = BuildAddRmsNormNode(graphBuilder, x1, x2, gamma);
    auto castY = es::Cast(addRmsNormOut.y, DT_FLOAT);
    auto dynamicQuantOut1 = BuildDynamicQuantNode(graphBuilder, addRmsNormOut.y, smooth1,
                                                  static_cast<int64_t>(DT_INT8));

    SetInputNodeOutputDesc(x1, shapeX, DT_FLOAT16);
    SetInputNodeOutputDesc(x2, shapeX, DT_FLOAT16);
    SetInputNodeOutputDesc(gamma, shapeGamma, DT_FLOAT16);
    SetInputNodeOutputDesc(smooth1, shapeSmooth, DT_FLOAT16);
    SetAddRmsNormDescs(*addRmsNormOut.y.GetProducer(), shapeX, shapeGamma, DT_FLOAT16);
    SetCastDescs(*castY.GetProducer(), shapeX, DT_FLOAT16);
    SetDynamicQuantDescs(*dynamicQuantOut1.y.GetProducer(), shapeX, DT_FLOAT16, true, shapeSmooth, DT_INT8);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {dynamicQuantOut1.y, castY, addRmsNormOut.y, addRmsNormOut.x, dynamicQuantOut1.scale});
    CustomPassContext passContext;
    ops::AddRmsNormDynamicQuantV2FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    bool findFused = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "AddRmsNormDynamicQuantV2") {
            findFused = true;
        }
    }
    EXPECT_EQ(findFused, true);
}

TEST_F(AddRmsNormDynamicQuantV2FusionPassTest, unsupportedPlatformFail)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 48;
    platformInfo.str_info.short_soc_version = "Ascend910_93";
    optiCompilationInfo.soc_version = "Ascend910_93";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> dimsX{1, 32, 256};
    std::vector<int64_t> dimsGamma{256};
    Shape shapeX(dimsX);
    Shape shapeGamma(dimsGamma);
    Shape shapeSmooth(dimsGamma);

    auto graphBuilder = es::EsGraphBuilder("addrmsnorm_dynamic_quant_v2_platform_fail");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto gamma = graphBuilder.CreateInput(2, "gamma", DT_FLOAT16, FORMAT_ND, shapeGamma.GetDims());
    auto smooth1 = graphBuilder.CreateInput(3, "smooth1", DT_FLOAT16, FORMAT_ND, shapeSmooth.GetDims());
    auto addRmsNormOut = BuildAddRmsNormNode(graphBuilder, x1, x2, gamma);
    auto castY = es::Cast(addRmsNormOut.y, DT_FLOAT);
    auto dynamicQuantOut1 = BuildDynamicQuantNode(graphBuilder, addRmsNormOut.y, smooth1,
                                                  static_cast<int64_t>(DT_INT8));

    SetInputNodeOutputDesc(x1, shapeX, DT_FLOAT16);
    SetInputNodeOutputDesc(x2, shapeX, DT_FLOAT16);
    SetInputNodeOutputDesc(gamma, shapeGamma, DT_FLOAT16);
    SetInputNodeOutputDesc(smooth1, shapeSmooth, DT_FLOAT16);
    SetAddRmsNormDescs(*addRmsNormOut.y.GetProducer(), shapeX, shapeGamma, DT_FLOAT16);
    SetCastDescs(*castY.GetProducer(), shapeX, DT_FLOAT16);
    SetDynamicQuantDescs(*dynamicQuantOut1.y.GetProducer(), shapeX, DT_FLOAT16, true, shapeSmooth, DT_INT8);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {dynamicQuantOut1.y, castY, addRmsNormOut.y, addRmsNormOut.x, dynamicQuantOut1.scale});
    CustomPassContext passContext;
    ops::AddRmsNormDynamicQuantV2FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddRmsNormDynamicQuantV2FusionPassTest, unsupportedDtypeFail)
{
    std::vector<int64_t> dimsX{1, 32, 256};
    std::vector<int64_t> dimsGamma{256};
    Shape shapeX(dimsX);
    Shape shapeGamma(dimsGamma);
    Shape shapeSmooth(dimsGamma);

    auto graphBuilder = es::EsGraphBuilder("addrmsnorm_dynamic_quant_v2_dtype_fail");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT, FORMAT_ND, shapeX.GetDims());
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT, FORMAT_ND, shapeX.GetDims());
    auto gamma = graphBuilder.CreateInput(2, "gamma", DT_FLOAT, FORMAT_ND, shapeGamma.GetDims());
    auto smooth1 = graphBuilder.CreateInput(3, "smooth1", DT_FLOAT, FORMAT_ND, shapeSmooth.GetDims());
    auto addRmsNormOut = BuildAddRmsNormNode(graphBuilder, x1, x2, gamma);
    auto castY = es::Cast(addRmsNormOut.y, DT_FLOAT);
    auto dynamicQuantOut1 = BuildDynamicQuantNode(graphBuilder, addRmsNormOut.y, smooth1,
                                                  static_cast<int64_t>(DT_INT8));

    SetInputNodeOutputDesc(x1, shapeX, DT_FLOAT);
    SetInputNodeOutputDesc(x2, shapeX, DT_FLOAT);
    SetInputNodeOutputDesc(gamma, shapeGamma, DT_FLOAT);
    SetInputNodeOutputDesc(smooth1, shapeSmooth, DT_FLOAT);
    SetAddRmsNormDescs(*addRmsNormOut.y.GetProducer(), shapeX, shapeGamma, DT_FLOAT);
    SetCastDescs(*castY.GetProducer(), shapeX, DT_FLOAT);
    SetDynamicQuantDescs(*dynamicQuantOut1.y.GetProducer(), shapeX, DT_FLOAT, true, shapeSmooth, DT_INT8);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {dynamicQuantOut1.y, castY, addRmsNormOut.y, addRmsNormOut.x, dynamicQuantOut1.scale});
    CustomPassContext passContext;
    ops::AddRmsNormDynamicQuantV2FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddRmsNormDynamicQuantV2FusionPassTest, unsupportedCastOutputDtypeFail)
{
    std::vector<int64_t> dimsX{1, 32, 256};
    std::vector<int64_t> dimsGamma{256};
    Shape shapeX(dimsX);
    Shape shapeGamma(dimsGamma);
    Shape shapeSmooth(dimsGamma);

    auto graphBuilder = es::EsGraphBuilder("addrmsnorm_dynamic_quant_v2_cast_fail");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto gamma = graphBuilder.CreateInput(2, "gamma", DT_FLOAT16, FORMAT_ND, shapeGamma.GetDims());
    auto smooth1 = graphBuilder.CreateInput(3, "smooth1", DT_FLOAT16, FORMAT_ND, shapeSmooth.GetDims());
    auto addRmsNormOut = BuildAddRmsNormNode(graphBuilder, x1, x2, gamma);
    auto castY = es::Cast(addRmsNormOut.y, DT_INT8);
    auto dynamicQuantOut1 = BuildDynamicQuantNode(graphBuilder, addRmsNormOut.y, smooth1,
                                                  static_cast<int64_t>(DT_INT8));

    SetInputNodeOutputDesc(x1, shapeX, DT_FLOAT16);
    SetInputNodeOutputDesc(x2, shapeX, DT_FLOAT16);
    SetInputNodeOutputDesc(gamma, shapeGamma, DT_FLOAT16);
    SetInputNodeOutputDesc(smooth1, shapeSmooth, DT_FLOAT16);
    SetAddRmsNormDescs(*addRmsNormOut.y.GetProducer(), shapeX, shapeGamma, DT_FLOAT16);
    SetCastDescs(*castY.GetProducer(), shapeX, DT_FLOAT16);
    TensorDesc castOutputDesc;
    castY.GetProducer()->GetOutputDesc(0, castOutputDesc);
    castOutputDesc.SetDataType(DT_INT8);
    castY.GetProducer()->UpdateOutputDesc(0, castOutputDesc);
    SetDynamicQuantDescs(*dynamicQuantOut1.y.GetProducer(), shapeX, DT_FLOAT16, true, shapeSmooth, DT_INT8);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {dynamicQuantOut1.y, castY, addRmsNormOut.y, addRmsNormOut.x, dynamicQuantOut1.scale});
    CustomPassContext passContext;
    ops::AddRmsNormDynamicQuantV2FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(AddRmsNormDynamicQuantV2FusionPassTest, unsupportedQuantOutputDtypeFail)
{
    std::vector<int64_t> dimsX{1, 32, 256};
    std::vector<int64_t> dimsGamma{256};
    Shape shapeX(dimsX);
    Shape shapeGamma(dimsGamma);
    Shape shapeSmooth(dimsGamma);

    auto graphBuilder = es::EsGraphBuilder("addrmsnorm_dynamic_quant_v2_quant_fail");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto gamma = graphBuilder.CreateInput(2, "gamma", DT_FLOAT16, FORMAT_ND, shapeGamma.GetDims());
    auto smooth1 = graphBuilder.CreateInput(3, "smooth1", DT_FLOAT16, FORMAT_ND, shapeSmooth.GetDims());
    auto addRmsNormOut = BuildAddRmsNormNode(graphBuilder, x1, x2, gamma);
    auto castY = es::Cast(addRmsNormOut.y, DT_FLOAT);
    auto dynamicQuantOut1 = BuildDynamicQuantNode(graphBuilder, addRmsNormOut.y, smooth1,
                                                  static_cast<int64_t>(DT_INT8));

    SetInputNodeOutputDesc(x1, shapeX, DT_FLOAT16);
    SetInputNodeOutputDesc(x2, shapeX, DT_FLOAT16);
    SetInputNodeOutputDesc(gamma, shapeGamma, DT_FLOAT16);
    SetInputNodeOutputDesc(smooth1, shapeSmooth, DT_FLOAT16);
    SetAddRmsNormDescs(*addRmsNormOut.y.GetProducer(), shapeX, shapeGamma, DT_FLOAT16);
    SetCastDescs(*castY.GetProducer(), shapeX, DT_FLOAT16);
    SetDynamicQuantDescs(*dynamicQuantOut1.y.GetProducer(), shapeX, DT_FLOAT16, true, shapeSmooth, DT_INT8);

    TensorDesc quantYDesc;
    dynamicQuantOut1.y.GetProducer()->GetOutputDesc(0, quantYDesc);
    quantYDesc.SetDataType(DT_FLOAT16);
    dynamicQuantOut1.y.GetProducer()->UpdateOutputDesc(0, quantYDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {dynamicQuantOut1.y, castY, addRmsNormOut.y, addRmsNormOut.x, dynamicQuantOut1.scale});
    CustomPassContext passContext;
    ops::AddRmsNormDynamicQuantV2FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}
