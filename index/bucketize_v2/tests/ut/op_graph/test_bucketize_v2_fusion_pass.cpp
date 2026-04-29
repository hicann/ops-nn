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
#include <vector>
#include <gtest/gtest.h>
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "../../../op_graph/fusion_pass/bucketize_v2_fusion_pass.h"
#include "register/register_custom_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace ops;

class BucketizeFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static GNode CreateBucketizeNode(es::EsGraphBuilder &graphBuilder, const std::string &nodeName)
    {
        auto *graph = graphBuilder.GetCGraphBuilder()->GetGraph();
        auto bucketize = es::CompliantNodeBuilder(graph)
                        .OpType("Bucketize")
                        .Name(nodeName.c_str())
                        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
        return bucketize;
    }

    static void SetBucketizeAttrs(GNode &bucketize, std::vector<float32_t> boundaries, bool right, DataType dtype)
    {
        AscendString attrNameBoundaries("boundaries");
        AscendString attrNameRight("right");
        AscendString attrNameDtype("dtype");
        bucketize.SetAttr(attrNameBoundaries, boundaries);
        bucketize.SetAttr(attrNameRight, right);
        bucketize.SetAttr(attrNameDtype, dtype);
    }

    static void InferShapeForBucketize(DataType dtype, Shape &shapeX,
        es::EsTensorHolder &x, GNode &bucketize)
    {
        TensorDesc xOutputDesc;
        x.GetProducer()->GetOutputDesc(0, xOutputDesc);
        xOutputDesc.SetDataType(dtype);
        xOutputDesc.SetShape(shapeX);
        x.GetProducer()->UpdateOutputDesc(0, xOutputDesc);

        TensorDesc bucketizeInputDesc;
        bucketize.GetInputDesc(0, bucketizeInputDesc);
        bucketizeInputDesc.SetDataType(dtype);
        bucketizeInputDesc.SetShape(shapeX);
        bucketize.UpdateInputDesc(0, bucketizeInputDesc);

        TensorDesc bucketizeOutputDesc;
        bucketize.GetOutputDesc(0, bucketizeOutputDesc);
        bucketizeOutputDesc.SetDataType(DT_INT32);
        bucketizeOutputDesc.SetShape(shapeX);
        bucketize.UpdateOutputDesc(0, bucketizeOutputDesc);
    }

    bool FindBucketizeV2Node(std::shared_ptr<Graph> &graph)
    {
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == "BucketizeV2") {
                return true;
            }
        }
        return false;
    }
};

TEST_F(BucketizeFusionPassTest, patternTest)
{
    ops::BucketizeFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_EQ(patterns.size(), 1);
}

TEST_F(BucketizeFusionPassTest, fusionSuccess950Float)
{
    std::vector<int64_t> dimsX{2, 3, 4};
    Shape shapeX(dimsX);
    DataType dtype = DT_FLOAT;
    DataType outputDtype = DT_INT32;
    std::vector<float32_t> boundaries{1.0f, 2.0f, 3.0f};

    auto graphBuilder = es::EsGraphBuilder("bucketize_fusion_test");
    auto x = graphBuilder.CreateInput(0, "x", dtype, FORMAT_ND, dimsX);
    auto bucketize = CreateBucketizeNode(graphBuilder, "BucketizeNode");

    SetBucketizeAttrs(bucketize, boundaries, false, outputDtype);
    auto status = es::AddEdgeAndUpdatePeerDesc(*graphBuilder.GetCGraphBuilder()->GetGraph(),
        *x.GetProducer(), x.GetProducerOutIndex(), bucketize, 0);
    ASSERT_EQ(status, GRAPH_SUCCESS);

    InferShapeForBucketize(dtype, shapeX, x, bucketize);

    auto *yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bucketize, 0);
    auto bucketizeTensor = es::EsTensorHolder(yHolder);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({bucketizeTensor});
    CustomPassContext passContext;
    ops::BucketizeFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, SUCCESS);
    EXPECT_TRUE(FindBucketizeV2Node(graph));
}

TEST_F(BucketizeFusionPassTest, fusionSuccess950Float16)
{
    std::vector<int64_t> dimsX{1, 128, 256};
    Shape shapeX(dimsX);
    DataType dtype = DT_FLOAT16;
    DataType outputDtype = DT_INT32;
    std::vector<float32_t> boundaries{0.0f, 10.0f, 100.0f};

    auto graphBuilder = es::EsGraphBuilder("bucketize_fusion_fp16_test");
    auto x = graphBuilder.CreateInput(0, "x", dtype, FORMAT_ND, dimsX);
    auto bucketize = CreateBucketizeNode(graphBuilder, "BucketizeNodeFP16");

    SetBucketizeAttrs(bucketize, boundaries, false, outputDtype);
    auto status = es::AddEdgeAndUpdatePeerDesc(*graphBuilder.GetCGraphBuilder()->GetGraph(),
        *x.GetProducer(), x.GetProducerOutIndex(), bucketize, 0);
    ASSERT_EQ(status, GRAPH_SUCCESS);

    InferShapeForBucketize(dtype, shapeX, x, bucketize);

    auto *yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bucketize, 0);
    auto bucketizeTensor = es::EsTensorHolder(yHolder);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({bucketizeTensor});
    CustomPassContext passContext;
    ops::BucketizeFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, SUCCESS);
    EXPECT_TRUE(FindBucketizeV2Node(graph));
}

TEST_F(BucketizeFusionPassTest, fusionSuccess950Int32)
{
    std::vector<int64_t> dimsX{4, 8, 16};
    Shape shapeX(dimsX);
    DataType dtype = DT_INT32;
    DataType outputDtype = DT_INT32;
    std::vector<float32_t> boundaries{5.0f, 15.0f, 25.0f};

    auto graphBuilder = es::EsGraphBuilder("bucketize_fusion_int32_test");
    auto x = graphBuilder.CreateInput(0, "x", dtype, FORMAT_ND, dimsX);
    auto bucketize = CreateBucketizeNode(graphBuilder, "BucketizeNodeInt32");

    SetBucketizeAttrs(bucketize, boundaries, false, outputDtype);
    auto status = es::AddEdgeAndUpdatePeerDesc(*graphBuilder.GetCGraphBuilder()->GetGraph(),
        *x.GetProducer(), x.GetProducerOutIndex(), bucketize, 0);
    ASSERT_EQ(status, GRAPH_SUCCESS);

    InferShapeForBucketize(dtype, shapeX, x, bucketize);

    auto *yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bucketize, 0);
    auto bucketizeTensor = es::EsTensorHolder(yHolder);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({bucketizeTensor});
    CustomPassContext passContext;
    ops::BucketizeFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, SUCCESS);
    EXPECT_TRUE(FindBucketizeV2Node(graph));
}

TEST_F(BucketizeFusionPassTest, fusionSuccess950Int64Output)
{
    std::vector<int64_t> dimsX{2, 4, 8};
    Shape shapeX(dimsX);
    DataType dtype = DT_FLOAT;
    DataType outputDtype = DT_INT64;
    std::vector<float32_t> boundaries{1.0f, 5.0f, 10.0f};

    auto graphBuilder = es::EsGraphBuilder("bucketize_fusion_int64_output_test");
    auto x = graphBuilder.CreateInput(0, "x", dtype, FORMAT_ND, dimsX);
    auto bucketize = CreateBucketizeNode(graphBuilder, "BucketizeNodeInt64Out");

    SetBucketizeAttrs(bucketize, boundaries, false, outputDtype);
    auto status = es::AddEdgeAndUpdatePeerDesc(*graphBuilder.GetCGraphBuilder()->GetGraph(),
        *x.GetProducer(), x.GetProducerOutIndex(), bucketize, 0);
    ASSERT_EQ(status, GRAPH_SUCCESS);

    InferShapeForBucketize(dtype, shapeX, x, bucketize);

    // Set output dtype to DT_INT64
    TensorDesc bucketizeOutputDesc;
    bucketize.GetOutputDesc(0, bucketizeOutputDesc);
    bucketizeOutputDesc.SetDataType(DT_INT64);
    bucketizeOutputDesc.SetShape(shapeX);
    bucketize.UpdateOutputDesc(0, bucketizeOutputDesc);

    auto *yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bucketize, 0);
    auto bucketizeTensor = es::EsTensorHolder(yHolder);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({bucketizeTensor});
    CustomPassContext passContext;
    ops::BucketizeFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, SUCCESS);
    EXPECT_TRUE(FindBucketizeV2Node(graph));
}

TEST_F(BucketizeFusionPassTest, unsupportedDtypeDoubleFail)
{
    std::vector<int64_t> dimsX{2, 3, 4};
    Shape shapeX(dimsX);
    DataType dtype = DT_DOUBLE;
    DataType outputDtype = DT_INT32;
    std::vector<float32_t> boundaries{1.0f, 2.0f, 3.0f};

    auto graphBuilder = es::EsGraphBuilder("bucketize_fusion_double_test");
    auto x = graphBuilder.CreateInput(0, "x", dtype, FORMAT_ND, dimsX);
    auto bucketize = CreateBucketizeNode(graphBuilder, "BucketizeNodeDouble");

    SetBucketizeAttrs(bucketize, boundaries, false, outputDtype);
    auto status = es::AddEdgeAndUpdatePeerDesc(*graphBuilder.GetCGraphBuilder()->GetGraph(),
        *x.GetProducer(), x.GetProducerOutIndex(), bucketize, 0);
    ASSERT_EQ(status, GRAPH_SUCCESS);

    InferShapeForBucketize(dtype, shapeX, x, bucketize);

    auto *yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bucketize, 0);
    auto bucketizeTensor = es::EsTensorHolder(yHolder);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({bucketizeTensor});
    CustomPassContext passContext;
    ops::BucketizeFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(FindBucketizeV2Node(graph));
}

TEST_F(BucketizeFusionPassTest, unsupportedPlatform910_93Fail)
{
    // Set platform to Ascend910_93 (not supported)
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_93";
    optiCompilationInfo.soc_version = "Ascend910_93";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> dimsX{2, 3, 4};
    Shape shapeX(dimsX);
    DataType dtype = DT_FLOAT;
    DataType outputDtype = DT_INT32;
    std::vector<float32_t> boundaries{1.0f, 2.0f, 3.0f};

    auto graphBuilder = es::EsGraphBuilder("bucketize_fusion_910_test");
    auto x = graphBuilder.CreateInput(0, "x", dtype, FORMAT_ND, dimsX);
    auto bucketize = CreateBucketizeNode(graphBuilder, "BucketizeNode910");

    SetBucketizeAttrs(bucketize, boundaries, false, outputDtype);
    auto status = es::AddEdgeAndUpdatePeerDesc(*graphBuilder.GetCGraphBuilder()->GetGraph(),
        *x.GetProducer(), x.GetProducerOutIndex(), bucketize, 0);
    ASSERT_EQ(status, GRAPH_SUCCESS);

    InferShapeForBucketize(dtype, shapeX, x, bucketize);

    auto *yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bucketize, 0);
    auto bucketizeTensor = es::EsTensorHolder(yHolder);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({bucketizeTensor});
    CustomPassContext passContext;
    ops::BucketizeFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(FindBucketizeV2Node(graph));
}

TEST_F(BucketizeFusionPassTest, unsupportedPlatform910BFail)
{
    // Set platform to Ascend910B (not supported)
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> dimsX{2, 3, 4};
    Shape shapeX(dimsX);
    DataType dtype = DT_FLOAT;
    DataType outputDtype = DT_INT32;
    std::vector<float32_t> boundaries{1.0f, 2.0f, 3.0f};

    auto graphBuilder = es::EsGraphBuilder("bucketize_fusion_910B_test");
    auto x = graphBuilder.CreateInput(0, "x", dtype, FORMAT_ND, dimsX);
    auto bucketize = CreateBucketizeNode(graphBuilder, "BucketizeNode910B");

    SetBucketizeAttrs(bucketize, boundaries, false, outputDtype);
    auto status = es::AddEdgeAndUpdatePeerDesc(*graphBuilder.GetCGraphBuilder()->GetGraph(),
        *x.GetProducer(), x.GetProducerOutIndex(), bucketize, 0);
    ASSERT_EQ(status, GRAPH_SUCCESS);

    InferShapeForBucketize(dtype, shapeX, x, bucketize);

    auto *yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bucketize, 0);
    auto bucketizeTensor = es::EsTensorHolder(yHolder);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({bucketizeTensor});
    CustomPassContext passContext;
    ops::BucketizeFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(FindBucketizeV2Node(graph));
}