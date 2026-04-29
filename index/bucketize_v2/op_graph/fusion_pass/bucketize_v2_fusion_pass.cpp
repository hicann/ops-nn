/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bucketize_v2_fusion_pass.h"

#include <string>
#include <vector>

#include "common/inc/error_util.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"

using namespace ge;
using namespace fe;
using namespace fusion;

/**
 * @brief Define fusion Bucketize pattern (with bias)
 * @details Only support Ascend950 with the datatype of input is not double
 *
 *           Bucketize  ==> BucketizeV2
 */
namespace ops {
namespace {
const std::string kPassName = "BucketizeFusionPass";
const char* kSourceOpTypes = "Bucketize";
}

es::EsTensorHolder CreatePatternBucketize(es::EsGraphBuilder &graphBuilder, const char *opType,
 	     const std::string &nodeName, const es::EsTensorHolder &x)
{
    auto *graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto bucketize = es::CompliantNodeBuilder(graph)
                    .OpType(opType)
                    .Name(nodeName.c_str())
                    .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .Build();

    OP_LOGE_IF(
        es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), bucketize, 0) != GRAPH_SUCCESS,
        es::EsTensorHolder(), kPassName.c_str(), "AddEdgeAndUpdatePeerDesc failed.");

    auto *y_holder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bucketize, 0);
    return es::EsTensorHolder(y_holder);
}

std::vector<PatternUniqPtr> BucketizeFusionPass::Patterns()
{
    OPS_LOG_D(kPassName.c_str(), "Enter Pattern for BucketizeFusionPass");
    std::vector<PatternUniqPtr> patterns;
    auto graph_builder = es::EsGraphBuilder(kSourceOpTypes);
    auto x = graph_builder.CreateInput(0, "x");
    auto bucketize = CreatePatternBucketize(graph_builder, kSourceOpTypes, std::string(kSourceOpTypes) + "Pattern", x);
    auto graph = graph_builder.BuildAndReset({bucketize});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*bucketize.GetProducer(), 0});
    patterns.emplace_back(std::move(pattern));
    return patterns;
}

bool IsTargetPlatform()
{
    PlatformInfo platform_info;
    OptionalInfo optional_info;
    OP_LOGE_IF(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS,
        false, kPassName.c_str(), "Get platform_info failed.");
    const std::string soc = platform_info.str_info.short_soc_version;
    bool is_platform950 = (soc == "Ascend950");
    OPS_LOG_D(kPassName.c_str(), "Platform short soc: %s", soc.c_str());
    if (!is_platform950) {
        OPS_LOG_D(kPassName.c_str(), "Platform is not support, only work on Ascend950.");
        return false;
    }
    return true;
}

bool IsShapeAndDtypeValid(const std::unique_ptr<MatchResult>& match_result)
{
    NodeIo bucketize_io;
    OP_LOGE_IF(
        match_result->GetCapturedTensor(0, bucketize_io) != SUCCESS, false, kPassName,
        "Failed to get bucketize in meetrequirements");
    auto bucketize_node = bucketize_io.node;

    TensorDesc bucketize_input_desc;
    OP_LOGE_IF(bucketize_node.GetInputDesc(0, bucketize_input_desc) != SUCCESS, false, kPassName.c_str(),
 	    "Get input desc failed.");
    TensorDesc bucketize_output_desc;
    OP_LOGE_IF(bucketize_node.GetOutputDesc(0, bucketize_output_desc) != SUCCESS, false, kPassName.c_str(),
 	    "Get output desc failed.");

    if (bucketize_input_desc.GetDataType() == DT_DOUBLE) {
        OPS_LOG_D(kPassName.c_str(), "Not support dtype DT_DOUBLE of input.");
        return false;
    }
    return true;
}

bool BucketizeFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter MeetRequirements for BucketizeFusionPass");

    if (!IsTargetPlatform()) {
        return false;
    }
    if (!IsShapeAndDtypeValid(match_result)){
        return false;
    }
    return true;
}

GraphUniqPtr BucketizeFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for BucketizeFusionPass");
    auto replace_graph_builder = es::EsGraphBuilder("BucketizeV2");

    NodeIo bucketize_io;
    OP_LOGE_IF(match_result->GetCapturedTensor(0, bucketize_io) != SUCCESS, nullptr, kPassName.c_str(),
  	    "Get bucketize node failed.");

    TensorDesc bucketize_input_desc;
    bucketize_io.node.GetInputDesc(0, bucketize_input_desc);

    auto x = replace_graph_builder.CreateInput(0, "x", bucketize_input_desc.GetDataType(), bucketize_input_desc.GetFormat(), 
            bucketize_input_desc.GetShape().GetDims());

    std::vector<float32_t> boundaries;
    bucketize_io.node.GetAttr("boundaries", boundaries);
    std::vector<int64_t> dims = {static_cast<int64_t>(boundaries.size())};
    auto boundaries_dtype = bucketize_input_desc.GetDataType();
    auto boundaries_node = replace_graph_builder.CreateConst(boundaries, dims, DT_FLOAT);
    auto boundaries_tensor = es::Cast(boundaries_node, boundaries_dtype);

    // 更新 boundaries_tensor 的 TensorDesc，设置 shape、dtype 和 origin 信息
    TensorDesc boundaries_desc;
    boundaries_tensor.GetProducer()->GetOutputDesc(0, boundaries_desc);
    boundaries_desc.SetShape(Shape(dims));
    boundaries_desc.SetOriginShape(Shape(dims));
    boundaries_desc.SetDataType(boundaries_dtype);
    boundaries_tensor.GetProducer()->UpdateOutputDesc(0, boundaries_desc);

    ge::DataType dtype = DT_UNDEFINED;
    OP_LOGE_IF(bucketize_io.node.GetAttr("dtype", dtype) != SUCCESS, nullptr, kPassName.c_str(),
  	    "Get bucketize attr dtype failed.");
    bool out_int32 = dtype == DT_INT32 ? true : false;

    bool right;
    OP_LOGE_IF(bucketize_io.node.GetAttr("right", right) != SUCCESS, nullptr, kPassName.c_str(),
  	    "Get bucketize attr right failed.");
    auto bucketize_v2 = es::BucketizeV2(x, boundaries_tensor, out_int32, right);

    TensorDesc bucketize_output_desc;
    bucketize_io.node.GetOutputDesc(0, bucketize_output_desc);
    auto out_node = bucketize_v2.GetProducer();
    out_node->UpdateInputDesc(0, bucketize_input_desc);
    out_node->UpdateInputDesc(1, boundaries_desc);
    out_node->UpdateOutputDesc(0, bucketize_output_desc);

    return replace_graph_builder.BuildAndReset({bucketize_v2});
}

REG_FUSION_PASS(BucketizeFusionPass).Stage(CustomPassStage::kAfterInferShape);
} //namespace ops
