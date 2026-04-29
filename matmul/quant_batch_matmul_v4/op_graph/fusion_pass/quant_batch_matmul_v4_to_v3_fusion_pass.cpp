/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>

#include "es_nn_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"

#include "common/inc/error_util.h"
#include "quant_batch_matmul_v4_to_v3_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {

namespace {
    constexpr char QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME[] = "QuantBatchMatmulV4ToV3FusionPass";
    constexpr int V4_IR_OPTIONAL_INPUT_NUMBER = 8;
    constexpr int V4_IR_REQUIRED_INPUT_NUMBER = 2;
    constexpr int V4_IR_INPUT_NUMBER = 10;
    constexpr int V3_IR_INPUT_NUMBER = 6;
    constexpr int V3_X1_INDEX = 0;
    constexpr int V3_X2_INDEX = 1;
    constexpr int V3_SCALE_INDEX = 2;
    constexpr int V3_OFFSET_INDEX = 3;
    constexpr int V3_BIAS_INDEX = 4;
    constexpr int V3_PERTOKENSCALE_INDEX = 5;
    constexpr int V4_X1_INDEX = 0;
    constexpr int V4_X2_INDEX = 1;
    constexpr int V4_BIAS_INDEX = 2;
    constexpr int V4_X1SCALE_INDEX = 3;
    constexpr int V4_X2SCALE_INDEX = 4;
    constexpr int V4_YSCALE_INDEX = 5;
    constexpr int V4_X1OFFSET_INDEX = 6;
    constexpr int V4_X2OFFSET_INDEX = 7;
    constexpr int V4_YOFFSET_INDEX = 8;
    constexpr int V4_X2TABLE_INDEX = 9;
    constexpr int QUANT_BATCH_MATMUL_V4_TO_V3_CAPTURE_TENSOR_INDEX = 0;
    const std::map<int, int> V4_V3_INDEX_MAP {
        {V4_X1_INDEX, V3_X1_INDEX},
        {V4_X2_INDEX, V3_X2_INDEX},
        {V4_BIAS_INDEX, V3_BIAS_INDEX},
        {V4_X1SCALE_INDEX, V3_PERTOKENSCALE_INDEX},
        {V4_X2SCALE_INDEX, V3_SCALE_INDEX},
        {V4_X2OFFSET_INDEX, V3_OFFSET_INDEX}
    };
}

std::vector<PatternUniqPtr> QuantBatchMatmulV4ToV3FusionPass::Patterns()
{
    OPS_LOG_D(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME, "Enter Patterns for QuantBatchMatmulV4ToV3FusionPass");
    std::vector<PatternUniqPtr> patternGraphs;

    int dtype = -1;
    // 构造可选输入数目为0-8的9种pattern
    for(int optionalInputNumber = 0; optionalInputNumber <= V4_IR_OPTIONAL_INPUT_NUMBER; optionalInputNumber++){
        auto graphBuilder = es::EsGraphBuilder(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME);
        auto quantBatchMatmulV4Inputs = graphBuilder.CreateInputs<V4_IR_INPUT_NUMBER>();
        for(int i = V4_IR_REQUIRED_INPUT_NUMBER; i < optionalInputNumber + V4_IR_REQUIRED_INPUT_NUMBER; i++){
            quantBatchMatmulV4Inputs[i] = nullptr;
        }
        auto y = es::QuantBatchMatmulV4(quantBatchMatmulV4Inputs[V4_X1_INDEX], quantBatchMatmulV4Inputs[V4_X2_INDEX],
            quantBatchMatmulV4Inputs[V4_BIAS_INDEX], quantBatchMatmulV4Inputs[V4_X1SCALE_INDEX],
            quantBatchMatmulV4Inputs[V4_X2SCALE_INDEX], quantBatchMatmulV4Inputs[V4_YSCALE_INDEX],
            quantBatchMatmulV4Inputs[V4_X1OFFSET_INDEX], quantBatchMatmulV4Inputs[V4_X2OFFSET_INDEX],
            quantBatchMatmulV4Inputs[V4_YOFFSET_INDEX], quantBatchMatmulV4Inputs[V4_X2TABLE_INDEX], dtype);
        auto graph = graphBuilder.BuildAndReset({y});
        auto pattern = std::make_unique<Pattern>(std::move(*graph));
        pattern->CaptureTensor({*y.GetProducer(), QUANT_BATCH_MATMUL_V4_TO_V3_CAPTURE_TENSOR_INDEX});
        patternGraphs.emplace_back(std::move(pattern));
    }
    OPS_LOG_D(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME, "end Patterns for QuantBatchMatmulV4ToV3FusionPass");
    return patternGraphs;
}

bool QuantBatchMatmulV4ToV3FusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    NodeIo y;
    if (match_result->GetCapturedTensor(QUANT_BATCH_MATMUL_V4_TO_V3_CAPTURE_TENSOR_INDEX, y) != SUCCESS) {
        OPS_LOG_E(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME, "Failed to GetCapture tensor.");
        return false;
    }
    GNode quantBatchMatmulV4GNode = y.node;
    TensorDesc x1Desc;
    if(quantBatchMatmulV4GNode.GetInputDesc(V4_X1_INDEX, x1Desc) != SUCCESS){
        OPS_LOG_E(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME, "Failed to GetInputDesc for x1.");
        return false;
    }
    TensorDesc x2Desc;
    if(quantBatchMatmulV4GNode.GetInputDesc(V4_X2_INDEX, x2Desc) != SUCCESS){
        OPS_LOG_E(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME, "Failed to GetInputDesc for x2.");
        return false;
    }
    TensorDesc yDesc;
    quantBatchMatmulV4GNode.GetOutputDesc(0, yDesc);
    DataType x1Dtype = x1Desc.GetDataType();
    DataType x2Dtype = x2Desc.GetDataType();
    DataType yDtype = yDesc.GetDataType();
    bool isV4Supported = (x1Dtype == DT_FLOAT8_E4M3FN) && (x2Dtype == DT_FLOAT4_E2M1 || x2Dtype == DT_FLOAT) &&
                       (yDtype == DT_BF16 || yDtype == DT_FLOAT16);
    if (isV4Supported) {
        OPS_LOG_D(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME,
                 "When x1's type is FLOAT8_E4M3FN, x2's type is FLOAT4_E2M1 or FLOAT, y's type is BF16 or FLOAT16, "
                 "there is no need for graph fusion");
        return false;
    }
    TensorDesc x1ScaleDesc;
    auto x1ScaleResult = quantBatchMatmulV4GNode.GetInputDesc(V4_X1SCALE_INDEX, x1ScaleDesc);
    TensorDesc x2ScaleDesc;
    auto x2ScaleResult = quantBatchMatmulV4GNode.GetInputDesc(V4_X2SCALE_INDEX, x2ScaleDesc);
    if (x1ScaleResult == SUCCESS && x2ScaleResult == SUCCESS) {
        bool isA8W8GBDtype = (x1Dtype == DT_INT8 && x2Dtype == DT_INT8 &&
                              x1ScaleDesc.GetDataType() == DT_FLOAT && x2ScaleDesc.GetDataType() == DT_FLOAT &&
                              yDtype == DT_BF16);
        auto x1Shape = x1Desc.GetOriginShape();
        auto x2Shape = x2Desc.GetOriginShape();
        auto x1ScaleShape = x1ScaleDesc.GetOriginShape();
        auto x2ScaleShape = x2ScaleDesc.GetOriginShape();
        bool isA8W8GBDim = x1Shape.GetDimNum() == x1ScaleShape.GetDimNum() && x2Shape.GetDimNum() == x2ScaleShape.GetDimNum();
        if (isA8W8GBDtype && isA8W8GBDim) {
            OPS_LOG_D(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME, "In G-B quantification, there is no need for graph fusion");
            return false;
        }
    }
    return true;
}

GraphUniqPtr QuantBatchMatmulV4ToV3FusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME, "Enter Replacement for QuantBatchMatmulV4ToV3FusionPass");

    NodeIo output;
    if (match_result->GetCapturedTensor(QUANT_BATCH_MATMUL_V4_TO_V3_CAPTURE_TENSOR_INDEX, output) != SUCCESS) {
        OPS_LOG_E(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME, "Failed to GetCapture tensor.");
        return nullptr;
    }

    GNode quantBatchMatmulV4GNode = output.node;
    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");

    int index = 0;
    TensorDesc v3InputDesc[V3_IR_INPUT_NUMBER];
    es::EsTensorHolder v3Input[V3_IR_INPUT_NUMBER];
    for(auto [v4idx, v3idx] : V4_V3_INDEX_MAP){
        if(quantBatchMatmulV4GNode.GetInputDesc(v4idx, v3InputDesc[v3idx]) == ge::GRAPH_SUCCESS){
            v3Input[v3idx] = replaceGraphBuilder.CreateInput(index++);
        }
    }
    TensorDesc outputDesc;
    quantBatchMatmulV4GNode.GetOutputDesc(0, outputDesc);

    auto dtype = 0;
    auto transpose_x1 = false;
    auto transpose_x2 = false;
    int64_t group_size = 0;

    OPS_LOG_D(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME, "set QuantBatchMatmulV4 attr start.");
    quantBatchMatmulV4GNode.GetAttr("dtype", dtype);
    quantBatchMatmulV4GNode.GetAttr("transpose_x1", transpose_x1);
    quantBatchMatmulV4GNode.GetAttr("transpose_x2", transpose_x2);
    quantBatchMatmulV4GNode.GetAttr("group_size", group_size);
    // V3 default group_size is 0,V4 default group_size is -1, set V3 group_size to 0 if V4 group_size is -1
    if(group_size == -1){
        group_size = 0;
    }
    OPS_LOG_D(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME, "set QuantBatchMatmulV4 attr end.");
    auto y = es::QuantBatchMatmulV3(v3Input[V3_X1_INDEX], v3Input[V3_X2_INDEX], v3Input[V3_SCALE_INDEX],
        v3Input[V3_OFFSET_INDEX], v3Input[V3_BIAS_INDEX], v3Input[V3_PERTOKENSCALE_INDEX], dtype,
        transpose_x1, transpose_x2, group_size);
    GNode quantBatchMatmulV3Node = *y.GetProducer();

    for(auto [v4idx, v3idx] : V4_V3_INDEX_MAP){
        if(quantBatchMatmulV4GNode.GetInputDesc(v4idx, v3InputDesc[v3idx]) == ge::GRAPH_SUCCESS){
            quantBatchMatmulV3Node.UpdateInputDesc(v3idx, v3InputDesc[v3idx]);
        }
    }
    quantBatchMatmulV3Node.UpdateOutputDesc(0, outputDesc);
    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({y});
    OPS_LOG_D(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME, "end Replacement for QuantBatchMatmulV4ToV3FusionPass");
    return std::move(replaceGraph);
}

REG_FUSION_PASS(QuantBatchMatmulV4ToV3FusionPass).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops
