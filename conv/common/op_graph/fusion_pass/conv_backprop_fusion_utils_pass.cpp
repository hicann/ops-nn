/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv_backprop_fusion_utils_pass.h"
#include "log/log.h"

namespace ops {
using namespace ge;
using namespace ge::es;
using namespace ConvBackpropFusionUtils;

std::vector<int64_t> ConvBackpropFusionUtilsPass::CalcTransposeShape(
    const std::vector<int64_t>& inputShape, const std::vector<int32_t>& perm)
{
    std::vector<int64_t> retShape;
    for (size_t i = 0; i < perm.size() && i < inputShape.size(); ++i) {
        if (perm[i] >= 0 && static_cast<size_t>(perm[i]) < inputShape.size()) {
            retShape.push_back(inputShape[perm[i]]);
        }
    }
    return retShape;
}

void ConvBackpropFusionUtilsPass::SetPlaceholderDesc(
    EsTensorHolder& tensorHolder, int64_t idx, const TensorDesc& desc)
{
    auto* producer = tensorHolder.GetProducer();
    if (producer == nullptr) {
        return;
    }
    producer->UpdateOutputDesc(static_cast<uint32_t>(idx), desc);
}

bool ConvBackpropFusionUtilsPass::InWhitelist(
    const std::vector<int64_t>& shape, const std::vector<std::vector<int64_t>>& whitelist)
{
    return std::find(whitelist.begin(), whitelist.end(), shape) != whitelist.end();
}

bool ConvBackpropFusionUtilsPass::CheckSocAndIntrinsic(
    const std::map<std::string, NpuArch>& supportSocList, NpuArch& npuArch)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        return false;
    }
    const std::string soc = platformInfo.str_info.short_soc_version;
    if (supportSocList.find(soc) == supportSocList.end()) {
        return false;
    }
    npuArch = supportSocList.at(soc);
    return true;
}

bool ConvBackpropFusionUtilsPass::CreateTransposeNode(
    EsGraphBuilder& builder, const TransposeNodeConfig& config,
    EsTensorHolder& output, TensorDesc& outDesc, const AscendString& opType)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    OP_CHECK_IF(graph == nullptr, OP_LOGE(opType.GetString(), "create transpose node failed"), return false);

    auto* producer = config.input.GetProducer();
    OP_CHECK_IF(producer == nullptr, OP_LOGE(opType.GetString(), "input producer is nullptr in CreateTransposeNode"),
                return false);

    TensorDesc inDesc;
    producer->GetOutputDesc(config.input.GetProducerOutIndex(), inDesc);
    auto transposeNode = ge::es::CompliantNodeBuilder(graph)
                             .OpType("Transpose")
                             .Name(config.name.c_str())
                             .IrDefInputs(
                                 {{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"perm", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .Build();
    OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*graph, *producer, TENSOR_DEFAULT_OUTPUT_INDEX, 
                transposeNode, TRANSPOSE_INPUT_X_INDEX) != GRAPH_SUCCESS,
                OP_LOGE(opType.GetString(), "Add edge for transpose input failed"), return false);
    transposeNode.UpdateInputDesc(TRANSPOSE_INPUT_X_INDEX, inDesc);
    auto permTensorHolder = builder.CreateVector(config.perm);
    auto* permTensorProducer = permTensorHolder.GetProducer();
    OP_CHECK_IF(permTensorProducer == nullptr, OP_LOGE(opType.GetString(), "perm producer is nullptr"), return false);
    OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*graph, *permTensorProducer, TENSOR_DEFAULT_OUTPUT_INDEX, 
                transposeNode, TRANSPOSE_INPUT_PERM_INDEX) != GRAPH_SUCCESS,
                OP_LOGE(opType.GetString(), "Add edge for transpose perm failed"), return false);

    TensorDesc permTensorDesc;
    permTensorProducer->GetOutputDesc(TENSOR_DEFAULT_OUTPUT_INDEX, permTensorDesc);
    transposeNode.UpdateInputDesc(TRANSPOSE_INPUT_PERM_INDEX, permTensorDesc);
    outDesc.SetDataType(inDesc.GetDataType());
    auto outShape = CalcTransposeShape(inDesc.GetShape().GetDims(), config.perm);
    outDesc.SetShape(Shape(outShape));
    outDesc.SetOriginShape(Shape(outShape));
    outDesc.SetFormat(config.format);
    outDesc.SetOriginFormat(config.format);
    transposeNode.UpdateOutputDesc(TRANSPOSE_OUTPUT_Y_INDEX, outDesc);
    output = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(transposeNode, TRANSPOSE_OUTPUT_Y_INDEX));

    return true;
}

} // namespace ops

