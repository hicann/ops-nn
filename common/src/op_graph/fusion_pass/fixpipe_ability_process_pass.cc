/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fixpipe_ability_process_pass.cc
 * \brief fixpipe ability process
 */

#include "fusion_pass/fixpipe_ability_process_pass.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "version/ge-compiler_version.h"

namespace ops {
namespace {
const char* const kFixpipeAbilityModule = "FixPipeAbility";
// legacy pass name, kept for the log/report alignment
const char* const kFixpipeAbilityPassName = "FixPipeAbilityProcessPass";
// the fixpipe multiple output platform config key
const char* const kFixpipeMultiOutputKey = "Intrinsic_fix_pipe_support_multiple_output";
} // namespace

bool FixPipeAbilityProcessPass::CheckMatmulNode(const ge::GNodePtr& cubeNode)
{
    if (kMatMulSet.count(GNodeGetType(cubeNode).GetString()) == 0U) {
        return true;
    }
    ge::TensorDesc outputDesc;
    if (cubeNode->GetOutputDesc(0, outputDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    // matmul itself used fixpipe, dont need to do fixpipe if output has one of the 4d/5d formats
    if (kMatmulNotSupportPostCubeFormatSet.count(outputDesc.GetFormat()) != 0U) {
        return false;
    }
    const ge::Shape& outShape = outputDesc.GetShape();
    // matmul fp32&c0=16 not support fixpipe fusion
    bool notSupportFixpipe = outputDesc.GetDataType() == ge::DT_FLOAT &&
                             ge::GetPrimaryFormat(outputDesc.GetFormat()) == ge::FORMAT_FRACTAL_NZ &&
                             outShape.GetDimNum() > 0 && outShape.GetDims().back() == SHAPE_NUMBER_16;
    if (notSupportFixpipe) {
        return false;
    }
    ge::TensorDesc inputDesc0;
    ge::TensorDesc inputDesc1;
    if (cubeNode->GetInputDesc(0, inputDesc0) != ge::GRAPH_SUCCESS ||
        cubeNode->GetInputDesc(1, inputDesc1) != ge::GRAPH_SUCCESS) {
        return false;
    }
    // matmul int32 in int32 out not support fixpipe fusion
    notSupportFixpipe = inputDesc0.GetDataType() == ge::DT_INT32 && inputDesc1.GetDataType() == ge::DT_INT32 &&
                        outputDesc.GetDataType() == ge::DT_INT32;
    if (notSupportFixpipe) {
        return false;
    }
    notSupportFixpipe = (ge::GetPrimaryFormat(inputDesc0.GetFormat()) == ge::FORMAT_ND) ||
                        (ge::GetPrimaryFormat(inputDesc1.GetFormat()) == ge::FORMAT_ND);
    if (notSupportFixpipe) {
        return false;
    }
    return true;
}

bool FixPipeAbilityProcessPass::CheckConv3DNode(const ge::GNodePtr& cubeNode)
{
    if (GNodeGetType(cubeNode).GetString() != kConv3D) {
        return true;
    }
    ge::TensorDesc inputDesc;
    if (cubeNode->GetInputDesc(0, inputDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    if (inputDesc.GetDataType() == ge::DT_FLOAT) {
        return false;
    }
    return true;
}

bool FixPipeAbilityProcessPass::IsUnknownShapeOp(const ge::GNodePtr& node)
{
    // equals UnknownShapeUtils::IsUnknownShapeOp of the legacy framework: any output dim < 0
    for (size_t idx = 0; idx < node->GetOutputsSize(); ++idx) {
        ge::TensorDesc outputDesc;
        if (node->GetOutputDesc(static_cast<int32_t>(idx), outputDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        const auto dims = outputDesc.GetShape().GetDims();
        for (const auto& dim : dims) {
            if (dim < 0) {
                return true;
            }
        }
    }
    return false;
}

bool FixPipeAbilityProcessPass::IsFixpipeSupportMultiOutput()
{
    fe::PlatFormInfos platformInfos;
    fe::OptionalInfos optionalInfos;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfos, optionalInfos) !=
        ge::GRAPH_SUCCESS) {
        return false;
    }
    std::string multiOutputStr;
    if (!platformInfos.GetPlatformRes("AICoreintrinsicDtypeMap", kFixpipeMultiOutputKey, multiOutputStr) ||
        multiOutputStr.empty()) {
        return false;
    }
    return multiOutputStr == "true";
}

ge::graphStatus FixPipeAbilityProcessPass::CheckCubeNode(const ge::GNodePtr& cubeNode)
{
    if (cubeNode == nullptr) {
        OPS_LOG_E(kFixpipeAbilityModule, "[GraphOpt][FixpipeAbltPrcsPss][Fusion] Cube node is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (IsUnknownShapeOp(cubeNode)) {
        return ge::GRAPH_NOT_CHANGED;
    }
    if (!CheckMatmulNode(cubeNode)) {
        return ge::GRAPH_NOT_CHANGED;
    }
    if (!CheckConv3DNode(cubeNode)) {
        return ge::GRAPH_NOT_CHANGED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixPipeAbilityProcessPass::SetConvFixpipeAbilityAttr(const ge::GNodePtr& cubeNode)
{
    if (kAtomicWriteCubeSet.count(GNodeGetType(cubeNode).GetString()) != 0U) {
        PostCubeComm::SetPostCubeAbilityAttr(cubeNode, PostCubeAbilityType::UseGmAtomicAdd);
    }
    if (!IsFixpipeSupportMultiOutput()) {
        PostCubeComm::UnSetPostCubeAbilityAttr(cubeNode, PostCubeAbilityType::SupportMultipleOutput);
        return ge::GRAPH_SUCCESS;
    }
    if (kSupportMultipleOutputCubeSet.count(GNodeGetType(cubeNode).GetString()) != 0U) {
        PostCubeComm::SetPostCubeAbilityAttr(cubeNode, PostCubeAbilityType::SupportMultipleOutput);
        const ge::GNodePtr mergeNode = PostCubeComm::GetMergeNodeByCube(cubeNode);
        if (mergeNode != nullptr) {
            PostCubeComm::SetPostCubeAbilityAttr(mergeNode, PostCubeAbilityType::SupportMultipleOutput);
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool FixPipeAbilityProcessPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    (void)matchResult;
    // Run is fully rewritten, the hook never takes effect
    return true;
}

ge::fusion::GraphUniqPtr FixPipeAbilityProcessPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    (void)matchResult;
    // Run is fully rewritten, the hook never takes effect
    return nullptr;
}

std::vector<ge::fusion::PatternUniqPtr> FixPipeAbilityProcessPass::Patterns()
{
    // Run is fully rewritten as a graph surgery flow which walks the graph by itself, no
    // pattern graph is needed and the hook never takes effect
    return {};
}

ge::Status FixPipeAbilityProcessPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    (void)passContext;
    OPS_LOG_D(kFixpipeAbilityModule, "Begin to do %s.", kFixpipeAbilityPassName);
    if (graph == nullptr) {
        OPS_LOG_E(kFixpipeAbilityModule, "Input graph is nullptr.");
        return ge::GRAPH_FAILED;
    }
    // collect the matched cube nodes (equals the kPatternConv of the legacy DefinePatterns)
    const std::set<std::string> matchTypes(kSupportPostCubeCubeTypeVec.begin(), kSupportPostCubeCubeTypeVec.end());
    std::vector<ge::GNodePtr> headNodes;
    for (const auto& node : graph->GetDirectNode()) {
        ge::AscendString nodeType;
        if (node.GetType(nodeType) != ge::GRAPH_SUCCESS || nodeType.GetString() == nullptr) {
            continue;
        }
        if (matchTypes.count(std::string(nodeType.GetString())) == 0U) {
            continue;
        }
        ge::AscendString nodeName;
        if (node.GetName(nodeName) != ge::GRAPH_SUCCESS || nodeName.GetString() == nullptr) {
            continue;
        }
        ge::GNodePtr nodePtr = graph->FindNodeByName(nodeName);
        if (nodePtr != nullptr) {
            headNodes.push_back(nodePtr);
        }
    }
    if (headNodes.empty()) {
        OPS_LOG_D(kFixpipeAbilityModule, "No matched cube node, exit.");
        return ge::GRAPH_NOT_CHANGED;
    }
    int32_t effectTimes = 0;
    for (const auto& headNode : headNodes) {
        // equals the legacy Fusion(): the limited cube node is marked NodeCantAccess and it
        // still counts as a changed graph, the ability process of the other nodes continues
        if (CheckCubeNode(headNode) != ge::GRAPH_SUCCESS) {
            PostCubeComm::SetPostCubeAbilityAttr(headNode, PostCubeAbilityType::NodeCantAccess);
            effectTimes++;
            continue;
        }
        // equals the GetFixpipeUtilFuncSupportFlag check of the legacy Fusion()
        if (PostCubeComm::GetPostCubeUtilFuncSupportFlag(headNode) != ge::GRAPH_SUCCESS) {
            continue; // platforminfo get failed or not in fixpipeUtilFunc op white list
        }
        (void)SetConvFixpipeAbilityAttr(headNode);
        effectTimes++;
    }
    OPS_LOG_D(kFixpipeAbilityModule, "%s completed, effect times = %d.", kFixpipeAbilityPassName, effectTimes);
    return effectTimes != 0 ? ge::GRAPH_SUCCESS : ge::GRAPH_NOT_CHANGED;
}

#if GE_COMPILER_VERSION_NUM >= 90300000U
REG_FUSION_PASS(FixPipeAbilityProcessPass).Stage(ge::CustomPassStage::kCompatibleInherited);
#endif

} // namespace ops
