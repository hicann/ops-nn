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
 * \file fixpipe_pass.cc
 * \brief Cube + pre_conv + pre_act + (post_eltwise + post_act + post_quant) + post_transform fusion pass
 */

#include "fusion_pass/fixpipe_pass.h"

#include <set>
#include <string>
#include <vector>

#include "version/ge-compiler_version.h"

namespace ops {
namespace {
const char* const kFixpipePassModule = "FixPipe";
// legacy pass name, kept for the log/report alignment
const char* const kFixpipePassName = "FIXPIPEFUSIONPASS";
} // namespace

bool FIXPIPEFUSIONPASS::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    (void)matchResult;
    // Run is fully rewritten, the hook never takes effect
    return true;
}

ge::fusion::GraphUniqPtr FIXPIPEFUSIONPASS::Replacement(const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    (void)matchResult;
    // Run is fully rewritten, the hook never takes effect
    return nullptr;
}

std::vector<ge::fusion::PatternUniqPtr> FIXPIPEFUSIONPASS::Patterns()
{
    // Run is fully rewritten as a graph surgery flow which walks the graph by itself, no
    // pattern graph is needed and the hook never takes effect
    return {};
}

ge::Status FIXPIPEFUSIONPASS::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    OPS_LOG_D(kFixpipePassModule, "Begin to do %s.", kFixpipePassName);
    if (graph == nullptr) {
        OPS_LOG_E(kFixpipePassModule, "Input graph is nullptr.");
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
        OPS_LOG_D(kFixpipePassModule, "No matched cube node, exit.");
        return ge::GRAPH_NOT_CHANGED;
    }
    int32_t effectTimes = 0;
    for (const auto& headNode : headNodes) {
        /*
         * equals the legacy FixPipePass::Fusion flow, every cube node owns an independent
         * PostCubeUtils instance (equals the ClearPasses + ReadConfig + InitInput + member
         * state of one legacy Fusion call):
         * - GetPostCubeNodeList: ReadConfig(skip trans by the isa arch version) + the merge
         *   head check + the NodeCantAccess ability check + the merge head switch + the
         *   GenerateMatchedPasses walk
         * - SelectPostCubeNodeList(true): the ModfiyMatchedPasses first-round cut
         * - CreatePostCubeNode: the InitInput strategies + the FixPipe node creation + the
         *   edge relink + the folded node removal + the not_support attr cleanup + the scope id
         */
        ops::PostCubeUtils postCubeUtils;
        if (postCubeUtils.GetPostCubeNodeList(headNode, passContext) != ge::GRAPH_SUCCESS) {
            OPS_LOG_D(kFixpipePassModule, "GetPostCubeNodeList failed on node[%s], skip it.",
                      GNodeGetName(headNode).GetString());
            continue;
        }
        // equals the legacy Fusion which does not check the ModfiyMatchedPasses return
        (void)postCubeUtils.SelectPostCubeNodeList(true);
        if (postCubeUtils.m_matchpasses_.empty()) {
            // equals the NOT_CHANGED of the legacy FusionImpl when no fixpipe node is created
            postCubeUtils.ResetNotSupportPostCubeFusionAttr();
            OPS_LOG_D(kFixpipePassModule, "No matched post cube pass on node[%s], skip it.",
                      GNodeGetName(headNode).GetString());
            continue;
        }
        std::vector<ge::GNodePtr> newNodes;
        const std::string passName = GNodeGetName(headNode).GetString();
        if (postCubeUtils.CreatePostCubeNode(passName, *graph, newNodes) != ge::GRAPH_SUCCESS) {
            OPS_LOG_W(kFixpipePassModule, "CreatePostCubeNode failed on node[%s], skip it.",
                      GNodeGetName(headNode).GetString());
            continue;
        }
        effectTimes++;
        OPS_LOG_D(kFixpipePassModule, "Node[%s] fixpipe fusion success.", GNodeGetName(headNode).GetString());
    }
    OPS_LOG_D(kFixpipePassModule, "%s completed, effect times = %d.", kFixpipePassName, effectTimes);
    return effectTimes != 0 ? ge::GRAPH_SUCCESS : ge::GRAPH_NOT_CHANGED;
}

#if GE_COMPILER_VERSION_NUM >= 90300000U
REG_FUSION_PASS(FIXPIPEFUSIONPASS).Stage(ge::CustomPassStage::kCompatibleInherited);
#endif

} // namespace ops
