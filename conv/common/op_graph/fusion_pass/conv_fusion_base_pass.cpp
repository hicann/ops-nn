/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv_fusion_base_pass.h"

#include "es_nn_ops.h"
#include "version/ge-compiler_version.h"

namespace ge {
namespace fusion {
class SubgraphRewriter {
public:
    static Status Replace(const SubgraphBoundary& subgraph, const Graph& replacement);
#if GE_COMPILER_VERSION_NUM >= 90100000U
    static Status Replace(const SubgraphBoundary& subgraph, const Graph& replacement, CustomPassContext& ctx)
        __attribute__((weak));
#endif
};
} // namespace fusion
} // namespace ge

namespace Ops {
namespace NN {
namespace Conv {
using namespace ConvFusionUtils;
using namespace ge;
using namespace fusion;

bool ConvFusionBasePass::CanFuseNodes(const std::vector<GNode>& nodesBeforeFuse)
{
#if GE_COMPILER_VERSION_NUM >= 90100000U
    if (ge::fusion::GraphFuseInspectorUtils::CanFuse == nullptr) {
        return true;
    }
    AscendString failedReason;
    if (!ge::fusion::GraphFuseInspectorUtils::CanFuse(nodesBeforeFuse, failedReason)) {
        OP_LOGD(convDescInfo.nodeNameStr, "CanFuse failed, reason: %s.", failedReason.GetString());
        return false;
    }
#endif
    return true;
}

bool ConvFusionBasePass::ReportFuseNodes(const std::vector<GNode>& nodesBeforeFuse,
                                         const std::vector<GNode>& nodesAfterFuse, CustomPassContext& passContext)
{
#if GE_COMPILER_VERSION_NUM >= 90100000U
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse == nullptr) {
        return true;
    }
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse(nodesBeforeFuse, nodesAfterFuse, passContext) != SUCCESS) {
        OP_LOGE(convDescInfo.nodeNameStr, "ReportFuse failed.");
        return false;
    }
#endif
    return true;
}

bool ConvFusionBasePass::DefaultConvFusionReplaceImpl(const GNode& convNode, CustomPassContext& passContext)
{
    auto boundary = ConstructBoundary(convNode);
    FUSION_PASS_CHECK(
        boundary == nullptr,
        OP_LOGE("ConvFusionBasePass", "Construct boundary for %s failed.", convDescInfo.nodeNameStr.c_str()),
        return false);

    auto replacement = Replacement(convNode);
    FUSION_PASS_CHECK(
        replacement == nullptr,
        OP_LOGE("ConvFusionBasePass", "Construct replacement for %s failed.", convDescInfo.nodeNameStr.c_str()),
        return false);
#if GE_COMPILER_VERSION_NUM >= 90100000U
    using ReplaceWithCtxFn = Status (*)(const SubgraphBoundary&, const Graph&, CustomPassContext&);
    auto replaceWithCtx = static_cast<ReplaceWithCtxFn>(&ge::fusion::SubgraphRewriter::Replace);
    if (replaceWithCtx != nullptr) {
        FUSION_PASS_CHECK(replaceWithCtx(*boundary, *replacement, passContext) != SUCCESS,
                          OP_LOGE("ConvFusionBasePass", "Replace for %s failed.", convDescInfo.nodeNameStr.c_str()),
                          return false);
        return true;
    }
#endif

    FUSION_PASS_CHECK(SubgraphRewriter::Replace(*boundary, *replacement) != SUCCESS,
                      OP_LOGE("ConvFusionBasePass", "Replace for %s failed.", convDescInfo.nodeNameStr.c_str()),
                      return false);

    return true;
}

Status ConvFusionBasePass::Run(GraphPtr& graph, CustomPassContext& pass_context)
{
    std::string fusionName = "ConvFusionBasePass";
    OP_LOGD(fusionName, "Begin to do %s.", fusionName.c_str());

    std::vector<GNode> matchedNodes = {};

    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetMatchedNodes(graph, matchedNodes, GetNodeTypes()), return FAILED);
    FUSION_PASS_CHECK(matchedNodes.empty(), OP_LOGD(fusionName, "No matched node, exit."), return GRAPH_NOT_CHANGED);

    int32_t effectTimes = 0;
    for (auto& node : matchedNodes) {
        InitMember();
        if (!CheckMatchStructure(node)) {
            OP_LOGD(fusionName, "structure not matched, skip.");
            continue;
        }
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvDescInfo(node, convDescInfo), return FAILED);

        if (!MeetRequirements(node)) {
            OP_LOGD(fusionName, "%s is not meet requirements, skip.", convDescInfo.nodeNameStr.c_str());
            continue;
        }

        Status preRes = ConvFusionPreImpl(graph, node, pass_context);
        if (preRes == FAILED) {
            OP_LOGE(fusionName, "ConvFusionPreImpl for %s get failed.", convDescInfo.nodeNameStr.c_str());
            return FAILED;
        }
        if (preRes == CONV_NOT_CHANGED) {
            OP_LOGD(fusionName, "ConvFusionPreImpl for %s get not changed.", convDescInfo.nodeNameStr.c_str());
            continue;
        }

        if (!ConvFusionReplaceImpl(graph, node, pass_context)) {
            OP_LOGE(fusionName, "ConvFusionReplaceImpl for %s failed.", convDescInfo.nodeNameStr.c_str());
            return FAILED;
        }

        PrintGraphStructure();

        effectTimes++;
        OP_LOGD(fusionName, "%s fusion success.", convDescInfo.nodeNameStr.c_str());
    }

    OP_LOGD(fusionName, "%s completed.", fusionName.c_str());

    return effectTimes != 0 ? SUCCESS : CONV_NOT_CHANGED;
}

} // namespace Conv
} // namespace NN
} // namespace Ops
