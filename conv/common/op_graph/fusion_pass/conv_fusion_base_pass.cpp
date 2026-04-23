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

#include "log/log.h"
#include "es_nn_ops.h"
#include "ge/fusion/graph_rewriter.h"

namespace Ops {
namespace NN {
namespace Conv {
using namespace ConvFusionUtils;
using namespace ge;
using namespace fusion;

Status ConvFusionBasePass::Run(GraphPtr &graph, CustomPassContext &pass_context)
{
    std::string fusionName = "ConvFusionBasePass";
    OP_LOGD(fusionName, "Begin to do %s.", fusionName.c_str());

    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::CheckSocSupport(GetSocSupportList(), npuArch),
        return GRAPH_NOT_CHANGED);

    std::vector<GNode> matchedNodes = {};

    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetMatchedNodes(graph, matchedNodes, GetNodeType()), return FAILED);
    FUSION_PASS_CHECK(matchedNodes.empty(), OP_LOGD(fusionName, "No matched node, exit."), return GRAPH_NOT_CHANGED);

    int32_t effectTimes = 0;
    for (auto &node : matchedNodes) {
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvDescInfo(node, convDescInfo), return FAILED);

        // Check fusion contidions.
        if (!MeetRequirements(node)) {
            OP_LOGD(fusionName, "%s is not meet requirements, skip.", convDescInfo.nodeNameStr.c_str());
            continue;
        }

        InitMember();

        // Transfer to PostCube node.
        if (!PostCubeFusionImpl(graph, node, pass_context)) {
            OP_LOGD(fusionName, "Transfer postcube for %s failed, skip.", convDescInfo.nodeNameStr.c_str());
            continue;
        }

        // Create sup graph boundary to be replaced.
        auto boundary = ConstructBoundary(node);
        FUSION_PASS_CHECK(boundary == nullptr,
            OP_LOGE(fusionName, "Construct boundary for %s failed.", convDescInfo.nodeNameStr.c_str()),
            return FAILED);

        auto replacement = Replacement(node);
        FUSION_PASS_CHECK(replacement == nullptr,
            OP_LOGE(fusionName, "Construct replacement for %s failed.", convDescInfo.nodeNameStr.c_str()),
            return FAILED);

        FUSION_PASS_CHECK(SubgraphRewriter::Replace(*boundary, *replacement) != SUCCESS,
            OP_LOGE(fusionName, "Replace for %s failed.", convDescInfo.nodeNameStr.c_str()), return FAILED);
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