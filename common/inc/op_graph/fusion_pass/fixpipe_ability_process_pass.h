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
 * \file fixpipe_ability_process_pass.h
 * \brief fixpipe ability process
 */

#ifndef COMMON_GRAPH_FUSION_CUBE_UTILS_FIXPIPE_ABILITY_PROCESS_PASS_H_
#define COMMON_GRAPH_FUSION_CUBE_UTILS_FIXPIPE_ABILITY_PROCESS_PASS_H_

#include <vector>

#include "cube_utils/cube_common.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
/*!
 * \brief fixpipe ability process pass
 *
 * set the fixpipe ability attrs on the cube nodes before the fixpipe fusion:
 * - NodeCantAccess: the cube node cannot enter the fixpipe fusion (unknown shape / the matmul
 *   format and dtype limits / the conv3d fp32 input)
 * - UseGmAtomicAdd: the conv backprop filter cube uses the gm atomic add
 * - SupportMultipleOutput: the conv family supports the multi output fixpipe when the platform
 *   allows it (the merge head inherits the same ability)
 */
class __attribute__((visibility("default"))) FixPipeAbilityProcessPass : public ge::fusion::PatternFusionPass {
public:
    FixPipeAbilityProcessPass() = default;
    ~FixPipeAbilityProcessPass() override = default;

    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext) override;

protected:
    // Run is fully rewritten as a graph surgery flow which walks the graph by itself, the
    // three framework hooks below are only placeholders and are never invoked
    std::vector<ge::fusion::PatternUniqPtr> Patterns() override;
    bool MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& matchResult) override;
    ge::fusion::GraphUniqPtr Replacement(const std::unique_ptr<ge::fusion::MatchResult>& matchResult) override;

private:
    // the matmul node limits of the fixpipe fusion (output format / fp32 nz c0 / int32 chain / nd input)
    static bool CheckMatmulNode(const ge::GNodePtr& cubeNode);
    // the conv3d fp32 input cannot enter the fixpipe fusion
    static bool CheckConv3DNode(const ge::GNodePtr& cubeNode);
    // returns GRAPH_NOT_CHANGED when the cube node cannot enter the fixpipe fusion
    static ge::graphStatus CheckCubeNode(const ge::GNodePtr& cubeNode);
    // set the UseGmAtomicAdd / SupportMultipleOutput ability attrs on the cube (and merge) node
    static ge::graphStatus SetConvFixpipeAbilityAttr(const ge::GNodePtr& cubeNode);
    // whether any output of the node has an unknown shape dim (equals UnknownShapeUtils::IsUnknownShapeOp)
    static bool IsUnknownShapeOp(const ge::GNodePtr& node);
    // whether the platform fixpipe supports multiple outputs
    // (equals PlatformUtils::Instance().IsFixpipeSupportMultiOutput)
    static bool IsFixpipeSupportMultiOutput();
};
} // namespace ops

#endif // COMMON_GRAPH_FUSION_CUBE_UTILS_FIXPIPE_ABILITY_PROCESS_PASS_H_
