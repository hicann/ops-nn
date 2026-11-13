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
 * \file fixpipe_pass.h
 * \brief Cube + pre_conv + pre_act + (post_eltwise + post_act + post_quant) + post_transform fusion pass
 */

#ifndef COMMON_GRAPH_FUSION_CUBE_UTILS_FIXPIPE_PASS_H_
#define COMMON_GRAPH_FUSION_CUBE_UTILS_FIXPIPE_PASS_H_

#include <string>
#include <vector>

#include "cube_utils/cube_utils.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
/*!
 * \brief fixpipe fusion pass
 *
 * fold the post op chain of the cube node into the FixPipe hardware unit:
 * Cube -> pre_conv -> pre_act -> (post_eltwise -> post_act -> post_quant) -> post_transform
 * ===> FixPipe.
 * the match/modify/create flow is provided by the PostCubeUtils of the common cube_utils
 * (the legacy FixPipePass logic has been migrated there), this pass only keeps the pass
 * entry: walk the cube nodes of the graph and drive the three-stage api.
 */
class __attribute__((visibility("default"))) FIXPIPEFUSIONPASS : public ge::fusion::PatternFusionPass {
public:
    FIXPIPEFUSIONPASS() = default;
    ~FIXPIPEFUSIONPASS() override = default;

    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext) override;

protected:
    // Run is fully rewritten as a graph surgery flow which walks the graph by itself, the
    // three framework hooks below are only placeholders and are never invoked
    std::vector<ge::fusion::PatternUniqPtr> Patterns() override;
    bool MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& matchResult) override;
    ge::fusion::GraphUniqPtr Replacement(const std::unique_ptr<ge::fusion::MatchResult>& matchResult) override;
};
} // namespace ops

#endif // COMMON_GRAPH_FUSION_CUBE_UTILS_FIXPIPE_PASS_H_
