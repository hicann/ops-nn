/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge/fusion/pass/pattern_fusion_pass.h"

#ifndef NN_PRELU_GRAD_FUSION_PASS_H
#define NN_PRELU_GRAD_FUSION_PASS_H

namespace ops {
using namespace ge;
using namespace fusion;

class __attribute__((visibility("default"))) PReluGradFusionPass : public PatternFusionPass {
protected:
    std::vector<PatternUniqPtr> Patterns() override;

    bool MeetRequirements(const std::unique_ptr<MatchResult>& match_result) override;

    std::unique_ptr<Graph> Replacement(const std::unique_ptr<MatchResult>& match_result) override;
};

static void GetInputsInfo(const std::vector<SubgraphInput>& subgraph_inputs, std::vector<Shape>& input_shapes,
    std::vector<DataType>& input_dtypes, std::vector<Format>& input_formats);
static Status InferShape(const GraphUniqPtr& replace_graph, const std::vector<SubgraphInput>& subgraph_inputs);

} //namespace ops
#endif // NN_PRELU_GRAD_FUSION_PASS_H
