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
 * \file fixpipe_relu_quant_pass.h
 * \brief Cube + relu + AscendQuant pass
 */

#ifndef COMMON_GRAPH_FUSION_CUBE_UTILS_FIXPIPE_RELU_QUANT_PASS_H_
#define COMMON_GRAPH_FUSION_CUBE_UTILS_FIXPIPE_RELU_QUANT_PASS_H_

#include <map>
#include <vector>

#include "fusion_pass/fixpipe_pre_quant_pass.h"

namespace ops {
/*!
 * \brief fixpipe relu quant fusion pass
 *
 * swap the relu and the tail AscendQuant of Cube -> relu -> AscendQuant so the quant
 * directly follows the cube and the cloned relu consumes the quant output (the fixpipe
 * unit requires the quant before the act):
 * - Cube -> Relu -> Quant -> consumers ===> Cube -> Quant -> Relu(fixpipe) -> consumers
 * the quant nodes sharing the same scale/offset/sqrt_mode/output dtype reuse the same
 * cloned relu, the stale relu is removed once it has no data successor left.
 */
class __attribute__((visibility("default"))) FIXPIPEARELUQUANTFUSIONPASS : public FIXPIPEAPREQUANTFUSIONPASS {
public:
    FIXPIPEARELUQUANTFUSIONPASS() = default;
    ~FIXPIPEARELUQUANTFUSIONPASS() override = default;

    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext) override;

private:
    // walk every quant successor of the relu, swap it before the cloned relu or reuse the
    // matched cloned relu (equals the legacy FusionImpl)
    ge::graphStatus ReluQuantFusionImpl(ge::Graph& graph, const ge::GNodePtr& cubeNode, const ge::GNodePtr& reluNode,
                                        std::vector<ge::GNodePtr>& newNodes) const;
    // clone the relu node whose input/output desc takes the quant output desc
    ge::GNodePtr CreateReluNode(ge::Graph& graph, const ge::GNodePtr& reluNode, const ge::GNodePtr& quantNode,
                                std::vector<ge::GNodePtr>& newNodes) const;
    // find the cloned relu of the processed quant node with the same scale/offset/sqrt_mode/dtype
    // (equals the legacy FindQuantOutAnchor on the anchor map)
    ge::GNodePtr FindMatchedReluNode(const ge::GNodePtr& quantNode,
                                     const std::map<ge::GNodePtr, ge::GNodePtr>& reluMap) const;
    // relink: cube -> quant -> cloned relu -> the consumers of quant (equals the legacy RelinkEdge)
    ge::graphStatus RelinkEdge(ge::Graph& graph, const ge::GNodePtr& cubeNode, const ge::GNodePtr& reluNode,
                               const ge::GNodePtr& quantNode, const ge::GNodePtr& newReluNode,
                               std::map<ge::GNodePtr, ge::GNodePtr>& reluMap) const;
    // remove the quant node whose consumers are moved to the matched cloned relu
    ge::graphStatus DeleteQuantNode(ge::Graph& graph, const ge::GNodePtr& quantNode,
                                    const ge::GNodePtr& matchedReluNode) const;
    // the quant value constraints of the relu branch (equals the legacy JudgeQuantReluValue)
    ge::graphStatus JudgeQuantReluValue(const ge::GNodePtr& reluNode, const ge::GNodePtr& quantNode) const;
    // eltwise cycle check of the multi output branch (equals the FusionCycleDetector logic)
    bool CheckEltwiseCycleExist(const ge::GNodePtr& node) const;
    // only Eltwise/Add/Sub supported by the fixpipe units of the platform needs the cycle check
    // (reads the platform config independently, equals the legacy needCheckCycle)
    bool NeedCheckCycle(const ge::GNodePtr& node) const;
    // whether the platform fixpipe supports multiple outputs
    bool IsFixpipeSupportMultiOutput() const;
    // atomic counter for the generated node name suffix (equals PatternFusionBasePass::GetAtomicId)
    static uint32_t GetAtomicId();
};
} // namespace ops

#endif // COMMON_GRAPH_FUSION_CUBE_UTILS_FIXPIPE_RELU_QUANT_PASS_H_
