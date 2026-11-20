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
 * \file fixpipe_pre_quant_pass.h
 * \brief Cube + AscendDequant + (relu) + AscendQuant fusion pass
 */

#ifndef COMMON_GRAPH_FUSION_CUBE_UTILS_FIXPIPE_PRE_QUANT_PASS_H_
#define COMMON_GRAPH_FUSION_CUBE_UTILS_FIXPIPE_PRE_QUANT_PASS_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "cube_utils/cube_common.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
/*!
 * \brief fixpipe pre quant fusion pass
 *
 * absorb the tail AscendQuant of Cube -> AscendDequant -> (relu family) -> AscendQuant:
 * - Cube -> Dequant -> Quant                 ===> Cube -> VectorMulScalar -> Dequant(requant)
 * - Cube -> Dequant -> Relu -> Quant         ===> Cube -> VectorMulScalar -> Dequant(requant) -> Relu(requant)
 * the quant scale/offset is folded into the cloned dequant, the dequant scale input is pre-multiplied
 * by a VectorMulScalar host op, so the fixpipe unit can fuse the whole chain on the soc.
 */
class __attribute__((visibility("default"))) FIXPIPEAPREQUANTFUSIONPASS : public ge::fusion::PatternFusionPass {
public:
    FIXPIPEAPREQUANTFUSIONPASS() = default;
    ~FIXPIPEAPREQUANTFUSIONPASS() override = default;

    /*!
     * \brief read the fixpipe unit config of the current platform
     * \param[in] context custom pass context, used to read the compile options
     * \return true when both the pre_conv and pre_act units exist on the platform
     */
    bool ReadConfig(const ge::CustomPassContext& context);

    // access the fixpipe unit map read by ReadConfig (equals the legacy GetFixPipeMap)
    const std::map<std::string, std::map<std::string, std::vector<CONFIGDTYPE>>>& GetFixPipeMap() const;

    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext) override;

protected:
    // Run is fully rewritten as a graph surgery flow, the three framework hooks below are
    // only placeholders to satisfy the PatternFusionPass interface and are never invoked
    std::vector<ge::fusion::PatternUniqPtr> Patterns() override;
    bool MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& matchResult) override;
    ge::fusion::GraphUniqPtr Replacement(const std::unique_ptr<ge::fusion::MatchResult>& matchResult) override;

private:
    // pattern0 parse: collect the direct quant successors and the relu->quant branches of dequant
    ge::graphStatus Pattern0Parse(const ge::GNodePtr& dequantNode, std::vector<ge::GNodePtr>& quants,
                                  std::map<ge::GNodePtr, std::vector<ge::GNodePtr>>& relus) const;
    void Pattern0ParseForRelu(const ge::GNodePtr& dequantNode, const ge::GNodePtr& reluNode,
                              std::vector<ge::GNodePtr>& reluquants) const;
    // judge whether the quant value constraints allow the requant absorb on the relu branch
    ge::graphStatus JudgeQuantReluValue(const ge::GNodePtr& dequantNode, const ge::GNodePtr& reluNode,
                                        const ge::GNodePtr& quantNode) const;
    // judge whether the multi output scene of the head node allows the requant absorb
    bool JudgeMultiOutNode(const ge::GNodePtr& headNode, const ge::GNodePtr& dequantNode,
                           const std::vector<ge::GNodePtr>& quants,
                           const std::map<ge::GNodePtr, std::vector<ge::GNodePtr>>& relus) const;
    // check whether the multi-branch size is allowed when the head node has no multiple output ability
    static bool CheckQuantsAndrelusSize(const ge::GNodePtr& dequantNode, const std::vector<ge::GNodePtr>& quants,
                                        const std::map<ge::GNodePtr, std::vector<ge::GNodePtr>>& relus);
    bool IsCubeVectorCombineFuse() const;
    // tiny platform check: Intrinsic_set_l1 supports f16 but not f32 (equals UtilPlatform::isTiny)
    bool IsTinyPlatform() const;
    // handle one Cube -> Dequant branch: parse, judge and run the requant absorb
    ge::graphStatus HandlePattern0(const ge::GNodePtr& headNode, ge::Graph& graph, const ge::GNodePtr& dequantNode,
                                   std::vector<ge::GNodePtr>& newNodes);
    ge::graphStatus ReDequantProcess(const ge::GNodePtr& headNode, ge::Graph& graph, const ge::GNodePtr& dequantNode,
                                     std::vector<ge::GNodePtr>& quants,
                                     std::map<ge::GNodePtr, std::vector<ge::GNodePtr>>& relus,
                                     std::vector<ge::GNodePtr>& newNodes);
    // clone the dequant / relu node per quant branch and relink the edges
    ge::graphStatus DealDequant(const ge::GNodePtr& headNode, ge::Graph& graph, const ge::GNodePtr& dequantNode,
                                std::vector<ge::GNodePtr>& quants,
                                std::map<ge::GNodePtr, std::vector<ge::GNodePtr>>& relus,
                                std::vector<ge::GNodePtr>& dequants, std::vector<ge::GNodePtr>& newNodes);
    // build the VectorMulScalar host op and the enhanced dequant inputs for one quant branch
    ge::GNodePtr DealDequantInput(const ge::GNodePtr& enhancedDequantNode, ge::Graph& graph,
                                  const ge::GNodePtr& dequantNode, const ge::GNodePtr& quantNode,
                                  std::vector<ge::GNodePtr>& newNodes) const;
    // relink: head -> enhanced dequant -> the consumers of quant
    ge::graphStatus RelinkDequantWithQuant(ge::Graph& graph, const ge::GNodePtr& headNode,
                                           const ge::GNodePtr& enhancedDequantNode, const ge::GNodePtr& dequantNode,
                                           const ge::GNodePtr& quantNode) const;
    // relink: head -> enhanced dequant -> cloned relu -> the consumers of quant
    ge::graphStatus RelinkReluWithQuant(ge::Graph& graph, const ge::GNodePtr& headNode,
                                        const ge::GNodePtr& enhancedDequantNode, const ge::GNodePtr& newReluNode,
                                        const ge::GNodePtr& reluNode, const ge::GNodePtr& quantNode) const;
    // move the data/control consumers of the quant output to the new producer
    ge::graphStatus RelinkQuantConsumers(ge::Graph& graph, const ge::GNodePtr& quantNode,
                                         const ge::GNodePtr& newProducer) const;
    ge::graphStatus DeleteNode(ge::Graph& graph, const ge::GNodePtr& toDeleteNode) const;
    ge::graphStatus DealQuant(ge::Graph& graph, std::vector<ge::GNodePtr>& quantNodes) const;
    ge::graphStatus DealRelus(ge::Graph& graph, const ge::GNodePtr& dequantNode,
                              std::map<ge::GNodePtr, std::vector<ge::GNodePtr>>& relus) const;
    ge::graphStatus DeleteDequant(ge::Graph& graph, const ge::GNodePtr& dequantNode) const;
    // walk all AscendDequant successors of the head node (the merge node of the cube when it exists)
    ge::graphStatus FusionImpl(const ge::GNodePtr& headNode, ge::Graph& graph, std::vector<ge::GNodePtr>& newNodes);
    // clone a node with the same type/attrs/input-output desc, returns the new node in the graph
    ge::GNodePtr CloneNode(ge::Graph& graph, const ge::GNodePtr& srcNode, const std::string& nodeName) const;
    // eltwise cycle check of the multi output branch (equals the FusionCycleDetector logic)
    bool CheckEltwiseCycleExist(const ge::GNodePtr& node) const;
    // only Eltwise/Add/Sub supported by the fixpipe units of the platform needs the cycle check
    bool NeedCheckCycle(const ge::GNodePtr& node) const;
    // whether the platform fixpipe supports multiple outputs
    bool IsFixpipeSupportMultiOutput() const;
    // whether the op is dump visible (equals IsDumpableOp of the legacy framework)
    static bool IsDumpableOp(const ge::GNodePtr& node);
    // atomic counter for the generated node name suffix (equals PatternFusionBasePass::GetAtomicId)
    static uint32_t GetAtomicId();

protected:
    // the relu types of the pre_act unit, shared with the derived fixpipe relu quant pass
    // (equals the protected config_relus_ of the legacy FixPipePreQuantPass)
    std::set<std::string> configRelus_;

private:
    std::vector<std::string> mUnits_;
    std::map<std::string, std::map<std::string, std::vector<CONFIGDTYPE>>> mFixpipeMap_;
};
} // namespace ops

#endif // COMMON_GRAPH_FUSION_CUBE_UTILS_FIXPIPE_PRE_QUANT_PASS_H_
