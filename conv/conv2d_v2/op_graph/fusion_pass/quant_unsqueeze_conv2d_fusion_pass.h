/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_UNSQUEEZE_CONV2D_FUSION_PASS_H
#define QUANT_UNSQUEEZE_CONV2D_FUSION_PASS_H

#include <set>

#include "../../conv/common/op_graph/fusion_pass/conv_fusion_base_pass.h"
#include "graph/gnode.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace QuantUnsqueezeConv2dFusion {

const ge::AscendString SQUEEZE = "Squeeze";
const ge::AscendString UNSQUEEZE = "Unsqueeze";

const std::set<ge::AscendString> BROADCAST_WHITELIST = {"Elu", "LeakyRelu", "PRelu", "Relu", "Sigmoid", "Tanh"};

const std::string FUSION_NAME = "QuantUnsqueezeConv2DFusionPass";
const std::string INTRINSIC_CONV_UB_TO_UB = "Intrinsic_conv_ub_to_ub";
const std::string INTRINSIC_DATA_MOVE_OUT2L1_DN2NZ = "Intrinsic_data_move_out2l1_dn2nz";

constexpr int64_t CONV1D_DIM_SIZE = 3;
constexpr uint32_t PLATFORM_INFO_OK = 0;
constexpr size_t FIRST_CONSUMER_IDX = 0;
constexpr size_t SINGLE_CONSUMER_CNT = 1;

} // namespace QuantUnsqueezeConv2dFusion

class __attribute__((visibility("default"))) QuantUnsqueezeConv2DFusionPass : public ConvFusionBasePass {
protected:
    void InitMember() override;
    bool CheckMatchStructure(const ge::GNode& matchNode) override;
    bool MeetRequirements(const ge::GNode& convNode) override;
    std::set<ge::AscendString> GetNodeTypes() const override;
    void PrintGraphStructure() const override;
    ge::Status ConvFusionPreImpl(ge::GraphPtr& graph, ge::GNode& convNode, ge::CustomPassContext& passContext) override;
    bool ConvFusionReplaceImpl(ge::GraphPtr& graph, ge::GNode& convNode, ge::CustomPassContext& passContext) override;

private:
    bool CheckSocCapability() const;
    bool CheckNodeNoControlAnchor(const ge::GNode& node) const;
    bool CheckUnsqueezeDim() const;
    bool UpdateQuantUnsqueezeDesc();
    bool UpdateSqueezeDequantDesc();
    ge::Status RelinkQuantUnsqueezeEdges(ge::Graph& graph);
    ge::Status RelinkSqueezeDequantEdges(ge::Graph& graph, ge::GNode& convNode);

    bool broadcastFlag = false;
    ge::GNodePtr broadcastNode = nullptr;
    ge::GNodePtr dequantNode = nullptr;
    ge::GNodePtr quantNode = nullptr;
    ge::GNodePtr squeezeNode = nullptr;
    ge::GNodePtr unsqueezeNode = nullptr;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // QUANT_UNSQUEEZE_CONV2D_FUSION_PASS_H
