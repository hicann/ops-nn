/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV2D_TO_CONV2DV2_FUSION_PASS_H
#define CONV2D_TO_CONV2DV2_FUSION_PASS_H

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "../../conv/common/op_graph/fusion_pass/conv_fusion_base_pass.h"
#include "ge/fusion/subgraph_boundary.h"
#include "op_common/op_host/util/op_const_def.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace Conv2dToConv2dV2Fusion {

const std::map<std::string, NpuArch> SUPPORT_SOC_LIST = {{"Ascend950", NpuArch::DAV_3510},
                                                         {"Ascend350", NpuArch::DAV_3510},
                                                         {"Ascend960DT", NpuArch::DAV_9201},
                                                         {"Ascend960PR", NpuArch::DAV_9202},
                                                         {"Ascend5162", Ops::Base::DAV_5162}};

} // namespace Conv2dToConv2dV2Fusion

class __attribute__((visibility("default"))) Conv2dToConv2dV2FusionPass : public ConvFusionBasePass {
protected:
    void InitMember() override;
    bool MeetRequirements(const ge::GNode& convNode) override;
    std::set<ge::AscendString> GetNodeTypes() const override;
    void PrintGraphStructure() const override;
    bool ConvFusionReplaceImpl(ge::GraphPtr& graph, ge::GNode& convNode, ge::CustomPassContext& passContext) override;

private:
    bool AdjustLayout(const ge::GraphPtr& graph, ge::GNode& fusedNode);
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& convNode) override;
    std::unique_ptr<ge::fusion::SubgraphBoundary> ConstructBoundary(const ge::GNode& convNode) override;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // CONV2D_TO_CONV2DV2_FUSION_PASS_H
