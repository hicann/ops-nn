/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV3D_QUANT_PROCESS_FUSION_PASS_H
#define CONV3D_QUANT_PROCESS_FUSION_PASS_H

#include <set>

#include "../../../common/op_graph/fusion_pass/conv_fusion_base_pass.h"
#include "../../../../common/inc/op_graph/fusion_pass/bias_optimize_quant_rollback_base.h"

namespace Ops {
namespace NN {
namespace Conv {

class __attribute__((visibility("default"))) Conv3DQuantProcessFusionPass : public ConvFusionBasePass,
                                                                            public ops::BiasOptimizeQuantRollbackBase {
public:
    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext) override;

protected:
    void InitMember() override;
    bool MeetRequirements(const ge::GNode& convNode) override;
    std::set<ge::AscendString> GetNodeTypes() const override;
    void PrintGraphStructure() const override;

    ge::Status GetCoValue(const ge::GNodePtr& cubeNode, int64_t& co) override;
    ge::Status SetDataTypeOfNodes(const ge::GNodePtr& cubeNode) override;
    ge::GNodePtr GetCubeNodeInputNode(const ge::GNodePtr& cubeNode) const override;

private:
    bool CheckQuantChain(const ge::GNode& cubeNode);
    ge::Status SetPadNodeDataType(ge::GNode& padNode) const;

    ge::GNodePtr quantNode_ = nullptr;
    ge::GNodePtr dequantNode_ = nullptr;
    ge::GNodePtr padNode_ = nullptr;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // CONV3D_QUANT_PROCESS_FUSION_PASS_H
