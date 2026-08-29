/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV2D_SQUEEZE_BIASADD_FUSION_PASS_H
#define CONV2D_SQUEEZE_BIASADD_FUSION_PASS_H

#include <memory>
#include <vector>

#include "../../conv/common/op_graph/fusion_pass/conv_fusion_base_pass.h"
#include "graph/gnode.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace Conv2DSqueezeBiasaddFusion {

const ge::AscendString SQUEEZE = "Squeeze";
const ge::AscendString BIASADD = "BiasAdd";
const ge::AscendString VARIABLE = "Variable";
const std::string FUSION_NAME = "Conv2DSqueezeBiasaddFusionPass";

constexpr int32_t BIASADD_X_INPUT_INDEX = 0;
constexpr int32_t BIASADD_BIAS_INPUT_INDEX = 1;
constexpr int32_t SQUEEZE_INPUT_INDEX = 0;
constexpr int32_t SQUEEZE_OUTPUT_INDEX = 0;
constexpr size_t BIAS_DIM = 1;
} // namespace Conv2DSqueezeBiasaddFusion

class __attribute__((visibility("default"))) Conv2DSqueezeBiasaddFusionPass : public ConvFusionBasePass {
protected:
    void InitMember() override;
    bool CheckMatchStructure(const ge::GNode& matchNode) override;
    bool MeetRequirements(const ge::GNode& matchNode) override;
    std::set<ge::AscendString> GetNodeTypes() const override;
    void PrintGraphStructure() const override;
    bool ConvFusionReplaceImpl(ge::GraphPtr& graph, ge::GNode& matchNode, ge::CustomPassContext& passContext) override;

private:
    bool UpdateBiasAddDesc();
    ge::Status RelinkEdges(ge::Graph& graph, ge::GNode& squeezeNode);

    ge::GNodePtr convNode = nullptr;
    ge::GNodePtr biasaddNode = nullptr;
    ge::AscendString squeezeNodeName;
    ge::Shape convOutputShape = {};
    ge::TensorDesc convOutputDesc = {};
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // CONV2D_SQUEEZE_BIASADD_FUSION_PASS_H
