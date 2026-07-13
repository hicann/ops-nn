/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV2D_TRANSPOSE_TO_V2_FUSION_PASS_H
#define CONV2D_TRANSPOSE_TO_V2_FUSION_PASS_H

#include "../../conv/common/op_graph/fusion_pass/conv_backprop_fusion_base_pass.h"

namespace ops {

class __attribute__((visibility("default"))) Conv2DTransposeToV2FusionPass : public ConvBackpropFusionBasePass {
public:
    explicit Conv2DTransposeToV2FusionPass(const std::vector<ge::AscendString>& opTypes)
        : ConvBackpropFusionBasePass(opTypes)
    {}

protected:
    ge::AscendString GetNodeType() const override;
    bool MeetRequirements(const ge::GNode& matchedNode) override;
    bool GetNodeDesc(const ge::GNode& node) override;
    bool GetNodeAttrs(const ge::GNode& node) override;

    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& matchedNode) override;

private:
    bool IsInputInt8(const ge::GNode& matchedNode);
    bool BuildConv3DTransposeNode(ge::es::EsGraphBuilder& builder, const ge::GNode& convTransposeNode,
                                  ge::es::EsTensorHolder& iInputSize,
                                  ConvBackpropFusionUtils::UnsqueezeNodeInfo& unsqueezeXInfo,
                                  ConvBackpropFusionUtils::UnsqueezeNodeInfo& unsqueezeFilterInfo,
                                  const std::string& nodeNamePrefix, ge::GNode& outNode, ge::TensorDesc& output3DDesc);
    bool ConnectOptionalInput(ge::es::EsGraphBuilder& builder, size_t nodeInputsSize, size_t inputIndex,
                              const ge::TensorDesc& inputDesc, ge::GNode& outNode);
    void SetConv3DTransposeAttrsAndDescs(ge::GNode& outNode, const ge::TensorDesc& unsqueezeXOutDesc,
                                         const ge::TensorDesc& unsqueezeFilterOutDesc,
                                         const ge::TensorDesc& output3DDesc);

    ge::TensorDesc biasDesc;
    ge::TensorDesc offsetWDesc;
    std::vector<int64_t> outputPadding;
    int64_t offsetX = 0;
};

} // namespace ops

#endif // CONV2D_TRANSPOSE_TO_V2_FUSION_PASS_H
