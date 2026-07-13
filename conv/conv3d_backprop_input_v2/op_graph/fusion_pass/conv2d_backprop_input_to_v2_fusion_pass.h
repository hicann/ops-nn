/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV2D_BACKPROP_INPUT_TO_V2_FUSION_PASS_H
#define CONV2D_BACKPROP_INPUT_TO_V2_FUSION_PASS_H

#include "../../conv/common/op_graph/fusion_pass/conv_backprop_fusion_base_pass.h"

namespace ops {

class __attribute__((visibility("default"))) Conv2DBackpropInputToV2FusionPass : public ConvBackpropFusionBasePass {
public:
    explicit Conv2DBackpropInputToV2FusionPass(const std::vector<ge::AscendString>& opTypes)
        : ConvBackpropFusionBasePass(opTypes)
    {}

protected:
    ge::AscendString GetNodeType() const override;
    bool GetNodeAttrs(const ge::GNode& node) override;

    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& convBpInputNode) override;

private:
    bool BuildConv3DBackpropInputNode(ge::es::EsGraphBuilder& builder, const ge::GNode& convBpInputNode,
                                      ge::es::EsTensorHolder& iInputSize,
                                      ConvBackpropFusionUtils::UnsqueezeNodeInfo& unsqueezeFilterInfo,
                                      ConvBackpropFusionUtils::UnsqueezeNodeInfo& unsqueezeGradOutputInfo,
                                      const std::string& nodeNamePrefix, ge::GNode& outNode,
                                      ge::TensorDesc& output3DDesc);
    void SetConv3DBackpropInputAttrsAndDescs(ge::GNode& outNode, const ge::TensorDesc& unsqueezeFilterOutDesc,
                                             const ge::TensorDesc& unsqueezeGradOutputOutDesc,
                                             const ge::TensorDesc& output3DDesc);
};

} // namespace ops

#endif // CONV2D_BACKPROP_INPUT_TO_V2_FUSION_PASS_H
