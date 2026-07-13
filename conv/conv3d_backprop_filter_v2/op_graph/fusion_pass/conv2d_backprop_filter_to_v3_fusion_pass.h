/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV2D_BACKPROP_FILTER_TO_V3_FUSION_PASS_H
#define CONV2D_BACKPROP_FILTER_TO_V3_FUSION_PASS_H

#include "../../conv/common/op_graph/fusion_pass/conv_backprop_fusion_base_pass.h"

namespace ops {

const ge::AscendString CONV2D_BACKPROP_FILTER = "Conv2DBackpropFilter";
const ge::AscendString CONV2D_BP_FILTER_TO_V3_PASS = "Conv2DBackpropFilterToV3FusionPass";

class __attribute__((visibility("default"))) Conv2DBackpropFilterToV3FusionPass : public ConvBackpropFusionBasePass {
public:
    explicit Conv2DBackpropFilterToV3FusionPass(const std::vector<ge::AscendString>& opTypes)
        : ConvBackpropFusionBasePass(opTypes)
    {}

protected:
    ge::AscendString GetNodeType() const override;
    bool GetNodeAttrs(const ge::GNode& node) override;

    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& convBpFilterNode) override;

private:
    bool BuildConv3DBackpropFilterNode(ge::es::EsGraphBuilder& builder, ge::es::EsTensorHolder& iFilterSize,
                                       ConvBackpropFusionUtils::UnsqueezeNodeInfo& unsqueezeXInfo,
                                       ConvBackpropFusionUtils::UnsqueezeNodeInfo& unsqueezeGradOutputInfo,
                                       const std::string& nodeNamePrefix, ge::GNode& outNode,
                                       ge::TensorDesc& output3DDesc);
    void SetConv3DBackpropFilterAttrsAndDescs(ge::GNode& outNode, const ge::TensorDesc& unsqueezeXOutDesc,
                                              const ge::TensorDesc& unsqueezeGradOutputOutDesc,
                                              const ge::TensorDesc& output3DDesc);
};

} // namespace ops

#endif // CONV2D_BACKPROP_FILTER_TO_V3_FUSION_PASS_H
