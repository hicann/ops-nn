/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEPTHWISE_DF_FUSION_PASS_H
#define DEPTHWISE_DF_FUSION_PASS_H

#include "../../conv/common/op_graph/fusion_pass/conv_backprop_fusion_base_pass.h"

namespace ops {

class __attribute__((visibility("default"))) DepthwiseDfFusionPass : public ConvBackpropFusionBasePass {
public:
    explicit DepthwiseDfFusionPass(const std::vector<ge::AscendString>& opTypes) : ConvBackpropFusionBasePass(opTypes)
    {}

protected:
    ge::AscendString GetNodeType() const override;
    bool MeetRequirements(const ge::GNode& matchedNode) override;
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& matchedNode) override;
    bool GetNodeAttrs(const ge::GNode& node) override;
    void SetNodeAttrs(ge::GNode& outNode) override;

    bool ValidateFilterDesc(const ge::GNode& matchedNode);
    bool GetResizeDepthwiseFilterShape(std::vector<int64_t>& resizeShape);

private:
    bool ValidateArch35Descs();
    void CreateBoundaryInputs(ge::es::EsGraphBuilder& builder, ge::es::EsTensorHolder& iFilterHolder,
                              ge::es::EsTensorHolder& iGradOutputHolder, ge::es::EsTensorHolder& iInputSizeHolder);
    bool BuildOptionalTranspose(ge::es::EsGraphBuilder& builder, const std::string& nodeNamePrefix,
                                const ge::es::EsTensorHolder& iFilterHolder, ge::es::EsTensorHolder& reshapeInputHolder,
                                ge::TensorDesc& reshapeInputDesc);
    bool BuildReshapeNode(ge::es::EsGraphBuilder& builder, const std::vector<int64_t>& filterResetShape,
                          const std::string& nodeNamePrefix, const ge::es::EsTensorHolder& reshapeInputHolder,
                          const ge::TensorDesc& reshapeInputDesc, ge::es::EsTensorHolder& reshapeOutput,
                          ge::TensorDesc& targetFilterDesc);
    ge::fusion::GraphUniqPtr BuildDynamicTargetNode(ge::es::EsGraphBuilder& builder, const std::string& targetOpType,
                                                    const std::string& targetNodeName,
                                                    const ge::es::EsTensorHolder& iInputSizeHolder,
                                                    const ge::es::EsTensorHolder& iGradOutputHolder,
                                                    const ge::es::EsTensorHolder& reshapeOutput,
                                                    const ge::TensorDesc& targetFilterDesc);
    ge::fusion::GraphUniqPtr BuildStaticTargetNode(ge::es::EsGraphBuilder& builder, const std::string& targetOpType,
                                                   const std::string& targetNodeName,
                                                   const ge::es::EsTensorHolder& iGradOutputHolder,
                                                   const ge::es::EsTensorHolder& reshapeOutput,
                                                   const ge::TensorDesc& targetFilterDesc);
};

} // namespace ops

#endif // DEPTHWISE_DF_FUSION_PASS_H
