/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEPTHWISE_DW_MUL_FUSION_PASS_H
#define DEPTHWISE_DW_MUL_FUSION_PASS_H

#include "../../conv/common/op_graph/fusion_pass/conv_backprop_fusion_base_pass.h"

namespace ops {

class __attribute__((visibility("default"))) DepthwiseDwMulFusionPass : public ConvBackpropFusionBasePass {
public:
    explicit DepthwiseDwMulFusionPass(const std::vector<ge::AscendString>& opTypes)
        : ConvBackpropFusionBasePass(opTypes)
    {}

protected:
    ge::AscendString GetNodeType() const override;
    bool MeetRequirements(const ge::GNode& matchedNode) override;
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& matchedNode) override;

private:
    bool ValidateFilterDesc(const ge::GNode& matchedNode);
    bool GetResizeDepthwiseFilterShape(std::vector<int64_t>& resizeShape);
    void CreateBoundaryInputs(ge::es::EsGraphBuilder& builder, ge::es::EsTensorHolder& iXHolder,
                              ge::es::EsTensorHolder& iFilterSizeHolder, ge::es::EsTensorHolder& iGradOutputHolder);
    bool BuildTargetNode(ge::es::EsGraphBuilder& builder, const std::string& targetOpType,
                         const std::string& targetNodeName, const ge::es::EsTensorHolder& iXHolder,
                         const ge::es::EsTensorHolder& iFilterSizeHolder,
                         const ge::es::EsTensorHolder& iGradOutputHolder, ge::GNode& targetNode,
                         ge::TensorDesc& targetOutDesc);
    bool BuildTargetGNode(ge::es::EsGraphBuilder& builder, const std::string& targetOpType,
                          const std::string& targetNodeName, const ge::es::EsTensorHolder& iXHolder,
                          const ge::es::EsTensorHolder& iFilterSizeHolder,
                          const ge::es::EsTensorHolder& iGradOutputHolder, ge::GNode& targetNode) const;
    bool UpdateTargetNodeDescs(ge::GNode& targetNode, ge::TensorDesc& targetOutDesc);
    bool BuildReshapeNode(ge::es::EsGraphBuilder& builder, const std::string& nodeNamePrefix, ge::GNode& targetNode,
                          const std::vector<int64_t>& reshapeOutShape, const ge::TensorDesc& reshapeInDesc,
                          ge::es::EsTensorHolder& reshapeOutput, ge::TensorDesc& reshapeOutDesc) const;
    bool BuildOptionalTranspose(ge::es::EsGraphBuilder& builder, ge::Format originFormat,
                                const std::string& nodeNamePrefix, const ge::es::EsTensorHolder& reshapeOutput,
                                const ge::TensorDesc& reshapeOutDesc, ge::TensorDesc& finalOutDesc,
                                ge::es::EsTensorHolder& finalOutput) const;
};

} // namespace ops

#endif // DEPTHWISE_DW_MUL_FUSION_PASS_H
