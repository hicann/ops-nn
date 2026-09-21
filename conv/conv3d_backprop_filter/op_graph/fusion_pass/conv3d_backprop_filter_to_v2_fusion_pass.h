/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV3D_BACKPROP_FILTER_TO_V2_FUSION_PASS_H
#define CONV3D_BACKPROP_FILTER_TO_V2_FUSION_PASS_H

#include "../../conv/common/op_graph/fusion_pass/conv_backprop_fusion_base_pass.h"

namespace ops {

const ge::AscendString CONV_BACKPROP_FILTER = "Conv3DBackpropFilter";
const ge::AscendString CONV_BACKPROP_FILTER_V2_PASS = "Conv3DBackpropFilterToV2FusionPass";

class __attribute__((visibility("default"))) Conv3DBackpropFilterToV2FusionPass : public ConvBackpropFusionBasePass {
public:
    explicit Conv3DBackpropFilterToV2FusionPass(const std::vector<ge::AscendString>& opTypes)
        : ConvBackpropFusionBasePass(opTypes)
    {}

protected:
    ge::AscendString GetNodeType() const override;
    bool CheckTransposeNeeded() override;
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& convBpFilterNode) override;

private:
    bool IsDynamicShape() const;
    bool GetOutputChannelDims(int64_t& cin, int64_t& cout) const;
    bool GetInputDepthDim(int64_t& di) const;
    bool IsShapeNeedTranspose() const;
    bool CreateOutputWithTranspose(ge::es::EsGraphBuilder& builder,
                                   const ge::es::EsTensorHolder& conv3dBackpropFilterV2,
                                   ge::GNode* conv3dBackpropFilterV2Node, ge::es::EsTensorHolder& transOutput);
};

} // namespace ops

#endif // CONV3D_BACKPROP_FILTER_TO_V2_FUSION_PASS_H
