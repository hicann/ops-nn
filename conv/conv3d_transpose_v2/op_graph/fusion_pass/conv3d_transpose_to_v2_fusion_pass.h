/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV3D_TRANSPOSE_TO_V2_FUSION_PASS_H
#define CONV3D_TRANSPOSE_TO_V2_FUSION_PASS_H

#include "../../conv/common/op_graph/fusion_pass/conv_backprop_fusion_base_pass.h"

namespace ops {

class __attribute__((visibility("default"))) Conv3DTransposeToV2FusionPass : public ConvBackpropFusionBasePass {
public:
    explicit Conv3DTransposeToV2FusionPass(const std::vector<ge::AscendString>& opTypes)
        : ConvBackpropFusionBasePass(opTypes)
    {}

protected:
    ge::AscendString GetNodeType() const override;
    bool GetNodeDesc(const ge::GNode& node) override;
    bool GetNodeAttrs(const ge::GNode& node) override;
    bool CheckTransposeNeeded() override;
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& convTransposeNode) override;

private:
    bool CheckDtypeSupported();
    bool CreateFilterTranspose(ge::es::EsGraphBuilder& builder, const ge::es::EsTensorHolder& filter,
                               ge::es::EsTensorHolder& transFilter, ge::TensorDesc& transFilterDesc);

    ge::TensorDesc biasDesc;
    ge::TensorDesc offsetWDesc;
    bool hasBias = false;
    bool hasOffsetW = false;
    std::vector<int64_t> outputPadding = {0, 0, 0, 0, 0};
    int64_t offsetX = 0;
};

} // namespace ops

#endif // CONV3D_TRANSPOSE_TO_V2_FUSION_PASS_H
