/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV3D_BACKPROP_INPUT_TO_V2_FUSION_PASS_H
#define CONV3D_BACKPROP_INPUT_TO_V2_FUSION_PASS_H

#include "../../conv/common/op_graph/fusion_pass/conv_backprop_fusion_base_pass.h"

namespace ops {

const ge::AscendString CONV_BACKPROP_INPUT = "Conv3DBackpropInput";
const ge::AscendString CONV_BACKPROP_INPUT_V2_PASS = "Conv3DBackpropInputToV2FusionPass";

class __attribute__((visibility("default"))) Conv3DBackpropInputToV2FusionPass : public ConvBackpropFusionBasePass {
public:
    explicit Conv3DBackpropInputToV2FusionPass(const std::vector<ge::AscendString>& opTypes)
        : ConvBackpropFusionBasePass(opTypes)
    {}

protected:
    ge::AscendString GetNodeType() const override;
    bool CheckTransposeNeeded() override;
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& convBpInputNode) override;

private:
    // InputV2特有的Shape白名单
    const std::vector<std::vector<int64_t>> outputShapeWhitelist = {
        {256, 1, 8, 8, 512},   {256, 1, 32, 32, 128}, {256, 1, 8, 8, 1024},  {256, 1, 16, 16, 256},
        {256, 1, 64, 64, 64},  {256, 1, 32, 32, 256}, {256, 1, 64, 64, 128}, {256, 1, 128, 128, 3},
        {256, 1, 16, 16, 512}, {64, 1, 32, 32, 128},  {64, 1, 64, 64, 32}};

    const std::vector<std::vector<int64_t>> filterShapeWhitelist = {
        {1, 4, 4, 512, 1037}, {1, 4, 4, 128, 269}, {1, 4, 4, 1024, 1037}, {1, 4, 4, 256, 525},
        {1, 4, 4, 64, 141},   {1, 4, 4, 256, 781}, {1, 4, 4, 128, 397},   {1, 4, 4, 3, 205},
        {1, 4, 4, 512, 1549}, {1, 5, 5, 128, 256}, {1, 5, 5, 32, 128}};

    const std::vector<std::vector<int64_t>> dedyShapeWhitelist = {
        {256, 1, 4, 4, 1037},  {256, 1, 16, 16, 269}, {256, 1, 4, 4, 1037},  {256, 1, 8, 8, 525},
        {256, 1, 32, 32, 141}, {256, 1, 16, 16, 781}, {256, 1, 32, 32, 397}, {256, 1, 64, 64, 205},
        {256, 1, 8, 8, 1549},  {64, 1, 16, 16, 256},  {64, 1, 32, 32, 128}};

    bool Createconv3dBpInputTransposeGraph(ge::es::EsGraphBuilder& builder, ge::es::EsTensorHolder& conv3dBpInputTrans,
                                           ge::es::EsTensorHolder& inputSize, ge::es::EsTensorHolder& filter,
                                           ge::es::EsTensorHolder& dedy);

    bool CreateOutputWithTranspose(ge::es::EsGraphBuilder& builder, const ge::es::EsTensorHolder& conv3dBpInputV2,
                                   ge::GNode* conv3dBpInputV2Node, ge::es::EsTensorHolder& transOutput);
};

} // namespace ops

#endif // CONV3D_BACKPROP_INPUT_TO_V2_FUSION_PASS_H
