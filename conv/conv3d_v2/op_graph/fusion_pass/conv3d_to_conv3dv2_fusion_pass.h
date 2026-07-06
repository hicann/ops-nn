/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV3D_TO_CONV3DV2_FUSION_PASS_H
#define CONV3D_TO_CONV3DV2_FUSION_PASS_H

#include <map>
#include <memory>
#include <vector>

#include "../../conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "ge/fusion/pass/decompose_pass.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace Conv3dToConv3dV2Fusion {
const std::string FUSION_NAME = "Conv3dToConv3dV2FusionPass";
constexpr int64_t C0_SIZE = 16;
constexpr int64_t CIDX = 4;

// Fmap Filter Output Bias
const std::vector<std::vector<ge::DataType>> CONV_SUPPORT_DTYPES_OUT2L1_DN2NZ = {
    {ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},
    {ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_BF16},
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16},
    {ge::DataType::DT_HIFLOAT8, ge::DataType::DT_HIFLOAT8, ge::DataType::DT_HIFLOAT8, ge::DataType::DT_FLOAT}};

// Fmap Filter Output Bias
const std::vector<std::vector<ge::DataType>> CONV_SUPPORT_DTYPES = {
    {ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},
    {ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_FLOAT},
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16}};
} // namespace Conv3dToConv3dV2Fusion

class __attribute__((visibility("default"))) Conv3dToConv3dV2FusionPass : public ge::fusion::DecomposePass {
public:
    explicit Conv3dToConv3dV2FusionPass(const std::vector<ge::AscendString>& opTypes) : DecomposePass(opTypes) {}

protected:
    bool MeetRequirements(const ge::GNode& convNode) override;
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& convNode) override;

private:
    bool CheckSocCapability();
    bool CheckPostCubeInOutNode(const ge::GNode& convNode) const;
    bool CheckTransDataInInputNode(const ge::GNode& convNode) const;
    bool CheckIFMRInSameOutputNode(const ge::GNode& convNode) const;
    void InitMember();

    bool supportOut2L1Dn2Nz = false;
    ConvFusionUtils::ConvDescInfo convDescInfo = {};
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // CONV3D_TO_CONV3DV2_FUSION_PASS_H