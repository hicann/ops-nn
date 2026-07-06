/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV2D_TO_CONV2DV2_FUSION_PASS_H
#define CONV2D_TO_CONV2DV2_FUSION_PASS_H

#include <map>
#include <memory>
#include <vector>

#include "../../conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "ge/fusion/pass/decompose_pass.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace Conv2dToConv2dV2Fusion {
const std::string FUSION_NAME = "Conv2dToConv2dV2FusionPass";

const std::map<std::string, NpuArch> SUPPORT_SOC_LIST = {{"Ascend950", NpuArch::DAV_3510}, {"MC62", NpuArch::DAV_5102}};

// Fmap Filter Output Bias
const std::vector<std::vector<ge::DataType>> CONV_SUPPORT_DTYPES_DAV_3510 = {
    {ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},
    {ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_BF16},
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16},
    {ge::DataType::DT_HIFLOAT8, ge::DataType::DT_HIFLOAT8, ge::DataType::DT_HIFLOAT8, ge::DataType::DT_FLOAT}};

// Fmap Filter Output Bias
const std::vector<std::vector<ge::DataType>> CONV_SUPPORT_DTYPES_DAV_5102 = {
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16},
    {ge::DataType::DT_INT8, ge::DataType::DT_INT8, ge::DataType::DT_INT32, ge::DataType::DT_INT32}};
} // namespace Conv2dToConv2dV2Fusion

class __attribute__((visibility("default"))) Conv2dToConv2dV2FusionPass : public ge::fusion::DecomposePass {
public:
    explicit Conv2dToConv2dV2FusionPass(const std::vector<ge::AscendString>& opTypes) : DecomposePass(opTypes) {}

protected:
    bool MeetRequirements(const ge::GNode& convNode) override;
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& convNode) override;

private:
    void InitMember();

    NpuArch npuArch = NpuArch::DAV_RESV;
    ConvFusionUtils::ConvDescInfo convDescInfo = {};
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // CONV2D_TO_CONV2DV2_FUSION_PASS_H