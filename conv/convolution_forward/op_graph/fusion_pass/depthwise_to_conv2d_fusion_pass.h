/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEPTHWISE_TO_CONV2D_FUSION_PASS_H
#define DEPTHWISE_TO_CONV2D_FUSION_PASS_H

#include <map>
#include <memory>
#include <vector>

#include "../../../common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/pass/decompose_pass.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace DepthwiseToConv2dFusion {
const std::string FUSION_NAME = "DepthwiseToConv2dFusionPass";
constexpr int32_t MAX_DIM_NUM = 4;
constexpr int32_t FMAP_CHANNEL_NCHW_INDEX = 1;
constexpr int32_t FMAP_CHANNEL_NHWC_INDEX = 3;

const std::map<std::string, NpuArch> ND_SOC_LIST = {
    {"Ascend950", NpuArch::DAV_3510},
    {"MC62", NpuArch::DAV_5102}
};
} // namespace DepthwiseToConv2dFusion

class __attribute__((visibility("default"))) DepthwiseToConv2dFusionPass : public ge::fusion::DecomposePass {
public:
    explicit DepthwiseToConv2dFusionPass(const std::vector<ge::AscendString> &opTypes)
        : DecomposePass(opTypes) {}

protected:
    bool MeetRequirements(const ge::GNode &depthwiseNode) override;
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode &depthwiseNode) override;

private:
    void InitMember();
    bool CheckDynamicShape() const;
    bool GetFmapChannel();
    bool GetDepthwiseConvAttrs(const ge::GNode &depthwiseNode);
    bool SetConv2dAttrs(ge::GNode &conv2dNode, const ge::GNode &depthwiseNode);
    bool UpdateConv2dDesc(ge::GNode &conv2dNode);

    NpuArch npuArch = NpuArch::DAV_RESV;
    bool isNdSoc = false;
    int64_t fmapChannel = 0;
    ConvFusionUtils::ConvDescInfo convDescInfo = {};
    ConvFusionUtils::ConvBaseAttrs depthwiseAttrs = {};
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // DEPTHWISE_TO_CONV2D_FUSION_PASS_H