/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PAD_DEPTHWISE_CONV2D_FUSION_PASS_H
#define PAD_DEPTHWISE_CONV2D_FUSION_PASS_H

#include <map>
#include <set>
#include <vector>

#include "../../conv/common/op_graph/fusion_pass/conv_fusion_base_pass.h"
#include "graph/gnode.h"
#include "graph/operator.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace PadDepthwiseConv2dFusion {

const ge::AscendString PAD = "Pad";
const ge::AscendString PADDINGS = "paddings";
const ge::AscendString VALID_PADDING = "VALID";
const ge::AscendString SAME_PADDING = "SAME";

const std::string FUSION_NAME = "PadDepthwiseConv2dFusionPass";

const std::map<std::string, NpuArch> SUPPORT_SOC_LIST = {{"Ascend950", NpuArch::DAV_3510}};

constexpr int64_t DIM_NUM4 = 4;
constexpr int64_t DIRECTION_COUNT = 2;
constexpr int64_t MIN_PADDING_VALUE = 0;
constexpr int64_t MAX_PADDING_VALUE = 255;
constexpr int32_t PAD_X_INPUT_INDEX = 0;
constexpr int32_t PAD_PADDINGS_INPUT_INDEX = 1;
constexpr int32_t CONV_FMAP_INPUT_INDEX = 0;
constexpr int64_t NCHW_H_INDEX = 2;
constexpr int64_t HWCN_H_INDEX = 0;
constexpr int64_t NCHW_PAD_H_INDEX = 2;
constexpr int64_t NCHW_PAD_W_INDEX = 3;
constexpr int64_t NHWC_PAD_H_INDEX = 1;
constexpr int64_t NHWC_PAD_W_INDEX = 2;
constexpr int64_t NCHW_NC_PAIR_INDEX = 0;
constexpr int64_t NCHW_C_AXIS_INDEX = 1;
constexpr int64_t NHWC_N_AXIS_INDEX = 0;
constexpr int64_t NHWC_C_AXIS_INDEX = 3;

} // namespace PadDepthwiseConv2dFusion

class __attribute__((visibility("default"))) PadDepthwiseConv2dFusionPass : public ConvFusionBasePass {
protected:
    void InitMember() override;
    bool CheckMatchStructure(const ge::GNode& matchNode) override;
    bool MeetRequirements(const ge::GNode& convNode) override;
    std::set<ge::AscendString> GetNodeTypes() const override;
    void PrintGraphStructure() const override;
    bool ConvFusionReplaceImpl(ge::GraphPtr& graph, ge::GNode& convNode, ge::CustomPassContext& passContext) override;

private:
    bool CheckPadDynamicShape() const;
    bool GetPaddingsFromConst();
    bool ExtractPaddingByFormat();
    bool CheckPaddingRange() const;
    bool CheckFilterVsPadding(const ge::GNode& convNode) const;
    bool CheckPadOutputsAllDepthwise() const;
    bool BuildPadsVector();
    bool IsAscend950() const;

    ge::GNodePtr padNode = nullptr;
    ge::GNodePtr padXProducer = nullptr;
    int32_t padXProducerOutIdx = 0;
    std::vector<std::vector<int64_t>> paddings;
    std::vector<int64_t> pads;
    int64_t paddingsT = 0;
    int64_t paddingsB = 0;
    int64_t paddingsL = 0;
    int64_t paddingsR = 0;
    bool isAscend950 = false;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // PAD_DEPTHWISE_CONV2D_FUSION_PASS_H
