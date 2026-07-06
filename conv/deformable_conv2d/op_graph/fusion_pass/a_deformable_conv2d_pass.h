/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef A_DEFORMABLE_CONV2D_PASS_H
#define A_DEFORMABLE_CONV2D_PASS_H

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../../../common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/pass/decompose_pass.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace DfmConv2dFusion {
const std::string FUSION_NAME = "ADeformableConv2dPass";

const std::map<std::string, NpuArch> ND_SOC_LIST = {
    {"Ascend950", NpuArch::DAV_3510}
};

const ge::AscendString DFM_CONV2D = "DeformableConv2D";
const ge::AscendString DFM_GROUPS = "deformable_groups";
const ge::AscendString MODULATED = "modulated";

constexpr int64_t DIM_SIZE = 4;
constexpr int64_t DYNAMIC_RANGE_LOWER = 1;
constexpr int64_t DYNAMIC_RANGE_UPPER = 4096;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t KNOWN_DIM_LOWER = 0;
constexpr int64_t DEFAULT_DFM_GROUPS = 1;
constexpr int64_t DEFAULT_KN = 0;
constexpr int64_t PAD_VALUE = 0;
constexpr int64_t DILATION_VALUE = 1;
constexpr int64_t DEFAULT_OFFSET_X = 0;
constexpr int64_t ENABLE_FLOAT32_EXECUTION = 0x20;
constexpr int64_t INVALID_SHAPE_RANGE = 0;
constexpr int32_t INPUT_OFFSETS_INDEX = 2;
constexpr int32_t DFM_OFFSET_INPUT_X_INDEX = 0;
constexpr int32_t DFM_OFFSET_INPUT_OFFSETS_INDEX = 1;
constexpr int32_t INPUT_DFM_BIAS_INDEX = 3;
constexpr size_t INPUT_COUNT_WITHOUT_BIAS = 3;
constexpr size_t INPUT_COUNT_WITH_BIAS = 4;
constexpr size_t KS_H_INDEX = 0;
constexpr size_t KS_W_INDEX = 1;

const std::set<ge::AscendString> FILTER_FORMATS = {ge::AscendString("HWCN"), ge::AscendString("NCHW")};

struct DfmPos {
    size_t xPosC = 0;
    size_t xPosH = 0;
    size_t xPosW = 0;
    size_t outPosH = 0;
    size_t outPosW = 0;
};
} // namespace DfmConv2dFusion

class __attribute__((visibility("default"))) ADeformableConv2dPass : public ge::fusion::DecomposePass {
public:
    explicit ADeformableConv2dPass(const std::vector<ge::AscendString> &opTypes)
        : DecomposePass(opTypes) {}

protected:
    bool MeetRequirements(const ge::GNode &convNode) override;
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode &convNode) override;

private:
    void InitMember();
    bool GetDfmDescInfo(const ge::GNode &convNode);
    bool ParseFilter();
    bool GetDfmAttrs(const ge::GNode &convNode);
    bool ResolveDfmPos();
    bool CalcDfmOutShape();
    bool SetConv2dAttrs(ge::GNode &conv2dNode);
    bool UpdateConv2dDesc(ge::GNode &conv2dNode);
    bool UpdateDfmOffsetDesc(ge::GNode &dfmOffsetNode);

    ConvFusionUtils::ConvDescInfo convDescInfo;
    ConvFusionUtils::ConvBaseAttrs baseAttrs;
    ge::TensorDesc offsetsDesc;
    ge::TensorDesc dfmOffsetOutDesc;
    DfmConv2dFusion::DfmPos dfmPos;
    std::vector<int64_t> ksize = {};
    int64_t dfmGroups = DfmConv2dFusion::DEFAULT_DFM_GROUPS;
    int64_t kn = DfmConv2dFusion::DEFAULT_KN;
    bool modulated = true;
    bool isDav3510 = false;
    NpuArch npuArch = NpuArch::DAV_RESV;
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // A_DEFORMABLE_CONV2D_PASS_H