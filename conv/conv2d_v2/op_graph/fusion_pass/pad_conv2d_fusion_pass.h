/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PAD_CONV2D_FUSION_PASS_H
#define PAD_CONV2D_FUSION_PASS_H

#include <map>
#include <set>
#include <string>
#include <vector>

#include "../../conv/common/op_graph/fusion_pass/conv_fusion_base_pass.h"
#include "graph/gnode.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace PadConv2dFusion {

const ge::AscendString PAD_OP = "Pad";
const ge::AscendString PADV3_OP = "PadV3";
const ge::AscendString CONV2D_BACKPROP_FILTER_D = "Conv2DBackpropFilterD";
const ge::AscendString CONV2D_BACKPROP_INPUT_D = "Conv2DBackpropInputD";
const ge::AscendString BN_TRAINING_REDUCE_GRAD = "BNTrainingReduceGrad";
const ge::AscendString SLICE_OP = "Slice";
const ge::AscendString SLICE_D_OP = "SliceD";

const ge::AscendString MODE = "mode";
const ge::AscendString PADDINGS_CONTIGUOUS = "paddings_contiguous";
const ge::AscendString INPUT_SIZE = "input_size";
const ge::AscendString CONSTANT = "constant";
const ge::AscendString EXPLICIT = "EXPLICIT";
const ge::AscendString SAME = "SAME";
const ge::AscendString NOTSET = "NOTSET";

const std::string FUSION_NAME = "PadConv2dFusionPass";
const std::string DN2NZ_INTRINSIC = "Intrinsic_data_move_out2l1_dn2nz";

// Only DAV_3510 needs a dedicated branch, unlisted socs keep the original behavior as DAV_RESV.
const std::map<std::string, NpuArch> SOC_LIST = {{"Ascend950", NpuArch::DAV_3510}};

constexpr int32_t PADDINGS_INPUT_INDEX = 1;
constexpr int32_t CONSTANT_VALUES_INPUT_INDEX = 2;
constexpr int32_t DW_OUT_BACKPROP_INDEX = 1;
constexpr size_t INDEX_0 = 0;
constexpr size_t INDEX_1 = 1;
constexpr size_t INDEX_2 = 2;
constexpr size_t INDEX_3 = 3;
constexpr size_t DIM_NUM4 = 4;
constexpr size_t DIRECTION_COUNT = 2;
constexpr size_t SINGLE_CONSUMER_CNT = 1;
constexpr int64_t SINGLE_CUBE_CNT = 1;
constexpr int64_t MIN_PADDING_VALUE = 0;
constexpr int64_t MAX_PADDING_VALUE = 255;
constexpr uint32_t PLATFORM_INFO_OK = 0;

} // namespace PadConv2dFusion

class __attribute__((visibility("default"))) PadConv2dFusionPass : public ConvFusionBasePass {
protected:
    void InitMember() override;
    bool CheckMatchStructure(const ge::GNode& matchNode) override;
    bool MeetRequirements(const ge::GNode& convNode) override;
    std::set<ge::AscendString> GetNodeTypes() const override;
    void PrintGraphStructure() const override;
    bool ConvFusionReplaceImpl(ge::GraphPtr& graph, ge::GNode& convNode, ge::CustomPassContext& passContext) override;

private:
    void InitPlatform();
    bool ValidatePadTopology(const ge::GNode& convNode);
    bool CheckPadControlEdges(const ge::GNode& convNode) const;
    bool CheckPadDynamicShape() const;
    bool CheckPadV3AndExtractPaddings();
    bool CheckPadV3ConstantValue() const;
    bool ExtractPaddingsData(const ge::Tensor& padTensor, std::vector<int64_t>& padValue) const;
    bool ValidateAndComputePads(const ge::GNode& convNode);
    bool CheckFilterHeight() const;
    bool DiscoverAndCheckBackward();
    bool HandleBackwardPath(ge::Graph& graph);
    bool UpdateCubeNodes(ge::Graph& graph, ge::GNode& convNode);
    bool SetPaddingAttrs(ge::GNode& cubeNode) const;

    bool hasDn2Nz = false;
    ge::GNodePtr padNode = nullptr;
    std::vector<std::vector<int64_t>> paddings = {};
    std::vector<int64_t> combinedPads = {};
    ge::GNodePtr backpropFilterNode = nullptr;
    ge::GNodePtr backpropInputNode = nullptr;
    ge::GNodePtr sliceNode = nullptr;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // PAD_CONV2D_FUSION_PASS_H
