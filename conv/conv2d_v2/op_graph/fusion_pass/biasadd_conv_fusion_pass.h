/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file biasadd_conv_fusion_pass.h
 * \brief conv + biasadd/add --> conv with bias
 */
#ifndef BIASADD_CONV_FUSION_PASS_H
#define BIASADD_CONV_FUSION_PASS_H

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "../../conv/common/op_graph/fusion_pass/conv_fusion_base_pass.h"
#include "graph/gnode.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace BiasaddConvFusion {

const std::string FUSION_NAME = "AABiasaddConvFusion";
const std::string INTRINSIC_FIX_PIPE_L0C2OUT = "Intrinsic_fix_pipe_l0c2out";

const ge::AscendString ADD = "Add";
const ge::AscendString BIASADD = "BiasAdd";
const ge::AscendString RESHAPE = "Reshape";
const ge::AscendString DATA = "Data";
const ge::AscendString HAS_BIAS_ATTR = "has_bias";
const ge::AscendString VARIABLE = "Variable";
const ge::AscendString STREAM_SWITCH = "StreamSwitch";

constexpr int32_t ADD_BIAS_INPUT_INDEX = 1;
constexpr int32_t ADD_INPUT_NUMS = 2;
constexpr int32_t ADD_CONV_INPUT_INDEX = 0;
const std::string ATTR_PARENT_NODE_INDEX = "_parent_node_index";

constexpr int32_t RESHAPE_BIAS_INPUT_INDEX = 0;
constexpr int32_t RESHAPE_SHAPE_INPUT_INDEX = 1;
constexpr size_t SINGLE_CONSUMER_COUNT = 1U;

constexpr size_t NHWC_CHANNEL_AXIS = 3U;
constexpr size_t HWCN_CHANNEL_AXIS = 2U;
constexpr size_t NDHWC_CHANNEL_AXIS = 4U;
constexpr int64_t MIN_ONE_DIM_COUNT = 4;
constexpr int64_t UNKNOWN_DIM_SENTINEL = -2;

const std::set<ge::DataType> supportedBiasDtypes = {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT,
                                                    ge::DataType::DT_BF16, ge::DataType::DT_INT32};

const std::vector<std::string> convTypes = {"Conv2D", "DepthwiseConv2D", "Conv3D"};

} // namespace BiasaddConvFusion

class __attribute__((visibility("default"))) AABiasaddConvFusion : public ConvFusionBasePass {
protected:
    void InitMember() override;
    void OrderMatchedNodes(std::vector<ge::GNode>& matchedNodes) override;
    bool CheckMatchStructure(const ge::GNode& matchNode) override;
    bool MeetRequirements(const ge::GNode& convNode) override;
    std::set<ge::AscendString> GetNodeTypes() const override;
    void PrintGraphStructure() const override;
    bool ConvFusionReplaceImpl(ge::GraphPtr& graph, ge::GNode& convNode, ge::CustomPassContext& passContext) override;

private:
    bool ResolveMatchStructure(const ge::AscendString& addType);
    bool LinkBiasSourceToConv(ge::Graph& graph, ge::GNode& convNode);
    bool RelinkAddConsumersToConv(ge::Graph& graph, ge::GNode& convNode);
    bool MigrateStreamSwitchControlEdges(ge::Graph& graph, ge::GNode& convNode);

    ge::GNodePtr addNode = nullptr;
    ge::GNodePtr reshapeNode = nullptr;
    std::string addNodeName;
    int32_t convInputPortOfAdd = -1;
    int32_t biasInputIndex = -1;
    bool isReversedAdd = false;
    bool isReshapeMode = false;
    bool reshapeHasOtherConsumer = false;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // BIASADD_CONV_FUSION_PASS_H
