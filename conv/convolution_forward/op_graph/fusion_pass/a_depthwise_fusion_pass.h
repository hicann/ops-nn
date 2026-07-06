/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef A_DEPTHWISE_FUSION_PASS_H
#define A_DEPTHWISE_FUSION_PASS_H

#include <set>
#include <string>
#include <vector>

#include "../../../common/op_graph/fusion_pass/conv_fusion_base_pass.h"
#include "ge/es_graph_builder.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace ADepthwiseFusion {
const ge::AscendString RESHAPE_OP = "Reshape";
const ge::AscendString QUANT_BIAS_OPTIMIZATION = "QuantBiasOptimization";
const ge::AscendString ASCEND_WEIGHT_QUANT = "AscendWeightQuant";
const ge::AscendString SHAPE_ATTR = "shape";
const std::string FUSION_NAME = "ADepthwiseFusionPass";

constexpr int32_t FILTER_INPUT_INDEX = 1;
constexpr int32_t RESHAPE_INPUT_INDEX = 0;
constexpr int32_t RESHAPE_OUTPUT_INDEX = 0;
constexpr int32_t FILTER_SHAPE_SIZE = 4;
constexpr int64_t ALREADY_CHANGED_C = 1;
constexpr int64_t INDEX_0 = 0;
constexpr int64_t INDEX_1 = 1;
constexpr int64_t INDEX_2 = 2;
constexpr int64_t INDEX_3 = 3;

const std::set<ge::AscendString> TARGET_TYPES = {QUANT_BIAS_OPTIMIZATION, ConvFusionUtils::DEPTHWISE_CONV2D};
const std::set<ge::AscendString> TARGET_TYPES_QUANT = {QUANT_BIAS_OPTIMIZATION, ASCEND_WEIGHT_QUANT};

struct FilterShapeResult {
    int64_t n = 0;
    int64_t c = 0;
    int64_t h = 0;
    int64_t w = 0;
    std::vector<int64_t> preShape;
    std::vector<int64_t> newShape;
};

struct ReshapeDescInfo {
    ge::TensorDesc inDesc;
    ge::TensorDesc outDesc;
    std::vector<int64_t> shapeAttr;
};

struct ReshapeEdgeInfo {
    ge::Graph &graph;
    ge::GNode &src;
    ge::GNode &dst;
    int32_t srcOutIdx;
    int32_t dstInIdx;
};
} // namespace ADepthwiseFusion

class __attribute__((visibility("default"))) ADepthwiseFusionPass : public ConvFusionBasePass {
protected:
    void InitMember() override;
    bool MeetRequirements(const ge::GNode &depthwiseNode) override;
    ge::AscendString GetNodeType() const override;
    void PrintGraphStructure() const override;
    ge::Status ConvFusionPreImpl(ge::GraphPtr &graph, ge::GNode &depthwiseNode,
        const ge::CustomPassContext &pass_context) override;

private:
    bool DealQuantNodeCase(ge::Graph &graph, ge::GNode &quantNode, ge::GNode &depthwiseNode);
    bool UpdateQuantAndDepthwiseFilterDesc(ge::GNode &quantNode, ge::GNode &depthwiseNode) const;
    bool InsertReshapeForConsumers(ge::Graph &graph, ge::GNode &producer,
        const std::set<ge::AscendString> &targetTypes);
    bool ResolveReshapeShapes(ge::GNode &producer, int32_t outIdx, bool &resolved, std::vector<int64_t> &curPreShape,
        std::vector<int64_t> &curNewShape);
    ADepthwiseFusion::ReshapeDescInfo BuildReshapeDescInfo(const ge::TensorDesc &consumerInDesc,
        const std::vector<int64_t> &curPreShape, const std::vector<int64_t> &curNewShape) const;
    bool UpdateConsumerInputShape(ge::GNode &consumer, int32_t consumerInIdx, const ge::TensorDesc &consumerInDesc,
        const std::vector<int64_t> &newShape) const;
    bool InsertReshapeBetween(const ADepthwiseFusion::ReshapeEdgeInfo &edge,
        const ADepthwiseFusion::ReshapeDescInfo &descInfo);
    bool CreateReshapeNode(ge::Graph &graph, const ge::AscendString &producerName,
        const ADepthwiseFusion::ReshapeDescInfo &descInfo, ge::GNode &reshapeNode);
    bool ComputeFilterShapes(const std::vector<int64_t> &filterShape, ge::Format originFormat,
        ADepthwiseFusion::FilterShapeResult &result) const;
    void SetTensorDescShape(ge::TensorDesc &desc, const std::vector<int64_t> &dims) const;

    int64_t reshapeCounter = 0;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // A_DEPTHWISE_FUSION_PASS_H