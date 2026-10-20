/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NN_ADD_LAYER_NORM_QUANT_FUSION_COMMON_H
#define OPS_NN_ADD_LAYER_NORM_QUANT_FUSION_COMMON_H

#include <array>
#include <string>
#include <vector>
#include "error_util.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/ge_utils.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "platform/platform_info.h"

namespace ops {
namespace aln_quant {
using ge::GNode;
using ge::TensorDesc;
using ge::es::EsGraphBuilder;
using ge::es::EsTensorHolder;
using namespace ge::fusion;

const std::string kSupportedNpuArch = "3510";
constexpr float kDefaultEpsilon = 1e-5f;

inline std::string Name(const GNode& node)
{
    ge::AscendString name;
    return node.GetName(name) == ge::GRAPH_SUCCESS && name.GetString() != nullptr ? name.GetString() : "";
}
inline std::string Type(const GNode& node)
{
    ge::AscendString type;
    return node.GetType(type) == ge::GRAPH_SUCCESS && type.GetString() != nullptr ? type.GetString() : "";
}
inline bool HasInput(const GNode& node, int index)
{
    // Check whether the input port is connected.
    return index >= 0 && node.GetInDataNodesAndPortIndexs(index).first != nullptr;
}
inline ge::DataType InputType(const GNode& node, int index)
{
    TensorDesc desc;
    return node.GetInputDesc(index, desc) == ge::GRAPH_SUCCESS ? desc.GetDataType() : ge::DT_UNDEFINED;
}
inline bool IsFloat(ge::DataType type) { return type == ge::DT_FLOAT16 || type == ge::DT_BF16 || type == ge::DT_FLOAT; }
// Support Ascend950 (NpuArch 3510) only.
inline bool IsSupportedPlatform()
{
    fe::PlatFormInfos infos;
    fe::OptionalInfos options;
    std::string arch;
    return fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(infos, options) == ge::GRAPH_SUCCESS &&
           infos.GetPlatformRes("version", "NpuArch", arch) && arch == kSupportedNpuArch;
}
struct Captures {
    GNode norm;
    std::vector<GNode> quant;
};
inline bool Capture(const MatchResult& match, Captures& nodes)
{
    NodeIo io;
    if (match.GetCapturedTensor(0, io) != ge::SUCCESS) {
        return false;
    }
    nodes.norm = io.node;
    for (size_t i = 1; i <= 2; ++i) {
        if (match.GetCapturedTensor(i, io) == ge::SUCCESS) {
            nodes.quant.push_back(io.node);
        }
    }
    return !nodes.quant.empty();
}
// Check subgraph boundaries.
inline bool CheckBoundary(const Captures& nodes, const std::string& pass)
{
    if (nodes.quant.size() == 2 && Name(nodes.quant[0]) == Name(nodes.quant[1])) {
        OPS_LOG_D(pass.c_str(), "guard_rejected : two pattern branches matched the same quant node.");
        return false;
    }
    if (nodes.norm.GetOutDataNodesAndPortIndexs(0).size() != nodes.quant.size()) {
        OPS_LOG_D(pass.c_str(), "guard_rejected : normalization y has an unmatched consumer.");
        return false;
    }
    if (!nodes.norm.GetOutDataNodesAndPortIndexs(1).empty() || !nodes.norm.GetOutDataNodesAndPortIndexs(2).empty()) {
        OPS_LOG_D(pass.c_str(), "guard_rejected : mean/rstd is used.");
        return false;
    }
    return true;
}
inline bool CheckNorm(const GNode& node, const std::string& pass)
{
    const auto dtype = InputType(node, 0);
    if (!IsFloat(dtype)) {
        OPS_LOG_D(pass.c_str(), "guard_rejected : unsupported normalization dtype.");
        return false;
    }
    for (int i = 1; i < (HasInput(node, 4) ? 5 : 4); ++i) {
        if (InputType(node, i) != dtype) {
            OPS_LOG_D(pass.c_str(), "guard_rejected : mixed normalization input dtype.");
            return false;
        }
    }
    return true;
}
inline bool Int8Output(const GNode& node)
{
    TensorDesc desc;
    return node.GetOutputDesc(0, desc) == ge::GRAPH_SUCCESS && desc.GetDataType() == ge::DT_INT8;
}
// Map matched input ports to replacement graph inputs.
class ReplacementInputs {
public:
    ReplacementInputs(EsGraphBuilder& builder, const MatchResult& match)
    {
        auto boundary = match.ToSubgraphBoundary();
        valid_ = boundary != nullptr && boundary->GetAllInputs(boundary_) == ge::SUCCESS;
        if (valid_) {
            inputs_ = builder.CreateInputs(boundary_.size());
            for (size_t i = 0; i < boundary_.size(); ++i) {
                const auto consumers = boundary_[i].GetAllInputs();
                TensorDesc desc;
                if (consumers.empty() ||
                    consumers[0].node.GetInputDesc(consumers[0].index, desc) != ge::GRAPH_SUCCESS ||
                    inputs_[i].GetProducer()->UpdateOutputDesc(0, desc) != ge::GRAPH_SUCCESS ||
                    inputs_[i].GetProducer()->UpdateInputDesc(0, desc) != ge::GRAPH_SUCCESS) {
                    valid_ = false;
                    break;
                }
                input_shapes_.emplace_back(desc.GetShape());
            }
        }
    }
    const std::vector<ge::Shape>& Shapes() const { return input_shapes_; }
    bool Valid() const { return valid_; }
    EsTensorHolder Get(const GNode& node, int port) const
    {
        for (size_t i = 0; i < boundary_.size(); ++i) {
            for (const auto& input : boundary_[i].GetAllInputs()) {
                if (input.index == port && Name(input.node) == Name(node)) {
                    return inputs_[i];
                }
            }
        }
        return nullptr;
    }

private:
    bool valid_ = false;
    std::vector<SubgraphInput> boundary_;
    std::vector<EsTensorHolder> inputs_;
    std::vector<ge::Shape> input_shapes_;
};
inline bool SetQuantInputDescs(const ge::es::AddLayerNormQuantOutput& fused)
{
    auto* node = fused.y1.GetProducer();
    if (node == nullptr) {
        return false;
    }
    // Visit all input slots, including optional inputs.
    constexpr int kQuantInputCount = 9;
    for (int port = 0; port < kQuantInputCount; ++port) {
        const auto source = node->GetInDataNodesAndPortIndexs(port);
        if (source.first == nullptr) {
            continue;
        }
        TensorDesc desc;
        if (source.first->GetOutputDesc(source.second, desc) != ge::GRAPH_SUCCESS ||
            node->UpdateInputDesc(port, desc) != ge::GRAPH_SUCCESS) {
            return false;
        }
    }
    return true;
}
inline bool SeedQuantOutputs(const ge::es::AddLayerNormQuantOutput& fused)
{
    // Initialize descriptors for y2 and the two scale outputs.
    TensorDesc unused_y(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT8);
    TensorDesc unused_scale(ge::Shape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
    return fused.y2.GetProducer()->UpdateOutputDesc(1, unused_y) == ge::GRAPH_SUCCESS &&
           fused.out_scales1.GetProducer()->UpdateOutputDesc(3, unused_scale) == ge::GRAPH_SUCCESS &&
           fused.out_scales2.GetProducer()->UpdateOutputDesc(4, unused_scale) == ge::GRAPH_SUCCESS;
}
inline GraphUniqPtr Finish(EsGraphBuilder& builder, const std::vector<EsTensorHolder>& outputs,
                           const MatchResult& match, const std::string& pass,
                           const std::vector<ge::Shape>& input_shapes)
{
    // Copy original output descriptors to replacement outputs.
    auto boundary = match.ToSubgraphBoundary();
    std::vector<SubgraphOutput> original_outputs;
    if (!boundary || boundary->GetAllOutputs(original_outputs) != ge::SUCCESS ||
        original_outputs.size() != outputs.size()) {
        OPS_LOG_E(pass.c_str(), "replacement_rejected : output boundary mismatch.");
        return nullptr;
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        NodeIo original;
        TensorDesc desc;
        if (original_outputs[i].GetOutput(original) != ge::SUCCESS ||
            original.node.GetOutputDesc(original.index, desc) != ge::GRAPH_SUCCESS ||
            outputs[i].GetProducer()->UpdateOutputDesc(outputs[i].GetProducerOutIndex(), desc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(pass.c_str(), "replacement_rejected : cannot preserve boundary output descriptor, index=%zu.", i);
            return nullptr;
        }
    }
    auto graph = builder.BuildAndReset(outputs);
    if (graph == nullptr) {
        OPS_LOG_E(pass.c_str(), "replacement_rejected : graph build failed.");
        return nullptr;
    }
    const auto status = ge::GeUtils::InferShape(*graph, input_shapes);
    if (status != ge::SUCCESS) {
        OPS_LOG_E(pass.c_str(), "replacement_rejected : cannot infer replacement graph, status=%u.",
                  static_cast<unsigned int>(status));
        return nullptr;
    }
    OPS_LOG_D(pass.c_str(), "replacement_done.");
    return graph;
}
inline PatternUniqPtr MakePattern(EsGraphBuilder& builder, const std::vector<EsTensorHolder>& outputs,
                                  const EsTensorHolder& norm, const std::vector<EsTensorHolder>& quant)
{
    // Record norm and quant outputs for pattern capture.
    std::vector<NodeIo> captures{{*norm.GetProducer(), norm.GetProducerOutIndex()}};
    for (const auto& q : quant) {
        captures.push_back({*q.GetProducer(), q.GetProducerOutIndex()});
    }
    auto graph = builder.BuildAndReset(outputs);
    if (graph == nullptr) {
        return nullptr;
    }
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    for (const auto& capture : captures) {
        pattern->CaptureTensor(capture);
    }
    return pattern;
}
} // namespace aln_quant
} // namespace ops
#endif
