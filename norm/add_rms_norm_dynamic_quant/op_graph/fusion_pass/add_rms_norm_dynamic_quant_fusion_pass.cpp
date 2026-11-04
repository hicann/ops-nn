/** Copyright (c) 2026 Huawei Technologies Co., Ltd. See LICENSE for licensing details. */
#include "add_rms_norm_dynamic_quant_fusion_pass.h"
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <set>
#include "acl/acl_rt.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "arch_utils.h"
#include "common/inc/error_util.h"
#include "ge/ge_utils.h"
#include "norm/common/op_graph/fusion_report_utils.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "version/ge-compiler_version.h"

namespace ops {
namespace {
using ge::GNode;
using ge::TensorDesc;
using ge::es::EsGraphBuilder;
using ge::es::EsTensorHolder;
using namespace ge::fusion;

const std::string kPass = "AddRmsNormDynamicQuantFusionPass";
const std::set<std::string> kSupportedNpuArchs = {"3510", "9201", "9202"};

constexpr float kDefaultEpsilon = 1e-6f;
constexpr int32_t kMinGeCompilerVersion = 90000000;
constexpr int kQuantInputCount = 6;
constexpr int kReshapeInputCount = 2;
std::string Name(const GNode& node)
{
    ge::AscendString value;
    return node.GetName(value) == ge::GRAPH_SUCCESS && value.GetString() ? value.GetString() : "";
}
TensorDesc InputDesc(const GNode& node, int port)
{
    TensorDesc desc;
    node.GetInputDesc(port, desc);
    return desc;
}
TensorDesc OutputDesc(const GNode& node, int port)
{
    TensorDesc desc;
    node.GetOutputDesc(port, desc);
    return desc;
}
bool SetDesc(const EsTensorHolder& tensor, const TensorDesc& desc)
{
    auto* node = tensor.GetProducer();
    const auto port = tensor.GetProducerOutIndex();
    if (node->UpdateOutputDesc(port, desc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kPass.c_str(), "Cannot update output descriptor: node=%s, port=%d.", Name(*node).c_str(),
                  static_cast<int>(port));
        return false;
    }
    return true;
}
bool SetInputDescs(const EsTensorHolder& output, int input_count)
{
    auto* node = output.GetProducer();
    if (node == nullptr) {
        OPS_LOG_E(kPass.c_str(), "Cannot synchronize input descriptors: producer is null.");
        return false;
    }
    // Visit fixed input slots and preserve descriptors of unconnected optional inputs.
    for (int port = 0; port < input_count; ++port) {
        const auto source = node->GetInDataNodesAndPortIndexs(port);
        if (source.first == nullptr) {
            continue;
        }
        TensorDesc desc;
        if (source.first->GetOutputDesc(source.second, desc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kPass.c_str(), "Cannot read source descriptor: node=%s, output=%d, target=%s, input=%d.",
                      Name(*source.first).c_str(), source.second, Name(*node).c_str(), port);
            return false;
        }
        if (node->UpdateInputDesc(port, desc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kPass.c_str(), "Cannot update input descriptor: node=%s, port=%d.", Name(*node).c_str(), port);
            return false;
        }
    }
    return true;
}
bool UpdateDescFromConsumers(const GNode& node, int port, TensorDesc& desc)
{
    const auto consumers = node.GetOutDataNodesAndPortIndexs(port);
    if (consumers.empty()) {
        return true;
    }
    const auto& consumer = consumers.front();
    if (consumer.first == nullptr) {
        OPS_LOG_E(kPass.c_str(), "Cannot read consumer descriptor: source=%s, output=%d, consumer is null.",
                  Name(node).c_str(), port);
        return false;
    }
    if (consumer.first->GetInputDesc(consumer.second, desc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kPass.c_str(), "Cannot read consumer descriptor: node=%s, input=%d, source=%s, output=%d.",
                  Name(*consumer.first).c_str(), consumer.second, Name(node).c_str(), port);
        return false;
    }
    return true;
}
std::string Type(const GNode& node)
{
    ge::AscendString value;
    return node.GetType(value) == ge::GRAPH_SUCCESS && value.GetString() ? value.GetString() : "";
}
bool HasInput(const GNode& node, int port) { return node.GetInDataNodesAndPortIndexs(port).first != nullptr; }
bool InferenceMode(const ge::CustomPassContext& context)
{
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90000000)
    int32_t runtime_version = 0;
    std::string package_name = "ge_compiler";
    if (aclsysGetVersionNum(&package_name[0], &runtime_version) != ACL_SUCCESS ||
        runtime_version < kMinGeCompilerVersion) {
        OPS_LOG_D(kPass.c_str(), "guard_rejected : GE compiler version is unavailable or below 9.0.0.");
        return false;
    }
    ge::AscendString mode;
    (void)context.GetOptionValue(ge::AscendString("ge.graphRunMode"), mode);
    if (std::strtol(mode.GetString() ? mode.GetString() : "", nullptr, 10) == 1) {
        OPS_LOG_D(kPass.c_str(), "guard_rejected : training mode.");
        return false;
    }
    return true;
#else
    (void)context;
    return false;
#endif
}
struct Candidate {
    GNode norm;
    std::vector<GNode> quant;
    std::vector<GNode> reshapes;
};
bool FindBranches(const GNode& norm, Candidate& nodes)
{
    nodes.norm = norm;
    const auto consumers = norm.GetOutDataNodesAndPortIndexs(0);
    if (consumers.empty() || consumers.size() > 2) {
        OPS_LOG_D(kPass.c_str(), "guard_rejected : node=%s, y consumer connections=%zu, expected 1 or 2.",
                  Name(norm).c_str(), consumers.size());
        return false;
    }
    for (const auto& consumer : consumers) {
        if (!consumer.first) {
            OPS_LOG_D(kPass.c_str(), "guard_rejected : node=%s has a null y consumer.", Name(norm).c_str());
            return false;
        }
        auto node = *consumer.first;
        if (Type(node) == "Reshape") {
            nodes.reshapes.push_back(node);
            const auto next = node.GetOutDataNodesAndPortIndexs(0);
            if (next.size() != 1 || !next[0].first) {
                OPS_LOG_D(kPass.c_str(), "guard_rejected : Reshape=%s must have one non-null consumer, count=%zu.",
                          Name(node).c_str(), next.size());
                return false;
            }
            node = *next[0].first;
        }
        if (Type(node) != "DynamicQuant") {
            OPS_LOG_D(kPass.c_str(), "guard_rejected : node=%s, expected DynamicQuant, got=%s.", Name(node).c_str(),
                      Type(node).c_str());
            return false;
        }
        nodes.quant.push_back(node);
    }
    if (!nodes.reshapes.empty() && nodes.reshapes.size() != nodes.quant.size()) {
        OPS_LOG_D(kPass.c_str(), "guard_rejected : node=%s has mixed direct and Reshape branches.", Name(norm).c_str());
        return false;
    }
    return true;
}
int64_t Destination(const GNode& quant)
{
    int64_t dtype = ge::DT_INT8;
    quant.GetAttr("dst_type", dtype);
    return dtype;
}
bool CheckNorm(const GNode& norm)
{
    const auto dtype = InputDesc(norm, 0).GetDataType();
    if ((dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) || InputDesc(norm, 2).GetShape().GetDims().size() != 1) {
        OPS_LOG_D(kPass.c_str(), "guard_rejected : x1 dtype or gamma rank.");
        return false;
    }
    return true;
}
bool CheckRequirements(const Candidate& nodes)
{
    if (HasInput(nodes.quant[0], 1)) {
        if (nodes.quant.size() == 2 && !HasInput(nodes.quant[1], 1)) {
            OPS_LOG_D(kPass.c_str(), "guard_rejected : second smooth input is absent.");
            return false;
        }
        const auto dtype = InputDesc(nodes.norm, 0).GetDataType();
        for (const auto& quant : nodes.quant) {
            TensorDesc smooth_desc;
            if (quant.GetInputDesc(1, smooth_desc) != ge::GRAPH_SUCCESS || smooth_desc.GetDataType() != dtype) {
                OPS_LOG_D(kPass.c_str(), "guard_rejected : smooth descriptor unavailable or dtype differs from x1.");
                return false;
            }
        }
    }
    const auto dst = Destination(nodes.quant[0]);
    for (const auto& quant : nodes.quant) {
        const auto quant_dst = Destination(quant);
        if ((quant_dst != ge::DT_INT8 && quant_dst != ge::DT_HIFLOAT8 && quant_dst != ge::DT_FLOAT8_E4M3FN &&
             quant_dst != ge::DT_FLOAT8_E5M2) ||
            quant_dst != dst) {
            OPS_LOG_D(kPass.c_str(), "guard_rejected : destination dtype.");
            return false;
        }
    }
    return true;
}
std::unique_ptr<SubgraphBoundary> MakeBoundary(const Candidate& nodes)
{
    auto boundary = std::make_unique<SubgraphBoundary>();
    int input_index = 0, output_index = 0;
    auto add_input = [&](const GNode& node, int port) {
        const int index = input_index++;
        if (boundary->AddInput(index, SubgraphInput({{node, port}})) != ge::SUCCESS) {
            OPS_LOG_E(kPass.c_str(), "Cannot add boundary input: index=%d, node=%s, port=%d.", index,
                      Name(node).c_str(), port);
            return false;
        }
        return true;
    };
    auto add_output = [&](const GNode& node, int port) {
        const int index = output_index++;
        if (boundary->AddOutput(index, SubgraphOutput({node, port})) != ge::SUCCESS) {
            OPS_LOG_E(kPass.c_str(), "Cannot add boundary output: index=%d, node=%s, port=%d.", index,
                      Name(node).c_str(), port);
            return false;
        }
        return true;
    };
    for (int port = 0; port < 3; ++port) {
        if (!add_input(nodes.norm, port)) {
            return nullptr;
        }
    }
    for (const auto& reshape : nodes.reshapes) {
        if (!add_input(reshape, 1)) {
            return nullptr;
        }
    }
    for (const auto& quant : nodes.quant) {
        // DynamicQuant has fixed smooth_scales and group_index ports.
        for (int port = 1; port <= 2; ++port) {
            if (HasInput(quant, port) && !add_input(quant, port)) {
                return nullptr;
            }
        }
        if (!add_output(quant, 0) || !add_output(quant, 1)) {
            return nullptr;
        }
    }
    if (!add_output(nodes.norm, 2)) {
        return nullptr;
    }
    return boundary;
}
std::unique_ptr<ge::Graph> Replacement(const Candidate& nodes, const SubgraphBoundary& boundary)
{
    EsGraphBuilder builder(kPass.c_str());
    std::vector<SubgraphInput> boundary_inputs;
    if (boundary.GetAllInputs(boundary_inputs) != ge::SUCCESS) {
        OPS_LOG_E(kPass.c_str(), "Cannot enumerate boundary inputs: node=%s.", Name(nodes.norm).c_str());
        return nullptr;
    }
    auto inputs = builder.CreateInputs(boundary_inputs.size());
    std::vector<ge::Shape> input_shapes;
    input_shapes.reserve(boundary_inputs.size());
    for (size_t i = 0; i < boundary_inputs.size(); ++i) {
        const auto consumers = boundary_inputs[i].GetAllInputs();
        if (consumers.empty()) {
            OPS_LOG_E(kPass.c_str(), "Cannot initialize boundary input %zu: no consumer.", i);
            return nullptr;
        }
        const auto desc = InputDesc(consumers[0].node, consumers[0].index);
        if (!SetDesc(inputs[i], desc)) {
            return nullptr;
        }
        if (inputs[i].GetProducer()->UpdateInputDesc(0, desc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kPass.c_str(), "Cannot update boundary Data input descriptor: index=%zu.", i);
            return nullptr;
        }
        input_shapes.emplace_back(desc.GetShape());
    }
    auto input = [&](const GNode& node, int port) -> EsTensorHolder {
        for (size_t i = 0; i < boundary_inputs.size(); ++i) {
            for (const auto& consumer : boundary_inputs[i].GetAllInputs()) {
                if (consumer.index == port && Name(consumer.node) == Name(node)) {
                    return inputs[i];
                }
            }
        }
        return nullptr;
    };
    float epsilon = kDefaultEpsilon;
    nodes.norm.GetAttr("epsilon", epsilon);
    const bool dual = nodes.quant.size() == 2;
    const bool has_smooth = HasInput(nodes.quant[0], 1);
    auto smooth1 = has_smooth ? input(nodes.quant[0], 1) : EsTensorHolder(nullptr);
    auto smooth2 = has_smooth && dual ? input(nodes.quant[1], 1) : EsTensorHolder(nullptr);
    auto fused = ge::es::AddRmsNormDynamicQuant(
        input(nodes.norm, 0), input(nodes.norm, 1), input(nodes.norm, 2), smooth1, smooth2, nullptr, epsilon,
        dual ? std::vector<uint8_t>{1, 1} : std::vector<uint8_t>{}, Destination(nodes.quant[0]));
    if (has_smooth) {
        const auto smooth_desc = InputDesc(nodes.quant[0], 1);
        auto* fused_node = fused.y1.GetProducer();
        if (fused_node == nullptr || fused_node->UpdateInputDesc(3, smooth_desc) != ge::GRAPH_SUCCESS ||
            (dual && fused_node->UpdateInputDesc(4, smooth_desc) != ge::GRAPH_SUCCESS)) {
            OPS_LOG_E(kPass.c_str(), "replacement_rejected : cannot set smooth input descriptors.");
            return nullptr;
        }
    }
    if (!SetInputDescs(fused.y1, kQuantInputCount)) {
        OPS_LOG_E(kPass.c_str(), "replacement_rejected : cannot set fusion input descriptors.");
        return nullptr;
    }
    auto y2_desc = OutputDesc(*fused.y2.GetProducer(), fused.y2.GetProducerOutIndex());
    auto scale2_desc = OutputDesc(*fused.scale2.GetProducer(), fused.scale2.GetProducerOutIndex());
    y2_desc.SetDataType(ge::DT_INT8);
    scale2_desc.SetDataType(ge::DT_FLOAT);
    if (!SetDesc(fused.y2, y2_desc) || !SetDesc(fused.scale2, scale2_desc)) {
        return nullptr;
    }
    std::vector<EsTensorHolder> outputs;
    const auto first_y_desc = OutputDesc(nodes.quant[0], 0);
    const auto first_scale_desc = OutputDesc(nodes.quant[0], 1);
    for (size_t i = 0; i < nodes.quant.size(); ++i) {
        auto y = i == 0 ? fused.y1 : fused.y2;
        auto scale = i == 0 ? fused.scale1 : fused.scale2;
        // Initialize both branches from DQ0, then update from each output's consumers.
        auto y_desc = first_y_desc;
        auto scale_desc = first_scale_desc;
        if (!UpdateDescFromConsumers(nodes.quant[i], 0, y_desc) ||
            !UpdateDescFromConsumers(nodes.quant[i], 1, scale_desc)) {
            return nullptr;
        }
        if (!nodes.reshapes.empty()) {
            if (!SetDesc(y, y_desc) || !SetDesc(scale, scale_desc)) {
                return nullptr;
            }
            y = ge::es::Reshape(y, builder.CreateVector(y_desc.GetShape().GetDims()));
            scale = ge::es::Reshape(scale, builder.CreateVector(scale_desc.GetShape().GetDims()));
            if (!SetInputDescs(y, kReshapeInputCount) || !SetInputDescs(scale, kReshapeInputCount)) {
                OPS_LOG_E(kPass.c_str(), "replacement_rejected : cannot set reshape input descriptors.");
                return nullptr;
            }
        }
        if (!SetDesc(y, y_desc) || !SetDesc(scale, scale_desc)) {
            return nullptr;
        }
        outputs.push_back(y);
        outputs.push_back(scale);
    }
    auto x_desc = OutputDesc(nodes.norm, 2);
    if (!UpdateDescFromConsumers(nodes.norm, 2, x_desc) || !SetDesc(fused.x, x_desc)) {
        return nullptr;
    }
    outputs.push_back(fused.x);
    auto graph = builder.BuildAndReset(outputs);
    if (!graph) {
        OPS_LOG_E(kPass.c_str(), "Cannot build replacement graph.");
        return nullptr;
    }
    const auto status = ge::GeUtils::InferShape(*graph, input_shapes);
    if (status != ge::SUCCESS) {
        OPS_LOG_E(kPass.c_str(), "Cannot infer replacement graph, status=%u.", static_cast<unsigned int>(status));
        return nullptr;
    }
    return graph;
}
} // namespace
ge::Status AddRmsNormDynamicQuantFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& context)
{
    if (!graph) {
        OPS_LOG_D(kPass.c_str(), "skip : input graph is null.");
        return ge::SUCCESS;
    }
    if (!ops::CheckNpuArch(kPass, kSupportedNpuArchs)) {
        return ge::SUCCESS;
    }
    if (!InferenceMode(context)) {
        return ge::SUCCESS;
    }
    std::vector<GNode> norms;
    for (const auto& node : graph->GetAllNodes()) {
        if (Type(node) == "AddRmsNorm") {
            norms.push_back(node);
        }
    }
    for (const auto& node : norms) {
        if (!node.GetOutDataNodesAndPortIndexs(1).empty()) {
            OPS_LOG_D(kPass.c_str(), "guard_rejected : rstd has consumers.");
            continue;
        }
        Candidate nodes;
        if (!CheckNorm(node) || !FindBranches(node, nodes) || !CheckRequirements(nodes)) {
            continue;
        }
        auto boundary = MakeBoundary(nodes);
        auto replacement = boundary ? Replacement(nodes, *boundary) : nullptr;
        if (!replacement) {
            OPS_LOG_E(kPass.c_str(), "Cannot build replacement for %s.", Name(node).c_str());
            return ge::FAILED;
        }
        const auto status = ops::fusion::ReplaceSubgraphWithReport(*boundary, *replacement, context);
        if (status != ge::SUCCESS) {
            OPS_LOG_E(kPass.c_str(), "Cannot replace %s.", Name(node).c_str());
            return ge::FAILED;
        }
        OPS_LOG_D(kPass.c_str(), "replacement_done.");
    }
    return ge::SUCCESS;
}
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90000000)
REG_FUSION_PASS(AddRmsNormDynamicQuantFusionPass).Stage(ge::CustomPassStage::kAfterBuiltinFusionPass);
#endif
} // namespace ops
