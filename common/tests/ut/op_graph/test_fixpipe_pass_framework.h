/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TEST_FIXPIPE_PASS_FRAMEWORK_H
#define TEST_FIXPIPE_PASS_FRAMEWORK_H

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cube_utils/cube_common.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/tensor.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"

// the op proto classes are required by the OperatorFactory::CreateOperator calls of the
// passes (CloneNode of the pre quant pass and CreateReluNode of the relu quant pass), they
// are declared with DISABLE_COMPILE_V1 so no static factory registration happens: a
// process-wide registration changes the es node builder behaviour of the other ut suites
// (the registered ir validation rejects their graphs), the runtime registration of
// RegisterCloneOpProtos below only covers the suites that call it (the pre quant and relu
// quant pass suites of the split fixpipe ut files)
#define DISABLE_COMPILE_V1
#include "../../../activation/relu/op_graph/relu_proto.h"
#include "../../../quant/ascend_dequant/op_graph/ascend_dequant_proto.h"
#undef DISABLE_COMPILE_V1

using namespace ge;
using namespace ops;

inline constexpr int32_t kOutPort = 0;
inline constexpr int32_t kCubeFmapPort = 0;
inline constexpr int32_t kCubeFilterPort = 1;
inline constexpr int32_t kQuantXPort = 0;
inline constexpr int32_t kDequantXPort = 0;
inline constexpr int32_t kDequantScalePort = 1;
inline constexpr int32_t kReluXPort = 0;

// the bits of the "support_fixpipe_ability" attr (equals the PostCubeAbilityType of cube_utils)
inline constexpr int64_t kAbilityMultipleOutput = 0x02;
inline constexpr int64_t kAbilityGmAtomicAdd = 0x04;
inline constexpr int64_t kAbilityCantAccess = 0x08;
// the ability attr name written by PostCubeComm::SetPostCubeAbilityAttr
inline const ge::AscendString kAbilityAttrName("support_fixpipe_ability");

// the fixpipe multiple output platform config key read by the ability pass
inline const char* const kFixpipeMultiOutputKey = "Intrinsic_fix_pipe_support_multiple_output";
// the intrinsic key read by PostCubeComm::GetSupportDN2NZSoc
inline const char* const kDn2nzKey = "Intrinsic_data_move_out2l1_dn2nz";

/*!
 * \brief the es graph builder of the fixpipe pass ut, same usage as the bias optimize ut
 */
class FixpipeTestGraph {
public:
    explicit FixpipeTestGraph(const std::string& name) : builder_(name.c_str())
    {
        rawGraph_ = builder_.GetCGraphBuilder()->GetGraph();
    }

    GNode& AddNode(const std::string& nodeName, const std::string& opType,
                   const std::vector<std::pair<std::string, bool>>& inputs)
    {
        std::vector<es::CompliantNodeBuilder::IrInputDef> inputDefs;
        for (const auto& in : inputs) {
            inputDefs.push_back({in.first,
                                 in.second ? es::CompliantNodeBuilder::kEsIrInputRequired :
                                             es::CompliantNodeBuilder::kEsIrInputOptional,
                                 ""});
        }
        GNode node = es::CompliantNodeBuilder(rawGraph_)
                         .OpType(opType.c_str())
                         .Name(nodeName.c_str())
                         .IrDefInputs(inputDefs)
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
        nodes_[nodeName] = node;
        return nodes_[nodeName];
    }

    void Connect(const std::string& from, int32_t fromPort, const std::string& to, int32_t toPort)
    {
        auto it = nodes_.find(from);
        auto toIt = nodes_.find(to);
        if (it == nodes_.end() || toIt == nodes_.end()) {
            return;
        }
        (void)es::AddEdgeAndUpdatePeerDesc(*rawGraph_, it->second, fromPort, toIt->second, toPort);
    }

    void ConnectInput(const es::EsTensorHolder& input, const std::string& to, int32_t toPort)
    {
        auto it = nodes_.find(to);
        if (it == nodes_.end()) {
            return;
        }
        (void)es::AddEdgeAndUpdatePeerDesc(*rawGraph_, *input.GetProducer(), kOutPort, it->second, toPort);
    }

    void SetInDesc(const std::string& nodeName, int32_t idx, DataType dtype, Format format,
                   const std::vector<int64_t>& dims)
    {
        TensorDesc desc;
        (void)nodes_[nodeName].GetInputDesc(idx, desc);
        desc.SetDataType(dtype);
        desc.SetFormat(format);
        desc.SetShape(Shape(dims));
        desc.SetOriginFormat(format);
        desc.SetOriginShape(Shape(dims));
        (void)nodes_[nodeName].UpdateInputDesc(idx, desc);
    }

    void SetOutDesc(const std::string& nodeName, int32_t idx, DataType dtype, Format format,
                    const std::vector<int64_t>& dims)
    {
        TensorDesc desc;
        (void)nodes_[nodeName].GetOutputDesc(idx, desc);
        desc.SetDataType(dtype);
        desc.SetFormat(format);
        desc.SetShape(Shape(dims));
        desc.SetOriginFormat(format);
        desc.SetOriginShape(Shape(dims));
        (void)nodes_[nodeName].UpdateOutputDesc(idx, desc);
    }

    std::shared_ptr<Graph> Build(const std::vector<std::string>& outputNodes)
    {
        std::vector<es::EsTensorHolder> holders;
        for (const auto& outputNode : outputNodes) {
            auto it = nodes_.find(outputNode);
            if (it == nodes_.end()) {
                continue;
            }
            holders.push_back(builder_.GetCGraphBuilder()->GetTensorHolderFromNode(it->second, kOutPort));
        }
        geGraph_ = builder_.BuildAndReset(holders);
        if (geGraph_ != nullptr) {
            rawGraph_ = geGraph_.get();
        }
        return geGraph_;
    }

    es::EsTensorHolder CreateInput(const std::string& name, DataType dtype, Format format,
                                   const std::vector<int64_t>& dims)
    {
        return builder_.CreateInput(inputIdx_++, name.c_str(), dtype, format, dims);
    }

    GNode& Node(const std::string& nodeName) { return nodes_[nodeName]; }

private:
    es::EsGraphBuilder builder_;
    Graph* rawGraph_ = nullptr;
    std::map<std::string, GNode> nodes_;
    int64_t inputIdx_ = 0;
    std::shared_ptr<Graph> geGraph_ = nullptr;
};

inline bool HasNodeType(const std::shared_ptr<Graph>& graph, const std::string& opType)
{
    for (auto node : graph->GetAllNodes()) {
        ge::AscendString type;
        if (node.GetType(type) == ge::GRAPH_SUCCESS && type.GetString() != nullptr && opType == type.GetString()) {
            return true;
        }
    }
    return false;
}

inline bool HasNodeName(const std::shared_ptr<Graph>& graph, const std::string& nodeName)
{
    for (auto node : graph->GetAllNodes()) {
        ge::AscendString name;
        if (node.GetName(name) == ge::GRAPH_SUCCESS && name.GetString() != nullptr && nodeName == name.GetString()) {
            return true;
        }
    }
    return false;
}

inline bool FindNodeByName(const std::shared_ptr<Graph>& graph, const std::string& nodeName, GNode& outNode)
{
    for (auto node : graph->GetAllNodes()) {
        ge::AscendString name;
        if (node.GetName(name) == ge::GRAPH_SUCCESS && name.GetString() != nullptr && nodeName == name.GetString()) {
            outNode = node;
            return true;
        }
    }
    return false;
}

inline int64_t GetAbilityAttr(GNode& node)
{
    int64_t ability = 0;
    (void)node.GetAttr(kAbilityAttrName, ability);
    return ability;
}

inline std::string GetInputProducerType(GNode& node, int32_t port)
{
    auto producer = node.GetInDataNodesAndPortIndexs(port);
    if (producer.first == nullptr) {
        return "";
    }
    ge::AscendString type;
    if (producer.first->GetType(type) != ge::GRAPH_SUCCESS || type.GetString() == nullptr) {
        return "";
    }
    return type.GetString();
}

inline std::string GetInputProducerName(GNode& node, int32_t port)
{
    auto producer = node.GetInDataNodesAndPortIndexs(port);
    if (producer.first == nullptr) {
        return "";
    }
    ge::AscendString name;
    if (producer.first->GetName(name) != ge::GRAPH_SUCCESS || name.GetString() == nullptr) {
        return "";
    }
    return name.GetString();
}

inline std::string GetNodeTypeOfName(const std::shared_ptr<Graph>& graph, const std::string& nodeName)
{
    GNode node;
    if (!FindNodeByName(graph, nodeName, node)) {
        return "";
    }
    ge::AscendString type;
    if (node.GetType(type) != ge::GRAPH_SUCCESS || type.GetString() == nullptr) {
        return "";
    }
    return type.GetString();
}

/*!
 * \brief the shared platform fixture of the fixpipe pass ut suites
 *
 * the platform is re-applied in every SetUp so the global PlatformInfoManager state never
 * leaks from the other op fusion suites of the same ut binary
 */
class FixpipePassTestBase : public testing::Test {
protected:
    void SetUp() override { ApplyPlatform(true, true); }

    // multiOutput: the Intrinsic_fix_pipe_support_multiple_output platform res value
    // preUnitsEnabled: false removes the pre_conv/pre_act units so the pre quant ReadConfig fails
    static void ApplyPlatform(bool multiOutput, bool preUnitsEnabled)
    {
        fe::PlatformInfo platformInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        // the dn2nz intrinsic enables the Conv2D/Conv2DTranspose util func white list
        platformInfo.ai_core_intrinsic_dtype_map[kDn2nzKey] = std::vector<std::string>();
        fe::OptionalInfo optiCompilationInfo;
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

        fe::PlatFormInfos platformInfos;
        fe::OptionalInfos optiCompilationInfos;
        platformInfos.Init();
        optiCompilationInfos.Init();
        optiCompilationInfos.SetSocVersion("Ascend950");
        std::map<std::string, std::vector<std::string>> fixpipeDtypeMap;
        // the unit layout follows the real Ascend950 platform config: the dequant/requant/
        // quant ops belong to the pre_conv unit and the relu family belongs to pre_act, the
        // FixPipe input strategies dispatch by the belong unit of the folded node
        if (preUnitsEnabled) {
            fixpipeDtypeMap["Intrinsic_fix_pipe_unit_list"] = {"pre_conv", "pre_act", "post_eltwise", "post_act",
                                                               "post_quant"};
        } else {
            fixpipeDtypeMap["Intrinsic_fix_pipe_unit_list"] = {"post_eltwise", "post_act", "post_quant"};
        }
        if (preUnitsEnabled) {
            fixpipeDtypeMap["Intrinsic_fix_pipe_pre_conv_func_list"] = {"requant", "dequant", "quant"};
            fixpipeDtypeMap["Intrinsic_fix_pipe_pre_conv_requant"] = {"s322s8"};
            fixpipeDtypeMap["Intrinsic_fix_pipe_pre_conv_dequant"] = {"s322f16"};
            fixpipeDtypeMap["Intrinsic_fix_pipe_pre_conv_quant"] = {"f322s8"};
            fixpipeDtypeMap["Intrinsic_fix_pipe_pre_act_func_list"] = {"normal_relu", "scalar_relu"};
            fixpipeDtypeMap["Intrinsic_fix_pipe_pre_act_normal_relu"] = {"f32"};
            fixpipeDtypeMap["Intrinsic_fix_pipe_pre_act_scalar_relu"] = {"f16"};
        }
        platformInfos.SetFixPipeDtypeMap(fixpipeDtypeMap);
        std::map<std::string, std::string> versionRes = {{"AIC_version", "AIC-C-310"}};
        platformInfos.SetPlatformRes("version", versionRes);
        std::map<std::string, std::string> intrinsicRes;
        intrinsicRes[kFixpipeMultiOutputKey] = multiOutput ? "true" : "false";
        platformInfos.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsicRes);
        fe::PlatformInfoManager::Instance().platform_infos_map_["Ascend950"] = platformInfos;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfos);
    }

    // remove the dn2nz intrinsic so the util func white list accepts every cube type
    static void SetDn2nzEnabled(bool enabled)
    {
        fe::PlatformInfo platformInfo = fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"];
        if (enabled) {
            platformInfo.ai_core_intrinsic_dtype_map[kDn2nzKey] = std::vector<std::string>();
        } else {
            platformInfo.ai_core_intrinsic_dtype_map.erase(kDn2nzKey);
        }
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    }

    // the runtime operator factory registration required by the clone paths of the passes:
    // it must only run inside the suites that call it from their SetUp (those files are
    // linked after every other suite of the ut binary) so the registered ir never changes
    // the behaviour of the other suites, the static locals keep the registration idempotent
    static void RegisterCloneOpProtos()
    {
        static const ge::OperatorCreatorRegister gRegisterAscendDequant(
            "AscendDequant", [](const std::string& name) { return ge::op::AscendDequant(name); });
        static const ge::OperatorCreatorRegister gRegisterRelu(
            "Relu", [](const std::string& name) { return ge::op::Relu(name); });
    }

    // the plain cube graph: x -> cube -> (graph output), only the descs matter for the ability pass
    static std::shared_ptr<Graph> BuildCubeGraph(const std::string& cubeType, const std::vector<int64_t>& outDims,
                                                 DataType outDtype = DT_FLOAT16, Format outFormat = FORMAT_NCHW,
                                                 DataType inDtype = DT_FLOAT16, Format inFormat = FORMAT_NCHW,
                                                 const std::vector<int64_t>& inDims = {1, 16, 16, 16},
                                                 DataType in1Dtype = DT_FLOAT16, Format in1Format = FORMAT_NCHW)
    {
        FixpipeTestGraph builder("cube_graph_" + cubeType);
        auto xInput = builder.CreateInput("x", inDtype, inFormat, inDims);
        builder.AddNode("cube", cubeType, {{"x", true}, {"filter", true}, {"bias", false}, {"offset_w", false}});
        builder.ConnectInput(xInput, "cube", kCubeFmapPort);
        builder.SetInDesc("cube", kCubeFmapPort, inDtype, inFormat, inDims);
        builder.SetInDesc("cube", kCubeFilterPort, in1Dtype, in1Format, inDims);
        builder.SetOutDesc("cube", kOutPort, outDtype, outFormat, outDims);
        return builder.Build({"cube"});
    }
};

#endif // TEST_FIXPIPE_PASS_FRAMEWORK_H
