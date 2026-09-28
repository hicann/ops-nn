/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <map>
#include <random>
#include <sstream>
#include <set>
#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#define private public
#include "platform/platform_info.h"
#undef private
#include "../../../op_graph/fusion_pass/quant_batch_matmul_inplace_add_transpose_fusion_pass.h"

namespace {
using namespace ge;
using Builder = ge::es::CompliantNodeBuilder;

TensorDesc Desc(std::vector<int64_t> dims, DataType type)
{
    TensorDesc result(Shape(dims), FORMAT_ND, type);
    result.SetOriginShape(Shape(dims));
    result.SetOriginFormat(FORMAT_ND);
    return result;
}

std::string Name(const GNode& node)
{
    AscendString name;
    node.GetName(name);
    return name.GetString();
}

struct Options {
    bool a = true;
    bool b = false;
    bool mx = true;
    bool bitcast = false;
    bool reshape = false;
    bool transposeD = true;
    DataType aType = DT_FLOAT8_E4M3FN;
    DataType bType = DT_FLOAT8_E5M2;
    int64_t m = 16;
    int64_t n = 32;
    int64_t k = 64;
    const char* type = "QuantBatchMatmulInplaceAdd";
};

class InplaceTransposeTest : public testing::Test {
protected:
    GraphPtr graph;
    GNode q;
    std::map<std::string, GNode> nodes;
    decltype(fe::PlatformInfoManager::Instance().platform_info_map_) savedPlatforms;
    fe::OptionalInfo savedOptional;
    ops::QuantBatchMatmulInplaceAddTransposeFusionPass pass;

    void SetUp() override
    {
        auto& manager = fe::PlatformInfoManager::Instance();
        savedPlatforms = manager.platform_info_map_;
        fe::PlatformInfo unused;
        manager.GetPlatformInfoWithOutSocVersion(unused, savedOptional);
        fe::PlatformInfo info;
        info.ai_core_intrinsic_dtype_map["Intrinsic_data_move_l12bt"] = {"bf16"};
        fe::OptionalInfo optional;
        optional.soc_version = "inplace_transpose_ut";
        manager.platform_info_map_.clear();
        manager.platform_info_map_[optional.soc_version] = info;
        manager.SetOptionalCompilationInfo(optional);
    }

    void TearDown() override
    {
        auto& manager = fe::PlatformInfoManager::Instance();
        manager.platform_info_map_ = savedPlatforms;
        manager.SetOptionalCompilationInfo(savedOptional);
    }

    GNode Make(const std::string& name, const char* type, const TensorDesc& desc, int inputs = 1)
    {
        Builder builder(graph.get());
        builder.OpType(type).Name(name.c_str());
        if (inputs == 1) {
            builder.IrDefInputs({{"x", Builder::kEsIrInputRequired, ""}});
        } else if (inputs == 2) {
            builder.IrDefInputs({{"x", Builder::kEsIrInputRequired, ""}, {"perm", Builder::kEsIrInputRequired, ""}});
        }
        auto node = builder.IrDefOutputs({{"y", Builder::kEsIrOutputRequired, ""}})
                        .InstanceOutputDataType("y", desc.GetDataType())
                        .InstanceOutputShape("y", desc.GetShape().GetDims())
                        .InstanceOutputFormat("y", FORMAT_ND)
                        .Build();
        EXPECT_EQ(node.UpdateOutputDesc(0, desc), GRAPH_SUCCESS);
        nodes[name] = node;
        return node;
    }

    void Link(GNode& source, GNode& destination, int port = 0)
    {
        TensorDesc desc;
        ASSERT_EQ(source.GetOutputDesc(0, desc), GRAPH_SUCCESS);
        ASSERT_EQ(graph->AddDataEdge(source, 0, destination, port), GRAPH_SUCCESS);
        ASSERT_EQ(destination.UpdateInputDesc(port, desc), GRAPH_SUCCESS);
    }

    GNode Chain(const std::string& name, TensorDesc desc, bool fold, bool scale, const Options& o)
    {
        const DataType outputType = desc.GetDataType();
        if (fold && o.bitcast) {
            desc.SetDataType(DT_UINT8);
        }
        auto source = Make(name, "Data", desc, 0);
        if (!fold) {
            return source;
        }
        auto dims = desc.GetShape().GetDims();
        std::swap(dims[0], dims[1]);
        auto out = Desc(dims, desc.GetDataType());
        const bool reshape = scale && o.reshape;
        auto trans = Make(name + "_trans", reshape ? "Reshape" : (o.transposeD ? "TransposeD" : "Transpose"), out,
                          !reshape && o.transposeD ? 1 : 2);
        Link(source, trans);
        std::vector<int64_t> perm = scale ? std::vector<int64_t>{1, 0, 2} : std::vector<int64_t>{1, 0};
        if (o.transposeD && !reshape) {
            EXPECT_EQ(trans.SetAttr("perm", perm), GRAPH_SUCCESS);
        } else {
            if (reshape) {
                perm = dims;
            }
            auto permDesc = Desc({static_cast<int64_t>(perm.size())}, DT_INT64);
            auto constant = Make(name + "_perm", "Const", permDesc, 0);
            Tensor value(permDesc, reinterpret_cast<const uint8_t*>(perm.data()), perm.size() * sizeof(int64_t));
            EXPECT_EQ(constant.SetAttr("value", value), GRAPH_SUCCESS);
            Link(constant, trans, 1);
        }
        if (!o.bitcast) {
            return trans;
        }
        out.SetDataType(outputType);
        auto cast = Make(name + "_bitcast", "Bitcast", out);
        Link(trans, cast);
        auto type = outputType;
        EXPECT_EQ(cast.SetAttr("type", type), GRAPH_SUCCESS);
        return cast;
    }

    void Build(const Options& o = {})
    {
        nodes.clear();
        graph = std::make_shared<Graph>("inplace_transpose");
        const auto group = o.k / 64 + (o.k % 64 != 0);
        auto a = Chain("a", Desc({o.k, o.m}, o.mx ? o.aType : DT_HIFLOAT8), o.a, false, o);
        auto b = Chain("b", Desc({o.k, o.n}, o.mx ? o.bType : DT_HIFLOAT8), o.b, false, o);
        auto sa = Chain("sa",
                        Desc(o.mx ? std::vector<int64_t>{group, o.m, 2} : std::vector<int64_t>{1},
                             o.mx ? DT_FLOAT8_E8M0 : DT_FLOAT),
                        o.mx && o.a, true, o);
        auto sb = Chain("sb",
                        Desc(o.mx ? std::vector<int64_t>{group, o.n, 2} : std::vector<int64_t>{1},
                             o.mx ? DT_FLOAT8_E8M0 : DT_FLOAT),
                        o.mx && o.b, true, o);
        auto y = Make("y", "Data", Desc({o.m, o.n}, DT_FLOAT), 0);
        q = Builder(graph.get())
                .OpType(o.type)
                .Name("q")
                .IrDefInputs({{"x1", Builder::kEsIrInputRequired, ""},
                              {"x2", Builder::kEsIrInputRequired, ""},
                              {"x2_scale", Builder::kEsIrInputRequired, ""},
                              {"y", Builder::kEsIrInputRequired, ""},
                              {"x1_scale", Builder::kEsIrInputOptional, ""}})
                .IrDefOutputs({{"y", Builder::kEsIrOutputRequired, ""}})
                .InstanceOutputDataType("y", DT_FLOAT)
                .InstanceOutputShape("y", {o.m, o.n})
                .InstanceOutputFormat("y", FORMAT_ND)
                .Build();
        Link(a, q, 0);
        Link(b, q, 1);
        Link(sb, q, 2);
        Link(y, q, 3);
        Link(sa, q, 4);
        q.UpdateOutputDesc(0, Desc({o.m, o.n}, DT_FLOAT));
        bool ta = !o.a;
        bool tb = o.b;
        int64_t gs = o.mx ? 32 : 0;
        q.SetAttr("transpose_x1", ta);
        q.SetAttr("transpose_x2", tb);
        q.SetAttr("group_size", gs);
        auto output = Make("consumer", "Identity", Desc({o.m, o.n}, DT_FLOAT));
        Link(q, output);
        nodes["q"] = q;
    }

    static std::string Snapshot(const GraphPtr& target)
    {
        std::map<std::string, std::string> entries;
        auto describe = [](std::ostringstream& out, const TensorDesc& desc) {
            out << desc.GetDataType() << ':' << desc.GetFormat() << ':' << desc.GetOriginFormat();
            for (auto dim : desc.GetShape().GetDims()) {
                out << ',' << dim;
            }
            out << '/';
            for (auto dim : desc.GetOriginShape().GetDims()) {
                out << ',' << dim;
            }
        };
        for (auto node : target->GetDirectNode()) {
            std::ostringstream out;
            AscendString type;
            node.GetType(type);
            out << type.GetString();
            for (size_t i = 0; i < node.GetInputsSize(); ++i) {
                const auto peer = node.GetInDataNodesAndPortIndexs(i);
                out << " in:" << i << ':' << (peer.first ? Name(*peer.first) : "null") << ':' << peer.second;
                TensorDesc desc;
                node.GetInputDesc(i, desc);
                describe(out, desc);
            }
            for (size_t i = 0; i < node.GetOutputsSize(); ++i) {
                out << " out:" << i;
                TensorDesc desc;
                node.GetOutputDesc(i, desc);
                describe(out, desc);
            }
            for (const auto& peer : node.GetInControlNodes()) {
                out << " ctrl:" << Name(*peer);
            }
            for (const auto* attr : {"transpose_x1", "transpose_x2"}) {
                bool value = false;
                auto status = node.GetAttr(attr, value);
                out << attr << ':' << status << ':' << value;
            }
            int64_t group = 0;
            const auto status = node.GetAttr("group_size", group);
            out << " group:" << status << ':' << group;
            entries[Name(node)] = out.str();
        }
        std::ostringstream result;
        for (const auto& entry : entries) {
            result << entry.first << '=' << entry.second << '\n';
        }
        return result.str();
    }

    std::string X2Branches()
    {
        std::set<std::string> names;
        std::vector<GNodePtr> pending;
        std::ostringstream result;
        for (int port : {1, 2}) {
            const auto peer = q.GetInDataNodesAndPortIndexs(port);
            pending.push_back(peer.first);
            result << port << ':' << (peer.first ? Name(*peer.first) : "null") << ':' << peer.second;
            TensorDesc desc;
            q.GetInputDesc(port, desc);
            result << ':' << desc.GetDataType() << ':' << desc.GetFormat() << ':' << desc.GetOriginFormat();
            for (auto dim : desc.GetShape().GetDims()) {
                result << ',' << dim;
            }
            result << '/';
            for (auto dim : desc.GetOriginShape().GetDims()) {
                result << ',' << dim;
            }
        }
        bool transpose = false;
        result << q.GetAttr("transpose_x2", transpose) << ':' << transpose << '\n';
        while (!pending.empty()) {
            auto node = pending.back();
            pending.pop_back();
            if (!node || !names.insert(Name(*node)).second) {
                continue;
            }
            for (size_t i = 0; i < node->GetInputsSize(); ++i) {
                pending.push_back(node->GetInDataNodesAndPortIndexs(i).first);
            }
        }
        std::istringstream snapshot(Snapshot(graph));
        std::string line;
        while (std::getline(snapshot, line)) {
            if (names.count(line.substr(0, line.find('=')))) {
                result << line << '\n';
            }
        }
        return result.str();
    }

    Status Run()
    {
        const auto before = X2Branches();
        CustomPassContext context;
        const auto result = pass.Run(graph, context);
        EXPECT_EQ(X2Branches(), before);
        return result;
    }

    void Rejected()
    {
        const auto before = Snapshot(graph);
        EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
        EXPECT_EQ(Snapshot(graph), before);
    }

    void Fused(const Options& o)
    {
        ASSERT_EQ(Run(), SUCCESS);
        bool a = false;
        bool b = false;
        q.GetAttr("transpose_x1", a);
        q.GetAttr("transpose_x2", b);
        EXPECT_TRUE(a);
        EXPECT_EQ(b, o.b); // Legacy subclass does not toggle transpose_x2.
        EXPECT_EQ(Name(*q.GetInDataNodesAndPortIndexs(3).first), "y");
        EXPECT_EQ(Name(*nodes["consumer"].GetInDataNodesAndPortIndexs(0).first), "q");
        TensorDesc desc;
        q.GetInputDesc(0, desc);
        EXPECT_EQ(desc.GetShape().GetDims(), (std::vector<int64_t>{o.k, o.m}));
        q.GetInputDesc(1, desc);
        EXPECT_EQ(desc.GetShape().GetDims(), (o.b ? std::vector<int64_t>{o.n, o.k} : std::vector<int64_t>{o.k, o.n}));
        Rejected();
    }
};

TEST_F(InplaceTransposeTest, LegacyDataTypeAndOutputMatrix)
{
    for (auto a : {DT_INT8, DT_INT4, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT4_E2M1}) {
        for (auto b : {DT_INT8, DT_INT4, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT4_E2M1}) {
            for (auto output : {DT_INT8, DT_FLOAT16, DT_BF16, DT_INT32, DT_FLOAT}) {
                SCOPED_TRACE(std::to_string(a) + "/" + std::to_string(b) + "/" + std::to_string(output));
                Options o;
                o.aType = a;
                o.bType = b;
                Build(o);
                q.UpdateOutputDesc(0, Desc({o.m, o.n}, output));
                Fused(o);
            }
        }
    }
}

TEST_F(InplaceTransposeTest, X1OnlyPatternMatrix)
{
    for (int sides = 1; sides < 4; ++sides) {
        for (int pattern = 0; pattern < 8; ++pattern) {
            for (bool mx : {false, true}) {
                SCOPED_TRACE(std::to_string(sides) + "/" + std::to_string(pattern) + "/" + std::to_string(mx));
                Options o;
                o.a = sides & 1;
                o.b = sides & 2;
                o.bitcast = pattern & 1;
                o.reshape = pattern & 2;
                o.transposeD = pattern & 4;
                o.mx = mx;
                Build(o);
                if (o.a && !o.b) {
                    Fused(o);
                } else {
                    Rejected();
                }
            }
        }
    }
}

TEST_F(InplaceTransposeTest, OnlySupportedFusedTransposeAttributesAreAccepted)
{
    for (bool a : {false, true}) {
        for (bool b : {false, true}) {
            Build();
            q.SetAttr("transpose_x1", a);
            q.SetAttr("transpose_x2", b);
            if (a || b) {
                Rejected();
                continue;
            }
            ASSERT_EQ(Run(), SUCCESS);
            bool actual = false;
            q.GetAttr("transpose_x1", actual);
            EXPECT_TRUE(actual);
            q.GetAttr("transpose_x2", actual);
            EXPECT_EQ(actual, b);
        }
    }
}

TEST_F(InplaceTransposeTest, NoAdditionalPermutationOrGroupChecks)
{
    Build();
    std::vector<int64_t> perm = {0, 1};
    nodes["a_trans"].SetAttr("perm", perm);
    int64_t group = 64;
    q.SetAttr("group_size", group);
    Fused({});
}

TEST_F(InplaceTransposeTest, OptionalX1ScaleMayBeDisconnected)
{
    Build();
    ASSERT_EQ(graph->RemoveEdge(nodes["sa_trans"], 0, q, 4), GRAPH_SUCCESS);
    Fused({});
}

TEST_F(InplaceTransposeTest, MissingRequiredOrTransformInputsLeaveGraphUnchanged)
{
    for (const int port : {0, 1, 2}) {
        Build();
        const auto peer = q.GetInDataNodesAndPortIndexs(port);
        ASSERT_EQ(graph->RemoveEdge(*peer.first, peer.second, q, port), GRAPH_SUCCESS);
        Rejected();
    }
    for (const bool bitcast : {false, true}) {
        Options o;
        o.bitcast = bitcast;
        for (const auto* name : {"a_trans", "sa_trans"}) {
            Build(o);
            auto& trans = nodes[name];
            const auto peer = trans.GetInDataNodesAndPortIndexs(0);
            ASSERT_EQ(graph->RemoveEdge(*peer.first, peer.second, trans, 0), GRAPH_SUCCESS);
            Rejected();
        }
    }
    Options o;
    o.bitcast = true;
    Build(o);
    ASSERT_EQ(graph->RemoveEdge(nodes["a_trans"], 0, nodes["a_bitcast"], 0), GRAPH_SUCCESS);
    Rejected();
}

TEST_F(InplaceTransposeTest, X2ScaleMustBeConnected)
{
    Build();
    ASSERT_EQ(graph->RemoveEdge(nodes["sb"], 0, q, 2), GRAPH_SUCCESS);
    Rejected();
}

TEST_F(InplaceTransposeTest, ScaleNeedNotHaveATransform)
{
    Build();
    ASSERT_EQ(graph->RemoveEdge(nodes["sa_trans"], 0, q, 4), GRAPH_SUCCESS);
    Link(nodes["sa"], q, 4);
    Fused({});
}

TEST_F(InplaceTransposeTest, StaticNonSingletonScaleReshapeIsRetained)
{
    Options o;
    o.reshape = true;
    o.k = 128;
    Build(o);
    Fused(o);
    EXPECT_EQ(Name(*q.GetInDataNodesAndPortIndexs(4).first), "sa_trans");
}

TEST_F(InplaceTransposeTest, DynamicScaleReshapeIsFused)
{
    Options o;
    o.reshape = true;
    o.m = -1;
    o.k = 128;
    Build(o);
    Fused(o);
    EXPECT_EQ(Name(*q.GetInDataNodesAndPortIndexs(4).first), "sa");
}

TEST_F(InplaceTransposeTest, RuntimeReshapeTargetIsNotAnAdditionalGate)
{
    Options o;
    o.reshape = true;
    Build(o);
    ASSERT_EQ(graph->RemoveEdge(nodes["sa_perm"], 0, nodes["sa_trans"], 1), GRAPH_SUCCESS);
    auto shape = Make("runtime_shape", "Data", Desc({3}, DT_INT64), 0);
    Link(shape, nodes["sa_trans"], 1);
    Fused(o);
}

TEST_F(InplaceTransposeTest, PlatformRestrictionRejectsBitcastOnlyWithoutL12btRequirement)
{
    auto& manager = fe::PlatformInfoManager::Instance();
    fe::PlatformInfo info;
    info.ai_core_intrinsic_dtype_map["Intrinsic_mmad"] = {"s8s4"};
    manager.platform_info_map_["inplace_transpose_ut"] = info;
    Build();
    Fused({});
    Options o;
    o.bitcast = true;
    Build(o);
    Rejected();
}

TEST_F(InplaceTransposeTest, MissingPlatformDoesNotAddARejection)
{
    Build();
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
    Fused({});
}

TEST_F(InplaceTransposeTest, SharedTransposeKeepsOtherConsumer)
{
    Build();
    TensorDesc desc;
    nodes["a_trans"].GetOutputDesc(0, desc);
    auto other = Make("other", "Identity", desc);
    Link(nodes["a_trans"], other);
    Fused({});
    EXPECT_EQ(Name(*other.GetInDataNodesAndPortIndexs(0).first), "a_trans");
}

TEST_F(InplaceTransposeTest, SharedBitcastAndLegacyDescriptorSideEffect)
{
    Options o;
    o.bitcast = true;
    Build(o);
    TensorDesc desc;
    nodes["a_bitcast"].GetOutputDesc(0, desc);
    auto other = Make("cast_consumer", "Identity", desc);
    Link(nodes["a_bitcast"], other);
    nodes["a_trans"].GetOutputDesc(0, desc);
    auto shared = Make("transpose_consumer", "Identity", desc);
    Link(nodes["a_trans"], shared);
    Fused(o);
    nodes["a_trans"].GetInputDesc(0, desc);
    EXPECT_EQ(desc.GetDataType(), o.aType);
    nodes["a_bitcast"].GetOutputDesc(0, desc);
    EXPECT_EQ(desc.GetShape().GetDims(), (std::vector<int64_t>{o.m, o.k}));
}

TEST_F(InplaceTransposeTest, TransformControlEdgesDoNotAddAnEligibilityGate)
{
    Build();
    ASSERT_EQ(graph->AddControlEdge(nodes["y"], nodes["a_trans"]), GRAPH_SUCCESS);
    ASSERT_EQ(Run(), SUCCESS);
}

TEST_F(InplaceTransposeTest, YRefAndTargetControlEdgesArePreserved)
{
    Build();
    auto before = Make("prior_update", "Identity", Desc({16, 32}, DT_FLOAT));
    auto after = Make("later_read", "Identity", Desc({16, 32}, DT_FLOAT));
    Link(nodes["y"], before);
    Link(nodes["y"], after);
    ASSERT_EQ(graph->AddControlEdge(before, q), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddControlEdge(q, after), GRAPH_SUCCESS);
    Fused({});
    EXPECT_EQ(Name(*q.GetInControlNodes()[0]), "prior_update");
    EXPECT_EQ(Name(*q.GetOutControlNodes()[0]), "later_read");
}

TEST_F(InplaceTransposeTest, NonNdAndHigherRankAreNotRejected)
{
    Build();
    q.UpdateInputDesc(1, Desc({2, 64, 32}, DT_FLOAT8_E5M2));
    TensorDesc desc;
    q.GetInputDesc(1, desc);
    desc.SetFormat(FORMAT_FRACTAL_NZ);
    q.UpdateInputDesc(1, desc);
    ASSERT_EQ(Run(), SUCCESS);
}

TEST_F(InplaceTransposeTest, LowerRankAndUnsupportedTypesAreRejected)
{
    for (auto type : {DT_FLOAT16, DT_FLOAT, DT_INT32}) {
        Build();
        q.UpdateInputDesc(0, Desc({64, 16}, type));
        Rejected();
    }
    Build();
    q.UpdateInputDesc(0, Desc({16}, DT_INT8));
    Rejected();
}

TEST_F(InplaceTransposeTest, DoesNotMatchQuantBatchMatmulV3)
{
    Options o;
    o.type = "QuantBatchMatmulV3";
    Build(o);
    Rejected();
}

TEST_F(InplaceTransposeTest, WideBitcastIsNotRejectedByThePass)
{
    Options o;
    o.bitcast = true;
    Build(o);
    nodes["a_trans"].UpdateInputDesc(0, Desc({64, 16}, DT_INT32));
    Fused(o);
}

TEST_F(InplaceTransposeTest, RandomShapesRetainLegacyBDescriptors)
{
    std::mt19937 random(20260923);
    for (int i = 0; i < 80; ++i) {
        Options o;
        o.m = random() % 200 + 1;
        o.n = random() % 200 + 1;
        o.k = random() % 300 + 1;
        o.b = i & 1;
        o.bitcast = i & 2;
        Build(o);
        if (o.b) {
            Rejected();
        } else {
            Fused(o);
        }
    }
}
TEST_F(InplaceTransposeTest, SharedConstantInputIsNotDeleted)
{
    Options o;
    o.transposeD = false;
    Build(o);
    auto other = Make("constant_consumer", "Identity", Desc({2}, DT_INT64));
    Link(nodes["a_perm"], other);
    Fused(o);
    EXPECT_EQ(Name(*other.GetInDataNodesAndPortIndexs(0).first), "a_perm");
}

TEST_F(InplaceTransposeTest, ConstantControlConsumerKeepsTheConstant)
{
    Options o;
    o.transposeD = false;
    Build(o);
    ASSERT_EQ(graph->AddControlEdge(nodes["a_perm"], nodes["consumer"]), GRAPH_SUCCESS);
    Fused(o);
}

TEST_F(InplaceTransposeTest, OutgoingTransformControlEdgeIsRelayed)
{
    Build();
    ASSERT_EQ(graph->AddControlEdge(nodes["a_trans"], nodes["consumer"]), GRAPH_SUCCESS);
    ASSERT_EQ(Run(), SUCCESS);
    EXPECT_EQ(Name(*nodes["consumer"].GetInControlNodes()[0]), "a");
}

TEST_F(InplaceTransposeTest, StreamLabelMismatchKeepsPatternUnchanged)
{
    Build();
    AscendString label("branch_one");
    nodes["a_trans"].SetAttr("_stream_label", label);
    Rejected();
}

TEST_F(InplaceTransposeTest, MatchingStreamLabelsDoNotBlockFusion)
{
    Build();
    AscendString label("branch_one");
    nodes["a_trans"].SetAttr("_stream_label", label);
    q.SetAttr("_stream_label", label);
    Fused({});
}

TEST_F(InplaceTransposeTest, DynamicRecognitionIsIndependentOfRunOrder)
{
    fe::PlatformInfo info;
    info.ai_core_intrinsic_dtype_map["Intrinsic_mmad"] = {"s8s4"};
    fe::PlatformInfoManager::Instance().platform_info_map_["inplace_transpose_ut"] = info;
    std::map<bool, std::string> expected;
    for (const bool dynamicFirst : {false, true}) {
        for (const bool dynamic : {dynamicFirst, !dynamicFirst, dynamicFirst}) {
            Options o;
            o.m = dynamic ? -1 : 16;
            o.k = 128;
            o.reshape = true;
            Build(o);
            ASSERT_EQ(graph->RemoveEdge(nodes["sa_trans"], 0, q, 4), GRAPH_SUCCESS);
            auto cast = Make("scale_only_bitcast", "Bitcast", Desc({o.m, 2, 2}, DT_FLOAT8_E8M0));
            Link(nodes["sa_trans"], cast);
            Link(cast, q, 4);
            if (dynamic) {
                Rejected(); // Current dynamic reshape selects the platform-restricted Bitcast pattern.
            } else {
                ASSERT_EQ(Run(), SUCCESS); // Static nonsingleton reshape does not select Bitcast mode.
            }
            const auto snapshot = Snapshot(graph);
            if (expected.count(dynamic) == 0) {
                expected[dynamic] = snapshot;
            } else {
                EXPECT_EQ(snapshot, expected[dynamic]);
            }
        }
    }
}

TEST_F(InplaceTransposeTest, ScaleOnlyBitcastUsesCurrentDynamicShape)
{
    Options o;
    o.k = 128;
    o.m = -1;
    o.reshape = true;
    Build(o);
    ASSERT_EQ(graph->RemoveEdge(nodes["sa_trans"], 0, q, 4), GRAPH_SUCCESS);
    auto cast = Make("scale_only_bitcast", "Bitcast", Desc({-1, 2, 2}, DT_FLOAT8_E8M0));
    Link(nodes["sa_trans"], cast);
    Link(cast, q, 4);
    ASSERT_EQ(Run(), SUCCESS);
    EXPECT_EQ(Name(*q.GetInDataNodesAndPortIndexs(4).first), "scale_only_bitcast");
    EXPECT_EQ(Name(*cast.GetInDataNodesAndPortIndexs(0).first), "sa");
}

TEST_F(InplaceTransposeTest, GenericScaleReshapeUsesLastTwoDimensions)
{
    Options o;
    o.reshape = true;
    Build(o);
    nodes["sa_trans"].UpdateInputDesc(0, Desc({8, 1}, DT_FLOAT));
    nodes["sa_trans"].UpdateOutputDesc(0, Desc({1, 8}, DT_FLOAT));
    q.UpdateInputDesc(4, Desc({1, 8}, DT_FLOAT));
    ASSERT_EQ(Run(), SUCCESS);
    EXPECT_EQ(Name(*q.GetInDataNodesAndPortIndexs(4).first), "sa");
}

TEST_F(InplaceTransposeTest, YRefAndGroupRemainOutsidePassEligibility)
{
    Build();
    ASSERT_EQ(graph->RemoveEdge(nodes["y"], 0, q, 3), GRAPH_SUCCESS);
    q.UpdateOutputDesc(0, Desc({99, 100}, DT_FLOAT16));
    int64_t group = -123;
    q.SetAttr("group_size", group);
    ASSERT_EQ(Run(), SUCCESS);
}

TEST_F(InplaceTransposeTest, OriginShapeMayDifferFromPhysicalShape)
{
    Build();
    auto desc = Desc({64, 16}, DT_FLOAT8_E4M3FN);
    desc.SetShape(Shape({1, 64, 16}));
    q.UpdateInputDesc(0, desc);
    ASSERT_EQ(Run(), SUCCESS);
}
TEST_F(InplaceTransposeTest, SharedDataScaleTransformPreservesLegacyFailure)
{
    Build();
    ASSERT_EQ(graph->RemoveEdge(nodes["sa_trans"], 0, q, 4), GRAPH_SUCCESS);
    Link(nodes["a_trans"], q, 4);
    ASSERT_EQ(Run(), GRAPH_FAILED);
}

TEST_F(InplaceTransposeTest, NonzeroSourceOutputPortIsPreserved)
{
    Build();
    const auto desc = Desc({64, 16}, DT_FLOAT8_E4M3FN);
    auto source = Builder(graph.get())
                      .OpType("TwoOutputs")
                      .Name("two_outputs")
                      .IrDefOutputs(
                          {{"first", Builder::kEsIrOutputRequired, ""}, {"second", Builder::kEsIrOutputRequired, ""}})
                      .InstanceOutputDataType("first", DT_FLOAT8_E4M3FN)
                      .InstanceOutputShape("first", {64, 16})
                      .InstanceOutputDataType("second", DT_FLOAT8_E4M3FN)
                      .InstanceOutputShape("second", {64, 16})
                      .Build();
    source.UpdateOutputDesc(0, desc);
    source.UpdateOutputDesc(1, desc);
    ASSERT_EQ(graph->RemoveEdge(nodes["a"], 0, nodes["a_trans"], 0), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddDataEdge(source, 1, nodes["a_trans"], 0), GRAPH_SUCCESS);
    ASSERT_EQ(Run(), SUCCESS);
    EXPECT_EQ(q.GetInDataNodesAndPortIndexs(0).second, 1);
}
TEST_F(InplaceTransposeTest, SharedTransformWithAlternatePathToTarget)
{
    Build();
    auto other = Make("alternate", "Identity", Desc({16, 64}, DT_FLOAT8_E4M3FN));
    Link(nodes["a_trans"], other);
    ASSERT_EQ(graph->RemoveEdge(nodes["b"], 0, q, 1), GRAPH_SUCCESS);
    Link(other, q, 1);
    ASSERT_EQ(Run(), SUCCESS);
}

TEST_F(InplaceTransposeTest, FusionAttributeCompatibility)
{
    for (const auto* key : {"_user_stream_label", "_super_kernel_scope", "_super_kernel_options", "_op_aicore_num",
                            "_op_vectorcore_num"}) {
        SCOPED_TRACE(key);
        Build();
        AscendString first("1");
        AscendString second("2");
        q.SetAttr(key, first);
        nodes["a_trans"].SetAttr(key, second);
        Rejected();
        nodes["a_trans"].SetAttr(key, first);
        ASSERT_EQ(Run(), SUCCESS);
        Build();
        q.SetAttr(key, first);
        EXPECT_EQ(Run(), SUCCESS);
    }
}

TEST_F(InplaceTransposeTest, EmptyStreamLabelPreservesLegacyMappingOrder)
{
    Build();
    AscendString empty("");
    AscendString label("branch_one");
    q.SetAttr("_stream_label", empty);
    nodes["a_trans"].SetAttr("_stream_label", label);
    ASSERT_EQ(Run(), SUCCESS);
    Build();
    q.SetAttr("_stream_label", label);
    nodes["a_trans"].SetAttr("_stream_label", empty);
    Rejected();
}

TEST_F(InplaceTransposeTest, DeterministicAttributeCompatibility)
{
    for (const auto* key : {"_deterministic", "_deterministic_level"}) {
        for (const auto* value : {"0", "1", "2", "3", "4", "bad"}) {
            SCOPED_TRACE(std::string(key) + "=" + value);
            Build();
            AscendString text(value);
            q.SetAttr(key, text);
            const bool valid = std::string(value) != "bad" &&
                               std::stoi(value) <= (std::string(key) == "_deterministic" ? 1 : 3);
            EXPECT_EQ(Run(), valid ? SUCCESS : GRAPH_NOT_CHANGED);
        }
        Build();
        int64_t integer = 1;
        q.SetAttr(key, integer);
        Rejected();
        Build();
        AscendString zero("0");
        AscendString one("1");
        q.SetAttr(key, zero);
        nodes["a_trans"].SetAttr(key, one);
        Rejected();
    }
}

TEST_F(InplaceTransposeTest, MultipleTargetsShareTransformsAndCacheMatches)
{
    for (const bool mismatchedLabel : {false, true}) {
        Build();
        auto second = Builder(graph.get())
                          .OpType("QuantBatchMatmulInplaceAdd")
                          .Name("q_second")
                          .IrDefInputs({{"x1", Builder::kEsIrInputRequired, ""},
                                        {"x2", Builder::kEsIrInputRequired, ""},
                                        {"x2_scale", Builder::kEsIrInputRequired, ""},
                                        {"y", Builder::kEsIrInputRequired, ""},
                                        {"x1_scale", Builder::kEsIrInputOptional, ""}})
                          .IrDefOutputs({{"y", Builder::kEsIrOutputRequired, ""}})
                          .InstanceOutputDataType("y", DT_FLOAT)
                          .InstanceOutputShape("y", {16, 32})
                          .Build();
        Link(nodes["a_trans"], second, 0);
        Link(nodes["b"], second, 1);
        Link(nodes["sb"], second, 2);
        Link(nodes["y"], second, 3);
        Link(nodes["sa_trans"], second, 4);
        bool transpose = false;
        second.SetAttr("transpose_x1", transpose);
        second.SetAttr("transpose_x2", transpose);
        second.UpdateOutputDesc(0, Desc({16, 32}, DT_FLOAT));
        if (mismatchedLabel) {
            AscendString label("another_stream");
            second.SetAttr("_stream_label", label);
            Rejected();
        } else {
            ASSERT_EQ(Run(), SUCCESS);
            EXPECT_EQ(Name(*q.GetInDataNodesAndPortIndexs(0).first), "a");
            EXPECT_EQ(Name(*second.GetInDataNodesAndPortIndexs(0).first), "a");
        }
    }
}
TEST_F(InplaceTransposeTest, X2BitcastDoesNotSelectBitcastMode)
{
    Build();
    Options right;
    right.bitcast = true;
    auto b = Chain("right", Desc({64, 32}, DT_FLOAT8_E5M2), true, false, right);
    graph->RemoveEdge(nodes["b"], 0, q, 1);
    Link(b, q, 1);
    AscendString label("right_only");
    nodes["right_trans"].SetAttr("_stream_label", label);
    fe::PlatformInfo info;
    info.ai_core_intrinsic_dtype_map["Intrinsic_mmad"] = {"s8s4"};
    fe::PlatformInfoManager::Instance().platform_info_map_["inplace_transpose_ut"] = info;
    ASSERT_EQ(Run(), SUCCESS);
    EXPECT_EQ(Name(*q.GetInDataNodesAndPortIndexs(1).first), "right_bitcast");
}

TEST_F(InplaceTransposeTest, SharedBitcastWithX2IsNotModified)
{
    for (int port : {1, 2}) {
        Options o;
        o.bitcast = true;
        Build(o);
        auto previous = q.GetInDataNodesAndPortIndexs(port);
        graph->RemoveEdge(*previous.first, previous.second, q, port);
        Link(nodes["a_bitcast"], q, port);
        Rejected();
    }
}

TEST_F(InplaceTransposeTest, SharedBitcastSourceWithX2IsNotModified)
{
    Options o;
    o.bitcast = true;
    Build(o);
    graph->RemoveEdge(nodes["b"], 0, q, 1);
    Link(nodes["a_trans"], q, 1);
    Rejected();
}

} // namespace
