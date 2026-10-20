/** Copyright (c) 2026 Huawei Technologies Co., Ltd. See LICENSE for licensing details. */
#include <gtest/gtest.h>
#include <cstdlib>
#include <dlfcn.h>
#include <map>
#include "../../../op_graph/fusion_pass/addlayernorm_dynamicquant_fusion_pass.h"
#include "../../../op_graph/fusion_pass/quantize_addlayernorm_fusion_pass.h"
#include "../../../op_graph/fusion_pass/add_layer_norm_quant_fusion_common.h"
#ifdef QUANT_ALN_STANDALONE_TEST
#include "base/registry/op_impl_space_registry_v2.h"
#endif

namespace ops {
namespace {
using namespace aln_quant;
struct Config {
    std::string type = "DynamicQuant";
    int branches = 1;
    bool bias = true;
    bool mul = false;
    int smooth_mask = 0;
    int swap_mask = 0;
    bool offset = true;
    bool reshape = false;
    bool shared_scale = false;
    bool mean = false;
    bool side_y = false;
    bool side_mul = false;
    bool double_smooth = false;
    bool group = false;
    std::string round = "round";
    bool sqrt = false;
    ge::DataType dtype = ge::DT_BF16;
    ge::DataType scale_dtype = ge::DT_FLOAT;
    int64_t width = 32;
    int64_t axis = -1;
    int64_t dst = ge::DT_INT8;
};
void Describe(const EsTensorHolder& tensor, ge::DataType dtype, const std::vector<int64_t>& shape)
{
    TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, dtype);
    ASSERT_EQ(tensor.GetProducer()->UpdateOutputDesc(tensor.GetProducerOutIndex(), desc), ge::GRAPH_SUCCESS);
}
ge::GraphPtr Build(const Config& c)
{
    EsGraphBuilder builder("quant_aln_test");
    int index = 0;
    auto input = [&](ge::DataType dtype, std::vector<int64_t> shape) {
        auto name = "input_" + std::to_string(index);
        return builder.CreateInput(index++, name.c_str(), dtype, ge::FORMAT_ND, shape);
    };
    const std::vector<int64_t> shape{2, c.width};
    auto x1 = input(c.dtype, shape), x2 = input(c.dtype, shape);
    auto gamma = input(c.dtype, {c.width}), beta = input(c.dtype, {c.width});
    auto bias = c.bias ? input(c.dtype, {c.width}) : EsTensorHolder(nullptr);
    auto norm = ge::es::AddLayerNorm(x1, x2, gamma, beta, bias, 0.0002f, true);
    Describe(norm.y, c.dtype, shape);
    Describe(norm.x, c.dtype, shape);
    Describe(norm.mean, ge::DT_FLOAT, {2, 1});
    Describe(norm.rstd, ge::DT_FLOAT, {2, 1});
    std::vector<EsTensorHolder> outputs, scales;
    EsTensorHolder shared_scale(nullptr);
    for (int i = 0; i < c.branches; ++i) {
        auto y = norm.y;
        if (c.mul) {
            auto smooth = input(c.dtype, {c.width});
            y = ((c.swap_mask >> i) & 1) ? ge::es::Mul(smooth, y) : ge::es::Mul(y, smooth);
            Describe(y, c.dtype, shape);
            if (c.side_mul) {
                outputs.push_back(y);
            }
        }
        if (c.reshape) {
            auto shape_tensor = builder.CreateVector(std::vector<int64_t>{1, 2 * c.width});
            y = ge::es::Reshape(y, shape_tensor);
            Describe(y, c.dtype, {1, 2 * c.width});
        }
        if (c.type == "DynamicQuant") {
            auto smooth = ((c.smooth_mask >> i) & 1) || (c.double_smooth && i == 1) ? input(c.dtype, {c.width}) :
                                                                                      EsTensorHolder(nullptr);
            auto group = c.group ? input(ge::DT_INT32, {2}) : EsTensorHolder(nullptr);
            auto q = ge::es::DynamicQuant(y, smooth, group, c.dst);
            Describe(q.y, static_cast<ge::DataType>(c.dst), shape);
            Describe(q.scale, ge::DT_FLOAT, {2});
            outputs.push_back(q.y);
            scales.push_back(q.scale);
        } else {
            auto scale = c.shared_scale && shared_scale.GetProducer() ? shared_scale : input(c.scale_dtype, {c.width});
            shared_scale = scale;
            auto zero = c.offset ? input(c.scale_dtype, {c.width}) : EsTensorHolder(nullptr);
            auto q = c.type == "Quantize" ?
                         ge::es::Quantize(y, scale, zero, "torch.qint8", c.axis) :
                         ge::es::AscendQuantV2(y, scale, zero, c.sqrt, c.round.c_str(), c.dst, c.axis);
            Describe(q, static_cast<ge::DataType>(c.dst), c.reshape ? std::vector<int64_t>{1, 2 * c.width} : shape);
            outputs.push_back(q);
        }
    }
    outputs.push_back(norm.x);
    outputs.insert(outputs.end(), scales.begin(), scales.end());
    if (c.mean) {
        outputs.push_back(norm.mean);
    }
    if (c.side_y) {
        outputs.push_back(norm.y);
    }
    auto graph = builder.BuildAndReset(outputs);
    // ES builds topology; the pass runs after shape inference, so supply real input descriptors too.
    for (auto node : graph->GetAllNodes()) {
        for (int port = 0; port < static_cast<int>(node.GetInputsSize()); ++port) {
            auto edge = node.GetInDataNodesAndPortIndexs(port);
            if (edge.first) {
                TensorDesc desc;
                EXPECT_EQ(edge.first->GetOutputDesc(edge.second, desc), ge::GRAPH_SUCCESS);
                EXPECT_EQ(node.UpdateInputDesc(port, desc), ge::GRAPH_SUCCESS);
            }
        }
    }
    return graph;
}
int Count(const ge::GraphPtr& graph, const std::string& type)
{
    int count = 0;
    for (const auto& node : graph->GetAllNodes()) {
        count += Type(node) == type;
    }
    return count;
}
class QuantAlnFusionTest : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
#ifdef QUANT_ALN_STANDALONE_TEST
        const char* opp = std::getenv("ASCEND_OPP_PATH");
        ASSERT_NE(opp, nullptr);
        auto registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
        for (const char* family : {"nn", "math"}) {
            const std::string path = std::string(opp) + "/built-in/op_impl/ai_core/tbe/op_host/lib/linux/" +
#if defined(__aarch64__)
                                     "aarch64/" +
#else
                                     "x86_64/" +
#endif
                                     "libophost_" + family + ".so";
            gert::OppSoDesc descriptor({ge::AscendString(path.c_str())}, family);
            ASSERT_EQ(registry->AddSoToRegistry(descriptor), ge::GRAPH_SUCCESS) << path;
        }
        ASSERT_EQ(gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(registry), ge::GRAPH_SUCCESS);
#endif
    }
    fe::OptionalInfo saved_legacy_options_;
    fe::OptionalInfos saved_options_;
    void SetUp() override
    {
        auto& manager = fe::PlatformInfoManager::Instance();
        saved_legacy_options_ = manager.opti_compilation_info_;
        saved_options_ = manager.opti_compilation_infos_;
        SetPlatform(true);
    }
    void TearDown() override
    {
        auto& manager = fe::PlatformInfoManager::Instance();
        manager.opti_compilation_info_ = saved_legacy_options_;
        manager.opti_compilation_infos_ = saved_options_;
        manager.platform_info_map_.erase("quant_aln_test");
        manager.platform_infos_map_.erase("quant_aln_test");
    }
    void SetPlatform(bool reg_base, const std::string& arch = "3510")
    {
        auto& manager = fe::PlatformInfoManager::Instance();
        fe::PlatformInfo legacy;
        legacy.str_info.short_soc_version = reg_base ? "Ascend950" : "Ascend910B";
        fe::OptionalInfo old_options;
        old_options.soc_version = "quant_aln_test";
        manager.platform_info_map_["quant_aln_test"] = legacy;
        manager.SetOptionalCompilationInfo(old_options);
        fe::PlatFormInfos info;
        ASSERT_TRUE(info.Init());
        std::map<std::string, std::string> version{{"NpuArch", reg_base ? arch : "2201"}};
        info.SetPlatformRes("version", version);
        manager.platform_infos_map_["quant_aln_test"] = info;
        fe::OptionalInfos options;
        ASSERT_TRUE(options.Init());
        options.SetSocVersion("quant_aln_test");
        manager.SetOptionalCompilationInfo(options);
    }
    void Check(const Config& config, const std::string& expected)
    {
        auto graph = Build(config);
        ASSERT_NE(graph, nullptr);
        const auto old_size = graph->GetAllNodes().size();
        std::vector<TensorDesc> output_descs;
        std::vector<std::string> scale_sources;
        for (const auto& node : graph->GetAllNodes()) {
            if (Type(node) != "NetOutput") {
                continue;
            }
            for (int i = 0; i < static_cast<int>(node.GetInputsSize()); ++i) {
                auto edge = node.GetInDataNodesAndPortIndexs(i);
                TensorDesc desc;
                ASSERT_EQ(edge.first->GetOutputDesc(edge.second, desc), ge::GRAPH_SUCCESS);
                output_descs.push_back(desc);
                if (i < config.branches) {
                    auto scale = edge.first->GetInDataNodesAndPortIndexs(1).first;
                    if (config.mul) {
                        auto mul = edge.first->GetInDataNodesAndPortIndexs(0).first;
                        auto first = mul->GetInDataNodesAndPortIndexs(0).first;
                        scale = mul->GetInDataNodesAndPortIndexs(Type(*first) == "AddLayerNorm" ? 1 : 0).first;
                    }
                    scale_sources.push_back(scale ? Name(*scale) : "");
                }
            }
        }
        ge::CustomPassContext context;
        const char* pass_name = config.type == "DynamicQuant" ? "AddLayerNormDynamicQuantFusionPass" :
                                                                "QuantizeAddLayerNormPass";
        context.SetPassName(pass_name);
        ge::Status status;
        if (config.type == "DynamicQuant") {
            AddLayerNormDynamicQuantFusionPass pass;
            status = pass.Run(graph, context);
        } else {
            QuantizeAddLayerNormPass pass;
            status = pass.Run(graph, context);
        }
        if (expected.empty()) {
            EXPECT_EQ(status, ge::GRAPH_NOT_CHANGED);
            EXPECT_EQ(Count(graph, "AddLayerNorm"), 1);
            EXPECT_EQ(graph->GetAllNodes().size(), old_size);
            return;
        }
        ASSERT_EQ(status, ge::SUCCESS);
        EXPECT_EQ(Count(graph, "AddLayerNorm"), 0);
        EXPECT_EQ(Count(graph, config.type), 0);
        ASSERT_EQ(Count(graph, expected), 1);
        for (auto node : graph->GetAllNodes()) {
            if (Type(node) != expected) {
                continue;
            }
            for (int port = 0; port < 4; ++port) {
                TensorDesc input_desc;
                ASSERT_EQ(node.GetInputDesc(port, input_desc), ge::GRAPH_SUCCESS);
                EXPECT_EQ(input_desc.GetDataType(), config.dtype);
                EXPECT_EQ(input_desc.GetShape().GetDims(),
                          port < 2 ? std::vector<int64_t>({2, config.width}) : std::vector<int64_t>({config.width}));
            }
            float epsilon = 0;
            bool additional = false;
            ASSERT_EQ(node.GetAttr("epsilon", epsilon), ge::GRAPH_SUCCESS);
            ASSERT_EQ(node.GetAttr("additional_output", additional), ge::GRAPH_SUCCESS);
            EXPECT_FLOAT_EQ(epsilon, 0.0002f);
            EXPECT_TRUE(additional);
            if (expected == "AddLayerNormQuant") {
                ge::AscendString mode("dynamic");
                node.GetAttr("quant_mode", mode);
                EXPECT_EQ(std::string(mode.GetString()), config.type == "DynamicQuant" ? "dynamic" : "static");
                if (config.type != "DynamicQuant") {
                    bool div = true; // IR default may be elided by the ES builder.
                    node.GetAttr("div_mode", div);
                    EXPECT_EQ(div, config.type == "Quantize");
                }
            }
        }
        EXPECT_EQ(Count(graph, "Reshape"), config.reshape ? 1 : 0);
        EXPECT_EQ(Count(graph, "Mul"), 0);
        // Every graph output must remain connected; quantized branch order and residual/scale ports are checked.
        for (const auto& node : graph->GetAllNodes()) {
            if (Type(node) != "NetOutput") {
                continue;
            }
            ASSERT_EQ(node.GetInputsSize(),
                      static_cast<size_t>(config.branches + 1 + (config.type == "DynamicQuant" ? config.branches : 0)));
            for (int i = 0; i < static_cast<int>(node.GetInputsSize()); ++i) {
                auto edge = node.GetInDataNodesAndPortIndexs(i);
                ASSERT_NE(edge.first, nullptr);
                TensorDesc actual;
                ASSERT_EQ(edge.first->GetOutputDesc(edge.second, actual), ge::GRAPH_SUCCESS);
                EXPECT_EQ(actual.GetDataType(), output_descs[i].GetDataType());
                EXPECT_EQ(actual.GetShape().GetDims(), output_descs[i].GetShape().GetDims());
                if (i < config.branches && !config.reshape) {
                    EXPECT_EQ(Type(*edge.first), expected);
                    EXPECT_GE(edge.second, 0);
                    EXPECT_LT(edge.second, config.branches);
                    int32_t port = -1;
                    const std::string scale_name = expected == "QuantizeAddLayerNorm" ?
                                                       "scales" :
                                                       "scales" + std::to_string(edge.second + 1);
                    // Absent optional input may be absent from the name map too.
                    edge.first->GetInputIndexByName(scale_name.c_str(), port);
                    auto scale = port >= 0 ? edge.first->GetInDataNodesAndPortIndexs(port).first : nullptr;
                    EXPECT_EQ(scale ? Name(*scale) : "", scale_sources[i]);
                }
                if (i == config.branches) {
                    EXPECT_EQ(Type(*edge.first), expected);
                    EXPECT_EQ(edge.second, expected == "QuantizeAddLayerNorm" ? 1 : 2);
                }
                if (i > config.branches) {
                    auto y_edge = node.GetInDataNodesAndPortIndexs(i - config.branches - 1);
                    EXPECT_EQ(edge.second, y_edge.second + 3);
                }
            }
        }
    }
};
TEST_F(QuantAlnFusionTest, DynamicVariants)
{
    for (bool reg : {true}) {
        SetPlatform(reg);
        for (int branches : {1, 2}) {
            for (bool bias : {false, true}) {
                for (bool mul : {false, true}) {
                    for (int mask = 0; mask < (1 << branches); ++mask) {
                        Config c;
                        c.branches = branches;
                        c.bias = bias;
                        c.mul = mul;
                        c.smooth_mask = mul ? 0 : mask;
                        c.swap_mask = mul ? mask : 0;
                        SCOPED_TRACE(std::to_string(reg) + "/" + std::to_string(branches) + "/" + std::to_string(bias) +
                                     "/" + std::to_string(mul) + "/" + std::to_string(mask));
                        Check(c, "AddLayerNormQuant");
                    }
                }
            }
        }
    }
}
TEST_F(QuantAlnFusionTest, StaticRegBaseVariants)
{
    for (const char* type : {"Quantize", "AscendQuantV2"}) {
        for (int branches : {1, 2}) {
            for (bool bias : {false, true}) {
                for (bool offset : {false, true}) {
                    Config c;
                    c.type = type;
                    c.branches = branches;
                    c.bias = bias;
                    c.offset = offset;
                    SCOPED_TRACE(std::string(type) + "/" + std::to_string(branches) + "/" + std::to_string(bias) + "/" +
                                 std::to_string(offset));
                    Check(c, "AddLayerNormQuant");
                }
            }
        }
    }
}
// Match legacy acceptance only; this does not assert kernel support or numerical equivalence.
TEST_F(QuantAlnFusionTest, DynamicGammaRankDoesNotRejectLegacyMatch)
{
    for (bool mul : {false, true}) {
        for (int branches : {1, 2}) {
            Config config;
            config.bias = false;
            config.mul = mul;
            config.branches = branches;
            config.smooth_mask = mul ? 0 : (1 << branches) - 1;
            auto graph = Build(config);
            ASSERT_NE(graph, nullptr);
            for (auto node : graph->GetAllNodes()) {
                if (Type(node) != "AddLayerNorm") {
                    continue;
                }
                for (int port : {2, 3}) {
                    TensorDesc desc(ge::Shape({1, config.width}), ge::FORMAT_ND, config.dtype);
                    auto source = node.GetInDataNodesAndPortIndexs(port);
                    ASSERT_NE(source.first, nullptr);
                    ASSERT_EQ(source.first->UpdateOutputDesc(source.second, desc), ge::GRAPH_SUCCESS);
                    ASSERT_EQ(node.UpdateInputDesc(port, desc), ge::GRAPH_SUCCESS);
                }
            }
            ge::CustomPassContext context;
            context.SetPassName("AddLayerNormDynamicQuantFusionPass");
            AddLayerNormDynamicQuantFusionPass pass;
            ASSERT_EQ(pass.Run(graph, context), ge::SUCCESS);
            EXPECT_EQ(Count(graph, "AddLayerNorm"), 0);
            EXPECT_EQ(Count(graph, "DynamicQuant"), 0);
            EXPECT_EQ(Count(graph, "AddLayerNormQuant"), 1);
        }
    }
}
TEST_F(QuantAlnFusionTest, SharedScalesAndSupportedDtypes)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        Config c;
        c.dtype = dtype;
        c.branches = 2;
        c.smooth_mask = 3;
        Check(c, "AddLayerNormQuant");
        c.type = "AscendQuantV2";
        c.shared_scale = true;
        c.scale_dtype = dtype;
        Check(c, "AddLayerNormQuant");
    }
}
TEST_F(QuantAlnFusionTest, RejectUsedIntermediateOutputs)
{
    for (const char* type : {"DynamicQuant", "Quantize", "AscendQuantV2"}) {
        Config c;
        c.type = type;
        c.mean = true;
        Check(c, "");
        c.mean = false;
        c.side_y = true;
        Check(c, "");
    }
    Config c;
    c.mul = true;
    c.side_mul = true;
    Check(c, "");
}
TEST_F(QuantAlnFusionTest, RejectSecondMulWithExtraSmooth)
{
    Config c;
    c.branches = 2;
    c.mul = true;
    c.double_smooth = true;
    Check(c, "");
}
// Legacy dynamic fusion does not reject INT4; target compilation is a separate concern.
TEST_F(QuantAlnFusionTest, DynamicOutputTypeDoesNotRejectLegacyMatch)
{
    for (bool mul : {false, true}) {
        for (int branches : {1, 2}) {
            Config c;
            c.mul = mul;
            c.branches = branches;
            c.smooth_mask = mul ? 0 : (1 << branches) - 1;
            c.dst = ge::DT_INT4;
            Check(c, "AddLayerNormQuant");
        }
    }
}
} // namespace
} // namespace ops
