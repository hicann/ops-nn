/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CONV_FUSION_PASS_FRAMEWORK_H
#define TEST_CONV_FUSION_PASS_FRAMEWORK_H

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "es_nn_ops.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/graph_rewriter.h"
#include "mmpa/mmpa_api.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "register/register_custom_pass.h"

namespace test_conv_fusion_framework {
using namespace ge;
using namespace fe;
using namespace es;

// ============================================================================
// 工具类：张量描述
// ============================================================================

struct TensorInfo {
    TensorInfo() = default;

    TensorInfo(DataType dataType, Format dataFormat, const std::vector<int64_t>& tensorShape, const std::string& tensorName = std::string())
        : dtype(dataType), format(dataFormat), shape(tensorShape), name(tensorName) {
        tensorDesc.SetDataType(dataType);
        tensorDesc.SetFormat(dataFormat);
    }

    TensorInfo& SetDtype(DataType dataType) { dtype = dataType; return *this; }
    TensorInfo& SetFormat(Format dataFormat) { format = dataFormat; return *this; }
    TensorInfo& SetShape(const std::vector<int64_t>& tensorShape) { shape = tensorShape; return *this; }
    TensorInfo& SetName(const std::string& tensorName) { name = tensorName; return *this; }
    TensorInfo& SetOptional(bool optional) { isOptional = optional; return *this; }
    TensorInfo& SetEnabled(bool isEnabled) { enabled = isEnabled; return *this; }

    DataType dtype = DT_FLOAT16;
    Format format = FORMAT_NCHW;
    std::vector<int64_t> shape = {};
    std::string name = "";
    bool isOptional = false;
    bool enabled = false;
    TensorDesc tensorDesc;
};

// ============================================================================
// SOC配置
// ============================================================================

struct SocConfig {
    SocConfig() = default;
    SocConfig(const std::string& shortSoc, const std::string& soc) : shortSocVersion(shortSoc), socVersion(soc) {}

    static SocConfig Ascend950() { return SocConfig("Ascend950", "Ascend950PR_9589"); }
    static SocConfig MC62CM12A() { return SocConfig("MC62CM12A", "Ascend950PR_9589"); }

    void Apply() const {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.str_info.short_soc_version = shortSocVersion;
        optiCompilationInfo.soc_version = socVersion;
        PlatformInfoManager::Instance().platform_info_map_[shortSocVersion] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

        PlatformInfoManager::Instance().InitializePlatformInfo();
        PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion(socVersion);
    }

    std::string shortSocVersion = "Ascend950";
    std::string socVersion = "Ascend950PR_9589";
};

// ============================================================================
// 基础节点配置（Fluent Interface）
// ============================================================================

struct NodeConfig {
    NodeConfig(const std::string& nodeName = "") : name(nodeName) {}

    NodeConfig& SetName(const std::string& nodeName) {name = nodeName; return *this; }

    NodeConfig& AddInput(const TensorInfo& info) {
        inputs.push_back(info);
        return *this;
    }

    NodeConfig& AddInput(DataType dataType, Format dataFormat, const std::vector<int64_t>& tensorShape,
                        const std::string& inputName = std::string(), bool optional = false) {
        inputs.emplace_back(dataType, dataFormat, tensorShape, inputName);
        inputs.back().isOptional = optional;
        return *this;
    }

    NodeConfig& AddOutput(const TensorInfo& info) {
        outputs.push_back(info);
        return *this;
    }

    NodeConfig& AddOutput(DataType dataType, Format dataFormat, const std::vector<int64_t>& tensorShape,
                         const std::string& outputName = std::string()) {
        outputs.emplace_back(dataType, dataFormat, tensorShape, outputName);
        return *this;
    }

    NodeConfig& SetAttr(const std::string& attrKey, int64_t attrValue) {
        intAttrs[attrKey] = attrValue;
        return *this;
    }

    NodeConfig& SetAttr(const std::string& attrKey, float attrValue) {
        floatAttrs[attrKey] = attrValue;
        return *this;
    }

    NodeConfig& SetAttr(const std::string& attrKey, bool attrValue) {
        boolAttrs[attrKey] = attrValue;
        return *this;
    }

    NodeConfig& SetAttr(const std::string& attrKey, const std::string& attrValue) {
        strAttrs[attrKey] = attrValue;
        return *this;
    }

    NodeConfig& SetAttr(const std::string& attrKey, const std::vector<int64_t>& attrValue) {
        listIntAttrs[attrKey] = attrValue;
        return *this;
    }

    NodeConfig& SetAttr(const std::string& attrKey, const std::vector<std::string>& attrValue) {
        listStrAttrs[attrKey] = attrValue;
        return *this;
    }

    std::string name;
    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;
    std::map<std::string, int64_t> intAttrs;
    std::map<std::string, float> floatAttrs;
    std::map<std::string, bool> boolAttrs;
    std::map<std::string, std::string> strAttrs;
    std::map<std::string, std::vector<int64_t>> listIntAttrs;
    std::map<std::string, std::vector<std::string>> listStrAttrs;
};

// ============================================================================
// 各算子特定配置（继承NodeConfig，提供静态工厂方法）
// ============================================================================

struct Conv2DConfig : public NodeConfig {
    Conv2DConfig() {
        name = "Conv2D";
        SetAttr("strides", std::vector<int64_t>{1, 1, 1, 1});
        SetAttr("pads", std::vector<int64_t>{1, 1, 1, 1});
        SetAttr("dilations", std::vector<int64_t>{1, 1, 1, 1});
        SetAttr("groups", int64_t(1));
        SetAttr("data_format", std::string("NHWC"));
        SetAttr("offset_x", int64_t(0));
    }

    static Conv2DConfig Basic(std::string nodeName, DataType dataType = DT_INT8,
                              DataType outDataType = DT_INT32, Format format = FORMAT_NCHW,
                              const std::vector<int64_t>& inputShape = {1, 16, 244, 244},
                              const std::vector<int64_t>& filterShape = {3, 16, 3, 3},
                              const std::vector<int64_t>& shape = {1, 3, 244, 244}) {
        Format filterFormat = format == FORMAT_NHWC ? FORMAT_HWCN : format;
        Conv2DConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, inputShape, nodeName + "_x")
            .AddInput(dataType, filterFormat, filterShape, nodeName + "_filter")
            .AddOutput(outDataType, format, shape, "y")
            .SetAttr("data_format", "NCHW");
        return config;
    }

    Conv2DConfig& WithBias(DataType dataType = DT_INT32, const std::vector<int64_t>& biasShape = {1}) {
        biasTensor = TensorInfo(dataType, FORMAT_ND, biasShape, this->name + "_bias");
        biasTensor.enabled = true;
        this->AddInput(biasTensor);

        return *this;
    }

    TensorInfo biasTensor;
};

struct Conv3DConfig : public NodeConfig {
    Conv3DConfig() {
        name = "Conv3D";
        SetAttr("strides", std::vector<int64_t>{1, 1, 1, 1, 1});
        SetAttr("pads", std::vector<int64_t>{1, 1, 1, 1, 1, 1});
        SetAttr("dilations", std::vector<int64_t>{1, 1, 1, 1, 1});
        SetAttr("groups", int64_t(1));
        SetAttr("data_format", std::string("NDHWC"));
        SetAttr("offset_x", int64_t(0));
    }

    static Conv3DConfig Basic(std::string nodeName,
                              DataType dataType = DT_INT8, Format format = FORMAT_NCDHW,
                              const std::vector<int64_t>& inputShape = {1, 16, 16, 244, 244},
                              const std::vector<int64_t>& filterShape = {3, 16, 3, 3, 3},
                              const std::vector<int64_t>& shape = {1, 3, 16, 244, 244}) {
        Format filterFormat = format == FORMAT_NDHWC ? FORMAT_DHWCN : format;
        Conv3DConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, inputShape, nodeName + "_x")
            .AddInput(dataType, filterFormat, filterShape, nodeName + "_filter")
            .AddOutput(DT_INT32, format, shape, "y");
        return config;
    }

    Conv3DConfig& WithBias(DataType dataType = DT_INT32, const std::vector<int64_t>& biasShape = {1}) {
        biasTensor = TensorInfo(dataType, FORMAT_ND, biasShape, this->name + "_bias");
        biasTensor.enabled = true;
        this->AddInput(biasTensor);

        return *this;
    }

    TensorInfo biasTensor;
};

struct AscendDequantConfig : public NodeConfig {
    AscendDequantConfig() {
        name = "AscendDequant";
        SetAttr("sqrt_mode", false);
        SetAttr("relu_flag", false);
        SetAttr("dtype", int64_t(DT_FLOAT));
    }

    static AscendDequantConfig Basic(std::string nodeName,
                                     DataType dataType = DT_FLOAT16, Format format = FORMAT_NCHW,
                                     const std::vector<int64_t>& shape = {1, 3, 244, 244},
                                     const std::vector<int64_t>& scaleShape = {1}) {
        AscendDequantConfig config;
        config.SetName(nodeName)
            .AddInput(DT_INT32, format, shape, nodeName + "_x")
            .AddInput(DT_UINT64, FORMAT_ND, scaleShape, nodeName + "_deq_scale")
            .AddOutput(dataType, format, shape, "y");
        return config;
    }
};

struct AscendQuantConfig : public NodeConfig {
    AscendQuantConfig() {
        name = "AscendQuant";
        SetAttr("sqrt_mode", false);
        SetAttr("round_mode", std::string("Round"));
        SetAttr("dst_type", int64_t(DT_INT8));
    }

    static AscendQuantConfig Basic(std::string nodeName,
                                   DataType dataType = DT_FLOAT16, Format format = FORMAT_NCHW,
                                   const std::vector<int64_t>& shape = {1, 3, 244, 244}) {
        AscendQuantConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, shape, nodeName + "_x")
            .AddOutput(DT_INT8, format, shape, nodeName + "_y")
            .SetAttr("scale", 1.0f)
            .SetAttr("offset", 0.0f);
        return config;
    }
};

struct AscendRequantConfig : public NodeConfig {
    AscendRequantConfig() {
        name = "AscendRequant";
        SetAttr("relu_flag", false);
    }

    static AscendRequantConfig Basic(std::string nodeName, Format format = FORMAT_NCHW,
                                     const std::vector<int64_t>& shape = {1, 3, 244, 244}) {
        AscendRequantConfig config;
        config.SetName(nodeName)
            .AddInput(DT_INT32, format, shape, nodeName + "_x")
            .AddInput(DT_UINT64, FORMAT_ND, {1}, nodeName + "_req_scale")
            .AddOutput(DT_INT8, format, shape, "y");
        return config;
    }
};

struct ReluConfig : public NodeConfig {
    ReluConfig() {
        name = "Relu";
    }

    static ReluConfig Basic(std::string nodeName,
                            DataType dataType = DT_INT32, Format format = FORMAT_NCHW,
                            const std::vector<int64_t>& shape = {1, 3, 244, 244}) {
        ReluConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, shape, nodeName + "_x")
            .AddOutput(dataType, format, shape, nodeName + "_y");
        return config;
    }
};

struct LeakyReluConfig : public NodeConfig {
    LeakyReluConfig() {
        name = "LeakyRelu";
        SetAttr("negative_slope", 0.0f);
    }

    static LeakyReluConfig Basic(std::string nodeName,
                                 DataType dataType = DT_FLOAT16, Format format = FORMAT_NCHW,
                                 const std::vector<int64_t>& shape = {1, 3, 244, 244}) {
        LeakyReluConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, shape, nodeName + "_x")
            .AddOutput(dataType, format, shape, nodeName + "_y")
            .SetAttr("negative_slope", 0.0f);
        return config;
    }
};

struct PostCubeConfig : public NodeConfig {
    PostCubeConfig() {
        name = "FixPipe";
    }

    static PostCubeConfig Basic(std::string nodeName,
                               DataType dataType = DT_FLOAT16, Format format = FORMAT_NCHW,
                               const std::vector<int64_t>& shape = {1, 3, 244, 244}) {
        PostCubeConfig config;
        config.SetName(nodeName)
            .AddInput(DT_INT32, format, shape, nodeName + "_x1")
            .AddOutput(dataType, format, shape, nodeName + "_output")
            .SetAttr("fusion_op_list", std::vector<std::string>{"AscendDequant"});

        return config;
    }

    PostCubeConfig& WithScale0(DataType dataType = DT_UINT64, const std::vector<int64_t>& shape = {1}) {
        quantScale0Tensor = TensorInfo(dataType, FORMAT_ND, shape, this->name + "_quant_scale_0");
        quantScale0Tensor.enabled = true;
        this->AddInput(quantScale0Tensor);
        return *this;
    }

    PostCubeConfig& WithScale1(DataType dataType = DT_UINT64, const std::vector<int64_t>& shape = {1}) {
        quantScale1Tensor = TensorInfo(dataType, FORMAT_ND, shape, this->name + "_quant_scale_1");
        quantScale1Tensor.enabled = true;
        this->AddInput(quantScale1Tensor);
        return *this;
    }

    PostCubeConfig& WithRelu0(DataType dataType = DT_FLOAT, const std::vector<int64_t>& shape = {1}) {
        reluWeight0Tensor = TensorInfo(dataType, FORMAT_ND, shape, this->name + "_relu_weight_0");
        reluWeight0Tensor.enabled = true;
        this->AddInput(reluWeight0Tensor);
        return *this;
    }

    PostCubeConfig& WithRelu1(DataType dataType = DT_FLOAT, const std::vector<int64_t>& shape = {1}) {
        reluWeight1Tensor = TensorInfo(dataType, FORMAT_ND, shape, this->name + "_relu_weight_1");
        reluWeight1Tensor.enabled = true;
        this->AddInput(reluWeight1Tensor);
        return *this;
    }

    TensorInfo quantScale0Tensor;
    TensorInfo reluWeight0Tensor;
    TensorInfo quantScale1Tensor;
    TensorInfo reluWeight1Tensor;
};

// ============================================================================
// 待处理节点信息
// ============================================================================

struct PendingNodeInfo {
    std::string nodeName;
    std::string opType;
    std::vector<CompliantNodeBuilder::IrInputDef> inputDefs;
    std::vector<CompliantNodeBuilder::IrOutputDef> outputDefs;
    std::map<int32_t, TensorDesc> inputDesc;
    std::map<int32_t, TensorDesc> outputDesc;
    std::map<std::string, int64_t> intAttrs;
    std::map<std::string, float> floatAttrs;
    std::map<std::string, bool> boolAttrs;
    std::map<std::string, std::string> strAttrs;
    std::map<std::string, std::vector<int64_t>> listIntAttrs;
    std::map<std::string, std::vector<std::string>> listStrAttrs;
};

struct PendingConnectionInfo {
    std::string fromNodeName;
    int fromOutputIndex;
    std::string toNodeName;
    int toInputIndex;
    es::EsTensorHolder graphInput;
};

// ============================================================================
// 测试图构建器（Builder模式）
// ============================================================================

class TestGraph {
public:
    TestGraph(const std::string& name = "test_graph") : graphName(name), graphBuilder(name.c_str()) {}

    TestGraph& SetSoc(const SocConfig& socConfig) {
        socConfig.Apply();
        return *this;
    }

    TestGraph& SetSocAscend950() {
        SetSoc(SocConfig::Ascend950());
        return *this;
    }

    TestGraph& SetSocMC62CM12A() {
        SetSoc(SocConfig::MC62CM12A());
        
        return *this;
    }

    es::EsTensorHolder CreateGraphInput(const TensorInfo& tensorInfo) {
        auto input = graphBuilder.CreateInput(inputIndex++, tensorInfo.name.c_str(), tensorInfo.dtype,
            tensorInfo.format, tensorInfo.shape);
        graphInputs.push_back(input);

        return input;
    }

    TestGraph& AddConv2D(const Conv2DConfig& conv2dConfig,
        bool autoFmap = true, bool autoFilter = true, bool autoBias = true) {
        std::vector<es::EsTensorHolder> inputTensors;
        PendingNodeInfo nodeInfo;

        if (autoFmap) {
            auto fmap = CreateGraphInput(conv2dConfig.inputs[0]);
            pendingConnections.push_back({"", 0, conv2dConfig.name, 0, fmap});
        }
        nodeInfo.inputDesc[0] = conv2dConfig.inputs[0].tensorDesc;
        if (autoFilter) {
            auto filter = CreateGraphInput(conv2dConfig.inputs[1]);
            pendingConnections.push_back({"", 0, conv2dConfig.name, 1, filter});
        }
        nodeInfo.inputDesc[1] = conv2dConfig.inputs[1].tensorDesc;
        if (autoBias && conv2dConfig.inputs.size() >= 2) { // bias idx 2
            if (conv2dConfig.biasTensor.enabled) {
                auto bias = CreateGraphInput(conv2dConfig.biasTensor);
                pendingConnections.push_back({"", 0, conv2dConfig.name, 2, bias}); // bias idx 2
                nodeInfo.inputDesc[2] = conv2dConfig.inputs[2].tensorDesc; // bias idx 2
            }
        }

        nodeInfo.outputDesc[0] = conv2dConfig.outputs[0].tensorDesc;

        nodeInfo.nodeName = conv2dConfig.name;
        nodeInfo.opType = "Conv2D";
        nodeInfo.inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                              {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        nodeInfo.inputDefs.push_back({"bias", CompliantNodeBuilder::kEsIrInputOptional, ""});
        nodeInfo.inputDefs.push_back({"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""});

        nodeInfo.outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        nodeInfo.intAttrs = conv2dConfig.intAttrs;
        nodeInfo.strAttrs = conv2dConfig.strAttrs;
        nodeInfo.listIntAttrs = conv2dConfig.listIntAttrs;
        pendingNodes.push_back(nodeInfo);

        return *this;
    }

    TestGraph& AddConv3D(const Conv3DConfig& conv3dConfig,
        bool autoFmap = true, bool autoFilter = true, bool autoBias = true) {
        PendingNodeInfo nodeInfo;

        if (autoFmap) {
            auto fmap = CreateGraphInput(conv3dConfig.inputs[0]);
            pendingConnections.push_back({"", 0, conv3dConfig.name, 0, fmap});
        }
        nodeInfo.inputDesc[0] = conv3dConfig.inputs[0].tensorDesc;
        if (autoFilter) {
            auto filter = CreateGraphInput(conv3dConfig.inputs[1]);
            pendingConnections.push_back({"", 0, conv3dConfig.name, 1, filter});
        }
        nodeInfo.inputDesc[1] = conv3dConfig.inputs[1].tensorDesc;

        if (autoBias && conv3dConfig.inputs.size() >= 2) { // bias idx 2
            if (conv3dConfig.biasTensor.enabled) {
                auto bias = CreateGraphInput(conv3dConfig.biasTensor);
                pendingConnections.push_back({"", 0, conv3dConfig.name, 2, bias}); // bias idx 2
                nodeInfo.inputDesc[2] = conv3dConfig.inputs[2].tensorDesc; // bias idx 2
            }
        }

        nodeInfo.outputDesc[0] = conv3dConfig.outputs[0].tensorDesc;

        nodeInfo.nodeName = conv3dConfig.name;
        nodeInfo.opType = "Conv3D";
        nodeInfo.inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                              {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        nodeInfo.inputDefs.push_back({"bias", CompliantNodeBuilder::kEsIrInputOptional, ""});
        nodeInfo.inputDefs.push_back({"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""});

        nodeInfo.outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        nodeInfo.intAttrs = conv3dConfig.intAttrs;
        nodeInfo.strAttrs = conv3dConfig.strAttrs;
        nodeInfo.listIntAttrs = conv3dConfig.listIntAttrs;
        pendingNodes.push_back(nodeInfo);

        return *this;
    }

    TestGraph& AddAscendDequant(const AscendDequantConfig& ascendDequantConfig,
        bool autoX = false, bool autoScale = true) {
        PendingNodeInfo nodeInfo;
        if (autoX) {
            auto x = CreateGraphInput(ascendDequantConfig.inputs[0]);
            pendingConnections.push_back({"", 0, ascendDequantConfig.name, 0, x});
        }
        nodeInfo.inputDesc[0] = ascendDequantConfig.inputs[0].tensorDesc;
        if (autoScale) {
            auto scale = CreateGraphInput(ascendDequantConfig.inputs[1]);
            pendingConnections.push_back({"", 0, ascendDequantConfig.name, 1, scale});
        }
        nodeInfo.inputDesc[1] = ascendDequantConfig.inputs[1].tensorDesc;
        nodeInfo.outputDesc[0] = ascendDequantConfig.outputs[0].tensorDesc;

        nodeInfo.nodeName = ascendDequantConfig.name;
        nodeInfo.opType = "AscendDequant";
        nodeInfo.inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                              {"deq_scale", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        nodeInfo.outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        nodeInfo.boolAttrs = ascendDequantConfig.boolAttrs;
        nodeInfo.intAttrs = ascendDequantConfig.intAttrs;
        pendingNodes.push_back(nodeInfo);

        return *this;
    }

    TestGraph& AddAscendRequant(const AscendRequantConfig& ascendRequantConfig,
        bool autoX = false, bool autoScale = true) {
        PendingNodeInfo nodeInfo;
        if (autoX) {
            auto x = CreateGraphInput(ascendRequantConfig.inputs[0]);
            pendingConnections.push_back({"", 0, ascendRequantConfig.name, 0, x});
        }
        nodeInfo.inputDesc[0] = ascendRequantConfig.inputs[0].tensorDesc;
        if (autoScale) {
            auto scale = CreateGraphInput(ascendRequantConfig.inputs[1]);
            pendingConnections.push_back({"", 0, ascendRequantConfig.name, 1, scale});
        }
        nodeInfo.inputDesc[1] = ascendRequantConfig.inputs[1].tensorDesc;
        nodeInfo.outputDesc[0] = ascendRequantConfig.outputs[0].tensorDesc;

        nodeInfo.nodeName = ascendRequantConfig.name;
        nodeInfo.opType = "AscendRequant";
        nodeInfo.inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                              {"req_scale", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        nodeInfo.outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        nodeInfo.boolAttrs = ascendRequantConfig.boolAttrs;
        pendingNodes.push_back(nodeInfo);

        return *this;
    }

    TestGraph& AddAscendQuant(const AscendQuantConfig& ascendQuantConfig, bool autoInputs = false) {
        PendingNodeInfo nodeInfo;
        if (autoInputs) {
            auto x = CreateGraphInput(ascendQuantConfig.inputs[0]);
            pendingConnections.push_back({"", 0, ascendQuantConfig.name, 0, x});
        }
        nodeInfo.inputDesc[0] = ascendQuantConfig.inputs[0].tensorDesc;
        nodeInfo.outputDesc[0] = ascendQuantConfig.outputs[0].tensorDesc;
        nodeInfo.nodeName = ascendQuantConfig.name;
        nodeInfo.opType = "AscendQuant";
        nodeInfo.inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        nodeInfo.outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        nodeInfo.floatAttrs = ascendQuantConfig.floatAttrs;
        nodeInfo.intAttrs = ascendQuantConfig.intAttrs;
        nodeInfo.boolAttrs = ascendQuantConfig.boolAttrs;
        nodeInfo.strAttrs = ascendQuantConfig.strAttrs;
        pendingNodes.push_back(nodeInfo);

        return *this;
    }

    TestGraph& AddRelu(const ReluConfig& reluConfig, bool autoInputs = false) {
        PendingNodeInfo nodeInfo;
        if (autoInputs) {
            auto x = CreateGraphInput(reluConfig.inputs[0]);
            pendingConnections.push_back({"", 0, reluConfig.name, 0, x});
        }
        nodeInfo.inputDesc[0] = reluConfig.inputs[0].tensorDesc;
        nodeInfo.outputDesc[0] = reluConfig.outputs[0].tensorDesc;
        nodeInfo.nodeName = reluConfig.name;
        nodeInfo.opType = "Relu";
        nodeInfo.inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        nodeInfo.outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        pendingNodes.push_back(nodeInfo);

        return *this;
    }

    TestGraph& AddLeakyRelu(const LeakyReluConfig& leakyReluConfig, bool autoInputs = false) {
        PendingNodeInfo nodeInfo;
        if (autoInputs) {
            auto x = CreateGraphInput(leakyReluConfig.inputs[0]);
            pendingConnections.push_back({"", 0, leakyReluConfig.name, 0, x});
        }
        nodeInfo.inputDesc[0] = leakyReluConfig.inputs[0].tensorDesc;
        nodeInfo.outputDesc[0] = leakyReluConfig.outputs[0].tensorDesc;
        nodeInfo.nodeName = leakyReluConfig.name;
        nodeInfo.opType = "LeakyRelu";
        nodeInfo.inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        nodeInfo.outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        nodeInfo.floatAttrs = leakyReluConfig.floatAttrs;
        pendingNodes.push_back(nodeInfo);

        return *this;
    }

    TestGraph& AddPostCube(const PostCubeConfig& postCubeConfig, bool autoInputs = true) {
        PendingNodeInfo nodeInfo;

        if (autoInputs) {
            if (postCubeConfig.quantScale0Tensor.enabled) {
                auto scale0 = CreateGraphInput(postCubeConfig.quantScale0Tensor);
                pendingConnections.push_back({"", 0, postCubeConfig.name, 2, scale0});
                nodeInfo.inputDesc[2] = postCubeConfig.quantScale0Tensor.tensorDesc; // quantScale0 idx 2
            }
            if (postCubeConfig.reluWeight0Tensor.enabled) {
                auto relu0 = CreateGraphInput(postCubeConfig.reluWeight0Tensor);
                pendingConnections.push_back({"", 0, postCubeConfig.name, 3, relu0});
                nodeInfo.inputDesc[3] = postCubeConfig.reluWeight0Tensor.tensorDesc; // relu0 idx 3
            }
            if (postCubeConfig.quantScale1Tensor.enabled) {
                auto scale1 = CreateGraphInput(postCubeConfig.quantScale0Tensor);
                pendingConnections.push_back({"", 0, postCubeConfig.name, 5, scale1});
                nodeInfo.inputDesc[5] = postCubeConfig.quantScale1Tensor.tensorDesc; // quantScale1 idx 5
            }
            if (postCubeConfig.reluWeight1Tensor.enabled) {
                auto relu1 = CreateGraphInput(postCubeConfig.reluWeight1Tensor);
                pendingConnections.push_back({"", 0, postCubeConfig.name, 6, relu1});
                nodeInfo.inputDesc[6] = postCubeConfig.reluWeight1Tensor.tensorDesc; // relu1 idx 6
            }
        }
        nodeInfo.inputDesc[0] = postCubeConfig.inputs[0].tensorDesc;
        nodeInfo.outputDesc[0] = postCubeConfig.outputs[0].tensorDesc;

        nodeInfo.nodeName = postCubeConfig.name;
        nodeInfo.opType = "FixPipe";
        nodeInfo.inputDefs = {{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        nodeInfo.inputDefs = {{"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        nodeInfo.inputDefs.push_back({"quant_scale_0", CompliantNodeBuilder::kEsIrInputOptional, ""});
        nodeInfo.inputDefs.push_back({"relu_weight_0", CompliantNodeBuilder::kEsIrInputOptional, ""});
        nodeInfo.inputDefs.push_back({"clip_value_0", CompliantNodeBuilder::kEsIrInputOptional, ""});
        nodeInfo.inputDefs.push_back({"quant_scale_1", CompliantNodeBuilder::kEsIrInputOptional, ""});
        nodeInfo.inputDefs.push_back({"relu_weight_1", CompliantNodeBuilder::kEsIrInputOptional, ""});
        nodeInfo.inputDefs.push_back({"clip_value_1", CompliantNodeBuilder::kEsIrInputOptional, ""});
        nodeInfo.inputDefs.push_back({"anti_quant_scale", CompliantNodeBuilder::kEsIrInputOptional, ""});
        nodeInfo.inputDefs.push_back({"anti_quant_offset", CompliantNodeBuilder::kEsIrInputOptional, ""});
        nodeInfo.outputDefs = {{"output", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        nodeInfo.strAttrs = postCubeConfig.strAttrs;
        nodeInfo.listStrAttrs = postCubeConfig.listStrAttrs;
        pendingNodes.push_back(nodeInfo);

        return *this;
    }

    TestGraph& Connect(const std::string& fromNodeName, int fromOutputIndex,
                        const std::string& toNodeName, int toInputIndex) {
        if (graphBuilt) {
            auto fromIt = nodeMap.find(fromNodeName);
            auto toIt = nodeMap.find(toNodeName);
            if (fromIt != nodeMap.end() && toIt != nodeMap.end()) {
                AddEdgeAndUpdatePeerDesc(*geGraph, fromIt->second, fromOutputIndex,
                                         toIt->second, toInputIndex);
            }
        } else {
            pendingConnections.push_back({fromNodeName, fromOutputIndex, toNodeName, toInputIndex, {}});
        }
        return *this;
    }

    TestGraph& ConnectInput(const es::EsTensorHolder& graphInput, int inputIndex,
                            const std::string& toNodeName, int toInputIndex) {
        auto it = nodeMap.find(toNodeName);
        if (it != nodeMap.end()) {
            AddEdgeAndUpdatePeerDesc(*geGraph, *graphInput.GetProducer(), inputIndex,
                                     it->second, toInputIndex);
        }
        return *this;
    }

    TestGraph& SetOutput(const std::string& nodeName, int outputIndex = 0) {
        outputs.emplace_back(std::make_pair(nodeName, outputIndex));
        return *this;
    }

    std::shared_ptr<Graph> Build() {
        if (!graphBuilt) {
            geGraph = graphBuilder.BuildAndReset();

            for (const auto& nodeInfo : pendingNodes) {
                auto node = CompliantNodeBuilder(geGraph.get())
                    .OpType(nodeInfo.opType.c_str())
                    .Name(nodeInfo.nodeName.c_str())
                    .IrDefInputs(nodeInfo.inputDefs)
                    .IrDefOutputs(nodeInfo.outputDefs)
                    .IrDefAttrs({})
                    .Build();

                SetNodeAttr(node, nodeInfo);
                UpdateDesc(node, nodeInfo);
                nodeMap[nodeInfo.nodeName] = node;
            }
            AddNodeEdge();
            SetGraphOutput();
            graphBuilt = true;
        }

        return geGraph;
    }

    GNode GetNode(const std::string& nodeName) {
        auto it = nodeMap.find(nodeName);
        return (it != nodeMap.end()) ? it->second : GNode();
    }

    const std::map<std::string, GNode>& GetAllNodes() const {
        return nodeMap;
    }

    void UpdateNodeInputDesc(const std::string& nodeName, int32_t index, DataType dtype, Format format) {
        TensorDesc tensorDesc;
        tensorDesc.SetDataType(dtype);
        tensorDesc.SetFormat(format);
        auto it = nodeMap.find(nodeName);

        it->second.UpdateInputDesc(index, tensorDesc);
    }

    void UpdateNodeOutputDesc(const std::string& nodeName, int32_t index, DataType dtype, Format format) {
        TensorDesc tensorDesc;
        tensorDesc.SetDataType(dtype);
        tensorDesc.SetFormat(format);
        auto it = nodeMap.find(nodeName);

        it->second.UpdateOutputDesc(index, tensorDesc);
    }

private:
    void SetNodeAttr(GNode &node, PendingNodeInfo nodeInfo) {
        for (const auto& attr : nodeInfo.intAttrs) {
            int64_t attrValue = attr.second;
            node.SetAttr(AscendString(attr.first.c_str()), attrValue);
        }
        for (const auto& attr : nodeInfo.strAttrs) {
            AscendString strAttr = AscendString(attr.second.c_str());
            node.SetAttr(AscendString(attr.first.c_str()), strAttr);
        }
        for (const auto& attr : nodeInfo.listIntAttrs) {
            std::vector<int64_t> attrValue = attr.second;
            node.SetAttr(AscendString(attr.first.c_str()), attrValue);
        }
        for (const auto& attr : nodeInfo.floatAttrs) {
            float attrValue = attr.second;
            node.SetAttr(AscendString(attr.first.c_str()), attrValue);
        }
        for (const auto& attr : nodeInfo.boolAttrs) {
            bool attrValue = attr.second;
            node.SetAttr(AscendString(attr.first.c_str()), attrValue);
        }
        for (const auto& attr : nodeInfo.listStrAttrs) {
            std::vector<AscendString> attrValue;
            for (const auto& str : attr.second) {
                attrValue.push_back(AscendString(str.c_str()));
            }
            node.SetAttr(AscendString(attr.first.c_str()), attrValue);
        }
    }

    void UpdateDesc(GNode &node, PendingNodeInfo nodeInfo) {
        for (auto desc : nodeInfo.inputDesc) {
            node.UpdateInputDesc(desc.first, desc.second);
        }

        for (auto desc : nodeInfo.outputDesc) {
            node.UpdateOutputDesc(desc.first, desc.second);
        }
    }

    void AddNodeEdge() {
        for (const auto& connInfo : pendingConnections) {
            if (connInfo.fromNodeName.empty()) {
                auto it = nodeMap.find(connInfo.toNodeName);
                if (it != nodeMap.end()) {
                    AddEdgeAndUpdatePeerDesc(*geGraph, *connInfo.graphInput.GetProducer(),
                                             connInfo.fromOutputIndex, it->second, connInfo.toInputIndex);
                }
            } else {
                auto fromIt = nodeMap.find(connInfo.fromNodeName);
                auto toIt = nodeMap.find(connInfo.toNodeName);
                if (fromIt != nodeMap.end() && toIt != nodeMap.end()) {
                    AddEdgeAndUpdatePeerDesc(*geGraph, fromIt->second, connInfo.fromOutputIndex,
                                             toIt->second, connInfo.toInputIndex);
                }
            }
        }
    }

    void SetGraphOutput() {
        std::vector<std::pair<GNode, int32_t>> outputNodes = {};
        for (auto out : outputs) {
            auto iter = nodeMap.find(out.first);
            if (iter == nodeMap.end()) continue;
            outputNodes.emplace_back(std::make_pair(iter->second, out.second));
        }
        geGraph->SetOutputs(outputNodes);
    }

private:
    std::string graphName;
    EsGraphBuilder graphBuilder;
    std::shared_ptr<Graph> geGraph = nullptr;
    std::vector<es::EsTensorHolder> graphInputs = {};
    std::map<std::string, GNode> nodeMap;
    int inputIndex = 0;
    std::vector<PendingNodeInfo> pendingNodes = {};
    std::vector<PendingConnectionInfo> pendingConnections = {};
    bool graphBuilt = false;
    std::vector<std::pair<std::string, int32_t>> outputs = {};
};

// ============================================================================
// 图验证工具
// ============================================================================

class GraphChecker {
public:
    static bool HasNode(std::shared_ptr<Graph>& graph, const std::string& nodeType) {
        for (auto node : graph->GetAllNodes()) {
            AscendString curType;
            node.GetType(curType);
            if (curType.GetString() == nodeType) {
                return true;
            }
        }
        return false;
    }

    static int CountNodes(std::shared_ptr<Graph>& graph, const std::string& nodeType) {
        int count = 0;
        for (auto node : graph->GetAllNodes()) {
            AscendString curType;
            node.GetType(curType);
            if (curType.GetString() == nodeType) {
                count++;
            }
        }
        return count;
    }

    static void Print(std::shared_ptr<Graph>& graph) {
        std::cout << "Graph: " << graph->GetName() << std::endl;
        for (auto node : graph->GetAllNodes()) {
            AscendString nodeName;
            node.GetName(nodeName);
            AscendString nodeType;
            node.GetType(nodeType);
            std::cout << "  " << nodeName.GetString() << " (" << nodeType.GetString() << ")" << std::endl;
        }
    }
};

} // namespace test_conv_fusion_framework

#endif // TEST_CONV_FUSION_PASS_FRAMEWORK_H