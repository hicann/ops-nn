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

    TensorInfo(DataType dataType, Format dataFormat, const std::vector<int64_t>& tensorShape,
               const std::string& tensorName = std::string())
        : dtype(dataType), format(dataFormat), shape(tensorShape), name(tensorName)
    {
        tensorDesc.SetDataType(dataType);
        tensorDesc.SetFormat(dataFormat);
    }

    TensorInfo& SetDtype(DataType dataType)
    {
        dtype = dataType;
        return *this;
    }
    TensorInfo& SetFormat(Format dataFormat)
    {
        format = dataFormat;
        return *this;
    }
    TensorInfo& SetShape(const std::vector<int64_t>& tensorShape)
    {
        shape = tensorShape;
        return *this;
    }
    TensorInfo& SetName(const std::string& tensorName)
    {
        name = tensorName;
        return *this;
    }
    TensorInfo& SetOptional(bool optional)
    {
        isOptional = optional;
        return *this;
    }
    TensorInfo& SetEnabled(bool isEnabled)
    {
        enabled = isEnabled;
        return *this;
    }
    TensorInfo& SetSlot(int32_t portSlot)
    {
        slot = portSlot;
        return *this;
    }
    TensorInfo& SetDesc(const TensorDesc& desc)
    {
        tensorDesc = desc;
        return *this;
    }

    DataType dtype = DT_FLOAT16;
    Format format = FORMAT_NCHW;
    std::vector<int64_t> shape = {};
    std::string name = "";
    bool isOptional = false;
    bool enabled = false;
    int32_t slot = -1;
    TensorDesc tensorDesc;
};

// ============================================================================
// SOC配置
// ============================================================================

struct SocConfig {
    SocConfig() = default;
    SocConfig(const std::string& shortSoc, const std::string& soc) : shortSocVersion(shortSoc), socVersion(soc) {}

    static SocConfig Ascend950() { return SocConfig("Ascend950", "Ascend950PR_9589"); }
    static SocConfig MC62() { return SocConfig("MC62", "Ascend950PR_9589"); }

    void Apply() const
    {
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

    NodeConfig& SetName(const std::string& nodeName)
    {
        name = nodeName;
        return *this;
    }

    NodeConfig& AddInput(const TensorInfo& info)
    {
        inputs.push_back(info);
        return *this;
    }

    NodeConfig& AddInput(DataType dataType, Format dataFormat, const std::vector<int64_t>& tensorShape,
                         const std::string& inputName = std::string(), bool optional = false)
    {
        inputs.emplace_back(dataType, dataFormat, tensorShape, inputName);
        inputs.back().isOptional = optional;
        return *this;
    }

    NodeConfig& AddOutput(const TensorInfo& info)
    {
        outputs.push_back(info);
        return *this;
    }

    NodeConfig& AddOutput(DataType dataType, Format dataFormat, const std::vector<int64_t>& tensorShape,
                          const std::string& outputName = std::string())
    {
        outputs.emplace_back(dataType, dataFormat, tensorShape, outputName);
        return *this;
    }

    NodeConfig& SetAttr(const std::string& attrKey, int64_t attrValue)
    {
        intAttrs[attrKey] = attrValue;
        return *this;
    }

    NodeConfig& SetAttr(const std::string& attrKey, float attrValue)
    {
        floatAttrs[attrKey] = attrValue;
        return *this;
    }

    NodeConfig& SetAttr(const std::string& attrKey, bool attrValue)
    {
        boolAttrs[attrKey] = attrValue;
        return *this;
    }

    NodeConfig& SetAttr(const std::string& attrKey, const std::string& attrValue)
    {
        strAttrs[attrKey] = attrValue;
        return *this;
    }

    NodeConfig& SetAttr(const std::string& attrKey, const std::vector<int64_t>& attrValue)
    {
        listIntAttrs[attrKey] = attrValue;
        return *this;
    }

    NodeConfig& SetAttr(const std::string& attrKey, const std::vector<std::string>& attrValue)
    {
        listStrAttrs[attrKey] = attrValue;
        return *this;
    }

    NodeConfig& SetAttr(const std::string& attrKey, const std::vector<float>& attrValue)
    {
        listFloatAttrs[attrKey] = attrValue;
        return *this;
    }

    std::string name;
    std::string opType;
    std::vector<CompliantNodeBuilder::IrInputDef> inputDefs;
    std::vector<CompliantNodeBuilder::IrOutputDef> outputDefs;
    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;
    std::map<std::string, int64_t> intAttrs;
    std::map<std::string, float> floatAttrs;
    std::map<std::string, bool> boolAttrs;
    std::map<std::string, std::string> strAttrs;
    std::map<std::string, std::vector<int64_t>> listIntAttrs;
    std::map<std::string, std::vector<std::string>> listStrAttrs;
    std::map<std::string, std::vector<float>> listFloatAttrs;
};

// ============================================================================
// 各算子特定配置（继承NodeConfig，提供静态工厂方法）
// ============================================================================

struct Conv2DConfig : public NodeConfig {
    Conv2DConfig()
    {
        name = "Conv2D";
        opType = "Conv2D";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""},
                     {"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("strides", std::vector<int64_t>{1, 1, 1, 1});
        SetAttr("pads", std::vector<int64_t>{1, 1, 1, 1});
        SetAttr("dilations", std::vector<int64_t>{1, 1, 1, 1});
        SetAttr("groups", int64_t(1));
        SetAttr("data_format", std::string("NHWC"));
        SetAttr("offset_x", int64_t(0));
    }

    static Conv2DConfig Basic(std::string nodeName, DataType dataType = DT_INT8, DataType outDataType = DT_INT32,
                              Format format = FORMAT_NCHW, const std::vector<int64_t>& inputShape = {1, 16, 244, 244},
                              const std::vector<int64_t>& filterShape = {3, 16, 3, 3},
                              const std::vector<int64_t>& shape = {1, 3, 244, 244})
    {
        Format filterFormat = format == FORMAT_NHWC ? FORMAT_HWCN : format;
        Conv2DConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, inputShape, nodeName + "_x")
            .AddInput(dataType, filterFormat, filterShape, nodeName + "_filter")
            .AddOutput(outDataType, format, shape, "y")
            .SetAttr("data_format", "NCHW");
        return config;
    }

    Conv2DConfig& WithBias(DataType dataType = DT_INT32, const std::vector<int64_t>& biasShape = {1})
    {
        biasTensor = TensorInfo(dataType, FORMAT_ND, biasShape, this->name + "_bias");
        biasTensor.enabled = true;
        this->AddInput(biasTensor);

        return *this;
    }

    TensorInfo biasTensor;
};

struct Conv3DConfig : public NodeConfig {
    Conv3DConfig()
    {
        name = "Conv3D";
        opType = "Conv3D";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""},
                     {"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("strides", std::vector<int64_t>{1, 1, 1, 1, 1});
        SetAttr("pads", std::vector<int64_t>{1, 1, 1, 1, 1, 1});
        SetAttr("dilations", std::vector<int64_t>{1, 1, 1, 1, 1});
        SetAttr("groups", int64_t(1));
        SetAttr("data_format", std::string("NDHWC"));
        SetAttr("offset_x", int64_t(0));
    }

    static Conv3DConfig Basic(std::string nodeName, DataType dataType = DT_INT8, Format format = FORMAT_NCDHW,
                              const std::vector<int64_t>& inputShape = {1, 16, 16, 244, 244},
                              const std::vector<int64_t>& filterShape = {3, 16, 3, 3, 3},
                              const std::vector<int64_t>& shape = {1, 3, 16, 244, 244})
    {
        Format filterFormat = format == FORMAT_NDHWC ? FORMAT_DHWCN : format;
        Conv3DConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, inputShape, nodeName + "_x")
            .AddInput(dataType, filterFormat, filterShape, nodeName + "_filter")
            .AddOutput(DT_INT32, format, shape, "y");
        return config;
    }

    Conv3DConfig& WithBias(DataType dataType = DT_INT32, const std::vector<int64_t>& biasShape = {1})
    {
        biasTensor = TensorInfo(dataType, FORMAT_ND, biasShape, this->name + "_bias");
        biasTensor.enabled = true;
        this->AddInput(biasTensor);

        return *this;
    }

    TensorInfo biasTensor;
};

struct AscendDequantConfig : public NodeConfig {
    AscendDequantConfig()
    {
        name = "AscendDequant";
        opType = "AscendDequant";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"deq_scale", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("sqrt_mode", false);
        SetAttr("relu_flag", false);
        SetAttr("dtype", int64_t(DT_FLOAT));
    }

    static AscendDequantConfig Basic(std::string nodeName, DataType dataType = DT_FLOAT16, Format format = FORMAT_NCHW,
                                     const std::vector<int64_t>& shape = {1, 3, 244, 244},
                                     const std::vector<int64_t>& scaleShape = {1})
    {
        AscendDequantConfig config;
        config.SetName(nodeName)
            .AddInput(DT_INT32, format, shape, nodeName + "_x")
            .AddInput(DT_UINT64, FORMAT_ND, scaleShape, nodeName + "_deq_scale")
            .AddOutput(dataType, format, shape, "y");
        return config;
    }
};

struct AscendQuantConfig : public NodeConfig {
    AscendQuantConfig()
    {
        name = "AscendQuant";
        opType = "AscendQuant";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("sqrt_mode", false);
        SetAttr("round_mode", std::string("Round"));
        SetAttr("dst_type", int64_t(DT_INT8));
    }

    static AscendQuantConfig Basic(std::string nodeName, DataType dataType = DT_FLOAT16, Format format = FORMAT_NCHW,
                                   const std::vector<int64_t>& shape = {1, 3, 244, 244})
    {
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
    AscendRequantConfig()
    {
        name = "AscendRequant";
        opType = "AscendRequant";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"req_scale", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("relu_flag", false);
    }

    static AscendRequantConfig Basic(std::string nodeName, Format format = FORMAT_NCHW,
                                     const std::vector<int64_t>& shape = {1, 3, 244, 244})
    {
        AscendRequantConfig config;
        config.SetName(nodeName)
            .AddInput(DT_INT32, format, shape, nodeName + "_x")
            .AddInput(DT_UINT64, FORMAT_ND, {1}, nodeName + "_req_scale")
            .AddOutput(DT_INT8, format, shape, "y");
        return config;
    }
};

struct ReluConfig : public NodeConfig {
    ReluConfig()
    {
        name = "Relu";
        opType = "Relu";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    }

    static ReluConfig Basic(std::string nodeName, DataType dataType = DT_INT32, Format format = FORMAT_NCHW,
                            const std::vector<int64_t>& shape = {1, 3, 244, 244})
    {
        ReluConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, shape, nodeName + "_x")
            .AddOutput(dataType, format, shape, nodeName + "_y");
        return config;
    }

    static ReluConfig FromTensor(const std::string& nodeName, const TensorInfo& tensorInfo)
    {
        ReluConfig config;
        config.SetName(nodeName)
            .AddInput(tensorInfo)
            .AddOutput(TensorInfo(tensorInfo.dtype, tensorInfo.format, tensorInfo.shape, nodeName + "_y")
                           .SetDesc(tensorInfo.tensorDesc));
        return config;
    }
};

struct DataConfig : public NodeConfig {
    DataConfig()
    {
        name = "Data";
        opType = "Data";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    }

    static DataConfig Basic(const std::string& nodeName, DataType dataType = DT_FLOAT, Format format = FORMAT_NCHW,
                            const std::vector<int64_t>& shape = {})
    {
        DataConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, shape, nodeName + "_src")
            .AddOutput(dataType, format, shape, nodeName + "_y");
        return config;
    }

    static DataConfig FromTensor(const std::string& nodeName, const TensorInfo& tensorInfo,
                                 const std::string& inputName = "")
    {
        DataConfig config;
        TensorInfo inputInfo = tensorInfo;
        if (!inputName.empty()) {
            inputInfo.name = inputName;
        }
        config.SetName(nodeName).AddInput(inputInfo).AddOutput(
            TensorInfo(tensorInfo.dtype, tensorInfo.format, tensorInfo.shape, nodeName + "_y")
                .SetDesc(tensorInfo.tensorDesc));
        return config;
    }
};

struct LeakyReluConfig : public NodeConfig {
    LeakyReluConfig()
    {
        name = "LeakyRelu";
        opType = "LeakyRelu";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("negative_slope", 0.0f);
    }

    static LeakyReluConfig Basic(std::string nodeName, DataType dataType = DT_FLOAT16, Format format = FORMAT_NCHW,
                                 const std::vector<int64_t>& shape = {1, 3, 244, 244})
    {
        LeakyReluConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, shape, nodeName + "_x")
            .AddOutput(dataType, format, shape, nodeName + "_y")
            .SetAttr("negative_slope", 0.0f);
        return config;
    }
};

struct PostCubeConfig : public NodeConfig {
    PostCubeConfig()
    {
        name = "FixPipe";
        opType = "FixPipe";
        inputDefs = {{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"quant_scale_0", CompliantNodeBuilder::kEsIrInputOptional, ""},
                     {"relu_weight_0", CompliantNodeBuilder::kEsIrInputOptional, ""},
                     {"clip_value_0", CompliantNodeBuilder::kEsIrInputOptional, ""},
                     {"quant_scale_1", CompliantNodeBuilder::kEsIrInputOptional, ""},
                     {"relu_weight_1", CompliantNodeBuilder::kEsIrInputOptional, ""},
                     {"clip_value_1", CompliantNodeBuilder::kEsIrInputOptional, ""},
                     {"anti_quant_scale", CompliantNodeBuilder::kEsIrInputOptional, ""},
                     {"anti_quant_offset", CompliantNodeBuilder::kEsIrInputOptional, ""}};
        outputDefs = {{"output", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    }

    static PostCubeConfig Basic(std::string nodeName, DataType dataType = DT_FLOAT16, Format format = FORMAT_NCHW,
                                const std::vector<int64_t>& shape = {1, 3, 244, 244})
    {
        PostCubeConfig config;
        config.SetName(nodeName)
            .AddInput(DT_INT32, format, shape, nodeName + "_x1")
            .AddOutput(dataType, format, shape, nodeName + "_output")
            .SetAttr("fusion_op_list", std::vector<std::string>{"AscendDequant"});

        return config;
    }

    PostCubeConfig& WithScale0(DataType dataType = DT_UINT64, const std::vector<int64_t>& shape = {1})
    {
        quantScale0Tensor = TensorInfo(dataType, FORMAT_ND, shape, this->name + "_quant_scale_0");
        quantScale0Tensor.enabled = true;
        quantScale0Tensor.slot = 2; // quant_scale_0 端口号
        this->AddInput(quantScale0Tensor);
        return *this;
    }

    PostCubeConfig& WithScale1(DataType dataType = DT_UINT64, const std::vector<int64_t>& shape = {1})
    {
        quantScale1Tensor = TensorInfo(dataType, FORMAT_ND, shape, this->name + "_quant_scale_1");
        quantScale1Tensor.enabled = true;
        quantScale1Tensor.slot = 5; // quant_scale_1 端口号
        this->AddInput(quantScale1Tensor);
        return *this;
    }

    PostCubeConfig& WithRelu0(DataType dataType = DT_FLOAT, const std::vector<int64_t>& shape = {1})
    {
        reluWeight0Tensor = TensorInfo(dataType, FORMAT_ND, shape, this->name + "_relu_weight_0");
        reluWeight0Tensor.enabled = true;
        reluWeight0Tensor.slot = 3; // relu_weight_0 端口号
        this->AddInput(reluWeight0Tensor);
        return *this;
    }

    PostCubeConfig& WithRelu1(DataType dataType = DT_FLOAT, const std::vector<int64_t>& shape = {1})
    {
        reluWeight1Tensor = TensorInfo(dataType, FORMAT_ND, shape, this->name + "_relu_weight_1");
        reluWeight1Tensor.enabled = true;
        reluWeight1Tensor.slot = 6; // relu_weight_1 端口号
        this->AddInput(reluWeight1Tensor);
        return *this;
    }

    TensorInfo quantScale0Tensor;
    TensorInfo reluWeight0Tensor;
    TensorInfo quantScale1Tensor;
    TensorInfo reluWeight1Tensor;
};

inline TensorDesc BuildTensorDesc(DataType dtype, Format format, const std::vector<int64_t>& shape,
                                  Format originFormat = FORMAT_RESERVED, const std::vector<int64_t>& originShape = {})
{
    TensorDesc desc;
    desc.SetDataType(dtype);
    desc.SetFormat(format);
    desc.SetShape(Shape(shape));
    if (!originShape.empty()) {
        desc.SetOriginFormat(originFormat);
        desc.SetOriginShape(Shape(originShape));
    } else {
        desc.SetOriginFormat(format);
        desc.SetOriginShape(Shape(shape));
    }
    return desc;
}

struct AscendWeightQuantConfig : public NodeConfig {
    AscendWeightQuantConfig()
    {
        name = "AscendWeightQuant";
        opType = "AscendWeightQuant";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"offset", CompliantNodeBuilder::kEsIrInputOptional, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    }

    static AscendWeightQuantConfig Basic(const std::string& nodeName, DataType dataType = DT_FLOAT,
                                         Format format = FORMAT_HWCN, const std::vector<int64_t>& shape = {1, 1, 16, 4})
    {
        AscendWeightQuantConfig config;
        TensorInfo xInfo(dataType, format, shape, nodeName + "_x");
        xInfo.SetDesc(BuildTensorDesc(dataType, format, shape, format, shape));
        config.SetName(nodeName).AddInput(xInfo).AddOutput(dataType, format, shape, nodeName + "_y");
        return config;
    }
};

struct QuantBiasOptimizationConfig : public NodeConfig {
    QuantBiasOptimizationConfig()
    {
        name = "QuantBiasOptimization";
        opType = "QuantBiasOptimization";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    }

    static QuantBiasOptimizationConfig Basic(const std::string& nodeName, DataType dataType = DT_FLOAT,
                                             Format format = FORMAT_HWCN,
                                             const std::vector<int64_t>& shape = {1, 1, 16, 4})
    {
        QuantBiasOptimizationConfig config;
        TensorInfo filterInfo(dataType, format, shape, nodeName + "_filter");
        filterInfo.SetDesc(BuildTensorDesc(dataType, format, shape, format, shape));
        config.SetName(nodeName)
            .AddInput(dataType, FORMAT_NCHW, {1, 16, 256, 256}, nodeName + "_x")
            .AddInput(filterInfo)
            .AddOutput(dataType, format, shape, nodeName + "_y");
        return config;
    }
};

struct DepthwiseConv2DConfig : public NodeConfig {
    DepthwiseConv2DConfig()
    {
        name = "DepthwiseConv2D";
        opType = "DepthwiseConv2D";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""},
                     {"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("strides", std::vector<int64_t>{1, 1, 1, 1});
        SetAttr("pads", std::vector<int64_t>{0, 0, 0, 0});
        SetAttr("dilations", std::vector<int64_t>{1, 1, 1, 1});
        SetAttr("data_format", std::string("NCHW"));
        SetAttr("offset_x", int64_t(0));
    }

    static DepthwiseConv2DConfig Basic(std::string nodeName, DataType dataType = DT_FLOAT, Format format = FORMAT_NCHW,
                                       const std::vector<int64_t>& inputShape = {-1, 16, 256, 256},
                                       const std::vector<int64_t>& filterShape = {1, 1, 1, 64},
                                       const std::vector<int64_t>& outputShape = {-1, 4, 256, 256})
    {
        DepthwiseConv2DConfig config;
        TensorInfo xInfo(dataType, format, inputShape, nodeName + "_x");
        xInfo.SetDesc(BuildTensorDesc(dataType, format, inputShape, format, inputShape));
        TensorInfo filterInfo(dataType, FORMAT_HWCN, filterShape, nodeName + "_filter");
        filterInfo.SetDesc(BuildTensorDesc(dataType, FORMAT_HWCN, filterShape, FORMAT_HWCN, filterShape));
        TensorInfo yInfo(dataType, format, outputShape, "y");
        yInfo.SetDesc(BuildTensorDesc(dataType, format, outputShape, format, outputShape));
        config.SetName(nodeName)
            .AddInput(xInfo)
            .AddInput(filterInfo)
            .AddOutput(yInfo)
            .SetAttr("data_format", format == FORMAT_NHWC ? "NHWC" : "NCHW");
        return config;
    }

    DepthwiseConv2DConfig& WithBias(DataType dataType = DT_FLOAT, const std::vector<int64_t>& biasShape = {64})
    {
        biasTensor = TensorInfo(dataType, FORMAT_ND, biasShape, name + "_bias");
        biasTensor.SetDesc(BuildTensorDesc(dataType, FORMAT_ND, biasShape, FORMAT_ND, biasShape));
        biasTensor.enabled = true;
        AddInput(biasTensor);
        return *this;
    }

    TensorInfo biasTensor;
};

struct TransDataConfig : public NodeConfig {
    TransDataConfig()
    {
        name = "TransData";
        opType = "TransData";
        inputDefs = {{"src", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"dst", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("src_format", std::string("NDHWC"));
        SetAttr("dst_format", std::string("NDC1HWC0"));
        SetAttr("groups", int64_t(1));
    }

    // NDHWC -> NDC1HWC0; originShapeC is channel dim (index 4) for C%16 alignment tests.
    static TransDataConfig NdhwcToNdc1hwc0(const std::string& nodeName, DataType dataType = DT_FLOAT16,
                                           int64_t originShapeC = 16)
    {
        TransDataConfig config;
        config.SetName(nodeName);
        config.srcDesc = BuildTensorDesc(dataType, FORMAT_NDHWC, {1, 32, 240, 352, originShapeC}, FORMAT_NDHWC,
                                         {1, 32, 240, 352, originShapeC});
        config.dstDesc = BuildTensorDesc(dataType, FORMAT_NDC1HWC0, {1, 32, 1, 240, 352, 16}, FORMAT_NDHWC,
                                         {1, 32, 240, 352, originShapeC});
        // 把 src/dst 完整 desc 折进 inputs/outputs，统一走 Build() 的 slot 寻址
        TensorInfo srcInfo(dataType, FORMAT_NDHWC, {}, nodeName + "_src");
        srcInfo.SetDesc(config.srcDesc);
        config.AddInput(srcInfo);
        TensorInfo dstInfo;
        dstInfo.name = "dst";
        dstInfo.SetDesc(config.dstDesc);
        config.AddOutput(dstInfo);
        return config;
    }

    TensorDesc srcDesc;
    TensorDesc dstDesc;
};

struct IFMRConfig : public NodeConfig {
    IFMRConfig()
    {
        name = "IFMR";
        opType = "IFMR";
        inputDefs = {{"data", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"data_min", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"data_max", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"cumsum", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"scale", CompliantNodeBuilder::kEsIrOutputRequired, ""},
                      {"offset", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("min_percentile", 0.01f);
        SetAttr("max_percentile", 0.99f);
        SetAttr("search_range", std::vector<float>{0.8f, 1.2f});
        SetAttr("search_step", 0.01f);
        SetAttr("with_offset", true);
    }

    static IFMRConfig Basic(const std::string& nodeName, DataType dataType = DT_FLOAT16)
    {
        IFMRConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, FORMAT_ND, {1}, nodeName + "_data")
            .AddInput(dataType, FORMAT_ND, {1}, nodeName + "_data_min")
            .AddInput(dataType, FORMAT_ND, {1}, nodeName + "_data_max")
            .AddInput(DT_INT32, FORMAT_ND, {256}, nodeName + "_cumsum")
            .AddOutput(DT_FLOAT, FORMAT_ND, {1}, nodeName + "_scale")
            .AddOutput(DT_FLOAT, FORMAT_ND, {1}, nodeName + "_offset");
        return config;
    }
};

struct AddConfig : public NodeConfig {
    AddConfig()
    {
        name = "Add";
        opType = "Add";
        inputDefs = {{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    }

    static AddConfig Basic(const std::string& nodeName, DataType dataType = DT_FLOAT16, Format format = FORMAT_NCDHW,
                           const std::vector<int64_t>& shape = {1, 16, 16, 244, 244})
    {
        AddConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, shape, nodeName + "_x1")
            .AddInput(dataType, format, shape, nodeName + "_x2")
            .AddOutput(dataType, format, shape, nodeName + "_y");
        return config;
    }
};

// ============================================================================
// 待处理连接信息
// ============================================================================

struct PendingConnectionInfo {
    std::string fromNodeName;
    int fromOutputIndex;
    std::string toNodeName;
    int toInputIndex;
    es::EsTensorHolder graphInput;
};

struct FilterBranchOptions {
    bool withRelu = true;
    std::string filterNodeName = "filter";
    std::string reluNodeName = "relu";
    int32_t filterInputIndex = 1;
    int32_t filterInputSlot = 1;
};

// ============================================================================
// 测试图构建器（Builder模式）
// ============================================================================

class TestGraph {
public:
    TestGraph(const std::string& name = "test_graph") : graphName(name), graphBuilder(name.c_str()) {}

    TestGraph& SetSoc(const SocConfig& socConfig)
    {
        socConfig.Apply();
        return *this;
    }

    TestGraph& SetSocAscend950()
    {
        SetSoc(SocConfig::Ascend950());
        return *this;
    }

    TestGraph& SetSocMC62()
    {
        SetSoc(SocConfig::MC62());
        return *this;
    }

    es::EsTensorHolder CreateGraphInput(const TensorInfo& tensorInfo)
    {
        auto input = graphBuilder.CreateInput(inputIndex++, tensorInfo.name.c_str(), tensorInfo.dtype,
                                              tensorInfo.format, tensorInfo.shape);
        graphInputs.push_back(input);

        return input;
    }

    TestGraph& AddNode(const NodeConfig& config, const std::vector<int>& autoInputPositions = {})
    {
        return AddNodeImpl(config, autoInputPositions);
    }

    TestGraph& AddConv2D(const Conv2DConfig& conv2dConfig, bool autoFmap = true, bool autoFilter = true,
                         bool autoBias = true)
    {
        std::vector<int> autoInputs;
        if (autoFmap)
            autoInputs.push_back(0);
        if (autoFilter)
            autoInputs.push_back(1);
        if (autoBias && conv2dConfig.biasTensor.enabled)
            autoInputs.push_back(2);
        return AddNodeImpl(conv2dConfig, autoInputs);
    }

    TestGraph& AddDepthwiseConv2D(const DepthwiseConv2DConfig& depthwiseConfig, bool autoFmap = true,
                                  bool autoFilter = true, bool autoBias = true)
    {
        std::vector<int> autoInputs;
        if (autoFmap)
            autoInputs.push_back(0);
        if (autoFilter)
            autoInputs.push_back(1);
        if (autoBias && depthwiseConfig.biasTensor.enabled)
            autoInputs.push_back(2);
        return AddNodeImpl(depthwiseConfig, autoInputs);
    }

    // filter_src -> Data(filter) -> [Relu] -> Conv.filter
    TestGraph& AddConvWithFilterBranch(const NodeConfig& convConfig,
                                       const FilterBranchOptions& options = FilterBranchOptions())
    {
        return AddConvWithFilterBranchImpl(convConfig, options);
    }

    TestGraph& AddConv2DWithFilterBranch(const Conv2DConfig& conv2dConfig, bool withRelu = true,
                                         const std::string& filterNodeName = "filter",
                                         const std::string& reluNodeName = "relu")
    {
        FilterBranchOptions options;
        options.withRelu = withRelu;
        options.filterNodeName = filterNodeName;
        options.reluNodeName = reluNodeName;
        return AddConvWithFilterBranch(conv2dConfig, options);
    }

    TestGraph& AddConv3DWithFilterBranch(const Conv3DConfig& conv3dConfig, bool withRelu = true,
                                         const std::string& filterNodeName = "filter",
                                         const std::string& reluNodeName = "relu")
    {
        FilterBranchOptions options;
        options.withRelu = withRelu;
        options.filterNodeName = filterNodeName;
        options.reluNodeName = reluNodeName;
        return AddConvWithFilterBranch(conv3dConfig, options);
    }

    TestGraph& AddDepthwiseConv2DWithFilterBranch(const DepthwiseConv2DConfig& depthwiseConfig, bool withRelu = true,
                                                  const std::string& filterNodeName = "filter",
                                                  const std::string& reluNodeName = "relu")
    {
        FilterBranchOptions options;
        options.withRelu = withRelu;
        options.filterNodeName = filterNodeName;
        options.reluNodeName = reluNodeName;
        return AddConvWithFilterBranch(depthwiseConfig, options);
    }

    TestGraph& AddConv3D(const Conv3DConfig& conv3dConfig, bool autoFmap = true, bool autoFilter = true,
                         bool autoBias = true)
    {
        std::vector<int> autoInputs;
        if (autoFmap)
            autoInputs.push_back(0);
        if (autoFilter)
            autoInputs.push_back(1);
        if (autoBias && conv3dConfig.biasTensor.enabled)
            autoInputs.push_back(2);
        return AddNodeImpl(conv3dConfig, autoInputs);
    }

    TestGraph& AddAscendDequant(const AscendDequantConfig& ascendDequantConfig, bool autoX = false,
                                bool autoScale = true)
    {
        std::vector<int> autoInputs;
        if (autoX)
            autoInputs.push_back(0);
        if (autoScale)
            autoInputs.push_back(1);
        return AddNodeImpl(ascendDequantConfig, autoInputs);
    }

    TestGraph& AddAscendRequant(const AscendRequantConfig& ascendRequantConfig, bool autoX = false,
                                bool autoScale = true)
    {
        std::vector<int> autoInputs;
        if (autoX)
            autoInputs.push_back(0);
        if (autoScale)
            autoInputs.push_back(1);
        return AddNodeImpl(ascendRequantConfig, autoInputs);
    }

    TestGraph& AddAscendQuant(const AscendQuantConfig& ascendQuantConfig, bool autoInputs = false)
    {
        return AddNodeImpl(ascendQuantConfig, autoInputs ? std::vector<int>{0} : std::vector<int>{});
    }

    TestGraph& AddRelu(const ReluConfig& reluConfig, bool autoInputs = false)
    {
        return AddNodeImpl(reluConfig, autoInputs ? std::vector<int>{0} : std::vector<int>{});
    }

    TestGraph& AddData(const DataConfig& dataConfig, bool autoInput = true)
    {
        return AddNodeImpl(dataConfig, autoInput ? std::vector<int>{0} : std::vector<int>{});
    }

    TestGraph& AddLeakyRelu(const LeakyReluConfig& leakyReluConfig, bool autoInputs = false)
    {
        return AddNodeImpl(leakyReluConfig, autoInputs ? std::vector<int>{0} : std::vector<int>{});
    }

    TestGraph& AddPostCube(const PostCubeConfig& postCubeConfig, bool autoInputs = true)
    {
        std::vector<int> autoIdx;
        if (autoInputs) {
            // 自动建图输入的是已使能的可选张量(scale/relu)，主输入 x1 由 Conv 经 Connect 提供
            for (size_t i = 0; i < postCubeConfig.inputs.size(); ++i) {
                if (postCubeConfig.inputs[i].enabled) {
                    autoIdx.push_back(static_cast<int>(i));
                }
            }
        }
        return AddNodeImpl(postCubeConfig, autoIdx);
    }

    TestGraph& Connect(const std::string& fromNodeName, int fromOutputIndex, const std::string& toNodeName,
                       int toInputIndex)
    {
        if (graphBuilt) {
            auto fromIt = nodeMap.find(fromNodeName);
            auto toIt = nodeMap.find(toNodeName);
            if (fromIt != nodeMap.end() && toIt != nodeMap.end()) {
                AddEdgeAndUpdatePeerDesc(*geGraph, fromIt->second, fromOutputIndex, toIt->second, toInputIndex);
            }
        } else {
            pendingConnections.push_back({fromNodeName, fromOutputIndex, toNodeName, toInputIndex, {}});
        }
        return *this;
    }

    TestGraph& ConnectInput(const es::EsTensorHolder& graphInput, int inputIndex, const std::string& toNodeName,
                            int toInputIndex)
    {
        auto it = nodeMap.find(toNodeName);
        if (it != nodeMap.end()) {
            AddEdgeAndUpdatePeerDesc(*geGraph, *graphInput.GetProducer(), inputIndex, it->second, toInputIndex);
        }
        return *this;
    }

    TestGraph& SetOutput(const std::string& nodeName, int outputIndex = 0)
    {
        outputs.emplace_back(std::make_pair(nodeName, outputIndex));
        return *this;
    }

    std::shared_ptr<Graph> Build()
    {
        if (!graphBuilt) {
            geGraph = graphBuilder.BuildAndReset();

            for (const auto& cfg : pendingEntries) {
                auto node = CompliantNodeBuilder(geGraph.get())
                                .OpType(cfg.opType.c_str())
                                .Name(cfg.name.c_str())
                                .IrDefInputs(cfg.inputDefs)
                                .IrDefOutputs(cfg.outputDefs)
                                .IrDefAttrs({})
                                .Build();

                SetNodeAttrsFromConfig(node, cfg);
                for (size_t i = 0; i < cfg.inputs.size(); ++i) {
                    int32_t slot = cfg.inputs[i].slot >= 0 ? cfg.inputs[i].slot : static_cast<int32_t>(i);
                    node.UpdateInputDesc(slot, cfg.inputs[i].tensorDesc);
                }
                for (size_t i = 0; i < cfg.outputs.size(); ++i) {
                    node.UpdateOutputDesc(static_cast<int32_t>(i), cfg.outputs[i].tensorDesc);
                }
                nodeMap[cfg.name] = node;
            }
            AddNodeEdge();
            SetGraphOutput();
            graphBuilt = true;
        }

        return geGraph;
    }

    GNode GetNode(const std::string& nodeName)
    {
        auto it = nodeMap.find(nodeName);
        return (it != nodeMap.end()) ? it->second : GNode();
    }

    const std::map<std::string, GNode>& GetAllNodes() const { return nodeMap; }

    void UpdateNodeInputDesc(const std::string& nodeName, int32_t index, DataType dtype, Format format)
    {
        TensorDesc tensorDesc;
        tensorDesc.SetDataType(dtype);
        tensorDesc.SetFormat(format);
        auto it = nodeMap.find(nodeName);

        it->second.UpdateInputDesc(index, tensorDesc);
    }

    void UpdateNodeOutputDesc(const std::string& nodeName, int32_t index, DataType dtype, Format format)
    {
        TensorDesc tensorDesc;
        tensorDesc.SetDataType(dtype);
        tensorDesc.SetFormat(format);
        auto it = nodeMap.find(nodeName);

        it->second.UpdateOutputDesc(index, tensorDesc);
    }

    void UpdateNodeInputDescEx(const std::string& nodeName, int32_t index, const TensorDesc& tensorDesc)
    {
        auto it = nodeMap.find(nodeName);
        if (it != nodeMap.end()) {
            it->second.UpdateInputDesc(index, tensorDesc);
        }
    }

    void UpdateNodeOutputDescEx(const std::string& nodeName, int32_t index, const TensorDesc& tensorDesc)
    {
        auto it = nodeMap.find(nodeName);
        if (it != nodeMap.end()) {
            it->second.UpdateOutputDesc(index, tensorDesc);
        }
    }

    TestGraph& AddTransData(const TransDataConfig& transDataConfig, bool autoSrc = true)
    {
        return AddNodeImpl(transDataConfig, autoSrc ? std::vector<int>{0} : std::vector<int>{});
    }

    TestGraph& AddIFMR(const IFMRConfig& ifmrConfig, bool autoInputs = true)
    {
        std::vector<int> autoIdx;
        if (autoInputs) {
            for (size_t i = 0; i < ifmrConfig.inputs.size(); ++i) {
                autoIdx.push_back(static_cast<int>(i));
            }
        }
        return AddNodeImpl(ifmrConfig, autoIdx);
    }

    TestGraph& AddAdd(const AddConfig& addConfig, bool autoInputs = false)
    {
        std::vector<int> autoIdx;
        if (autoInputs) {
            for (size_t i = 0; i < addConfig.inputs.size(); ++i) {
                autoIdx.push_back(static_cast<int>(i));
            }
        }
        return AddNodeImpl(addConfig, autoIdx);
    }

private:
    TestGraph& AddConvWithFilterBranchImpl(const NodeConfig& convConfig, const FilterBranchOptions& options)
    {
        const TensorInfo& filterInfo = convConfig.inputs[static_cast<size_t>(options.filterInputIndex)];
        AddNodeImpl(convConfig, {0});

        DataConfig filterCfg = DataConfig::FromTensor(options.filterNodeName, filterInfo,
                                                      options.filterNodeName + "_src");
        AddData(filterCfg, true);

        if (options.withRelu) {
            AddRelu(ReluConfig::FromTensor(options.reluNodeName, filterInfo), false);
            Connect(options.filterNodeName, 0, options.reluNodeName, 0);
            Connect(options.reluNodeName, 0, convConfig.name, options.filterInputSlot);
        } else {
            Connect(options.filterNodeName, 0, convConfig.name, options.filterInputSlot);
        }
        return *this;
    }

    // 统一的节点登记：按 autoInputPositions 自动创建图输入，其余信息存入 pendingEntries(值拷贝，
    // 切片到 NodeConfig 基类，Build() 时只读基类字段，避免悬空指针)。
    TestGraph& AddNodeImpl(const NodeConfig& config, const std::vector<int>& autoInputPositions)
    {
        for (int pos : autoInputPositions) {
            const auto& info = config.inputs[pos];
            int32_t slot = info.slot >= 0 ? info.slot : pos;
            auto input = CreateGraphInput(info);
            pendingConnections.push_back({"", 0, config.name, slot, input});
        }
        pendingEntries.push_back(config);
        return *this;
    }

    void SetNodeAttrsFromConfig(GNode& node, const NodeConfig& cfg)
    {
        for (const auto& attr : cfg.intAttrs) {
            int64_t attrValue = attr.second;
            node.SetAttr(AscendString(attr.first.c_str()), attrValue);
        }
        for (const auto& attr : cfg.floatAttrs) {
            float attrValue = attr.second;
            node.SetAttr(AscendString(attr.first.c_str()), attrValue);
        }
        for (const auto& attr : cfg.boolAttrs) {
            bool attrValue = attr.second;
            node.SetAttr(AscendString(attr.first.c_str()), attrValue);
        }
        for (const auto& attr : cfg.strAttrs) {
            AscendString strAttr = AscendString(attr.second.c_str());
            node.SetAttr(AscendString(attr.first.c_str()), strAttr);
        }
        for (const auto& attr : cfg.listIntAttrs) {
            std::vector<int64_t> attrValue = attr.second;
            node.SetAttr(AscendString(attr.first.c_str()), attrValue);
        }
        for (const auto& attr : cfg.listFloatAttrs) {
            std::vector<float> attrValue = attr.second;
            node.SetAttr(AscendString(attr.first.c_str()), attrValue);
        }
        for (const auto& attr : cfg.listStrAttrs) {
            std::vector<AscendString> attrValue;
            for (const auto& str : attr.second) {
                attrValue.push_back(AscendString(str.c_str()));
            }
            node.SetAttr(AscendString(attr.first.c_str()), attrValue);
        }
    }

    void AddNodeEdge()
    {
        for (const auto& connInfo : pendingConnections) {
            if (connInfo.fromNodeName.empty()) {
                auto it = nodeMap.find(connInfo.toNodeName);
                if (it != nodeMap.end()) {
                    AddEdgeAndUpdatePeerDesc(*geGraph, *connInfo.graphInput.GetProducer(), connInfo.fromOutputIndex,
                                             it->second, connInfo.toInputIndex);
                }
            } else {
                auto fromIt = nodeMap.find(connInfo.fromNodeName);
                auto toIt = nodeMap.find(connInfo.toNodeName);
                if (fromIt != nodeMap.end() && toIt != nodeMap.end()) {
                    AddEdgeAndUpdatePeerDesc(*geGraph, fromIt->second, connInfo.fromOutputIndex, toIt->second,
                                             connInfo.toInputIndex);
                }
            }
        }
    }

    void SetGraphOutput()
    {
        std::vector<std::pair<GNode, int32_t>> outputNodes = {};
        for (auto out : outputs) {
            auto iter = nodeMap.find(out.first);
            if (iter == nodeMap.end())
                continue;
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
    std::vector<NodeConfig> pendingEntries = {};
    std::vector<PendingConnectionInfo> pendingConnections = {};
    bool graphBuilt = false;
    std::vector<std::pair<std::string, int32_t>> outputs = {};
};

// ============================================================================
// 图验证工具
// ============================================================================

class GraphChecker {
public:
    static bool HasNode(std::shared_ptr<Graph>& graph, const std::string& nodeType)
    {
        for (auto node : graph->GetAllNodes()) {
            AscendString curType;
            node.GetType(curType);
            if (curType.GetString() == nodeType) {
                return true;
            }
        }
        return false;
    }

    static int CountNodes(std::shared_ptr<Graph>& graph, const std::string& nodeType)
    {
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

    static bool FindFirstNodeByOpType(std::shared_ptr<Graph>& graph, const std::string& opType, GNode& outNode)
    {
        for (auto node : graph->GetAllNodes()) {
            AscendString curType;
            node.GetType(curType);
            if (curType.GetString() == opType) {
                outNode = node;
                return true;
            }
        }
        return false;
    }

    static bool FindNodeByNameSuffix(std::shared_ptr<Graph>& graph, const std::string& suffix, GNode& outNode)
    {
        for (auto node : graph->GetAllNodes()) {
            AscendString nodeName;
            if (node.GetName(nodeName) != GRAPH_SUCCESS) {
                continue;
            }
            std::string nameStr = nodeName.GetString();
            if (nameStr.size() >= suffix.size() &&
                nameStr.compare(nameStr.size() - suffix.size(), suffix.size(), suffix) == 0) {
                outNode = node;
                return true;
            }
        }
        return false;
    }

    static bool GetOriginShape(const GNode& node, int32_t index, bool isOutput, std::vector<int64_t>& dims)
    {
        TensorDesc desc;
        if (isOutput) {
            if (node.GetOutputDesc(index, desc) != GRAPH_SUCCESS) {
                return false;
            }
        } else if (node.GetInputDesc(index, desc) != GRAPH_SUCCESS) {
            return false;
        }
        dims = desc.GetOriginShape().GetDims();
        return true;
    }

    static bool GetNodeBoolAttr(const GNode& node, const char* attrName, bool& value)
    {
        return node.GetAttr(AscendString(attrName), value) == GRAPH_SUCCESS;
    }

    static bool GetNodeStringAttr(const GNode& node, const char* attrName, std::string& value)
    {
        AscendString attr;
        if (node.GetAttr(AscendString(attrName), attr) != GRAPH_SUCCESS) {
            return false;
        }
        value = attr.GetString();
        return true;
    }

    static void Print(std::shared_ptr<Graph>& graph)
    {
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