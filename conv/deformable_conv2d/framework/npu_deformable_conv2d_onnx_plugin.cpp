/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "op_nn_proto_extend.h"

using namespace ge;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsNPUDeformableConv2D(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE("DeformableConv2D", "Dynamic cast op_src to NodeProto failed");
        return FAILED;
    }
    int op_input_size = node->input_size();
    int op_output_size = node->output_size();
    op_dest.DynamicInputRegister("x", op_input_size);
    op_dest.DynamicOutputRegister("y", op_output_size);

    op_dest.SetAttr("original_type", "npu::1::NPUDeformableConv2d");

    // 3.set attr if needed
    std::vector<int> strides = {};
    std::vector<int> pads = {};
    std::vector<int> dilations = {1, 1, 1, 1};
    std::string data_format = "NHWC";
    int deformable_groups = 1;
    int groups = 1;
    bool modulated = true;
    bool set_strides = false;
    bool set_pads = false;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
            int num = attr.ints_size();
            for (int i = 0; i < num; ++i) {
                strides.push_back(attr.ints(i));
            }
            set_strides = true;
        }
        if (attr.name() == "paddings" && attr.type() == ge::onnx::AttributeProto::INTS) {
            int num = attr.ints_size();
            for (int i = 0; i < num; ++i) {
                pads.push_back(attr.ints(i));
            }
            set_pads = true;
        }
        if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS) {
            dilations.clear();
            int num = attr.ints_size();
            for (int i = 0; i < num; ++i) {
                dilations.push_back(attr.ints(i));
            }
        }
        if (attr.name() == "data_format" && attr.type() == ge::onnx::AttributeProto::STRING) {
            data_format = attr.s();
        }
        if (attr.name() == "deformable_groups" && attr.type() == ge::onnx::AttributeProto::INT) {
            deformable_groups = attr.i();
        }
        if (attr.name() == "groups" && attr.type() == ge::onnx::AttributeProto::INT) {
            groups = attr.i();
        }
        if (attr.name() == "modulated" && attr.type() == ge::onnx::AttributeProto::INT) {
            modulated = attr.i();
        }
    }
    if (set_strides) {
        op_dest.SetAttr("strides", strides);
    } else {
        OP_LOGE("DeformableConv2D", "onnx DeformableConv2D op has no strides attr.");
        return FAILED;
    }
    if (set_pads) {
        op_dest.SetAttr("pads", pads);
    } else {
        OP_LOGE("DeformableConv2D", "onnx DeformableConv2D op has no pads attr.");
        return FAILED;
    }

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("dilations", dilations);
    op_dest.SetAttr("groups", groups);
    op_dest.SetAttr("data_format", data_format);
    op_dest.SetAttr("deformable_groups", deformable_groups);
    op_dest.SetAttr("modulated", modulated);
    return SUCCESS;
}

static Status ParseOpToGraphDeformableConv2D(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }
    std::vector<int64_t> strides = {};
    std::vector<int64_t> pads = {};
    std::vector<int64_t> dilations = {1, 1, 1, 1};
    std::vector<int64_t> set_dilations = {};
    std::string data_format = "NHWC";
    int deformable_groups = 1;
    int groups = 1;
    bool modulated = true;

    if (op.GetAttr("strides", strides) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get strides from op failed");
        return FAILED;
    }
    if (op.GetAttr("pads", pads) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get pads from op failed");
        return FAILED;
    }
    if (op.GetAttr("dilations", set_dilations) == SUCCESS) {
        op.GetAttr("dilations", dilations);
    }

    op.GetAttr("groups", groups);
    op.GetAttr("data_format", data_format);
    op.GetAttr("deformable_groups", deformable_groups);
    op.GetAttr("modulated", modulated);

    auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto identity_0 = ge::op::Identity((ori_name + "_identity_0").c_str()).set_input_x(data0);
    auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
    auto identity_1 = ge::op::Identity((ori_name + "_identity_1").c_str()).set_input_x(data1);
    auto data2 = ge::op::Data((ori_name + "_data2").c_str()).set_attr_index(2);
    auto identity_2 = ge::op::Identity((ori_name + "_identity_2").c_str()).set_input_x(data2);
    auto data3 = ge::op::Data((ori_name + "_data3").c_str()).set_attr_index(3);
    auto identity_3 = ge::op::Identity((ori_name + "_identity_3").c_str()).set_input_x(data3);

    auto DeformableConv2D = ge::op::DeformableConv2D((ori_name + "DeformableConv2D").c_str())
                                .set_input_x(identity_0)
                                .set_input_filter(identity_1)
                                .set_input_offsets(identity_2)
                                .set_input_bias(identity_3)
                                .set_attr_strides(strides)
                                .set_attr_pads(pads)
                                .set_attr_dilations(dilations)
                                .set_attr_groups(groups)
                                .set_attr_data_format(data_format)
                                .set_attr_deformable_groups(deformable_groups)
                                .set_attr_modulated(modulated);
    if (ChangeFormatFromOnnx(DeformableConv2D, 0, ge::FORMAT_NCHW, true) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update input format failed.");
        return FAILED;
    }
    if (ChangeFormatFromOnnx(DeformableConv2D, 1, ge::FORMAT_NCHW, true) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update weight format failed.");
        return FAILED;
    }
    if (ChangeFormatFromOnnx(DeformableConv2D, 2, ge::FORMAT_NCHW, true) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update offset format failed.");
        return FAILED;
    }
    if (ChangeFormatFromOnnx(DeformableConv2D, 0, ge::FORMAT_NCHW, false) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update output format failed.");
        return FAILED;
    }

    std::vector<ge::Operator> inputs{data0, data1, data2, data3};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(DeformableConv2D, std::vector<std::size_t>{0});
    float output_data = 0.0;
    ge::Tensor scalar_zero = CreateScalar(output_data, ge::DT_FLOAT);
    auto data_output = ge::op::Const((ori_name + "_output2").c_str()).set_attr_value(scalar_zero);
    outputs.emplace_back(data_output, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);

    return SUCCESS;
}

// register DeformableConv2D op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("npu::1::NPUDeformableConv2d"), ge::AscendString("ai.onnx::9::NPUDeformableConv2d"),
         ge::AscendString("ai.onnx::10::NPUDeformableConv2d"), ge::AscendString("ai.onnx::11::NPUDeformableConv2d"),
         ge::AscendString("ai.onnx::12::NPUDeformableConv2d"), ge::AscendString("ai.onnx::13::NPUDeformableConv2d"),
         ge::AscendString("ai.onnx::14::NPUDeformableConv2d"), ge::AscendString("ai.onnx::15::NPUDeformableConv2d"),
         ge::AscendString("ai.onnx::16::NPUDeformableConv2d")})
    .ParseParamsFn(ParseParamsNPUDeformableConv2D)
    .ParseOpToGraphFn(ParseOpToGraphDeformableConv2D)
    .ImplyType(ImplyType::TVM);
} // namespace domi
