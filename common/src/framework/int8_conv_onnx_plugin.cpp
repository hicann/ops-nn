/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

namespace {
constexpr int INSERT_INDEX = 2;
constexpr int SCALE_INDEX = 2;
} // namespace

namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;

struct OnnxInt8ConvInfo {
    int groups = 1;
    int64_t Y_zero_point = 0;
    float Y_scale = 0.0;
    std::string order = "NCHW";
    std::vector<float> scale = {};
    std::vector<int> strides = {};
    std::vector<int> pads = {};
    std::vector<int> dilations = {};
};

static OnnxInt8ConvInfo SetInt8ConvInfo(const NodeProto* node, const Message* op_src, ge::Operator& op_dest)
{
    OnnxInt8ConvInfo tbeAttr;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                tbeAttr.strides.push_back(attr.ints(i));
            }
        } else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                tbeAttr.pads.push_back(attr.ints(i));
            }
        } else if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                tbeAttr.dilations.push_back(attr.ints(i));
            }
        } else if (attr.name() == "groups" && attr.type() == ge::onnx::AttributeProto::INT) {
            tbeAttr.groups = attr.i();
        } else if (attr.name() == "Y_scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            tbeAttr.Y_scale = attr.f();
        } else if (attr.name() == "Y_zero_point" && attr.type() == ge::onnx::AttributeProto::INT) {
            tbeAttr.Y_zero_point = attr.i();
        } else if (attr.name().find("scale") != std::string::npos) {
            tbeAttr.scale.push_back(attr.f());
        } else if (attr.name() == "order" && attr.type() == ge::onnx::AttributeProto::STRING) {
            tbeAttr.order = attr.s();
        }
    }

    int temp_num = 1;
    if (tbeAttr.order == "NCHW") {
        tbeAttr.strides.insert(tbeAttr.strides.begin(), INSERT_INDEX, temp_num);
        tbeAttr.dilations.insert(tbeAttr.dilations.begin(), INSERT_INDEX, temp_num);
    } else if (tbeAttr.order == "NHWC") {
        tbeAttr.strides.insert(tbeAttr.strides.begin(), temp_num);
        tbeAttr.strides.insert(tbeAttr.strides.end(), temp_num);
        tbeAttr.dilations.insert(tbeAttr.dilations.begin(), temp_num);
        tbeAttr.dilations.insert(tbeAttr.dilations.end(), temp_num);
    } else {
        OP_LOGW("Int8Conv", "get attr order foramt is failed.");
    }

    return tbeAttr;
}

static Status ParseParamsInt8Conv(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE("Int8Conv", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    op_dest.DynamicInputRegister("x", 3);
    op_dest.DynamicOutputRegister("output", 1);
    op_dest.SetAttr("original_type", "ai.onnx::11::Int8Conv");

    OnnxInt8ConvInfo Int8ConvAttr = SetInt8ConvInfo(node, op_src, op_dest);

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("strides", Int8ConvAttr.strides);
    op_dest.SetAttr("pads", Int8ConvAttr.pads);
    op_dest.SetAttr("dilations", Int8ConvAttr.dilations);
    op_dest.SetAttr("groups", Int8ConvAttr.groups);
    op_dest.SetAttr("Y_scale", Int8ConvAttr.Y_scale);
    op_dest.SetAttr("Y_zero_point", Int8ConvAttr.Y_zero_point);
    op_dest.SetAttr("scale", Int8ConvAttr.scale[SCALE_INDEX]);
    op_dest.SetAttr("order", Int8ConvAttr.order);
    return SUCCESS;
}

static Status ParseOpToGraphInt8Conv(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data(ori_name + "_data0").set_attr_index(0);
    auto data1 = op::Data(ori_name + "_data1").set_attr_index(1);
    auto data2 = op::Data(ori_name + "_data2").set_attr_index(2);

    std::vector<int64_t> strides = {};
    if (op.GetAttr("strides", strides) != SUCCESS) {
        OP_LOGE("Int8Conv", "get strides from op failed");
        return FAILED;
    }
    std::vector<int64_t> pads = {};
    if (op.GetAttr("pads", pads) != SUCCESS) {
        OP_LOGE("Int8Conv", "get pads from op failed");
        return FAILED;
    }
    std::vector<int64_t> dilations = {};
    if (op.GetAttr("dilations", dilations) != SUCCESS) {
        OP_LOGE("Int8Conv", "get dilations from op failed");
        return FAILED;
    }
    int groups = 1;
    if (op.GetAttr("groups", groups) != SUCCESS) {
        OP_LOGE("Int8Conv", "get groups from op failed");
        return FAILED;
    }
    float Y_scale = 0.0;
    if (op.GetAttr("Y_scale", Y_scale) != SUCCESS) {
        OP_LOGE("Int8Conv", "get Y_scale from op failed");
        return FAILED;
    }
    int64_t Y_zero_point = 0;
    if (op.GetAttr("Y_zero_point", Y_zero_point) != SUCCESS) {
        OP_LOGE("Int8Conv", "get Y_zero_point from op failed");
        return FAILED;
    }
    float scale = 0.0;
    if (op.GetAttr("scale", scale) != SUCCESS) {
        OP_LOGE("Int8Conv", "get scale from op failed");
        return FAILED;
    }
    std::string order = "";
    if (op.GetAttr("order", order) != SUCCESS) {
        OP_LOGE("Int8Conv", "get order from op failed");
        return FAILED;
    }

    ge::Format filter_format = ge::FORMAT_NCHW;
    if (order == "NHWC") {
        filter_format = ge::FORMAT_NHWC;
    }
    std::map<std::string, ge::Format> kvmap = {{"NCHW", ge::FORMAT_NCHW}, {"NHWC", ge::FORMAT_NHWC}};
    auto order_map = kvmap.find(order);
    if (order_map == kvmap.end()) {
        OP_LOGE("Int8Conv", "only support NCHW/NHWC, but got %s", order.c_str());
        return FAILED;
    }

    auto conv = op::Conv2D(ori_name + "_Conv2D")
                    .set_input_x(data0)
                    .set_input_filter(data1)
                    .set_input_bias(data2)
                    .set_attr_strides(strides)
                    .set_attr_pads(pads)
                    .set_attr_dilations(dilations)
                    .set_attr_groups(groups)
                    .set_attr_data_format(order);

    auto ret_x = ChangeFormatFromOnnx(conv, 0, order_map->second, true);
    if (ret_x != ge::GRAPH_SUCCESS) {
        OP_LOGE("Int8Conv", "update x format failed.");
        return FAILED;
    }
    auto ret_w = ChangeFormatFromOnnx(conv, 1, filter_format, true);
    if (ret_w != ge::GRAPH_SUCCESS) {
        OP_LOGE("Int8Conv", "update filter format failed.");
        return FAILED;
    }
    auto ret_y = ChangeFormatFromOnnx(conv, 0, order_map->second, false);
    if (ret_y != ge::GRAPH_SUCCESS) {
        OP_LOGE("Int8Conv", "update output format failed.");
        return FAILED;
    }

    ge::Tensor scalar_const_data_deq = CreateScalar(scale, ge::DT_FLOAT16);
    auto data_deq = op::Const(ori_name + "_data_deq").set_attr_value(scalar_const_data_deq);
    auto ascendD = op::AscendDequant(ori_name + "_AscendDequant")
                       .set_input_x(conv)
                       .set_input_deq_scale(data_deq)
                       .set_attr_dtype(1);
    auto ret_deq_x = ChangeFormatFromOnnx(ascendD, 0, order_map->second, true);
    if (ret_deq_x != ge::GRAPH_SUCCESS) {
        OP_LOGE("Intconv8", "update intput x format failed");
        return FAILED;
    }
    auto ret_deq = ChangeFormatFromOnnx(ascendD, 1, order_map->second, true);
    if (ret_deq != ge::GRAPH_SUCCESS) {
        OP_LOGE("Intconv8", "update intput deq format failed");
        return FAILED;
    }
    auto ret_deq_y = ChangeFormatFromOnnx(ascendD, 0, order_map->second, false);
    if (ret_deq_y != ge::GRAPH_SUCCESS) {
        OP_LOGE("Intconv8", "update output y format failed");
        return FAILED;
    }

    float offset = Y_zero_point;
    auto ascendQ = op::AscendQuant(ori_name + "_AscendQuant")
                       .set_input_x(ascendD)
                       .set_attr_scale(Y_scale)
                       .set_attr_offset(offset);
    auto ret_quant_x = ChangeFormatFromOnnx(ascendQ, 0, order_map->second, true);
    if (ret_quant_x != ge::GRAPH_SUCCESS) {
        OP_LOGE("Intconv8", "update intput x format failed");
        return FAILED;
    }
    auto ret_quant_y = ChangeFormatFromOnnx(ascendQ, 0, ge::FORMAT_NC1HWC0, false);
    if (ret_quant_y != ge::GRAPH_SUCCESS) {
        OP_LOGE("Intconv8", "update output y format failed");
        return FAILED;
    }

    std::vector<ge::Operator> inputs{data0, data1, data2};
    std::vector<std::pair<ge::Operator, std::vector<size_t> > > outputs;
    outputs.emplace_back(ascendQ, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Int8Conv", "ai.onnx::9::Int8Conv", "ai.onnx::10::Int8Conv", "ai.onnx::11::Int8Conv",
                   "ai.onnx::12::Int8Conv", "ai.onnx::13::Int8Conv", "ai.onnx::14::Int8Conv", "ai.onnx::15::Int8Conv",
                   "ai.onnx::16::Int8Conv"})
    .ParseParamsFn(ParseParamsInt8Conv)
    .ParseOpToGraphFn(ParseOpToGraphInt8Conv)
    .ImplyType(ImplyType::TVM);
} // namespace domi
