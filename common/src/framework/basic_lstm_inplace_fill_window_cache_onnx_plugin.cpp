/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file basic_lstm_inplace_fill_window_cache.cpp
 * \brief
 */

#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
struct BasicLSTMInplaceFillWindowCacheAttr {
    bool quant_sqrt_mode_x = false;
    bool quant_sqrt_mode_h = false;
    std::vector<float> activation_alpha;
    std::vector<float> activation_beta;
    std::vector<std::string> activations;
    std::string direction = "forward";
};

static Status BasicLSTMInplaceFillWindowCacheAttrFromOnnx(
    const NodeProto* node, BasicLSTMInplaceFillWindowCacheAttr& node_attr, ge::Operator& op_dest)
{
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "hidden_size" && attr.type() == ge::onnx::AttributeProto::INT) {
            op_dest.SetAttr("hidden_size", attr.i());
        }
        if (attr.name() == "activation_alpha" && attr.type() == ge::onnx::AttributeProto::FLOATS) {
            for (auto s : attr.floats()) {
                node_attr.activation_alpha.push_back(s);
            }
            op_dest.SetAttr("activation_alpha", node_attr.activation_alpha);
        }
        if (attr.name() == "activation_beta" && attr.type() == ge::onnx::AttributeProto::FLOATS) {
            for (auto s : attr.floats()) {
                node_attr.activation_beta.push_back(s);
            }
            op_dest.SetAttr("activation_beta", node_attr.activation_beta);
        }
        if (attr.name() == "activations" && attr.type() == ge::onnx::AttributeProto::STRINGS) {
            for (auto s : attr.strings()) {
                node_attr.activations.push_back(s);
            }
            op_dest.SetAttr("activations", node_attr.activations);
        }
        if (attr.name() == "clip" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            op_dest.SetAttr("clip", attr.f());
        }
        if (attr.name() == "direction" && attr.type() == ge::onnx::AttributeProto::STRING) {
            node_attr.direction = attr.s();
            op_dest.SetAttr("direction", node_attr.direction);
        }
        if (attr.name() == "input_forget" && attr.type() == ge::onnx::AttributeProto::INT) {
            op_dest.SetAttr("input_forget", attr.i());
        }
    }
    return SUCCESS;
}

static Status QuantUpdateAttrFromOnnx(
    const NodeProto* node, BasicLSTMInplaceFillWindowCacheAttr& node_attr, ge::Operator& op_dest)
{
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "quant_scale_x" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            op_dest.SetAttr("quant_scale_x", attr.f());
        }
        if (attr.name() == "quant_offset_x" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            op_dest.SetAttr("quant_offset_x", attr.f());
        }
        if (attr.name() == "quant_sqrt_mode_x" && attr.i() != 0) {
            node_attr.quant_sqrt_mode_x = true;
        }
        if (attr.name() == "quant_scale_h" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            op_dest.SetAttr("quant_scale_h", attr.f());
        }
        if (attr.name() == "quant_offset_h" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            op_dest.SetAttr("quant_offset_h", attr.f());
        }
        if (attr.name() == "quant_sqrt_mode_h" && attr.i() != 0) {
            node_attr.quant_sqrt_mode_h = true;
        }
        if (attr.name() == "quant_dtype" && attr.type() == ge::onnx::AttributeProto::INT) {
            op_dest.SetAttr("quant_dtype", attr.i());
        }
        op_dest.SetAttr("quant_sqrt_mode_x", node_attr.quant_sqrt_mode_x);
        op_dest.SetAttr("quant_sqrt_mode_h", node_attr.quant_sqrt_mode_h);
    }
    return SUCCESS;
}

static Status ParseParamBasicLSTMInplaceFillWindowCache(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    BasicLSTMInplaceFillWindowCacheAttr node_attr;
    if (BasicLSTMInplaceFillWindowCacheAttrFromOnnx(node, node_attr, op_dest) != SUCCESS) {
        return FAILED;
    }
    if (QuantUpdateAttrFromOnnx(node, node_attr, op_dest) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("BasicLSTMInplaceFillWindowCache")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::BasicLSTMInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::9::BasicLSTMInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::10::BasicLSTMInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::11::BasicLSTMInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::12::BasicLSTMInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::13::BasicLSTMInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::14::BasicLSTMInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::15::BasicLSTMInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::16::BasicLSTMInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::17::BasicLSTMInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::18::BasicLSTMInplaceFillWindowCache")})
    .ParseParamsFn(ParseParamBasicLSTMInplaceFillWindowCache)
    .ImplyType(ImplyType::TVM);
} // namespace domi
