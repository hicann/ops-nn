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
 * \file basic_gru_inplace_fill_window_cache.cpp
 * \brief
 */

#include "onnx_common.h"

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
struct BasicGRUInplaceFillWindowCacheAttr {
    bool quant_sqrt_mode_x = false;
    bool quant_sqrt_mode_h = false;
    int linear_before_reset = 1;
    std::vector<float> activation_alpha;
    std::vector<float> activation_beta;
    std::vector<std::string> activations;
    std::string direction = "forward";
};

static Status BasicGRUInplaceFillWindowCacheAttrFromOnnx(
    const NodeProto* node, BasicGRUInplaceFillWindowCacheAttr& nodeAttr, ge::Operator& op_dest)
{
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "hidden_size" && attr.type() == ge::onnx::AttributeProto::INT) {
            op_dest.SetAttr("hidden_size", attr.i());
        } else if (attr.name() == "activation_alpha" && attr.type() == ge::onnx::AttributeProto::FLOATS) {
            for (auto s : attr.floats()) {
                nodeAttr.activation_alpha.push_back(s);
            }
            op_dest.SetAttr("activation_alpha", nodeAttr.activation_alpha);
        } else if (attr.name() == "activation_beta" && attr.type() == ge::onnx::AttributeProto::FLOATS) {
            for (auto s : attr.floats()) {
                nodeAttr.activation_beta.push_back(s);
            }
            op_dest.SetAttr("activation_beta", nodeAttr.activation_beta);
        } else if (attr.name() == "activations" && attr.type() == ge::onnx::AttributeProto::STRINGS) {
            for (auto s : attr.strings()) {
                nodeAttr.activations.push_back(s);
            }
            op_dest.SetAttr("activations", nodeAttr.activations);
        } else if (attr.name() == "clip" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            op_dest.SetAttr("clip", attr.f());
        } else if (attr.name() == "direction" && attr.type() == ge::onnx::AttributeProto::STRING) {
            nodeAttr.direction = attr.s();
            op_dest.SetAttr("direction", nodeAttr.direction);
        } else if (attr.name() == "linear_before_reset" && attr.type() == ge::onnx::AttributeProto::INT) {
            op_dest.SetAttr("linear_before_reset", attr.i());
        }
    }
    return SUCCESS;
}

static Status QuantUpdateAttrFromOnnx(
    const NodeProto* node, BasicGRUInplaceFillWindowCacheAttr& nodeAttr, ge::Operator& op_dest)
{
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "quant_scale_x" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            op_dest.SetAttr("quant_scale_x", attr.f());
        } else if (attr.name() == "quant_offset_x" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            op_dest.SetAttr("quant_offset_x", attr.f());
        } else if (attr.name() == "quant_sqrt_mode_x" && attr.i() != 0) {
            nodeAttr.quant_sqrt_mode_x = true;
        } else if (attr.name() == "quant_scale_h" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            op_dest.SetAttr("quant_scale_h", attr.f());
        } else if (attr.name() == "quant_offset_h" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            op_dest.SetAttr("quant_offset_h", attr.f());
        } else if (attr.name() == "quant_sqrt_mode_h" && attr.i() != 0) {
            nodeAttr.quant_sqrt_mode_h = true;
        } else if (attr.name() == "quant_dtype" && attr.type() == ge::onnx::AttributeProto::INT) {
            op_dest.SetAttr("quant_dtype", attr.i());
        }
        op_dest.SetAttr("quant_sqrt_mode_x", nodeAttr.quant_sqrt_mode_x);
        op_dest.SetAttr("quant_sqrt_mode_h", nodeAttr.quant_sqrt_mode_h);
    }
    return SUCCESS;
}

static Status ParseParamBasicGRUInplaceFillWindowCache(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    BasicGRUInplaceFillWindowCacheAttr nodeAttr;
    if (BasicGRUInplaceFillWindowCacheAttrFromOnnx(node, nodeAttr, op_dest) != SUCCESS) {
        return FAILED;
    }
    if (QuantUpdateAttrFromOnnx(node, nodeAttr, op_dest) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("BasicGRUInplaceFillWindowCache")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::BasicGRUInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::9::BasicGRUInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::10::BasicGRUInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::11::BasicGRUInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::12::BasicGRUInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::13::BasicGRUInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::14::BasicGRUInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::15::BasicGRUInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::16::BasicGRUInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::17::BasicGRUInplaceFillWindowCache"),
         ge::AscendString("ai.onnx::18::BasicGRUInplaceFillWindowCache")})
    .ParseParamsFn(ParseParamBasicGRUInplaceFillWindowCache)
    .ImplyType(ImplyType::TVM);
} // namespace domi
