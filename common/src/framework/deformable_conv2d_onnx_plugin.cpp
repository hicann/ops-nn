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
namespace domi {
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsDeformableConv2D(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE("DeformableConv2D", "Dynamic cast op_src to NodeProto failed");
        return FAILED;
    }

    std::vector<int> strides = {};
    std::vector<int> pads = {};
    std::vector<int> dilations = {1, 1, 1, 1};
    std::string data_format = "NHWC";
    int deformable_groups = 1;
    int groups = 0;
    bool modulated = true;
    bool set_strides = false;
    bool set_pads = false;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
            if (attr.ints_size() == 1) {
                strides.push_back(1);
                strides.push_back(1);
                strides.push_back(attr.ints(0));
                strides.push_back(attr.ints(0));
            } else if (attr.ints_size() == 2) {
                strides.push_back(1);
                strides.push_back(1);
                strides.push_back(attr.ints(0));
                strides.push_back(attr.ints(1));
            } else {
                OP_LOGE("DeformableConv2D", "the strides attr shape is wrong.");
                return FAILED;
            }

            set_strides = true;
        }
        if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
            if (attr.ints_size() == 1) {
                pads.push_back(attr.ints(0));
                pads.push_back(attr.ints(0));
                pads.push_back(attr.ints(0));
                pads.push_back(attr.ints(0));
            } else if (attr.ints_size() == 2) {
                pads.push_back(attr.ints(0));
                pads.push_back(attr.ints(0));
                pads.push_back(attr.ints(1));
                pads.push_back(attr.ints(1));
            } else {
                OP_LOGE("DeformableConv2D", "the pads attr shape is wrong.");
                return FAILED;
            }

            set_pads = true;
        }

        if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS) {
            dilations.clear();
            if (attr.ints_size() == 1) {
                dilations.push_back(1);
                dilations.push_back(1);
                dilations.push_back(attr.ints(0));
                dilations.push_back(attr.ints(0));
            } else if (attr.ints_size() == 2) {
                dilations.push_back(1);
                dilations.push_back(1);
                dilations.push_back(attr.ints(0));
                dilations.push_back(attr.ints(1));
            } else {
                OP_LOGE("DeformableConv2D", "the dilations attr shape is wrong.");
                return FAILED;
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
    }
    if (set_pads) {
        op_dest.SetAttr("pads", pads);
    } else {
        OP_LOGE("DeformableConv2D", "onnx DeformableConv2D op has no pads attr.");
    }

    op_dest.SetAttr("dilations", dilations);
    op_dest.SetAttr("groups", groups);
    op_dest.SetAttr("data_format", data_format);
    op_dest.SetAttr("deformable_groups", deformable_groups);
    op_dest.SetAttr("modulated", modulated);

    if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, true) != SUCCESS) {
        return FAILED;
    }

    if (ChangeFormatFromOnnx(op_dest, 1, ge::FORMAT_NCHW, true) != SUCCESS) {
        return FAILED;
    }

    if (ChangeFormatFromOnnx(op_dest, 2, ge::FORMAT_NCHW, true) != SUCCESS) {
        return FAILED;
    }

    if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, false) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("DeformableConv2D")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::DeformableConv2D", "ai.onnx::10::DeformableConv2D", "ai.onnx::11::DeformableConv2D",
                   "ai.onnx::12::DeformableConv2D", "ai.onnx::13::DeformableConv2D", "ai.onnx::14::DeformableConv2D",
                   "ai.onnx::15::DeformableConv2D", "ai.onnx::16::DeformableConv2D"})
    .ParseParamsFn(ParseParamsDeformableConv2D)
    .ImplyType(ImplyType::TVM);
} // namespace domi
