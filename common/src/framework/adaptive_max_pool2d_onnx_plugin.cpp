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
 * \file adaptive_max_pool2d__plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {
static const uint32_t ATTR_LOOP_SIZE = 2;

static Status ParseParamsAdaptiveMaxPool2d(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    std::vector<int> v_output_size = {};
    bool set_output_size_flag = false;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "output_size" && attr.type() == ge::onnx::AttributeProto::INTS) {
            if (attr.ints_size() == ATTR_LOOP_SIZE) {
                for (int i = 0; i < attr.ints_size(); i++) {
                    v_output_size.push_back(attr.ints(i));
                }
            } else {
                OP_LOGE(GetOpName(op_dest).c_str(), "length of output_size must be 2.");
            }
            set_output_size_flag = true;
        }
    }

    if (set_output_size_flag) {
        op_dest.SetAttr("output_size", v_output_size);
    } else {
        OP_LOGE(GetOpName(op_dest).c_str(), "onnx AdaptiveMaxPool2d op has no output_size attr.");
    }
    return SUCCESS;
}

// register AdaptiveMaxPool2d op info to GE
REGISTER_CUSTOM_OP("AdaptiveMaxPool2d")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("aten::adaptive_max_pool2d"), 
                   ge::AscendString("ai.onnx::8::AdaptiveMaxPool2d"),
                   ge::AscendString("ai.onnx::9::AdaptiveMaxPool2d"), 
                   ge::AscendString("ai.onnx::10::AdaptiveMaxPool2d"),
                   ge::AscendString("ai.onnx::11::AdaptiveMaxPool2d"), 
                   ge::AscendString("ai.onnx::12::AdaptiveMaxPool2d"),
                   ge::AscendString("ai.onnx::13::AdaptiveMaxPool2d"), 
                   ge::AscendString("ai.onnx::14::AdaptiveMaxPool2d"),
                   ge::AscendString("ai.onnx::15::AdaptiveMaxPool2d"), 
                   ge::AscendString("ai.onnx::16::AdaptiveMaxPool2d")})
    .ParseParamsFn(ParseParamsAdaptiveMaxPool2d)
    .ImplyType(ImplyType::TVM);
} // namespace domi
