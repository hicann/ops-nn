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

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsAdaCast(const Message* op_src, ge::Operator& op_dst)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dst).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    int pixel_value = 65535;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "pixel" && attr.type() == ge::onnx::AttributeProto::INT) {
            pixel_value = attr.i();
        }
    }
    op_dst.SetAttr("pixel", pixel_value);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("AdaCast")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::AdaCast"), 
                   ge::AscendString("ai.onnx::9::AdaCast"), 
                   ge::AscendString("ai.onnx::10::AdaCast"), 
                   ge::AscendString("ai.onnx::11::AdaCast"),
                   ge::AscendString("ai.onnx::12::AdaCast"), 
                   ge::AscendString("ai.onnx::13::AdaCast"), 
                   ge::AscendString("ai.onnx::14::AdaCast"), 
                   ge::AscendString("ai.onnx::15::AdaCast"),
                   ge::AscendString("ai.onnx::16::AdaCast"), 
                   ge::AscendString("ai.onnx::17::AdaCast"), 
                   ge::AscendString("ai.onnx::18::AdaCast")})
    .ParseParamsFn(ParseParamsAdaCast)
    .ImplyType(ImplyType::TVM);
} // namespace domi