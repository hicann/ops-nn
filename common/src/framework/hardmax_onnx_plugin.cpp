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
static Status parse_params_hard_max(const Message *op_src, ge::Operator &op_dest)
{
    const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    int axis = -1;
    for (auto attr : node->attribute()) {
        if (attr.name() == "axis") {
            axis = attr.i();
        }
    }
    op_dest.SetAttr("axis", axis);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("HardMax")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Hardmax"),
                   ge::AscendString("ai.onnx::9::Hardmax"),
                   ge::AscendString("ai.onnx::10::Hardmax"),
                   ge::AscendString("ai.onnx::11::Hardmax"),
                   ge::AscendString("ai.onnx::12::Hardmax"),
                   ge::AscendString("ai.onnx::13::Hardmax"),
                   ge::AscendString("ai.onnx::14::Hardmax"),
                   ge::AscendString("ai.onnx::15::Hardmax"),
                   ge::AscendString("ai.onnx::16::Hardmax"),
                   ge::AscendString("ai.onnx::17::Hardmax"),
                   ge::AscendString("ai.onnx::18::Hardmax")})
    .ParseParamsFn(parse_params_hard_max)
    .ImplyType(ImplyType::TVM);
}
