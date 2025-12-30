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
static const float EPSILON_DEFAULT = 1e-05;

static Status ParseParamsInstanceNormalization(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    float epsilon = EPSILON_DEFAULT;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "epsilon" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            epsilon = attr.f();
        }
    }
    op_dest.SetAttr("epsilon", epsilon);
    op_dest.SetAttr("data_format", "NCHW");

    return SUCCESS;
}
// register Add op info to GE
REGISTER_CUSTOM_OP("InstanceNorm")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::InstanceNormalization"), 
                   ge::AscendString("ai.onnx::9::InstanceNormalization"), 
                   ge::AscendString("ai.onnx::10::InstanceNormalization"),
                   ge::AscendString("ai.onnx::11::InstanceNormalization"), 
                   ge::AscendString("ai.onnx::12::InstanceNormalization"),
                   ge::AscendString("ai.onnx::13::InstanceNormalization"), 
                   ge::AscendString("ai.onnx::14::InstanceNormalization"),
                   ge::AscendString("ai.onnx::15::InstanceNormalization"), 
                   ge::AscendString("ai.onnx::16::InstanceNormalization"),
                   ge::AscendString("ai.onnx::17::InstanceNormalization"), 
                   ge::AscendString("ai.onnx::18::InstanceNormalization")})
    .ParseParamsFn(ParseParamsInstanceNormalization)
    .ImplyType(ImplyType::TVM);
} // namespace domi