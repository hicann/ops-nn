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
 * \file batch_normalization_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {

static Status ParseParamsBatchNorm(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    float epsilon = 0.00001;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "epsilon" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            epsilon = attr.f();
            break;
        }
    }
    op_dest.SetAttr("epsilon", epsilon);

    const std::string data_format = "NCHW";
    op_dest.SetAttr("data_format", data_format);

    const bool is_training = false;
    op_dest.SetAttr("is_training", is_training);
    op_dest.SetAttr("onnx", "onnx");
    return SUCCESS;
}

// register ReduceMean op info to GE
REGISTER_CUSTOM_OP("BatchNorm")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::BatchNormalization"), 
                   ge::AscendString("ai.onnx::9::BatchNormalization"), 
                   ge::AscendString("ai.onnx::10::BatchNormalization"),
                   ge::AscendString("ai.onnx::11::BatchNormalization"), 
                   ge::AscendString("ai.onnx::12::BatchNormalization"), 
                   ge::AscendString("ai.onnx::13::BatchNormalization"),
                   ge::AscendString("ai.onnx::14::BatchNormalization"), 
                   ge::AscendString("ai.onnx::15::BatchNormalization"), 
                   ge::AscendString("ai.onnx::16::BatchNormalization"),
                   ge::AscendString("ai.onnx::17::BatchNormalization"), 
                   ge::AscendString("ai.onnx::18::BatchNormalization")})
    .ParseParamsFn(ParseParamsBatchNorm)
    .ImplyType(ImplyType::TVM);
} // namespace domi
