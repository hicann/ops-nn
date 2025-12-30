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
 * \file gru_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {

static Status ParseParamsCommonGRU(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);

    if (nullptr == node) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int hidden_size = 0;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "hidden_size" && attr.type() == ge::onnx::AttributeProto::INT) {
            hidden_size = attr.i();
        }
        if (attr.name() == "direction" && attr.type() == ge::onnx::AttributeProto::STRING) {
            op_dest.SetAttr("direction", attr.s());
        }
    }
    op_dest.SetAttr("hidden_size", hidden_size);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("CommonGRU")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::GRU"), 
                   ge::AscendString("ai.onnx::9::GRU"), 
                   ge::AscendString("ai.onnx::10::GRU"), 
                   ge::AscendString("ai.onnx::11::GRU"), 
                   ge::AscendString("ai.onnx::12::GRU"),
                   ge::AscendString("ai.onnx::13::GRU"), 
                   ge::AscendString("ai.onnx::14::GRU"), 
                   ge::AscendString("ai.onnx::15::GRU"), 
                   ge::AscendString("ai.onnx::16::GRU"), 
                   ge::AscendString("ai.onnx::17::GRU"),
                   ge::AscendString("ai.onnx::18::GRU")})
    .ParseParamsFn(ParseParamsCommonGRU)
    .ImplyType(ImplyType::TVM);

} // namespace domi