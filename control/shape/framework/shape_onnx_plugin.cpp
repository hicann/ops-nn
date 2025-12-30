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
 * \file shape_plugin.cpp
 * \brief
 */

#include "onnx_common.h"

namespace domi {

static Status  ParseParamsShape(const Message* op_src, ge::Operator& op_dest) {
  op_dest.SetAttr("dtype", static_cast<uint32_t>(ge::DT_INT64));
  return SUCCESS;
}

// register Add op info to GE
REGISTER_CUSTOM_OP("Shape")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Shape"),
                   ge::AscendString("ai.onnx::9::Shape"),
                   ge::AscendString("ai.onnx::10::Shape"),
                   ge::AscendString("ai.onnx::11::Shape"),
                   ge::AscendString("ai.onnx::12::Shape"),
                   ge::AscendString("ai.onnx::13::Shape"),
                   ge::AscendString("ai.onnx::14::Shape"),
                   ge::AscendString("ai.onnx::15::Shape"),
                   ge::AscendString("ai.onnx::16::Shape"),
                   ge::AscendString("ai.onnx::17::Shape"),
                   ge::AscendString("ai.onnx::18::Shape")})
    .ParseParamsFn(ParseParamsShape)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi