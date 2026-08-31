/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "plugin_util.h"
#include "register/register.h"
#include "graph/operator.h"

namespace domi {
static Status ParseParamsBatchMatMul(const ge::Operator&, ge::Operator& opDst)
{
    // onnx doesn't have transpose attr
    opDst.SetAttr("adj_x1", false);
    opDst.SetAttr("adj_x2", false);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("BatchMatMul")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::BatchMatMul"), ge::AscendString("ai.onnx::9::BatchMatMul"),
                   ge::AscendString("ai.onnx::10::BatchMatMul"), ge::AscendString("ai.onnx::11::BatchMatMul"),
                   ge::AscendString("ai.onnx::12::BatchMatMul"), ge::AscendString("ai.onnx::13::BatchMatMul"),
                   ge::AscendString("ai.onnx::14::BatchMatMul"), ge::AscendString("ai.onnx::15::BatchMatMul"),
                   ge::AscendString("ai.onnx::16::BatchMatMul")})
    .ParseParamsByOperatorFn(ParseParamsBatchMatMul)
    .ImplyType(ImplyType::TVM);
} // namespace domi
