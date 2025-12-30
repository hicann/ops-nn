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
#include "nlohmann/json.hpp"

using namespace ge;
using json = nlohmann::json;
namespace domi {
static Status ParseOnnxParamsLSTMP(const ge::Operator& op_src, ge::Operator& op_dest)
{
    return SUCCESS;
}

REGISTER_CUSTOM_OP("LSTMP")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::LSTMP"), 
                   ge::AscendString("ai.onnx::9::LSTMP"),
                   ge::AscendString("ai.onnx::10::LSTMP"), 
                   ge::AscendString("ai.onnx::11::LSTMP"),
                   ge::AscendString("ai.onnx::12::LSTMP"), 
                   ge::AscendString("ai.onnx::13::LSTMP"),
                   ge::AscendString("ai.onnx::14::LSTMP"), 
                   ge::AscendString("ai.onnx::15::LSTMP"),
                   ge::AscendString("ai.onnx::16::LSTMP"), 
                   ge::AscendString("ai.onnx::18::LSTMP")})
    .ParseParamsByOperatorFn(ParseOnnxParamsLSTMP)
    .ImplyType(ImplyType::TVM);
} // namespace domi