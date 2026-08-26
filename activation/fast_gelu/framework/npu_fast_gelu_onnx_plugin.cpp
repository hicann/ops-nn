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
 * \file npu_fast_gelu.cc
 * \brief
 */

#include "plugin_util.h"
#include "register/register.h"
#include "graph/operator.h"

namespace domi {
static Status ParseParamsNpuFastGelu(const ge::Operator&, ge::Operator&) { return SUCCESS; }

REGISTER_CUSTOM_OP("FastGelu")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUFastGelu"), ge::AscendString("ai.onnx::11::NPUFastGelu"),
                   ge::AscendString("ai.onnx::12::NPUFastGelu"), ge::AscendString("ai.onnx::13::NPUFastGelu"),
                   ge::AscendString("ai.onnx::14::NPUFastGelu"), ge::AscendString("ai.onnx::15::NPUFastGelu"),
                   ge::AscendString("ai.onnx::16::NPUFastGelu"), ge::AscendString("ai.onnx::17::NPUFastGelu"),
                   ge::AscendString("ai.onnx::18::NPUFastGelu")})
    .ParseParamsByOperatorFn(ParseParamsNpuFastGelu)
    .ImplyType(ImplyType::TVM);
} // namespace domi
