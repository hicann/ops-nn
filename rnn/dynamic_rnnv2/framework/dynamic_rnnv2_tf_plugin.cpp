/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_rnnv2_tf_plugin.cpp
 * \brief DynamicRnnV2 / DynamicRNNV2 TensorFlow plugin mapping.
 */
#include "register/register.h"

namespace domi {
static Status ParseParamsDynamicRNN(const ge::Operator& op_src, ge::Operator& op_dest)
{
    AutoMappingByOpFn(op_src, op_dest);
    op_dest.SetAttr("is_misplaced", true);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("DynamicRNN")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DynamicRnnV2")
    .ParseParamsByOperatorFn(ParseParamsDynamicRNN)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("DynamicRNNV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DynamicRnnv2WithoutSeqlength")
    .ParseParamsByOperatorFn(ParseParamsDynamicRNN)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("DynamicRNNV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DynamicRnnv2WithSeqlength")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::TVM);
} // namespace domi
