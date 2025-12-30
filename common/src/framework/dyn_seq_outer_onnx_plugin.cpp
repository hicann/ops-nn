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
 * \file dyn_seq_outer.cc
 * \brief
 */
#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsDynSeqOuter(const Message *op_src, ge::Operator &op_dest) {
  return SUCCESS;
}

REGISTER_CUSTOM_OP("DynSeqOuter")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::8::DynSeqOuter"),
                 ge::AscendString("ai.onnx::9::DynSeqOuter"),
                 ge::AscendString("ai.onnx::10::DynSeqOuter"),
                 ge::AscendString("ai.onnx::11::DynSeqOuter"),
                 ge::AscendString("ai.onnx::12::DynSeqOuter"),
                 ge::AscendString("ai.onnx::13::DynSeqOuter"),
                 ge::AscendString("ai.onnx::14::DynSeqOuter"),
                 ge::AscendString("ai.onnx::15::DynSeqOuter"),
                 ge::AscendString("ai.onnx::16::DynSeqOuter")})
  .ParseParamsFn(ParseParamsDynSeqOuter)
  .ImplyType(ImplyType::TVM);
} // namespace domi
