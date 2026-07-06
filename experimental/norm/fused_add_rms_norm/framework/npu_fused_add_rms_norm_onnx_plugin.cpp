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
 * \file npu_fused_add_rms_norm_onnx_plugin.cpp
 * \brief ONNX plugin for FusedAddRmsNorm.
 */
#include "onnx_common.h"

namespace domi {
namespace {
using NodeProto = ge::onnx::NodeProto;
constexpr float kDefaultEpsilon = 1e-6F;
constexpr float kDefaultScale = 1.0F;

float GetOptionalFloatAttr(const NodeProto &node, const char *attrName, float defaultValue)
{
    for (const auto &attr : node.attribute()) {
        if (attr.name() != attrName) {
            continue;
        }
        if (attr.type() == ge::onnx::AttributeProto::FLOAT) {
            return attr.f();
        }
        break;
    }
    return defaultValue;
}
}  // namespace

static Status ParseParamsNpuFusedAddRmsNorm(const Message *opSrc, ge::Operator &opDest)
{
    const auto *node = dynamic_cast<const NodeProto *>(opSrc);
    if (node == nullptr) {
        OP_LOGE(GetOpName(opDest).c_str(), "Convert op source to ONNX NodeProto failed.");
        return FAILED;
    }

    opDest.SetAttr("epsilon", GetOptionalFloatAttr(*node, "epsilon", kDefaultEpsilon));
    opDest.SetAttr("scale", GetOptionalFloatAttr(*node, "scale", kDefaultScale));
    return SUCCESS;
}

// register npu_fused_add_rms_norm op info to GE
REGISTER_CUSTOM_OP("FusedAddRmsNorm")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUFusedAddRmsNorm"),
        ge::AscendString("ai.onnx::11::NPUFusedAddRmsNorm"),
        ge::AscendString("ai.onnx::12::NPUFusedAddRmsNorm"),
        ge::AscendString("ai.onnx::13::NPUFusedAddRmsNorm"),
        ge::AscendString("ai.onnx::14::NPUFusedAddRmsNorm"),
        ge::AscendString("ai.onnx::15::NPUFusedAddRmsNorm"),
        ge::AscendString("ai.onnx::16::NPUFusedAddRmsNorm"),
        ge::AscendString("ai.onnx::17::NPUFusedAddRmsNorm"),
        ge::AscendString("ai.onnx::18::NPUFusedAddRmsNorm")})
    .ParseParamsFn(ParseParamsNpuFusedAddRmsNorm)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
