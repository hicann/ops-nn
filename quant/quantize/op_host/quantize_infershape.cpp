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
 * \file quantize_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_elewise_util.h"

using namespace ge;
namespace ops {
namespace {
static const size_t ATTR_INDEX_OF_DST_TYPE = 0;

std::map<std::string, ge::DataType> QUANTIZE_SUPPORT_MAP = {
    {"torch.quint8", ge::DataType::DT_UINT8},     {"torch.qint8", ge::DataType::DT_INT8},
    {"torch.qint32", ge::DataType::DT_UINT32},    {"torch.hifloat8", ge::DataType::DT_HIFLOAT8},
    {"torch.float8_e4m3fn", ge::DataType::DT_FLOAT8_E4M3FN},
    {"torch.float8_e5m2", ge::DataType::DT_FLOAT8_E5M2}};

static ge::graphStatus QuantizeInferDataType(gert::InferDataTypeContext* context) {
    OP_LOGD(context->GetNodeName(), "QuantizeInferDataType begin");
    const auto& attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto dtype = attrs->GetAttrPointer<char>(ATTR_INDEX_OF_DST_TYPE);
    std::string dstDtypeStr = dtype;

    context->SetOutputDataType(0, QUANTIZE_SUPPORT_MAP[dstDtypeStr]);
    OP_LOGD(context->GetNodeName(), "QuantizeInferDataType end");
    return GRAPH_SUCCESS;
};

static ge::graphStatus InferShapeForQuantize(gert::InferShapeContext *context) {
    OP_LOGD("Begin InferShapeForQuantize");
    return Ops::Base::InferShape4Elewise(context);
}
}; // namespace

IMPL_OP_INFERSHAPE(Quantize).InferShape(InferShapeForQuantize).InferDataType(QuantizeInferDataType);
}  // namespace ops