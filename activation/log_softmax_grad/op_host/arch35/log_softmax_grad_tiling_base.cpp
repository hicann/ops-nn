/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file log_softmax_grad_tiling_base.cpp
 * \brief
 */

#include "tiling_base/tiling_templates_registry.h"
#include "log_softmax_grad_tiling.h"

namespace optiling
{
ge::graphStatus LogSoftmaxGradTilingBase::GetAndCheckDtypes()
{
    OP_LOGD("LogSoftmaxGradTilingBase", "check dtypes enter");
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto gradDesc = context_->GetInputDesc(CONST_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradDesc);
    auto gradDtype = gradDesc->GetDataType();

    auto xDesc = context_->GetInputDesc(CONST_ONE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDtype_ = gradDesc->GetDataType();

    auto yDesc = context_->GetOutputDesc(CONST_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    yDtype_ = yDesc->GetDataType();

    OP_CHECK_IF((gradDtype != yDtype_) || (xDtype_ != yDtype_),
                    OP_LOGE(context_->GetNodeName(),
                                                    "Input grad dtype [%s], input x dtype [%s] and output y dtype [%s] should be same.",
                                                    ge::TypeUtils::DataTypeToSerialString(gradDtype).c_str(),
                                                    ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str(),
                                                    ge::TypeUtils::DataTypeToSerialString(yDtype_).c_str()),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF((xDtype_ != ge::DT_FLOAT16) && (xDtype_ != ge::DT_FLOAT) && (xDtype_ != ge::DT_BF16),
                    OP_LOGE(
                        context_->GetNodeName(),
                        "Input x dtype is [%s], only support dtype ge::ge::DT_FLOAT16, ge::DT_FLOAT or ge::DT_BF16.",
                        ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str()),
                    return ge::GRAPH_FAILED);

    xDtypeSize_ = xDtype_ == ge::DT_FLOAT ? FLOAT32_BYTES : FLOAT16_BYTES;
    yDtypeSize_ = xDtypeSize_;

    return ge::GRAPH_SUCCESS;
}

// 复用SoftmaxGrad的tiling解析
IMPL_OP_OPTILING(LogSoftmaxGrad)
    .Tiling(TilingForSoftmaxGrad)
    .TilingParse<SoftmaxGradCompileInfo>(TilingPrepareForSoftmaxGrad);  

}  // namespace optiling
