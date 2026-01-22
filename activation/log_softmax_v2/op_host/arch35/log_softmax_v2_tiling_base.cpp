/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file log_softmax_v2_tiling_base.cpp
 * \brief
 */

#include "log_softmax_v2_tiling.h"

namespace optiling
{
ge::graphStatus LogSoftmaxV2TilingBase::GetAndCheckDtypes()
{
    OP_LOGD("LogSoftmaxV2TilingBase", "check dtypes enter");
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto xDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDtype_ = xDesc->GetDataType();

    auto yDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    yDtype_ = yDesc->GetDataType();

    OP_CHECK_IF(xDtype_ != yDtype_,
                    OP_LOGE(context_->GetNodeName(),
                                                    "Input x dtype [%s] and Output y dtype [%s] should be same.",
                                                    ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str(),
                                                    ge::TypeUtils::DataTypeToSerialString(yDtype_).c_str()),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDtype_ != ge::DT_FLOAT16 && xDtype_ != ge::DT_FLOAT && xDtype_ != ge::DT_BF16,
                    OP_LOGE(
                        context_->GetNodeName(),
                        "Input x dtype is [%s], only support dtype ge::ge::DT_FLOAT16, ge::DT_FLOAT or ge::DT_BF16.",
                        ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str()),
                    return ge::GRAPH_FAILED);

    xDtypeSize_ = xDtype_ == ge::DT_FLOAT ? FLOAT32_BYTES : FLOAT16_BYTES;
    yDtypeSize_ = xDtypeSize_;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LogSoftmaxV2)
    .Tiling(TilingForSoftmaxV2)
    .TilingParse<SoftmaxV2CompileInfo>(TilingPrepareForSoftmaxV2);
}  // namespace optiling
