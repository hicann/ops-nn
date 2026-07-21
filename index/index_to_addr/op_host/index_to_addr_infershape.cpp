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
 * \file index_to_addr_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t kAttrBlockSizeIdx = 1U;
constexpr size_t kBlockSizeDimNum = 2U;
constexpr size_t kOutputColNum = 4U;
} // namespace

static ge::graphStatus InferShapeForIndexToAddr(gert::InferShapeContext* context)
{
    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto blockSize = attrs->GetAttrPointer<gert::ContinuousVector>(kAttrBlockSizeIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockSize);
    OP_CHECK_IF(blockSize->GetSize() != kBlockSizeDimNum,
                OP_LOGE(context->GetNodeName(), "Attr block_size size[%zu] must be 2.", blockSize->GetSize()),
                return ge::GRAPH_FAILED);

    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    const auto blockSizeData = static_cast<const int64_t*>(blockSize->GetData());
    outShape->SetDimNum(kBlockSizeDimNum);
    outShape->SetDim(0, blockSizeData[0]);
    outShape->SetDim(1, kOutputColNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForIndexToAddr(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IndexToAddr).InferShape(InferShapeForIndexToAddr).InferDataType(InferDataTypeForIndexToAddr);
} // namespace ops
