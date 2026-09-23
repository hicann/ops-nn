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
 * \file embedding_hash_table_evict_infershape.cpp
 * \brief embedding_hash_table_evict infer
 */

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace {
constexpr uint32_t INPUT_TABLE_HANDLE_IDX = 0;
constexpr uint32_t INPUT_KEYS_IDX = 1;
constexpr uint32_t INPUT_SAMPLED_VALUES_IDX = 2;
} // namespace

namespace ops {
ge::graphStatus InferDataTypeForEmbeddingHashTableEvict(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to infer data type for EmbeddingHashTableEvict.");

    auto tableHandleDtype = context->GetInputDataType(INPUT_TABLE_HANDLE_IDX);
    OP_CHECK_IF(tableHandleDtype != DT_INT64,
                OP_LOGE(context->GetNodeName(), "table_handle dtype [%s] is not required type: int64.",
                        TypeUtils::DataTypeToSerialString(tableHandleDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto keysDtype = context->GetInputDataType(INPUT_KEYS_IDX);
    OP_CHECK_IF(keysDtype != DT_INT64,
                OP_LOGE(context->GetNodeName(), "keys dtype [%s] is not required type: int64.",
                        TypeUtils::DataTypeToSerialString(keysDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto sampledValuesDtype = context->GetOptionalInputDataType(INPUT_SAMPLED_VALUES_IDX);
    OP_CHECK_IF(sampledValuesDtype != DT_UNDEFINED && sampledValuesDtype != DT_FLOAT,
                OP_LOGE(context->GetNodeName(), "sampled_values dtype [%s] is not required type: float.",
                        TypeUtils::DataTypeToSerialString(sampledValuesDtype).c_str()),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "End to infer data type for EmbeddingHashTableEvict.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EmbeddingHashTableEvict).InferDataType(InferDataTypeForEmbeddingHashTableEvict);
} // namespace ops
