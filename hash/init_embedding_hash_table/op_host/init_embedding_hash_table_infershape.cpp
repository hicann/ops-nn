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
 * \file init_embedding_hash_table_infershape.cpp
 * \brief init_embedding_hash_table
 */

#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
ge::graphStatus InferShape4InitEmbeddingHashTable([[maybe_unused]] gert::InferShapeContext* context)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4InitEmbeddingHashTable([[maybe_unused]] gert::InferDataTypeContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(InitEmbeddingHashTable)
    .InferShape(InferShape4InitEmbeddingHashTable)
    .InferDataType(InferDataType4InitEmbeddingHashTable);
} // namespace ops