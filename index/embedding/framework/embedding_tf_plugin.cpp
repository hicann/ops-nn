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
 * \file embedding_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {
// The following 8 ops use AutoMappingFn for simple mapping and can be migrated directly.
// The remaining 7 ops use custom AutoMappingFnHashMapInputs/AutoMappingFnHashMapOutputs and are not migrated.
// LookupTableFind maps directly from the TensorFlow op of the same name; auto operator mapping suffices.
REGISTER_CUSTOM_OP("LookupTableFind")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LookupTableFindV2")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::AI_CPU);

// FakeRemoteLookupUniqued maps directly from the TensorFlow op of the same name; auto operator mapping suffices.
REGISTER_CUSTOM_OP("FakeRemoteLookupUniqued")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FakeRemoteLookupUniqued")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::TVM);

// InitEmbeddingHashmapV2 maps directly from the TensorFlow op of the same name; auto operator mapping suffices.
REGISTER_CUSTOM_OP("InitEmbeddingHashmapV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("InitEmbeddingHashmapV2")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::TVM);

// DeinitEmbeddingHashmapV2 maps directly from the TensorFlow op of the same name; auto operator mapping suffices.
REGISTER_CUSTOM_OP("DeinitEmbeddingHashmapV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DeinitEmbeddingHashmapV2")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::TVM);

// TableToResourceV2 maps directly from the TensorFlow op of the same name; auto operator mapping suffices.
REGISTER_CUSTOM_OP("TableToResourceV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TableToResourceV2")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::TVM);

// EmbeddingHashmapSize maps directly from the TensorFlow op of the same name; auto operator mapping suffices.
REGISTER_CUSTOM_OP("EmbeddingHashmapSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("EmbeddingHashmapSize")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::TVM);

// EmbeddingHashmapFileSize maps directly from the TensorFlow op of the same name; auto operator mapping suffices.
REGISTER_CUSTOM_OP("EmbeddingHashmapFileSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("EmbeddingHashmapFileSize")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::TVM);

// EmbeddingHashTableApplyAdamW maps directly from the TensorFlow op of the same name; auto operator mapping suffices.
REGISTER_CUSTOM_OP("EmbeddingHashTableApplyAdamW")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("EmbeddingHashTableApplyAdamW")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::TVM);
} // namespace domi
