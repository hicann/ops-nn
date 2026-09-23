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
 * \file embedding_hash_table_evict_tiling_arch35.cpp
 * \brief embedding_hash_table_evict_tiling
 */

#include "embedding_hash_table_evict_tiling_arch35.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"

namespace {
#ifdef __DAV_FPGA__
constexpr uint32_t MAX_THREAD_NUM = 128;
#else
constexpr uint32_t MAX_THREAD_NUM = 512;
#endif
constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;

constexpr int64_t INIT_MODE_CONST = 0;
constexpr int64_t INIT_MODE_RANDOM = 1;

constexpr uint32_t INPUT_KEYS_IDX = 1;
constexpr uint32_t INPUT_SAMPLED_VALUES_IDX = 2;

constexpr uint32_t ATTR_TABLE_CAP_IDX = 0;
constexpr uint32_t ATTR_EMBEDDING_DIM_IDX = 1;
constexpr uint32_t ATTR_INIT_MODE_IDX = 2;
constexpr uint32_t ATTR_CONST_VAL_IDX = 3;
} // namespace

namespace optiling {
using namespace Ops::Base;

ge::graphStatus TilingForEvict(gert::TilingContext* context)
{
    const auto* compileInfo = reinterpret_cast<const EvictCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto tiling = EvictTilingData();

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto tableCap = attrs->GetAttrPointer<int64_t>(ATTR_TABLE_CAP_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableCap);
    tiling.set_tableCap(static_cast<int64_t>(*tableCap));

    auto embeddingDim = attrs->GetAttrPointer<int64_t>(ATTR_EMBEDDING_DIM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, embeddingDim);
    tiling.set_embeddingDim(static_cast<int64_t>(*embeddingDim));

    auto constVal = attrs->GetAttrPointer<float>(ATTR_CONST_VAL_IDX);
    tiling.set_constVal(constVal == nullptr ? 0.0f : static_cast<float>(*constVal));

    auto initMode = attrs->GetAttrPointer<char>(ATTR_INIT_MODE_IDX);
    int64_t mode = (initMode != nullptr && std::strcmp(initMode, "random") == 0) ? INIT_MODE_RANDOM : INIT_MODE_CONST;
    tiling.set_initMode(mode);

    const auto* keysShape = context->GetInputShape(INPUT_KEYS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keysShape);
    int64_t keyNum = keysShape->GetStorageShape().GetShapeSize();
    tiling.set_keyNum(keyNum);

    uint32_t usedThreadNum = std::min<uint32_t>(compileInfo->maxThreadNum, MAX_THREAD_NUM);
    tiling.set_usedThreadNum(usedThreadNum);

    auto* keysDesc = context->GetInputDesc(INPUT_KEYS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keysDesc);
    auto keysDtype = keysDesc->GetDataType();
    OP_CHECK_IF(keysDtype != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "keys",
                                          ge::TypeUtils::DataTypeToSerialString(keysDtype).c_str(), "int64"),
                return ge::GRAPH_FAILED);

    auto* sampledValuesDesc = context->GetOptionalInputDesc(INPUT_SAMPLED_VALUES_IDX);
    if (mode == INIT_MODE_RANDOM) {
        OP_CHECK_NULL_WITH_CONTEXT(context, sampledValuesDesc);
    }
    if (sampledValuesDesc != nullptr) {
        auto sampledValuesDtype = sampledValuesDesc->GetDataType();
        OP_CHECK_IF(
            sampledValuesDtype != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "sampled_values",
                                      ge::TypeUtils::DataTypeToSerialString(sampledValuesDtype).c_str(), "float"),
            return ge::GRAPH_FAILED);
    }
    if (mode == INIT_MODE_RANDOM) {
        auto* sampledValuesShape = context->GetOptionalInputShape(INPUT_SAMPLED_VALUES_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context, sampledValuesShape);
        int64_t expectSampledValuesShapeSize = keyNum * static_cast<int64_t>(*embeddingDim);
        int64_t sampledValuesShapeSize = sampledValuesShape->GetStorageShape().GetShapeSize();
        OP_CHECK_IF(sampledValuesShapeSize != expectSampledValuesShapeSize,
                    OP_LOGE_FOR_INVALID_SHAPESIZE(
                        context->GetNodeName(), "sampled_values", std::to_string(sampledValuesShapeSize).c_str(),
                        ("init_mode is random, sampled_values shape size should equal keys shape size * "
                         "embedding_dim, which is " +
                         std::to_string(expectSampledValuesShapeSize))
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }

    context->SetTilingKey(0);
    context->SetBlockDim(
        std::min<uint32_t>(static_cast<uint32_t>(CeilDiv<int64_t>(keyNum, usedThreadNum)), compileInfo->coreNumAiv));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = ASCENDC_TOOLS_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForEvict(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<EvictCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    compileInfo->maxThreadNum = GetSimtMaxThreadNum(context);
    OP_CHECK_IF((compileInfo->maxThreadNum <= 0),
                OP_LOGE(context->GetNodeName(), "Failed to get valid maxThreadNum in TilingParse func."),
                return ge::GRAPH_FAILED);

    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNumAiv <= 0),
                OP_LOGE(context->GetNodeName(), "Failed to get valid coreNumAiv in TilingParse func."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EmbeddingHashTableEvict).Tiling(TilingForEvict).TilingParse<EvictCompileInfo>(TilingPrepareForEvict);
} // namespace optiling
