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
 * \file embedding_hash_table_import_tiling_arch35.cpp
 * \brief
 */
#include "embedding_hash_table_import_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
static constexpr uint64_t IN_TABLE_HANDLES_IDX = 0;
static constexpr uint64_t IN_EMBEDDING_DIMS_IDX = 1;
static constexpr uint64_t IN_BUCKET_SIZES_IDX = 2;
static constexpr uint64_t IN_KEYS_IDX = 3;
static constexpr uint64_t IN_COUNTERS_IDX = 4;
static constexpr uint64_t IN_FILTER_FLAGS_IDX = 5;
static constexpr uint64_t IN_VALUES_IDX = 6;
static constexpr int64_t SIMT_DCACHE_SIZE = static_cast<int64_t>(32 * 1024);

ge::graphStatus EmbeddingHashTableImportTiling::GetPlatformInfo()
{
    auto compileInfoPtr = reinterpret_cast<const EmbeddingHashTableImportCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((compileInfoPtr->aivNum <= 0), OP_LOGE(opName, "embeddingHashTableImportTiling fail to get coreNum."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((compileInfoPtr->ubSize <= REGBASE_CCEC_CACHE_SIZE),
        OP_LOGE(opName, "ub size less than REGBASE_CCEC_CACHE_SIZE Size, please check."),
        return ge::GRAPH_FAILED);
    coreNum_ = static_cast<int64_t>(compileInfoPtr->aivNum);
    ubSize_ = static_cast<int64_t>(compileInfoPtr->ubSize - REGBASE_CCEC_CACHE_SIZE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableImportTiling::GetShapeAttrsInfo()
{
    auto resIn = GetInputInfo();
    if (resIn != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto resInList = GetInputInfoOfTensorList();
    if (resInList != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto resData = CheckInputData();
    if (resData != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableImportTiling::GetInputInfo()
{
    // check table_handles
    auto tableHandleDesc = context_->GetRequiredInputDesc(IN_TABLE_HANDLES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tableHandleDesc);
    auto tableHandleDtype = tableHandleDesc->GetDataType();
    OP_CHECK_IF((tableHandleDtype != ge::DataType::DT_INT64),
        OP_LOGE(opName, "table handle dtype should be int64, please check."),
        return ge::GRAPH_FAILED);

    // check embedding_dims
    auto embeddingDimsDesc = context_->GetRequiredInputDesc(IN_EMBEDDING_DIMS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, embeddingDimsDesc);
    auto embeddingDimsDtype = embeddingDimsDesc->GetDataType();
    OP_CHECK_IF((embeddingDimsDtype != ge::DataType::DT_INT64),
        OP_LOGE(opName, "embedding dims dtype should be int64, please check."),
        return ge::GRAPH_FAILED);

    // check bucket_sizes
    auto bucketSizesDesc = context_->GetRequiredInputDesc(IN_BUCKET_SIZES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, bucketSizesDesc);
    auto bucketSizesDtype = bucketSizesDesc->GetDataType();
    OP_CHECK_IF((bucketSizesDtype != ge::DataType::DT_INT64),
        OP_LOGE(opName, "bucket sizes dtype should be int64, please check."),
        return ge::GRAPH_FAILED);

    auto bucketSizesTensor = context_->GetRequiredInputTensor(IN_BUCKET_SIZES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, bucketSizesTensor);
    tablesCount_ = static_cast<int64_t>(bucketSizesTensor->GetShapeSize());
    OP_CHECK_IF(tablesCount_ <= 0, OP_LOGE(opName, "tables count should bigger than 0, but got %ld.", tablesCount_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableImportTiling::GetInputInfoOfTensorList()
{
    // check keys
    auto keysDesc = context_->GetRequiredInputDesc(IN_KEYS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keysDesc);
    auto keysDtype = keysDesc->GetDataType();
    OP_CHECK_IF((keysDtype != ge::DataType::DT_INT64),
        OP_LOGE(opName, "keys dtype should be int64, please check."), return ge::GRAPH_FAILED);

    // check counters
    auto countersDesc = context_->GetRequiredInputDesc(IN_COUNTERS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, countersDesc);
    auto countersDtype = countersDesc->GetDataType();
    OP_CHECK_IF((countersDtype != ge::DataType::DT_UINT64),
        OP_LOGE(opName, "counters dtype should be uint64, please check."), return ge::GRAPH_FAILED);

    // check filter_flags
    auto filterFlagsDesc = context_->GetRequiredInputDesc(IN_FILTER_FLAGS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, filterFlagsDesc);
    auto filterFlagDtype = filterFlagsDesc->GetDataType();
    OP_CHECK_IF((filterFlagDtype != ge::DataType::DT_UINT8),
        OP_LOGE(opName, "filter flags dtype should be uint8, please check."),
        return ge::GRAPH_FAILED);

    // check values
    auto valuesDesc = context_->GetRequiredInputDesc(IN_VALUES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valuesDesc);
    auto valuesDtype = valuesDesc->GetDataType();
    OP_CHECK_IF((valuesDtype != ge::DataType::DT_FLOAT),
        OP_LOGE(opName, "values dtype should be float32, please check."),
        return ge::GRAPH_FAILED);
    bitWidth_ = static_cast<int64_t>(ge::GetSizeByDataType(valuesDtype));

    auto valuesTensor = context_->GetRequiredInputTensor(IN_VALUES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valuesTensor);
    valuesNum_ = static_cast<int64_t>(valuesTensor->GetShapeSize());
    OP_CHECK_IF(valuesNum_ <= 0, OP_LOGE(opName, "values dim number should bigger than 0, but got %ld.", valuesNum_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableImportTiling::CheckInputData()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableImportTiling::DoOpTiling()
{
    blockNum_ = static_cast<int64_t>(coreNum_);
    m_tilingData_.set_bitWidth(bitWidth_);
    m_tilingData_.set_tablesCount(tablesCount_);
    m_tilingData_.set_blockNum(blockNum_);
    m_tilingData_.set_coreNum(coreNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableImportTiling::DoLibApiTiling()
{
    OP_LOGD(context_->GetNodeName(), "bitWidth = %ld, tablesCount = %ld, blockNum = %ld", m_tilingData_.get_bitWidth(),
        m_tilingData_.get_tablesCount(), m_tilingData_.get_blockNum());

    return ge::GRAPH_SUCCESS;
}

uint64_t EmbeddingHashTableImportTiling::GetTilingKey() const
{
    uint64_t tilingKey = TILINGKEY_INIT_VALUE;
    OP_LOGD(context_->GetNodeName(), "tilingKey = %lu.", tilingKey);
    return tilingKey;
}

ge::graphStatus EmbeddingHashTableImportTiling::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableImportTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(m_tilingData_.get_blockNum());
    context_->SetLocalMemorySize(ubSize_ - SIMT_DCACHE_SIZE);
    if (m_tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    m_tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(m_tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4EmbeddingHashTableImport(gert::TilingContext *context)
{
    EmbeddingHashTableImportTiling tilingObj(context);
    return tilingObj.DoTiling();
}

static ge::graphStatus TilingPrepare4EmbeddingHashTableImport(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "op [EmbeddingHashTableImport] tiling start.");
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<EmbeddingHashTableImportCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EmbeddingHashTableImport)
    .InputsDataDependency({IN_BUCKET_SIZES_IDX})
    .Tiling(Tiling4EmbeddingHashTableImport)
    .TilingParse<EmbeddingHashTableImportCompileInfo>(TilingPrepare4EmbeddingHashTableImport);
} // namespace optiling