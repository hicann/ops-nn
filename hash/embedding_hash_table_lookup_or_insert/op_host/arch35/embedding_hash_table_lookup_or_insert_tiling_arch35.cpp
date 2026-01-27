/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_hash_table_lookup_or_insert_tiling_arch35.cpp
 * \brief embedding_hash_table_lookup_or_insert_tiling
 */
#include <string>
#include <sstream>
#include <unordered_set>
#include "embedding_hash_table_lookup_or_insert_tiling_arch35.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "tiling/tiling_api.h"

namespace {
inline uint32_t nextPowerOf2(uint32_t x)
{
    if (x == 0) {
        return 1;
    }
    x--;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x + 1;
}

// SIMT最大线程数
#ifdef __DAV_FPGA__
constexpr uint32_t MAX_THREAD_NUM = 128;
#else
constexpr uint32_t MAX_THREAD_NUM = 512;
#endif
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t MIN_UB_BLOCK_SIZE = 32;

// 输入属性的索引
constexpr uint32_t INPUT_BUCKET_SIZE_IDX = 0;
constexpr uint32_t INPUT_EMBEDDINGDIM_IDX = 1;
constexpr uint32_t INPUT_FILTER_MODE_IDX = 2;
constexpr uint32_t INPUT_FILTER_FREQ_IDX = 3;
constexpr uint32_t INPUT_DEFAULT_KEY_OR_VALUE_IDX = 4;
constexpr uint32_t INPUT_DEFAULT_KEY_IDX = 5;
constexpr uint32_t INPUT_DEFAULT_VALUE_IDX = 6;
constexpr uint32_t INPUT_FILTER_KEY_FLAG_IDX = 7;
constexpr uint32_t INPUT_filter_KEY_IDX = 8;

// 框架配置
constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = 16777216; // 16 * 1024 * 1024;

// TilingKey
constexpr uint32_t EMBEDDING_HASH_TABLE_LOOKUP_OR_INSERT_TILING_KEY_GENERAL = 1001;
constexpr uint32_t EMBEDDING_HASH_TABLE_LOOKUP_OR_INSERT_TILING_KEY_OPT_DIM = 1002;
}; // namespace

namespace optiling {
using namespace Ops::Base;
struct EmbeddingHashTableLookupOrInsertCompileInfo {
    uint32_t maxThreadNum;
    uint32_t coreNumAiv;
};

inline void LogLookupOrInsertTilingData(const gert::TilingContext* context, LookupOrInsertTilingData* pTiling)
{
    std::stringstream stream;
    stream << "\n[EmbeddingHashTableLookupOrInsertTilingData]\n";
    stream << "tiling->size = " << static_cast<int64_t>(pTiling->get_size()) << "\n";
    stream << "tiling->size = " << static_cast<int64_t>(pTiling->get_size()) << "\n";
    stream << "tiling->embeddingDim = " << static_cast<int64_t>(pTiling->get_embeddingDim()) << "\n";
    stream << "tiling->filterFreq = " << static_cast<int64_t>(pTiling->get_filterFreq()) << "\n";
    stream << "tiling->keyNum = " << static_cast<int64_t>(pTiling->get_keyNum()) << "\n";
    stream << "tiling->filterMode = " << static_cast<int64_t>(pTiling->get_filterMode()) << "\n";
    stream << "tiling->defaultKeyOrValue = " << static_cast<int64_t>(pTiling->get_defaultKeyOrValue()) << "\n";
    stream << "tiling->defaultKey = " << static_cast<int64_t>(pTiling->get_defaultKey()) << "\n";
    stream << "tiling->defaultValue = " << static_cast<int64_t>(pTiling->get_defaultValue()) << "\n";
    stream << "tiling->filterKeyFlag = " << static_cast<int64_t>(pTiling->get_filterKeyFlag()) << "\n";
    stream << "tiling->filterKey = " << static_cast<int64_t>(pTiling->get_filterKey()) << "\n";
    stream << "tiling->threadXNum = " << static_cast<int64_t>(pTiling->get_threadXNum()) << "\n";
    stream << "tiling->threadYNum = " << static_cast<int64_t>(pTiling->get_threadYNum()) << "\n";
    OP_LOGI(context, "%s", stream.str().c_str());
}

inline bool IsLookupOrInsertOptDim(const int64_t embeddingDim)
{
    const std::unordered_set<int64_t> optEmbeddingDims = {1LL, 2LL, 4LL, 8LL, 16LL, 32LL};
    return optEmbeddingDims.count(embeddingDim) != 0;
}

inline uint32_t ComputeReservedUBSizeForSIMD(const gert::TilingContext* context, const uint32_t threadYNum)
{
    const uint32_t ubBlockSize = std::max<uint32_t>(Ops::Base::GetUbBlockSize(context), MIN_UB_BLOCK_SIZE);
    // 给aiv的每个Y线程（负责遍历查询/插入key的线程）预留一个B64大小，用于存储每个线程计数完的insertCounts
    uint32_t useBytes = threadYNum * sizeof(size_t);
    useBytes = Ops::Base::CeilAlign<uint32_t>(useBytes, ubBlockSize);
    return useBytes;
}

inline ge::graphStatus GetCheckLookupOrInsertInputs(
    const gert::TilingContext* context, LookupOrInsertTilingData* pTiling)
{
    // 获取shape信息
    const auto *inShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    const auto inShapeVal = inShape->GetStorageShape();
    int64_t keyNum = inShapeVal.GetShapeSize();
    pTiling->set_keyNum(keyNum);

    // 校验dtype
    auto *keyDtypePtr = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyDtypePtr);
    ge::DataType keyType = keyDtypePtr->GetDataType();
    if (keyType != ge::DT_INT64) {
        OP_LOGE(context->GetNodeName(), "Invalid datatype of input key, only support int64");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus GetSetLookupOrInsertAttrs(const gert::TilingContext* context, LookupOrInsertTilingData* pTiling)
{
    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto* size = attrs->GetAttrPointer<int64_t>(INPUT_BUCKET_SIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, size);
    auto* embeddingDim = attrs->GetAttrPointer<int64_t>(INPUT_EMBEDDINGDIM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, embeddingDim);
    auto* filterMode = attrs->GetAttrPointer<char>(INPUT_FILTER_MODE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterMode);
    auto* filterFreq = attrs->GetAttrPointer<int64_t>(INPUT_FILTER_FREQ_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterFreq);
    auto* defaultKeyOrValue = attrs->GetAttrPointer<bool>(INPUT_DEFAULT_KEY_OR_VALUE_IDX);
    auto* defaultKey = attrs->GetAttrPointer<int64_t>(INPUT_DEFAULT_KEY_IDX);
    auto* defaultValue = attrs->GetAttrPointer<float>(INPUT_DEFAULT_VALUE_IDX);
    auto* filterKeyFlag = attrs->GetAttrPointer<bool>(INPUT_FILTER_KEY_FLAG_IDX);
    auto* filterKey = attrs->GetAttrPointer<int64_t>(INPUT_filter_KEY_IDX);
    if (strcmp(filterMode, "counter") == 0) {
        OP_LOGI(context, "Filter mode takes effect");
        pTiling->set_filterMode(1);
    } else {
        pTiling->set_filterMode(0);
    }
    pTiling->set_size(static_cast<int64_t>(*size));
    pTiling->set_embeddingDim(static_cast<int64_t>(*embeddingDim));
    pTiling->set_filterFreq(static_cast<int64_t>(*filterFreq));
    pTiling->set_defaultKeyOrValue(defaultKeyOrValue == nullptr ? 0 : static_cast<uint32_t>(*defaultKeyOrValue));
    pTiling->set_defaultKey(defaultKey == nullptr ? 0 : static_cast<int64_t>(*defaultKey));
    pTiling->set_defaultValue(defaultValue == nullptr ? 0.0f : static_cast<float>(*defaultValue));
    pTiling->set_filterKeyFlag(filterKeyFlag == nullptr ? 0 : static_cast<uint32_t>(*filterKeyFlag));
    pTiling->set_filterKey(filterKey == nullptr ? -1 : static_cast<int64_t>(*filterKey));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4LookupOrInsert(gert::TilingContext* context)
{
    const EmbeddingHashTableLookupOrInsertCompileInfo* compileInfo =
        reinterpret_cast<const EmbeddingHashTableLookupOrInsertCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    LookupOrInsertTilingData tiling;
    ge::graphStatus ret;
    // 获取并校验输入Shape和Dtype，写入相关TilingData
    ret = GetCheckLookupOrInsertInputs(context, &tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // 获取输入属性并写入TilingData
    ret = GetSetLookupOrInsertAttrs(context, &tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // 设置TilingData中thread数相关，这里要求threadNum是2的幂次（1、2、4、8、16...）
    uint32_t threadNum = MAX_THREAD_NUM;
    uint32_t threadXNum = tiling.get_embeddingDim();
    threadXNum = nextPowerOf2(threadXNum);
    if (threadXNum > WARP_SIZE) {
        threadXNum = WARP_SIZE; // threadXNum是不超过WARP_SIZE的2的幂次
    }
    uint32_t threadYNum = threadNum / std::max<uint32_t>(threadXNum, 1u);
    tiling.set_threadXNum(threadXNum);
    tiling.set_threadYNum(threadYNum);
    // 设置tilingKey
    if (IsLookupOrInsertOptDim(tiling.get_embeddingDim())) {
        context->SetTilingKey(EMBEDDING_HASH_TABLE_LOOKUP_OR_INSERT_TILING_KEY_OPT_DIM);
    } else {
        context->SetTilingKey(EMBEDDING_HASH_TABLE_LOOKUP_OR_INSERT_TILING_KEY_GENERAL);
    }
    // 打日志
    OP_LOGI(context, "TilingKey = %ld\n", static_cast<int64_t>(context->GetTilingKey()));
    LogLookupOrInsertTilingData(context, &tiling);
    // 设置SIMD用的UB大小
    context->SetLocalMemorySize(ComputeReservedUBSizeForSIMD(context, threadYNum));
    // 设置启动核数
    context->SetBlockDim(std::min(
        static_cast<uint32_t>(CeilDiv<uint32_t>(static_cast<uint32_t>(tiling.get_keyNum()), threadYNum)),
        compileInfo->coreNumAiv));
    // 保存TilingData
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    // 设置Workspace大小
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4LookupOrInsert(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<EmbeddingHashTableLookupOrInsertCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    compileInfo->maxThreadNum = GetSimtMaxThreadNum(context);
    OP_CHECK_IF(
        (compileInfo->maxThreadNum <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to get valid maxThreadNum in TilingParse func."),
        return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNumAiv <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to get valid coreNumAiv in TilingParse func."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EmbeddingHashTableLookupOrInsert)
    .Tiling(Tiling4LookupOrInsert)
    .TilingParse<EmbeddingHashTableLookupOrInsertCompileInfo>(TilingPrepare4LookupOrInsert);
} // namespace optiling