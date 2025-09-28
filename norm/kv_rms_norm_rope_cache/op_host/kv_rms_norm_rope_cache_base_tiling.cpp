/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_rms_norm_rope_cache_tiling_base.cpp
 * \brief
 */
#include "kv_rms_norm_rope_cache_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {
std::tuple<int64_t, int64_t, int64_t, int64_t> KvRmsNormRopeCacheTilingBase::GetShapeTuple(
    const gert::TilingContext* context, const int64_t index)
{
    const gert::StorageShape* shapePtr = context->GetInputShape(index);
    OP_CHECK_IF(shapePtr == nullptr, OP_LOGE(context, "Shape is nullptr."), return std::make_tuple(0, 0, 0, 0));
    // check shape length is DIM_SIZE
    OP_CHECK_IF(
        shapePtr->GetStorageShape().GetDimNum() != DIM_SIZE, OP_LOGE(context, "Shape must be (B,N,S,D)."),
        return std::make_tuple(0, 0, 0, 0));
    return std::make_tuple(
        shapePtr->GetStorageShape().GetDim(SHAPE_IDX_B), shapePtr->GetStorageShape().GetDim(SHAPE_IDX_N),
        shapePtr->GetStorageShape().GetDim(SHAPE_IDX_S), shapePtr->GetStorageShape().GetDim(SHAPE_IDX_D));
}

bool KvRmsNormRopeCacheTilingBase::IsB1SD(const gert::TilingContext* context)
{
    auto kvShapeTuple = GetShapeTuple(context, KV_INDEX);
    auto cosShapeTuple = GetShapeTuple(context, COS_INDEX);
    int64_t seqLen = std::get<SHAPE_IDX_S>(kvShapeTuple);
    if (seqLen > 1 && std::get<SHAPE_IDX_S>(kvShapeTuple) == std::get<SHAPE_IDX_S>(cosShapeTuple)) {
        return true;
    } else {
        return false;
    }
}

bool KvRmsNormRopeCacheTilingBase::CheckKvValid(
    const gert::TilingContext* context, int64_t batchSize, int64_t numHead, int64_t seqLen, int64_t headSize)
{
    auto kvShapeTuple = GetShapeTuple(context, KV_INDEX);
    bool isValid = true;
    isValid = isValid && (std::get<SHAPE_IDX_B>(kvShapeTuple) == batchSize);
    isValid = isValid && (std::get<SHAPE_IDX_N>(kvShapeTuple) == numHead);
    isValid = isValid && (std::get<SHAPE_IDX_S>(kvShapeTuple) == seqLen);
    isValid = isValid && (std::get<SHAPE_IDX_D>(kvShapeTuple) == headSize);

    return isValid;
}

bool KvRmsNormRopeCacheTilingBase::CheckCosSinValid(
    const gert::TilingContext* context, int64_t batchSize, int64_t numHead, int64_t seqLen, int64_t headSize)
{
    auto cosShapeTuple = GetShapeTuple(context, COS_INDEX);
    auto sinShapeTuple = GetShapeTuple(context, SIN_INDEX);
    bool isValid = true;
    isValid = isValid && (std::get<SHAPE_IDX_B>(cosShapeTuple) == batchSize);
    isValid = isValid && (std::get<SHAPE_IDX_N>(cosShapeTuple) == numHead);
    isValid = isValid && (std::get<SHAPE_IDX_D>(cosShapeTuple) == headSize);
    isValid = isValid && (std::get<SHAPE_IDX_B>(sinShapeTuple) == batchSize);
    isValid = isValid && (std::get<SHAPE_IDX_N>(sinShapeTuple) == numHead);
    isValid = isValid && (std::get<SHAPE_IDX_D>(sinShapeTuple) == headSize);
    auto cosSeq = std::get<SHAPE_IDX_S>(cosShapeTuple);
    auto sinSeq = std::get<SHAPE_IDX_S>(sinShapeTuple);
    isValid = isValid && (cosSeq == sinSeq);
    if ((cosSeq != seqLen)) {
        if (cosSeq == 1) {
            cosSinNeedBrc_ = 1;
        } else {
            return false;
        }
    }

    if (currentCacheMode_ == CacheMode::PA_BLK_NZ || currentCacheMode_ == CacheMode::PA_NZ) {
        auto kcacheDesc = context_->GetInputDesc(K_CACHE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, kcacheDesc);
        ge::DataType kcacheDtype = kcacheDesc->GetDataType();
        if (kcacheDtype == ge::DT_INT8) {
            isValid = isValid && (std::get<SHAPE_IDX_D>(sinShapeTuple) % INT8_BLOCK_ALIGN_NUM == 0);
        } else {
            isValid = isValid && (std::get<SHAPE_IDX_D>(sinShapeTuple) % FP16_BLOCK_ALIGN_NUM == 0);
        }
    }
    return isValid;
}

bool KvRmsNormRopeCacheTilingBase::CheckGammaValid(const gert::TilingContext* context, int64_t headSize)
{
    const gert::StorageShape* gammaShapePtr = context->GetInputShape(GAMMA_INDEX);
    bool isValid = true;
    isValid = isValid && (gammaShapePtr != nullptr);
    isValid = isValid && (gammaShapePtr->GetStorageShape().GetDimNum() == 1);
    isValid = isValid && (gammaShapePtr->GetStorageShape().GetDim(0) == headSize);

    if (currentCacheMode_ == CacheMode::PA_BLK_NZ || currentCacheMode_ == CacheMode::PA_NZ) {
        auto vcacheDesc = context_->GetInputDesc(V_CACHE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, vcacheDesc);
        ge::DataType vcacheDtype = vcacheDesc->GetDataType();
        if (vcacheDtype == ge::DT_INT8) {
            isValid = isValid && (gammaShapePtr->GetStorageShape().GetDim(0) % INT8_BLOCK_ALIGN_NUM == 0);
        } else {
            isValid = isValid && (gammaShapePtr->GetStorageShape().GetDim(0) % FP16_BLOCK_ALIGN_NUM == 0);
        }
    }
    return isValid;
}

bool KvRmsNormRopeCacheTilingBase::CheckKCacheValid(
    const gert::TilingContext* context, int64_t batchSize, int64_t numHead, int64_t cacheLen, int64_t headSize)
{
    auto kCacheShapeTuple = GetShapeTuple(context, K_CACHE_INDEX);
    bool isValid = true;
    isValid = isValid && (std::get<SHAPE_IDX_B>(kCacheShapeTuple) == batchSize);
    isValid = isValid && (std::get<SHAPE_IDX_N>(kCacheShapeTuple) == numHead);
    isValid = isValid && (std::get<SHAPE_IDX_S>(kCacheShapeTuple) == cacheLen);
    isValid = isValid && (std::get<SHAPE_IDX_D>(kCacheShapeTuple) == headSize);
    return isValid;
}

bool KvRmsNormRopeCacheTilingBase::CheckVCacheValid(
    const gert::TilingContext* context, int64_t batchSize, int64_t numHead, int64_t cacheLen, int64_t headSize)
{
    auto vCacheShapeTuple = GetShapeTuple(context, V_CACHE_INDEX);
    bool isValid = true;
    isValid = isValid && (std::get<SHAPE_IDX_B>(vCacheShapeTuple) == batchSize);
    isValid = isValid && (std::get<SHAPE_IDX_N>(vCacheShapeTuple) == numHead);
    isValid = isValid && (std::get<SHAPE_IDX_S>(vCacheShapeTuple) == cacheLen);
    isValid = isValid && (std::get<SHAPE_IDX_D>(vCacheShapeTuple) == headSize);
    return isValid;
}

bool KvRmsNormRopeCacheTilingBase::CheckKCacheValidPA(
    const gert::TilingContext* context, int64_t numHead, int64_t headSize)
{
    auto kCacheShapeTuple = GetShapeTuple(context, K_CACHE_INDEX);
    bool isValid = true;
    isValid = isValid && (std::get<SHAPE_IDX_S>(kCacheShapeTuple) == numHead);
    isValid = isValid && (std::get<SHAPE_IDX_D>(kCacheShapeTuple) == headSize);

    auto kcacheDesc = context_->GetInputDesc(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kcacheDesc);
    ge::DataType kcacheDtype = kcacheDesc->GetDataType();
    if (kcacheDtype == ge::DT_INT8) {
        isValid = isValid && (std::get<SHAPE_IDX_N>(kCacheShapeTuple) % INT8_BLOCK_ALIGN_NUM == 0);
    } else {
        isValid = isValid && (std::get<SHAPE_IDX_N>(kCacheShapeTuple) % FP16_BLOCK_ALIGN_NUM == 0);
    }

    return isValid;
}

bool KvRmsNormRopeCacheTilingBase::CheckVCacheValidPA(
    const gert::TilingContext* context, int64_t numHead, int64_t headSize)
{
    auto vCacheShapeTuple = GetShapeTuple(context, V_CACHE_INDEX);
    bool isValid = true;
    isValid = isValid && (std::get<SHAPE_IDX_S>(vCacheShapeTuple) == numHead);
    isValid = isValid && (std::get<SHAPE_IDX_D>(vCacheShapeTuple) == headSize);

    auto vcacheDesc = context_->GetInputDesc(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vcacheDesc);
    ge::DataType vcacheDtype = vcacheDesc->GetDataType();
    if (vcacheDtype == ge::DT_INT8) {
        isValid = isValid && (std::get<SHAPE_IDX_N>(vCacheShapeTuple) % INT8_BLOCK_ALIGN_NUM == 0);
    } else {
        isValid = isValid && (std::get<SHAPE_IDX_N>(vCacheShapeTuple) % FP16_BLOCK_ALIGN_NUM == 0);
    }

    return isValid;
}

bool KvRmsNormRopeCacheTilingBase::CheckIndexValid(
    const gert::TilingContext* context, int64_t batchSize, int64_t seqLen, int64_t pageSize, CacheMode mode)
{
    bool isValid = true;
    const gert::StorageShape* indexShapePtr = context->GetInputShape(INDEX_INDEX);
    isValid = isValid && (indexShapePtr != nullptr);
    OP_CHECK_IF(!isValid, OP_LOGE(context, "Index shape pointer is null."), return false);
    auto shapeSize = indexShapePtr->GetStorageShape().GetShapeSize();
    if (mode == CacheMode::Norm || mode == CacheMode::PA || mode == CacheMode::PA_NZ) {
        isValid = isValid && (shapeSize == batchSize * seqLen);
        OP_CHECK_IF(!isValid, OP_LOGE(context, "Index's shape size must equal with B*S."), return false);
    } else {
        int64_t page_num = (seqLen + pageSize - 1) / pageSize;
        isValid = isValid && (shapeSize == batchSize * page_num);
        OP_CHECK_IF(
            !isValid, OP_LOGE(context, "Index's shape size must equal with batchSize * page_num."), return false);
    }
    return isValid;
}

int64_t KvRmsNormRopeCacheTilingBase::GetQuantMode(const gert::TilingContext* context)
{
    auto scale1Shape = context->GetOptionalInputShape(K_ROPE_SCALE_IDX);
    auto scale2Shape = context->GetOptionalInputShape(C_KV_SCALE_IDX);
    auto offset1Shape = context->GetOptionalInputShape(K_ROPE_OFFSET_IDX);
    auto offset2Shape = context->GetOptionalInputShape(C_KV_OFFSET_IDX);
    bool allNullPtr =
        (scale1Shape == nullptr) && (scale2Shape == nullptr) && (offset1Shape == nullptr) && (offset2Shape == nullptr);
    bool symmetricQuant = ((scale1Shape != nullptr) && (offset1Shape == nullptr)) ||
                          ((scale2Shape != nullptr) && (offset2Shape == nullptr));
    bool asymmetricQuant = ((scale1Shape != nullptr) && (offset1Shape != nullptr)) ||
                           ((scale2Shape != nullptr) && (offset2Shape != nullptr));

    if (allNullPtr) {
        return NON_QUANT_MODE;
    } else if (symmetricQuant) {
        return SYMMETRIC_QUANT_MODE;
    } else if (asymmetricQuant) {
        return ASYMMETRIC_QUANT_MODE;
    }
    return -1;
}

ge::graphStatus KvRmsNormRopeCacheTilingBase::GetPlatformInfo()
{
    vlFp32_ = Ops::Base::GetVRegSize(context_) / FLOAT32_BYTES;
    vlFp16_ = Ops::Base::GetVRegSize(context_) / FLOAT16_BYTES;
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<KvRmsNormRopeCacheCompileInfo>();
        OP_CHECK_IF(
            compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "CompileInfo is nullptr."),
            return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ubSize_ = ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvRmsNormRopeCacheTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        context_ == nullptr, OP_LOGE(context_->GetNodeName(), "context_ can not be nullptr."), return ge::GRAPH_FAILED);
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    isRegbase_ = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_95;
    // GetQuantMode
    quantMode_ = GetQuantMode(context_);
    // Basic info
    auto kvShapeTuple = GetShapeTuple(context_, KV_INDEX);
    auto kCacheShapeTuple = GetShapeTuple(context_, K_CACHE_INDEX);
    // Dk
    auto cosShapeTuple = GetShapeTuple(context_, COS_INDEX);
    // Dv
    const gert::StorageShape* gammaShapePtr = context_->GetInputShape(GAMMA_INDEX);
    kv_ = std::get<SHAPE_IDX_D>(kvShapeTuple);
    dv_ = gammaShapePtr->GetStorageShape().GetDim(0);
    dk_ = std::get<SHAPE_IDX_D>(cosShapeTuple);
    reciprocal_ = 1.0 / static_cast<float>(dv_);
    auto kvDesc = context_->GetInputDesc(KV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvDesc);
    kvDtype_ = kvDesc->GetDataType();
    if (kvDtype_ == ge::DT_FLOAT) {
        kvDtypeSize_ = FLOAT32_BYTES;
    } else if (kvDtype_ == ge::DT_FLOAT16 || kvDtype_ == ge::DT_BF16) {
        kvDtypeSize_ = FLOAT16_BYTES;
    }

    int64_t batchSize = std::get<SHAPE_IDX_B>(kvShapeTuple);
    int64_t numHead = std::get<SHAPE_IDX_N>(kvShapeTuple);
    int64_t seqLen = std::get<SHAPE_IDX_S>(kvShapeTuple);
    cacheLength_ = std::get<SHAPE_IDX_S>(kCacheShapeTuple);
    blockSize_ = std::get<SHAPE_IDX_BLOCK_SIZE>(kCacheShapeTuple);
    isMTP_ = (seqLen > 1);
    OP_CHECK_IF(batchSize < 1, OP_LOGE(context_->GetNodeName(), "batchSize should >= 1."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(seqLen < 1, OP_LOGE(context_->GetNodeName(), "seqLen should >= 1."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(numHead != 1, OP_LOGE(context_->GetNodeName(), "numHead should == 1."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckKvValid(context_, batchSize, numHead, seqLen, kv_),
        OP_LOGE(context_->GetNodeName(), "kv shape is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckCosSinValid(context_, batchSize, numHead, seqLen, dk_),
        OP_LOGE(context_->GetNodeName(), "cos or sin shape is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckGammaValid(context_, dv_), OP_LOGE(context_->GetNodeName(), "gamma shape is invalid."),
        return ge::GRAPH_FAILED);
    // Attrs info
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilon = attrs->GetFloat(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, epsilon);

    epsilon_ = *epsilon;
    const char* tmpmode = attrs->GetStr(CACHE_MODE_IDX);
    if (tmpmode != nullptr) {
        std::string cacheMode = tmpmode;
        isPagedAttention_ = (cacheMode == "PA" || cacheMode == "PA_BNSD");
        std::unordered_map<std::string, CacheMode> cacheModeMap = {
            {"PA", CacheMode::PA},
            {"PA_BNSD", CacheMode::PA},
            {"PA_NZ", CacheMode::PA_NZ},
            {"PA_BLK_BNSD", CacheMode::PA_BLK_BNSD},
            {"PA_BLK_NZ", CacheMode::PA_BLK_NZ}};
        auto getCacheMode = [&cacheModeMap](const std::string& mode) -> CacheMode {
            auto it = cacheModeMap.find(mode);
            return (it != cacheModeMap.end()) ? it->second : CacheMode::Norm;
        };
        currentCacheMode_ = getCacheMode(cacheMode);
    } else {
        isPagedAttention_ = false;
        currentCacheMode_ = CacheMode::Norm;
    }
    const bool* isOutputKv = attrs->GetBool(IS_OUTPUT_KV_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, isOutputKv);
    isOutputKv_ = *isOutputKv;
    OP_CHECK_IF(
        !CheckIndexValid(context_, batchSize, seqLen, blockSize_, currentCacheMode_),
        OP_LOGE(context_->GetNodeName(), "index shape is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

uint64_t KvRmsNormRopeCacheTilingBase::GetTilingKey() const
{
    return tilingKey_;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ge::graphStatus Tiling4KvRmsNormRopeCache(gert::TilingContext* context)
{
    OP_LOGD(context, "TilingForKvRmsNormRopeCache running.");
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4KvRmsNormRopeCache(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4KvRmsNormRopeCache running.");
    auto compileInfo = context->GetCompiledInfo<KvRmsNormRopeCacheCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        compileInfo->coreNum <= 0, OP_LOGE(context, "coreNum must be greater than 0."), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF(compileInfo->ubSize <= 0, OP_LOGE(context, "ubSize must be greater than 0."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "coreNum: %ld, ubSize: %ld", compileInfo->coreNum, compileInfo->ubSize);
    OP_LOGD(context, "TilingPrepare4KvRmsNormRopeCache success.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvRmsNormRopeCache)
    .Tiling(Tiling4KvRmsNormRopeCache)
    .TilingParse<KvRmsNormRopeCacheCompileInfo>(TilingPrepare4KvRmsNormRopeCache);

} // namespace optiling
