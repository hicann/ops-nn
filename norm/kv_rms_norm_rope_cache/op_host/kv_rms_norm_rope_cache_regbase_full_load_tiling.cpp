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
 * \file kv_rms_norm_rope_cache_regbase_full_load_tiling.cpp
 * \brief
 */
#include "kv_rms_norm_rope_cache_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

namespace optiling {
constexpr int64_t K_SCALE_OFFSET_UB_NUM = 4;
constexpr int64_t COS_SIN_UB_NUM = 4;
constexpr int64_t V_SCALE_OFFSET_UB_NUM = 2;
constexpr int64_t R_MAX_VALUE = 16384;
constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_THREE = 3;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_FIVE = 5;
constexpr static int64_t CONST_SIX = 6;
constexpr static int64_t CONST_SEVEN = 7;
constexpr static int64_t CONST_EIGHT = 8;
constexpr static int64_t CONST_SIXTY_THREE = 63;

constexpr static int64_t CONST_BRCFLAG_ZERO = 1;
constexpr static int64_t CONST_BRCFLAG_ONE = 1;
constexpr static int64_t CONST_BRCFLAG_TWO = 1;

using namespace Ops::Base;

bool KvRmsNormRopeCacheRegbaseFullLoadTiling::IsCapable()
{
    return isRegbase_;
}

bool KvRmsNormRopeCacheRegbaseFullLoadTiling::CheckScaleOffsetShape(
    const gert::StorageShape* inShape, int64_t lastDim, int64_t& brcFlag)
{
    if (inShape == nullptr) {
        brcFlag = CONST_BRCFLAG_ZERO;
        return true;
    }
    auto shapeSize = inShape->GetStorageShape().GetShapeSize();
    if (shapeSize == lastDim) {
        brcFlag = CONST_BRCFLAG_TWO;
        return true;
    }
    // Regbase模板支持Brc场景
    if (isRegbase_ && shapeSize == 1) {
        brcFlag = CONST_BRCFLAG_ONE;
        return true;
    }
    return false;
}

bool KvRmsNormRopeCacheRegbaseFullLoadTiling::CheckInputDtype()
{
    // kv dtype
    auto kvDesc = context_->GetInputDesc(KV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvDesc);
    ge::DataType kvDtype = kvDesc->GetDataType();
    OP_CHECK_IF(
        (kvDtype != ge::DT_FLOAT16 && kvDtype != ge::DT_BF16), OP_LOGE(context_->GetNodeName(), "kv dtype is invalid."),
        return false);

    // gamma dtype
    auto gammaDesc = context_->GetInputDesc(GAMMA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    ge::DataType gammaDtype = gammaDesc->GetDataType();
    OP_CHECK_IF((gammaDtype != kvDtype), OP_LOGE(context_->GetNodeName(), "gamma dtype is invalid."), return false);

    // cos dtype sin dtype
    auto cosDesc = context_->GetInputDesc(COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosDesc);
    ge::DataType cosDtype = cosDesc->GetDataType();
    auto sinDesc = context_->GetInputDesc(SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sinDesc);
    ge::DataType sinDtype = sinDesc->GetDataType();
    OP_CHECK_IF(
        ((sinDtype != cosDtype) || (sinDtype != kvDtype)),
        OP_LOGE(context_->GetNodeName(), "the dtype of sin or cos is invalid."), return false);

    // index dtype
    auto indexDesc = context_->GetInputDesc(INDEX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexDesc);
    ge::DataType indexDtype = indexDesc->GetDataType();
    OP_CHECK_IF(
        (indexDtype != ge::DT_INT64), OP_LOGE(context_->GetNodeName(), "the dtype of index is invalid."), return false);

    // k_cache dtype
    auto kcacheDesc = context_->GetInputDesc(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kcacheDesc);
    ge::DataType kcacheDtype = kcacheDesc->GetDataType();
    OP_CHECK_IF(
        ((kcacheDtype != kvDtype) && (kcacheDtype != ge::DT_INT8)),
        OP_LOGE(context_->GetNodeName(), "the dtype of k_cache is invalid."), return false);
    if (kcacheDtype == ge::DT_INT8) {
        // k_rope_scale
        auto kRopeScaleDesc = context_->GetOptionalInputDesc(K_ROPE_SCALE_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, kRopeScaleDesc);
        ge::DataType kRopeScaleDtype = kRopeScaleDesc->GetDataType();
        OP_CHECK_IF(
            kRopeScaleDtype != ge::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "the dtype of k_rope_scale is invalid."),
            return false);
        // k_rope_offset
        auto kRopeOffsetDesc = context_->GetOptionalInputDesc(K_ROPE_OFFSET_IDX);
        if (kRopeOffsetDesc != nullptr) {
            ge::DataType kRopeOffsetDtype = kRopeOffsetDesc->GetDataType();
            OP_CHECK_IF(
                kRopeOffsetDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "the dtype of k_rope_offset is invalid."), return false);
        }
    }

    // ckv_cache dtype
    auto vcacheDesc = context_->GetInputDesc(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vcacheDesc);
    ge::DataType vcacheDtype = vcacheDesc->GetDataType();
    OP_CHECK_IF(
        ((vcacheDtype != kvDtype) && (vcacheDtype != ge::DT_INT8)),
        OP_LOGE(context_->GetNodeName(), "the dtype of ckv_cache is invalid."), return false);
    if (vcacheDtype == ge::DT_INT8) {
        // c_kv_scale
        auto ckvScaleDesc = context_->GetOptionalInputDesc(C_KV_SCALE_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, ckvScaleDesc);
        ge::DataType ckvScaleDtype = ckvScaleDesc->GetDataType();
        OP_CHECK_IF(
            ckvScaleDtype != ge::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "the dtype of c_kv_scale is invalid."),
            return false);
        // v_kv_offset
        auto vKvOffsetDesc = context_->GetOptionalInputDesc(C_KV_OFFSET_IDX);
        if (vKvOffsetDesc != nullptr) {
            ge::DataType vKvOffsetDtype = vKvOffsetDesc->GetDataType();
            OP_CHECK_IF(
                vKvOffsetDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "the dtype of v_kv_offset is invalid."), return false);
        }
    }
    return true;
}

ge::graphStatus KvRmsNormRopeCacheRegbaseFullLoadTiling::DoOpTiling()
{
    auto kvShapeTuple = GetShapeTuple(context_, KV_INDEX);
    int64_t batchSize = std::get<SHAPE_IDX_B>(kvShapeTuple);
    int64_t seqLen = std::get<SHAPE_IDX_S>(kvShapeTuple);
    int64_t numHead = std::get<SHAPE_IDX_N>(kvShapeTuple);
    auto scale1Shape = context_->GetOptionalInputShape(K_ROPE_SCALE_IDX);
    auto scale2Shape = context_->GetOptionalInputShape(C_KV_SCALE_IDX);
    auto offset1Shape = context_->GetOptionalInputShape(K_ROPE_OFFSET_IDX);
    auto offset2Shape = context_->GetOptionalInputShape(C_KV_OFFSET_IDX);
    OP_CHECK_IF(
        !CheckScaleOffsetShape(scale1Shape, dk_, kScaleType_),
        OP_LOGE(context_->GetNodeName(), "k_rope_scale shape invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckScaleOffsetShape(scale2Shape, dv_, vScaleType_),
        OP_LOGE(context_->GetNodeName(), "c_kv_scale shape invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckScaleOffsetShape(offset1Shape, dk_, kOffsetType_),
        OP_LOGE(context_->GetNodeName(), "k_rope_scale shape invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckScaleOffsetShape(offset2Shape, dv_, vOffsetType_),
        OP_LOGE(context_->GetNodeName(), "c_kv_scale shape invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (dk_ % CONST_TWO) != 0, OP_LOGE(context_->GetNodeName(), "headsize of cos is invalid."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(numHead != 1, OP_LOGE(context_->GetNodeName(), "N must be one"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckInputDtype(), OP_LOGE(context_->GetNodeName(), "kvrmsnormrope dtype is invalid."),
        return ge::GRAPH_FAILED);
    if (currentCacheMode_ == CacheMode::Norm) {
        OP_CHECK_IF(
            !CheckKCacheValid(context_, batchSize, numHead, cacheLength_, dk_),
            OP_LOGE(context_->GetNodeName(), "k_cache shape invalid."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            !CheckVCacheValid(context_, batchSize, numHead, cacheLength_, dv_),
            OP_LOGE(context_->GetNodeName(), "ckv_cache shape invalid."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            !CheckKCacheValidPA(context_, numHead, dk_), OP_LOGE(context_->GetNodeName(), "k_cache shape invalid."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            !CheckVCacheValidPA(context_, numHead, dv_), OP_LOGE(context_->GetNodeName(), "ckv_cache shape invalid."),
            return ge::GRAPH_FAILED);
    }

    // N = 1
    int64_t bs = batchSize * seqLen * numHead;
    tilingData_.set_batchSize(batchSize);
    tilingData_.set_numHead(numHead);
    tilingData_.set_seqLength(seqLen);
    tilingData_.set_cacheLength(cacheLength_);
    tilingData_.set_dk(dk_);
    tilingData_.set_halfDk(dk_ / CONST_TWO);
    int64_t dkAlign = CeilAlign(dk_, static_cast<int64_t>(ubBlockSize_ / kvDtypeSize_));
    int64_t dkB8Align = CeilAlign(dk_, static_cast<int64_t>(ubBlockSize_ / sizeof(int8_t)));
    int64_t halfDkAlign = CeilAlign(tilingData_.get_halfDk(), (ubBlockSize_ / kvDtypeSize_));
    tilingData_.set_dkAlign(dkAlign);
    tilingData_.set_dkB8Align(dkB8Align);
    tilingData_.set_halfDkAlign(halfDkAlign);
    tilingData_.set_dv(dv_);
    int64_t dvAlign = CeilAlign(dv_, static_cast<int64_t>(ubBlockSize_ / kvDtypeSize_));
    tilingData_.set_dvAlign(dvAlign);
    int64_t dvB8Align = CeilAlign(dvAlign, static_cast<int64_t>(ubBlockSize_ / sizeof(int8_t)));
    tilingData_.set_dvB8Align(dvB8Align);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_reciprocal(reciprocal_);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_cosSinNeedBrc(cosSinNeedBrc_);
    tilingData_.set_kScaleType(kScaleType_);
    tilingData_.set_kOffsetType(kOffsetType_);
    tilingData_.set_vScaleType(vScaleType_);
    tilingData_.set_vOffsetType(vOffsetType_);
    tilingData_.set_cacheMode(currentCacheMode_);
    int64_t kScaleOffsetUbSize = kScaleType_ > 0 ? K_SCALE_OFFSET_UB_NUM * halfDkAlign * sizeof(float) : 0;
    int64_t vScaleOffsetUbSize = vScaleType_ > 0 ? V_SCALE_OFFSET_UB_NUM * dvAlign * sizeof(float) : 0;
    int64_t gammaUbSize = dvAlign * kvDtypeSize_;
    int64_t inCosSinUbSize = COS_SIN_UB_NUM * halfDkAlign * kvDtypeSize_;
    int64_t inUbSize = std::max(dkAlign, dvAlign) * kvDtypeSize_;
    int64_t kOutUbsize = 0;
    int64_t vOutUbsize = 0;
    // k量化场景
    if (kScaleType_ > 0) {
        if (isOutputKv_) {
            kOutUbsize = dkB8Align * kvDtypeSize_ + dkB8Align * sizeof(int8_t);
        } else {
            kOutUbsize = dkB8Align * sizeof(int8_t);
        }
    } else {
        kOutUbsize = dkAlign * kvDtypeSize_;
    }
    // v量化场景
    if (vScaleType_ > 0) {
        if (isOutputKv_) {
            vOutUbsize = dvB8Align * kvDtypeSize_ + dvB8Align * sizeof(int8_t);
        } else {
            vOutUbsize = dvB8Align * kvDtypeSize_ + dvB8Align * sizeof(int8_t);
        }
    } else {
        vOutUbsize = dvAlign * kvDtypeSize_;
    }
    int64_t outUbSize = std::max(kOutUbsize, vOutUbsize);
    int64_t ropeWspSize = 2 * vlFp32_ * sizeof(float);
    int64_t rmsNormWspSize = dvAlign * sizeof(float);
    int64_t ubFactor =
        (ubSize_ - UB_RESERVED_BYTE - kScaleOffsetUbSize - vScaleOffsetUbSize - gammaUbSize - ropeWspSize) /
        (DOUBLE_BUFFER * inUbSize + DOUBLE_BUFFER * inCosSinUbSize + DOUBLE_BUFFER * outUbSize + rmsNormWspSize);
    OP_CHECK_IF(
        (ubFactor <= 0),
        OP_LOGI(context_->GetNodeName(), "D full load template is not capable. dv is %ld, dk is %ld", dv_, dk_),
        return ge::GRAPH_PARAM_INVALID);
    tilingData_.set_inUbSize(inUbSize);
    tilingData_.set_outUbSize(outUbSize);
    tilingData_.set_rmsNormWspSize(rmsNormWspSize);

    int64_t blockFactor = (bs + coreNum_ - 1) / coreNum_;
    ubFactor = ubFactor < blockFactor ? ubFactor : blockFactor;
    usedCoreNum_ = (bs + blockFactor - 1) / blockFactor;
    tilingData_.set_blockFactor(blockFactor);
    tilingData_.set_ubFactor(ubFactor);
    int64_t outputKvKey = isOutputKv_ == true ? 1 : 0;
    tilingData_.set_isOutputKv(outputKvKey);
    tilingKey_ = FULL_LOAD_BASE_TILING_KEY;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvRmsNormRopeCacheRegbaseFullLoadTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(usedCoreNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("KvRmsNormRopeCache", KvRmsNormRopeCacheRegbaseFullLoadTiling, TEMPLATE_D_FULL_LOAD_PRIORITY);
} // namespace optiling
